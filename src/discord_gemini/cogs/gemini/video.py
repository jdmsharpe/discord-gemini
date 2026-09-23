"""Video generation helpers for the Gemini cog."""

import asyncio
import re
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any

from discord import Attachment, Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types
from PIL import Image

from ...config.auth import SHOW_COST_EMBEDS
from ...util import (
    VIDEO_GENERATION_TIMEOUT,
    VideoGenerationParameters,
    calculate_omni_video_cost,
    calculate_video_cost,
    truncate_text,
)
from . import attachments, embeds, state
from .embed_delivery import send_embed_batches
from .responses import APICallError

if TYPE_CHECKING:
    from .cog import GeminiCog

# Gemini Omni generates video via the Interactions API (not the Veo
# generate_videos path). Every id in OMNI_VIDEO_MODELS takes this code path; the
# GA gemini-omni-1.1-flash, the default video model, is the only one.
# The request runs in background mode and is polled every OMNI_VIDEO_POLL_INTERVAL
# seconds (see _generate_video_with_omni). Omni renders 720p unless a resolution is
# requested. No duration is ever derived from the token count: it does not scale
# with length (a 3 s 1080p clip billed the same 57,920 video tokens as a
# default-length 720p clip, live probe 2026-08-28).
DEFAULT_OMNI_VIDEO_MODEL = "gemini-omni-1.1-flash"
OMNI_VIDEO_MODELS = frozenset({"gemini-omni-1.1-flash"})
OMNI_VIDEO_POLL_INTERVAL = 5
OMNI_DEFAULT_VIDEO_RESOLUTION = "720p"
# Interaction statuses that end polling (mirrors research.py).
_OMNI_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "requires_action", "budget_exceeded"}
)

VEO_3_1_MODELS = frozenset(
    {
        "veo-3.1-lite-generate-preview",
        "veo-3.1-generate-preview",
        "veo-3.1-fast-generate-preview",
    }
)


VIDEO_SUPPORTED_RESOLUTIONS: dict[str, frozenset[str]] = {
    "veo-3.1-lite-generate-preview": frozenset({"720p", "1080p"}),
    "veo-3.1-generate-preview": frozenset({"720p", "1080p", "4k"}),
    "veo-3.1-fast-generate-preview": frozenset({"720p", "1080p", "4k"}),
    # Omni 1.1 (GA) honoured 1080p with a real 1920x1080 avc1 MP4 (live probe
    # 2026-08-28). Its docs also list 360p and 4k (upscaled), but neither was probed
    # and no per-resolution price exists, so they are refused as not yet supported.
    "gemini-omni-1.1-flash": frozenset({"720p", "1080p"}),
}


def _build_veo_image(data: bytes, attachment: Attachment) -> types.Image:
    """Wrap raw attachment bytes in a `types.Image` for the Veo image inputs.

    A `PIL.Image` cannot be used here: pydantic coerces it to an all-`None`
    `types.Image` without raising, which the API rejects with "Input instance with
    image should contain both bytesBase64Encoded and mimeType".
    """

    mime_type = (attachment.content_type or "").split(";")[0].strip().lower()
    if not mime_type.startswith("image/"):
        with Image.open(BytesIO(data)) as probe:
            detected = probe.format
        if not detected:
            raise ValueError("Could not determine the image format of the attachment.")
        mime_type = f"image/{detected.lower()}"
    return types.Image(image_bytes=data, mime_type=mime_type)


def _validate_video_request(
    *,
    model: str,
    aspect_ratio: str,
    resolution: str | None,
    number_of_videos: int,
    duration_seconds: int | None,
    has_last_frame: bool,
) -> str | None:
    """Validate request combinations against the current Veo model limits."""
    if has_last_frame and model not in VEO_3_1_MODELS:
        return "The `last_frame` parameter is only supported on Veo 3.1 models."

    if number_of_videos != 1:
        return "The `number_of_videos` parameter only supports `1` on Veo 3.x and Veo 3.1 models."

    if duration_seconds is not None and duration_seconds not in {4, 6, 8}:
        return "This Veo model only supports 4, 6, or 8 second videos."

    if resolution is not None:
        supported_resolutions = VIDEO_SUPPORTED_RESOLUTIONS.get(model, frozenset())
        if resolution not in supported_resolutions:
            supported_list = (
                ", ".join(sorted(supported_resolutions)) or "no explicit resolution values"
            )
            return f"The `{resolution}` resolution is not supported on `{model}`. Supported values: {supported_list}."

        if resolution == "4k" and aspect_ratio != "16:9":
            return "4k output is only supported with a `16:9` aspect ratio."

        if resolution in {"1080p", "4k"} and duration_seconds not in (None, 8):
            return f"The `{resolution}` resolution only supports 8 second videos."

    if has_last_frame and duration_seconds not in (None, 8):
        return "Interpolation with `last_frame` only supports 8 second videos on Veo 3.1 models."

    return None


async def _generate_video_with_veo(
    cog: "GeminiCog",
    video_params: VideoGenerationParameters,
    attachment: Attachment | None = None,
    last_frame_attachment: Attachment | None = None,
) -> list[bytes]:
    """Generate videos using Veo models with generate_videos.

    Returns the raw bytes of each generated video held in memory. Nothing is
    written to disk, so concurrent invocations cannot collide and no temporary
    files are ever left behind.
    """

    config_dict = video_params.to_dict()

    if last_frame_attachment:
        last_frame_data = await attachments._fetch_attachment_bytes(cog, last_frame_attachment)
        if last_frame_data:
            try:
                config_dict["last_frame"] = _build_veo_image(last_frame_data, last_frame_attachment)
            except Exception as error:
                cog.logger.warning("Failed to open last_frame attachment: %s", error)

    # The prompt/image arguments are deprecated since google-genai 2.14.0; the
    # inputs go through `source` instead. `last_frame` stays on the config.
    source_image: types.Image | None = None
    if attachment:
        image_data = await attachments._fetch_attachment_bytes(cog, attachment)
        if image_data:
            try:
                source_image = _build_veo_image(image_data, attachment)
            except Exception as error:
                cog.logger.warning("Failed to open attachment for video generation: %s", error)

    operation = await cog.client.aio.models.generate_videos(
        model=video_params.model,
        source=types.GenerateVideosSource(prompt=video_params.prompt, image=source_image),
        config=types.GenerateVideosConfig(**config_dict),
    )
    cog.logger.info("Started video generation operation: %s", operation.name)

    start_time = time.time()
    poll_interval = 20
    while not operation.done:
        if time.time() - start_time > VIDEO_GENERATION_TIMEOUT:
            raise TimeoutError("Video generation timed out after 10 minutes")
        await asyncio.sleep(poll_interval)
        operation = await cog.client.aio.operations.get(operation)
        cog.logger.debug("Operation status: %s", operation.done)

    generated_videos: list[bytes] = []
    if (
        hasattr(operation, "response")
        and operation.response
        and hasattr(operation.response, "generated_videos")
        and operation.response.generated_videos
    ):
        for index, generated_video in enumerate(operation.response.generated_videos):
            if hasattr(generated_video, "video") and generated_video.video:
                try:
                    await asyncio.to_thread(cog.client.files.download, file=generated_video.video)
                    video_bytes = generated_video.video.video_bytes
                    if video_bytes:
                        generated_videos.append(video_bytes)
                        cog.logger.info(
                            "Downloaded video %d (%d bytes)", index + 1, len(video_bytes)
                        )
                    else:
                        cog.logger.error("Video %d downloaded but contained no bytes", index + 1)
                except Exception as error:
                    cog.logger.error("Failed to download video %d: %s", index + 1, error)
    return generated_videos


def _interaction_error_message(interaction: Any) -> str:
    """Join the `errors[].message` entries of a terminal interaction (may be empty)."""

    messages = [
        str(getattr(error, "message", None) or "")
        for error in getattr(interaction, "errors", None) or []
    ]
    return "; ".join(message for message in messages if message)


async def _generate_video_with_omni(
    cog: "GeminiCog",
    video_params: VideoGenerationParameters,
) -> tuple[list[bytes], int]:
    """Generate a video via the Interactions API (Gemini Omni).

    The request is submitted with ``background=True`` and polled through
    ``interactions.get`` every ``OMNI_VIDEO_POLL_INTERVAL`` seconds until it reaches a
    terminal status, bounded by ``VIDEO_GENERATION_TIMEOUT`` exactly like the Veo
    poller. A synchronous ``interactions.create`` on ``gemini-omni-1.1-flash`` is
    closed server-side before a typical generation finishes, with no interaction id
    returned, so background mode is required. The completed interaction carries a URI
    to the generated MP4 (downloaded via ``files.download`` without waiting for the
    file to become ACTIVE) plus exact video-modality output-token usage. Returns the
    downloaded video bytes and the video output-token count (for exact costing).
    """

    response_format: dict[str, Any] = {
        "type": "video",
        "aspect_ratio": video_params.aspect_ratio,
        "delivery": "uri",
    }
    if video_params.resolution:
        response_format["resolution"] = video_params.resolution

    # interactions.create/get return Interaction | AsyncStream; this path never
    # streams, so narrow to Any to keep .id/.status access well-typed.
    interaction: Any = await cog.client.aio.interactions.create(
        model=video_params.model,
        input=video_params.prompt,
        response_format=response_format,
        background=True,
    )
    cog.logger.info("Started Omni video generation: %s", interaction.id)

    start_time = time.time()
    while interaction.status not in _OMNI_TERMINAL_STATUSES:
        if time.time() - start_time > VIDEO_GENERATION_TIMEOUT:
            raise TimeoutError("Video generation timed out after 10 minutes")
        await asyncio.sleep(OMNI_VIDEO_POLL_INTERVAL)
        interaction = await cog.client.aio.interactions.get(interaction.id)
        cog.logger.debug("Omni video %s status: %s", interaction.id, interaction.status)

    if interaction.status != "completed":
        summary = {
            "failed": "Omni video generation failed",
            "cancelled": "Omni video generation was cancelled",
        }.get(interaction.status, f"Omni video generation stopped ({interaction.status})")
        detail = _interaction_error_message(interaction)
        raise APICallError(f"{summary}: {detail}" if detail else f"{summary}.")

    video_output_tokens = 0
    usage = getattr(interaction, "usage", None)
    for modality in getattr(usage, "output_tokens_by_modality", None) or []:
        if getattr(modality, "modality", None) == "video":
            video_output_tokens = getattr(modality, "tokens", 0) or 0

    generated_videos: list[bytes] = []
    output_video = getattr(interaction, "output_video", None)
    uri = getattr(output_video, "uri", None)
    if not uri:
        cog.logger.error(
            "Omni interaction returned no video URI (status=%s)",
            getattr(interaction, "status", "?"),
        )
        return generated_videos, video_output_tokens

    match = re.search(r"/files/([^:/?]+)", uri)
    if not match:
        cog.logger.error("Could not parse file name from Omni video URI: %s", uri)
        return generated_videos, video_output_tokens

    file_name = f"files/{match.group(1)}"
    try:
        video_bytes = await asyncio.to_thread(cog.client.files.download, file=file_name)
        if video_bytes:
            generated_videos.append(video_bytes)
            cog.logger.info(
                "Downloaded Omni video (%d bytes, %d video tokens)",
                len(video_bytes),
                video_output_tokens,
            )
        else:
            cog.logger.error("Omni video downloaded but contained no bytes")
    except Exception as error:
        cog.logger.error("Failed to download Omni video: %s", error)
    return generated_videos, video_output_tokens


def _validate_omni_video_request(
    video_params: VideoGenerationParameters,
    attachment: Attachment | None,
    last_frame: Attachment | None,
) -> str | None:
    """Reject Veo-only options Gemini Omni does not support.

    Applies to every id in `OMNI_VIDEO_MODELS`, each of which must have a
    `VIDEO_SUPPORTED_RESOLUTIONS` entry. Omni (Interactions API) is exposed here as
    text-to-video with an aspect ratio and a `resolution`: duration, negative prompts,
    person-generation control, image/first-or-last-frame inputs, multiple videos, and
    resize modes are Veo-only.

    `resolution` must be one the model's `VIDEO_SUPPORTED_RESOLUTIONS` entry lists. The
    Omni docs also list `360p` and `4k` (upscaled), but those are unverified and have no
    per-resolution pricing row, so they are refused as not yet supported.

    `duration` is deliberately NOT exposed although the GA id accepts it (`"<n>s"`
    format, minimum 3 s — `1s` 400s with "less than the minimum allowed 3s"): a 3 s
    1080p clip was billed 57,920 video tokens, identical to a default-length 720p clip
    (2026-08-28), so a shorter clip saves nothing.
    """

    unsupported: list[str] = []
    supported_resolutions = VIDEO_SUPPORTED_RESOLUTIONS.get(video_params.model, frozenset())
    if video_params.resolution and video_params.resolution not in supported_resolutions:
        supported_list = ", ".join(sorted(supported_resolutions, key=lambda value: int(value[:-1])))
        return (
            f"The `{video_params.resolution}` resolution is not yet supported for "
            f"Gemini Omni. Supported values on `{video_params.model}`: {supported_list}."
        )
    if video_params.duration_seconds is not None:
        unsupported.append("`duration`")
    if video_params.negative_prompt:
        unsupported.append("`negative_prompt`")
    if video_params.number_of_videos and video_params.number_of_videos > 1:
        unsupported.append("`number_of_videos` > 1")
    if video_params.person_generation and video_params.person_generation != "allow_adult":
        unsupported.append("`person_generation`")
    if video_params.image_resize_mode:
        unsupported.append("`image_resize_mode`")
    if attachment:
        unsupported.append("image `attachment`")
    if last_frame:
        unsupported.append("`last_frame`")

    if unsupported:
        joined = ", ".join(unsupported)
        return (
            "Gemini Omni supports text-to-video with an `aspect_ratio` and a `resolution`. "
            f"Remove {joined}, or choose a Veo 3.1 model for those features."
        )
    return None


async def _create_video_response_embed(
    cog: "GeminiCog",
    video_params: VideoGenerationParameters,
    generated_videos: list[bytes],
    attachment: Attachment | None,
) -> tuple[Embed, list[File]]:
    """Create the embed and files for generated videos."""

    files: list[File] = []
    for index, video_bytes in enumerate(generated_videos):
        try:
            files.append(File(BytesIO(video_bytes), filename=f"generated_video_{index + 1}.mp4"))
        except Exception as error:
            cog.logger.error("Failed to create file for video %d: %s", index + 1, error)

    truncated_prompt = truncate_text(video_params.prompt, 2000)
    description = f"**Prompt:** {truncated_prompt}\n"
    description += f"**Model:** {video_params.model}\n"
    if attachment and video_params.has_last_frame:
        description += "**Mode:** Interpolation (First + Last Frame)\n"
    elif attachment:
        description += "**Mode:** Image-to-Video\n"
    elif video_params.has_last_frame:
        description += "**Mode:** Last Frame Constrained\n"
    else:
        description += "**Mode:** Text-to-Video\n"
    description += f"**Number of Videos:** {len(generated_videos)}"
    if video_params.number_of_videos and video_params.number_of_videos > len(generated_videos):
        description += f" (requested: {video_params.number_of_videos})"
    if video_params.aspect_ratio:
        description += f"\n**Aspect Ratio:** {video_params.aspect_ratio}"
    if video_params.resolution:
        description += f"\n**Resolution:** {video_params.resolution}"
    if video_params.image_resize_mode and (attachment or video_params.has_last_frame):
        description += f"\n**Image Resize Mode:** {video_params.image_resize_mode}"
    if video_params.person_generation and video_params.person_generation != "allow_adult":
        description += f"\n**Person Generation:** {video_params.person_generation}"
    if video_params.duration_seconds:
        description += f"\n**Duration:** {video_params.duration_seconds} seconds"
    if video_params.negative_prompt:
        description += f"\n**Negative Prompt:** {video_params.negative_prompt}"
    if video_params.enhance_prompt is not None:
        description += (
            f"\n**Prompt Enhancement:** {'Enabled' if video_params.enhance_prompt else 'Disabled'}"
        )

    return (
        Embed(title="Video Generation", description=description, color=embeds.GEMINI_BLUE),
        files,
    )


async def video_command(
    cog: "GeminiCog",
    ctx: ApplicationContext,
    prompt: str,
    model: str,
    aspect_ratio: str,
    resolution: str | None,
    person_generation: str,
    attachment: Attachment | None,
    last_frame: Attachment | None,
    number_of_videos: int,
    duration_seconds: int | None,
    negative_prompt: str | None,
    enhance_prompt: bool | None,
    image_resize_mode: str | None = None,
) -> None:
    """Run the `/gemini-media video` command."""

    await ctx.defer()
    try:
        for current_attachment in (attachment, last_frame):
            if current_attachment:
                validation_error = attachments._validate_attachment_size(current_attachment)
                if validation_error:
                    await send_embed_batches(
                        ctx.send_followup,
                        embed=embeds.build_error_embed(validation_error),
                        logger=cog.logger,
                    )
                    return

        video_params = VideoGenerationParameters(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            person_generation=person_generation,
            negative_prompt=negative_prompt,
            number_of_videos=number_of_videos,
            duration_seconds=duration_seconds,
            enhance_prompt=enhance_prompt,
            has_last_frame=last_frame is not None,
            image_resize_mode=image_resize_mode,
        )

        is_omni = model in OMNI_VIDEO_MODELS
        validation_error = (
            _validate_omni_video_request(video_params, attachment, last_frame)
            if is_omni
            else _validate_video_request(
                model=model,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                number_of_videos=number_of_videos,
                duration_seconds=duration_seconds,
                has_last_frame=last_frame is not None,
            )
        )
        if validation_error:
            await send_embed_batches(
                ctx.send_followup,
                embed=embeds.build_error_embed(validation_error),
                logger=cog.logger,
            )
            return

        omni_video_tokens = 0
        if is_omni:
            generated_videos, omni_video_tokens = await _generate_video_with_omni(cog, video_params)
        else:
            generated_videos = await _generate_video_with_veo(
                cog,
                video_params,
                attachment,
                last_frame,
            )

        if generated_videos:
            num_videos = len(generated_videos)
            est_duration = video_params.duration_seconds or 8
            if is_omni:
                # Exact from usage. The token count says nothing about the clip
                # length, so no duration is derived or shown for Omni.
                cost = calculate_omni_video_cost(model, omni_video_tokens)
                logged_resolution = video_params.resolution or OMNI_DEFAULT_VIDEO_RESOLUTION
                cost_details: dict[str, Any] = {"video_tokens": omni_video_tokens}
            else:
                cost = calculate_video_cost(
                    model,
                    est_duration,
                    num_videos,
                    resolution=video_params.resolution,
                )
                logged_resolution = video_params.resolution or "model_default"
                cost_details = {"duration_seconds": est_duration}
            daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
            cog._log_cost(
                "video",
                ctx.author.id,
                model,
                cost,
                daily_cost,
                videos=num_videos,
                resolution=logged_resolution,
                **cost_details,
            )

            embed, files = await _create_video_response_embed(
                cog,
                video_params=video_params,
                generated_videos=generated_videos,
                attachment=attachment,
            )
            response_embeds = [embed]
            if SHOW_COST_EMBEDS:
                if is_omni:
                    pricing_desc = (
                        f"${cost:.2f} · {num_videos} video{'s' if num_videos != 1 else ''} "
                        f"· {logged_resolution} · {omni_video_tokens:,} video tokens "
                        f"· daily ${daily_cost:.2f}"
                    )
                else:
                    pricing_desc = (
                        f"${cost:.2f} · {num_videos} video{'s' if num_videos != 1 else ''} "
                        f"× {est_duration}s · daily ${daily_cost:.2f}"
                    )
                    if video_params.resolution:
                        pricing_desc = (
                            f"${cost:.2f} · {num_videos} video{'s' if num_videos != 1 else ''} "
                            f"× {est_duration}s · {video_params.resolution} · daily ${daily_cost:.2f}"
                        )
                response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))

            await send_embed_batches(
                ctx.send_followup,
                embeds=response_embeds,
                files=files,
                logger=cog.logger,
            )
            return

        await send_embed_batches(
            ctx.send_followup,
            embed=Embed(
                title="No Videos Generated",
                description=(
                    "The model did not generate any videos. This may be due to resource "
                    "constraints or safety filters. Please try again with a different prompt "
                    "or parameters."
                ),
                color=Colour.orange(),
            ),
            logger=cog.logger,
        )
    except Exception as error:
        await cog._send_error_followup(ctx, error, "video")


__all__ = [
    "_create_video_response_embed",
    "_generate_video_with_omni",
    "_generate_video_with_veo",
    "_validate_omni_video_request",
    "_validate_video_request",
    "video_command",
]
