"""Video generation helpers for the Gemini cog."""

import asyncio
import base64
import binascii
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

from discord import Attachment, Colour, Embed, File
from discord.commands import ApplicationContext
from PIL import Image

from ...config.auth import SHOW_COST_EMBEDS
from ...cost_line import count_label, format_cost_line
from ...util import (
    VIDEO_GENERATION_TIMEOUT,
    VideoGenerationParameters,
    calculate_omni_video_cost,
    truncate_text,
)
from . import attachments, embeds, state
from .embed_delivery import send_embed_batches
from .image import _upload_limit_for
from .responses import APICallError

if TYPE_CHECKING:
    from .cog import GeminiCog

# Gemini Omni generates video via the Interactions API. Every id in OMNI_VIDEO_MODELS
# takes this code path. gemini-omni-1.2-flash is the default; gemini-omni-1.1-flash
# stays selectable.
# The request runs in background mode and is polled every OMNI_VIDEO_POLL_INTERVAL
# seconds (see _generate_video_with_omni). Omni renders 720p unless a resolution is
# requested. Billing differs by model: 1.2 bills video tokens per second of output
# (so the duration and resolution change the cost), while 1.1 bills a flat token
# count per clip. No duration is derived from the token count; the cost line shows
# the requested duration, or the model's default length when none was requested.
DEFAULT_OMNI_VIDEO_MODEL = "gemini-omni-1.2-flash"
OMNI_VIDEO_MODELS = frozenset({"gemini-omni-1.2-flash", "gemini-omni-1.1-flash"})
OMNI_VIDEO_POLL_INTERVAL = 5
OMNI_DEFAULT_VIDEO_RESOLUTION = "720p"
# Length the model produces when no duration is sent.
OMNI_DEFAULT_VIDEO_DURATION_SECONDS = 10
OMNI_MIN_VIDEO_DURATION_SECONDS = 3
# Longest clip each model accepts. First and last frame interpolation is capped lower.
OMNI_MAX_VIDEO_DURATION_SECONDS: dict[str, int] = {
    "gemini-omni-1.2-flash": 15,
    "gemini-omni-1.1-flash": 10,
}
OMNI_MAX_INTERPOLATION_DURATION_SECONDS = 10
# Average MP4 size per second of video at each resolution. Busy scenes encode larger,
# so these are estimates, used to refuse a request that cannot fit the server's
# upload limit before any paid call is made.
OMNI_VIDEO_BYTES_PER_SECOND: dict[str, int] = {
    "360p": 450_847,
    "720p": 756_522,
    "1080p": 1_018_427,
    "4k": 2_705_897,
}
# Prompt tags from the Omni prompt guide that bind the input images to their roles:
# the first image part is the starting frame, the second the final frame. Without
# them the model decides how to use an image and may treat it as a subject
# reference instead of a frame. `<LAST_FRAME>` is only valid together with
# `<FIRST_FRAME>`.
OMNI_FIRST_FRAME_TAG = "<FIRST_FRAME>"
OMNI_LAST_FRAME_TAG = "<LAST_FRAME>"
# Frame images go inline as base64 in the request body, so an image larger than
# this is refused before it is downloaded or any request is made.
OMNI_MAX_FRAME_IMAGE_BYTES = 20 * 1024 * 1024
# Interaction statuses that end polling (mirrors research.py).
_OMNI_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "requires_action", "budget_exceeded"}
)

VIDEO_SUPPORTED_RESOLUTIONS: dict[str, frozenset[str]] = {
    # 1.2 renders all four resolutions (1080p and 4k are upsampled) and bills them per
    # second of output.
    # 1.1 is limited to the two resolutions it was verified at; its flat per-clip
    # billing has no 360p or 4k rate.
    "gemini-omni-1.2-flash": frozenset({"360p", "720p", "1080p", "4k"}),
    "gemini-omni-1.1-flash": frozenset({"720p", "1080p"}),
}


def _attachment_image_mime_type(data: bytes, attachment: Attachment) -> str:
    """Image MIME type of an attachment: Discord's content type, else what PIL detects.

    Raises when the bytes are not an image PIL can identify.
    """

    mime_type = (attachment.content_type or "").split(";")[0].strip().lower()
    if not mime_type.startswith("image/"):
        with Image.open(BytesIO(data)) as probe:
            detected = probe.format
        if not detected:
            raise ValueError("Could not determine the image format of the attachment.")
        mime_type = f"image/{detected.lower()}"
    return mime_type


def _interaction_error_message(interaction: Any) -> str:
    """Join the `errors[].message` entries of a terminal interaction (may be empty)."""

    messages = [
        str(getattr(error, "message", None) or "")
        for error in getattr(interaction, "errors", None) or []
    ]
    return "; ".join(message for message in messages if message)


@dataclass
class OmniVideoUsage:
    """Token counts of one Omni interaction, split by the rate each is billed at.

    ``input_tokens`` is the prompt plus any input images ($1.50/M);
    ``video_output_tokens`` is the video-modality output (the model's video rate); and
    ``text_output_tokens`` is thinking plus any non-video output ($9.00/M, the text
    rate, which the pricing page says includes thinking tokens).
    """

    input_tokens: int = 0
    video_output_tokens: int = 0
    text_output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def output_tokens(self) -> int:
        """Every billed output token: video, non-video output and thinking."""

        return self.video_output_tokens + self.text_output_tokens


def _extract_omni_video_usage(interaction: Any) -> OmniVideoUsage:
    """Read the billed token counts from a completed interaction's ``usage``.

    ``total_input_tokens`` covers the prompt text and the input images.
    ``total_output_tokens`` includes the video tokens and excludes thinking, which
    the Interactions API reports separately as ``total_thought_tokens``. The
    remainder of ``total_output_tokens`` after the video tokens (several hundred
    tokens per clip) has no modality entry in ``output_tokens_by_modality``; it is
    billed at the text output rate, the only non-video output rate the pricing page
    lists.
    """

    usage = getattr(interaction, "usage", None)
    video_output_tokens = 0
    for modality in getattr(usage, "output_tokens_by_modality", None) or []:
        if getattr(modality, "modality", None) == "video":
            video_output_tokens += int(getattr(modality, "tokens", 0) or 0)
    total_output_tokens = int(getattr(usage, "total_output_tokens", 0) or 0)
    thinking_tokens = int(getattr(usage, "total_thought_tokens", 0) or 0)
    return OmniVideoUsage(
        input_tokens=int(getattr(usage, "total_input_tokens", 0) or 0),
        video_output_tokens=video_output_tokens,
        text_output_tokens=max(total_output_tokens - video_output_tokens, 0) + thinking_tokens,
        thinking_tokens=thinking_tokens,
    )


async def _build_omni_video_input(
    cog: "GeminiCog",
    prompt: str,
    attachment: Attachment | None,
    last_frame: Attachment | None,
) -> str | list[dict[str, Any]]:
    """Build the Interactions ``input`` for an Omni video request.

    Text-to-video sends the prompt string. With a first-frame ``attachment`` (and
    optionally a ``last_frame``), each image goes in as a base64 ``image`` part, in
    that order, followed by the prompt as a ``text`` part that starts with the
    ``<FIRST_FRAME>`` (and ``<LAST_FRAME>``) tags the prompt does not already
    contain. An image that cannot be read stops the request before the paid call.
    """

    if attachment is None:
        return prompt

    parts: list[dict[str, Any]] = []
    for option, frame in (("attachment", attachment), ("last_frame", last_frame)):
        if frame is None:
            continue
        data = await attachments._fetch_attachment_bytes(cog, frame)
        if not data:
            raise ValueError(f"Could not download the `{option}` image. Please try again.")
        try:
            mime_type = _attachment_image_mime_type(data, frame)
        except Exception as error:
            raise ValueError(f"The `{option}` attachment is not a readable image.") from error
        parts.append(
            {
                "type": "image",
                "data": base64.b64encode(data).decode("ascii"),
                "mime_type": mime_type,
            }
        )

    parts.append({"type": "text", "text": _tag_omni_frame_prompt(prompt, last_frame is not None)})
    return parts


def _tag_omni_frame_prompt(prompt: str, has_last_frame: bool) -> str:
    """Prefix the frame tags the prompt does not already contain.

    `<FIRST_FRAME>` is added when missing, and `<LAST_FRAME>` is added when there is a
    last frame and the prompt lacks it. A tag the prompt already places is never
    added a second time.
    """

    missing = [] if OMNI_FIRST_FRAME_TAG in prompt else [OMNI_FIRST_FRAME_TAG]
    if has_last_frame and OMNI_LAST_FRAME_TAG not in prompt:
        missing.append(OMNI_LAST_FRAME_TAG)
    return f"{' '.join(missing)} {prompt}" if missing else prompt


def _decode_omni_video_data(raw: Any) -> bytes | None:
    """Decode the base64 ``output_video.data`` the polled interaction carries.

    Strict decoding runs first. When it fails (for example on line breaks inside the
    payload), the lenient decoder that skips characters outside the base64 alphabet
    runs before the clip is given up.
    """

    if isinstance(raw, bytes | bytearray):
        return bytes(raw) or None
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return base64.b64decode(raw, validate=True) or None
    except (binascii.Error, ValueError):
        pass
    try:
        return base64.b64decode(raw) or None
    except (binascii.Error, ValueError):
        return None


async def _download_omni_video_uri(cog: "GeminiCog", uri: str) -> tuple[bytes | None, str | None]:
    """Download the MP4 behind an ``output_video.uri`` through the Files API.

    Returns the bytes and ``None``, or ``None`` and the reason the download failed.
    """

    match = re.search(r"/files/([^:/?]+)", uri)
    if not match:
        return None, "could not parse a file name from the URI"
    try:
        video_bytes = await asyncio.to_thread(
            cog.client.files.download, file=f"files/{match.group(1)}"
        )
    except Exception as error:
        return None, f"download failed: {error}"
    if not video_bytes:
        return None, "download returned no bytes"
    return video_bytes, None


async def _generate_video_with_omni(
    cog: "GeminiCog",
    video_params: VideoGenerationParameters,
    attachment: Attachment | None = None,
    last_frame: Attachment | None = None,
) -> tuple[list[bytes], OmniVideoUsage]:
    """Generate a video via the Interactions API (Gemini Omni).

    The request is submitted with ``background=True`` and polled through
    ``interactions.get`` every ``OMNI_VIDEO_POLL_INTERVAL`` seconds until it reaches a
    terminal status, bounded by ``VIDEO_GENERATION_TIMEOUT`` exactly like the Veo
    poller. A synchronous ``interactions.create`` on ``gemini-omni-1.1-flash`` is
    closed server-side before a typical generation finishes, with no interaction id
    returned, so background mode is required.

    ``attachment`` is sent as the first frame (image-to-video) and ``last_frame`` as
    the final frame (first and last frame interpolation); see
    ``_build_omni_video_input``.

    The completed interaction's ``output_video`` carries either a ``uri`` (downloaded
    via ``files.download`` without waiting for the file to become ACTIVE) or the MP4
    as base64 in ``data``. Google documents that ``interactions.get`` returns the
    base64 ``data`` even for ``delivery: "uri"``, and that the ``uri`` is only
    guaranteed on the creation response, so the bytes are decoded from ``data`` when
    there is no ``uri`` or its download fails. Returns the video bytes and the
    interaction's billed token counts.
    """

    response_format: dict[str, Any] = {
        "type": "video",
        "aspect_ratio": video_params.aspect_ratio,
        "delivery": "uri",
    }
    if video_params.resolution:
        response_format["resolution"] = video_params.resolution
    if video_params.duration is not None:
        response_format["duration"] = f"{video_params.duration}s"

    video_input = await _build_omni_video_input(cog, video_params.prompt, attachment, last_frame)

    # interactions.create/get return Interaction | AsyncStream; this path never
    # streams, so narrow to Any to keep .id/.status access well-typed.
    interaction: Any = await cog.client.aio.interactions.create(
        model=video_params.model,
        input=video_input,
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

    usage = _extract_omni_video_usage(interaction)
    output_video = getattr(interaction, "output_video", None)
    uri = getattr(output_video, "uri", None)
    video_bytes: bytes | None = None
    download_error: str | None = None
    source = "uri"
    if uri:
        video_bytes, download_error = await _download_omni_video_uri(cog, uri)
        if download_error:
            cog.logger.warning(
                "Omni video URI %s could not be downloaded (%s); trying the inline data",
                uri,
                download_error,
            )
    if video_bytes is None:
        video_bytes = _decode_omni_video_data(getattr(output_video, "data", None))
        source = "inline base64 data"
    if video_bytes is None:
        if uri:
            cog.logger.error(
                "Omni video URI %s could not be downloaded (%s) and the interaction "
                "carried no decodable inline data (%d video tokens billed)",
                uri,
                download_error,
                usage.video_output_tokens,
            )
        else:
            cog.logger.error(
                "Omni interaction returned no video URI and no decodable inline data "
                "(status=%s, %d video tokens billed)",
                getattr(interaction, "status", "?"),
                usage.video_output_tokens,
            )
        return [], usage

    cog.logger.info(
        "Received Omni video from %s (%d bytes, %d video tokens)",
        source,
        len(video_bytes),
        usage.video_output_tokens,
    )
    return [video_bytes], usage


def _validate_omni_video_request(
    video_params: VideoGenerationParameters,
    attachment: Attachment | None,
    last_frame: Attachment | None,
) -> str | None:
    """Reject options Gemini Omni has no equivalent for.

    Applies to every id in `OMNI_VIDEO_MODELS`, each of which must have a
    `VIDEO_SUPPORTED_RESOLUTIONS` and an `OMNI_MAX_VIDEO_DURATION_SECONDS` entry. Omni
    (Interactions API) takes an aspect ratio, a `resolution`, a `duration`, a
    first-frame image `attachment` (image-to-video) and a `last_frame` image (first and
    last frame interpolation). A `last_frame` needs an `attachment`: the prompt guide's
    `<LAST_FRAME>` tag is only valid together with `<FIRST_FRAME>`. A frame image over
    `OMNI_MAX_FRAME_IMAGE_BYTES` (checked from Discord's `size`, before any download) is
    refused. Negative prompts (the Omni docs say to write negatives into the prompt
    instead), person-generation control, resize modes, prompt enhancement and multiple
    videos are Veo-only.

    `resolution` must be one the model's `VIDEO_SUPPORTED_RESOLUTIONS` entry lists.
    `duration` must be at least `OMNI_MIN_VIDEO_DURATION_SECONDS` (the API rejects
    shorter clips), at most the model's maximum, and at most
    `OMNI_MAX_INTERPOLATION_DURATION_SECONDS` when a `last_frame` is given.
    """

    if video_params.model not in OMNI_VIDEO_MODELS:
        return "Choose a supported Gemini Omni video model."
    supported_resolutions = VIDEO_SUPPORTED_RESOLUTIONS.get(video_params.model, frozenset())
    if video_params.resolution and video_params.resolution not in supported_resolutions:
        supported_list = ", ".join(sorted(supported_resolutions, key=lambda value: int(value[:-1])))
        return (
            f"The `{video_params.resolution}` resolution is not yet supported for "
            f"Gemini Omni. Supported values on `{video_params.model}`: {supported_list}."
        )
    duration = video_params.duration
    if duration is not None:
        maximum = OMNI_MAX_VIDEO_DURATION_SECONDS[video_params.model]
        if duration < OMNI_MIN_VIDEO_DURATION_SECONDS:
            return f"`duration` must be at least {OMNI_MIN_VIDEO_DURATION_SECONDS} seconds."
        if duration > maximum:
            return f"`duration` can be at most {maximum} seconds on `{video_params.model}`."
        if last_frame and duration > OMNI_MAX_INTERPOLATION_DURATION_SECONDS:
            return (
                f"With `last_frame`, `duration` can be at most "
                f"{OMNI_MAX_INTERPOLATION_DURATION_SECONDS} seconds."
            )
    if last_frame and not attachment:
        return (
            "On Gemini Omni, `last_frame` needs an `attachment` as the first frame. "
            "Add a first-frame image, or remove `last_frame`."
        )
    for option, frame in (("attachment", attachment), ("last_frame", last_frame)):
        size = getattr(frame, "size", None)
        if isinstance(size, int) and size > OMNI_MAX_FRAME_IMAGE_BYTES:
            return (
                f"The `{option}` image is {size / (1024 * 1024):.1f} MB. Gemini Omni frame "
                f"images are sent inline and must be "
                f"{OMNI_MAX_FRAME_IMAGE_BYTES // (1024 * 1024)} MB or smaller."
            )
    return None


def _estimate_omni_video_bytes(resolution: str | None, duration: int | None) -> int:
    """Estimated MP4 size for a request, from the average bytes per second."""

    rate = OMNI_VIDEO_BYTES_PER_SECOND[resolution or OMNI_DEFAULT_VIDEO_RESOLUTION]
    return rate * (duration or OMNI_DEFAULT_VIDEO_DURATION_SECONDS)


def _megabytes(size: float) -> float:
    return size / (1024 * 1024)


def _upload_limit_error(video_params: VideoGenerationParameters, upload_limit: int) -> str | None:
    """Refuse a request whose estimated file would exceed the reply's upload limit.

    Google bills the clip even when Discord cannot take the file, and the Files API
    link needs the API key, so there is nothing to give the user afterwards.
    """

    resolution = video_params.resolution or OMNI_DEFAULT_VIDEO_RESOLUTION
    duration = video_params.duration or OMNI_DEFAULT_VIDEO_DURATION_SECONDS
    estimate = _estimate_omni_video_bytes(resolution, duration)
    if estimate <= upload_limit:
        return None
    message = (
        f"A {duration}-second {resolution} video is estimated at about "
        f"{_megabytes(estimate):.1f} MB, which is over this server's upload limit of "
        f"{_megabytes(upload_limit):g} MB. "
    )
    longest = upload_limit // OMNI_VIDEO_BYTES_PER_SECOND[resolution]
    if longest >= OMNI_MIN_VIDEO_DURATION_SECONDS:
        return message + (
            f"Try a duration of {longest} seconds or less at {resolution}, or a lower resolution."
        )
    return message + (
        f"Even {OMNI_MIN_VIDEO_DURATION_SECONDS} seconds at {resolution} would not fit. "
        "Try a lower resolution."
    )


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
    else:
        description += "**Mode:** Text-to-Video\n"
    description += f"**Number of Videos:** {len(generated_videos)}"
    if video_params.aspect_ratio:
        description += f"\n**Aspect Ratio:** {video_params.aspect_ratio}"
    if video_params.resolution:
        description += f"\n**Resolution:** {video_params.resolution}"
    description += (
        f"\n**Duration:** {video_params.duration or OMNI_DEFAULT_VIDEO_DURATION_SECONDS}s"
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
    attachment: Attachment | None,
    last_frame: Attachment | None,
    duration: int | None = None,
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
            has_last_frame=last_frame is not None,
            duration=duration,
        )

        validation_error = _validate_omni_video_request(video_params, attachment, last_frame)
        if validation_error:
            await send_embed_batches(
                ctx.send_followup,
                embed=embeds.build_error_embed(validation_error),
                logger=cog.logger,
            )
            return

        upload_limit = _upload_limit_for(ctx)
        limit_error = _upload_limit_error(video_params, upload_limit)
        if limit_error:
            await send_embed_batches(
                ctx.send_followup,
                embed=embeds.build_error_embed(limit_error),
                logger=cog.logger,
            )
            return

        generated_videos, omni_usage = await _generate_video_with_omni(
            cog, video_params, attachment, last_frame
        )

        # A video over the upload limit is billed but cannot be attached. The count of
        # generated videos is kept so the cost line still reports what was billed.
        generated_count = len(generated_videos)
        too_large_bytes = max(
            (len(video) for video in generated_videos if len(video) > upload_limit), default=None
        )
        if too_large_bytes is not None:
            generated_videos = []

        delivered = bool(generated_videos)
        # Google bills a completed Omni clip whether or not the bot can deliver it.
        billed_undelivered = not delivered and (
            omni_usage.video_output_tokens > 0 or too_large_bytes is not None
        )
        if not delivered and not billed_undelivered:
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="No Videos Generated",
                    description=(
                        "The model did not generate any videos. This may be due to resource "
                        "constraints or safety filters. Please try again with a different "
                        "prompt or parameters."
                    ),
                    color=Colour.orange(),
                ),
                logger=cog.logger,
            )
            return

        num_videos = generated_count
        cost = calculate_omni_video_cost(
            model,
            omni_usage.video_output_tokens,
            input_tokens=omni_usage.input_tokens,
            text_output_tokens=omni_usage.text_output_tokens,
        )
        logged_resolution = video_params.resolution or OMNI_DEFAULT_VIDEO_RESOLUTION
        logged_duration = video_params.duration or OMNI_DEFAULT_VIDEO_DURATION_SECONDS
        cost_details: dict[str, Any] = {
            "video_tokens": omni_usage.video_output_tokens,
            "input_tokens": omni_usage.input_tokens,
            "text_output_tokens": omni_usage.text_output_tokens,
            "thinking_tokens": omni_usage.thinking_tokens,
            "delivered": delivered,
            "too_large": too_large_bytes is not None,
        }
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "video",
            ctx.author.id,
            model,
            cost,
            daily_cost,
            videos=num_videos,
            resolution=logged_resolution,
            duration_seconds=logged_duration,
            **cost_details,
        )

        files: list[File] = []
        if delivered:
            embed, files = await _create_video_response_embed(
                cog,
                video_params=video_params,
                generated_videos=generated_videos,
                attachment=attachment,
            )
        elif too_large_bytes is not None:
            embed = Embed(
                title="Video Too Large to Upload",
                description=(
                    f"Gemini Omni generated a {_megabytes(too_large_bytes):.1f} MB video and "
                    f"Google billed it, but this server's upload limit is "
                    f"{_megabytes(upload_limit):g} MB. Try a shorter duration or a lower "
                    "resolution."
                ),
                color=Colour.orange(),
            )
        else:
            embed = Embed(
                title="Video Not Delivered",
                description=(
                    "Gemini Omni generated the video and Google billed it, but the bot could "
                    "not retrieve the file from the API. Please try again."
                ),
                color=Colour.orange(),
            )
        response_embeds = [embed]
        if SHOW_COST_EMBEDS:
            video_details = [count_label(num_videos, "video")]
            video_details.extend([logged_resolution, f"{logged_duration}s"])
            pricing_desc = format_cost_line(
                cost,
                daily_cost,
                input_tokens=omni_usage.input_tokens or None,
                output_tokens=omni_usage.output_tokens or None,
                thinking_tokens=omni_usage.thinking_tokens,
                details=video_details,
            )
            response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))

        await send_embed_batches(
            ctx.send_followup,
            embeds=response_embeds,
            files=files or None,
            logger=cog.logger,
        )
    except Exception as error:
        await cog._send_error_followup(ctx, error, "video")


__all__ = [
    "OmniVideoUsage",
    "_create_video_response_embed",
    "_generate_video_with_omni",
    "_validate_omni_video_request",
    "video_command",
]
