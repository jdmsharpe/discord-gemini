"""Video generation helpers for the Gemini cog."""

import asyncio
import time
from io import BytesIO
from typing import TYPE_CHECKING

from discord import Attachment, Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types
from PIL import Image

from ...config.auth import SHOW_COST_EMBEDS
from ...util import (
    VIDEO_GENERATION_TIMEOUT,
    VideoGenerationParameters,
    calculate_video_cost,
    truncate_text,
)
from . import attachments, embeds, state

if TYPE_CHECKING:
    from .cog import GeminiCog


async def _generate_video_with_veo(
    cog: "GeminiCog",
    video_params: VideoGenerationParameters,
    attachment: Attachment | None = None,
    last_frame_attachment: Attachment | None = None,
) -> list[str]:
    """Generate videos using Veo models with generate_videos."""

    config_dict = video_params.to_dict()

    if last_frame_attachment:
        last_frame_data = await attachments._fetch_attachment_bytes(cog, last_frame_attachment)
        if last_frame_data:
            try:
                config_dict["last_frame"] = Image.open(BytesIO(last_frame_data))
            except Exception as error:
                cog.logger.warning("Failed to open last_frame attachment: %s", error)

    kwargs = {
        "model": video_params.model,
        "prompt": video_params.prompt,
        "config": types.GenerateVideosConfig(**config_dict),
    }

    if attachment:
        image_data = await attachments._fetch_attachment_bytes(cog, attachment)
        if image_data:
            try:
                kwargs["image"] = Image.open(BytesIO(image_data))
            except Exception as error:
                cog.logger.warning("Failed to open attachment for video generation: %s", error)

    operation = await cog.client.aio.models.generate_videos(**kwargs)
    cog.logger.info("Started video generation operation: %s", operation.name)

    start_time = time.time()
    poll_interval = 20
    while not operation.done:
        if time.time() - start_time > VIDEO_GENERATION_TIMEOUT:
            raise TimeoutError("Video generation timed out after 10 minutes")
        await asyncio.sleep(poll_interval)
        operation = await cog.client.aio.operations.get(operation)
        cog.logger.debug("Operation status: %s", operation.done)

    generated_videos: list[str] = []
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
                    video_path = f"temp_video_{index}.mp4"
                    generated_video.video.save(video_path)
                    generated_videos.append(video_path)
                    cog.logger.info("Downloaded video %d: %s", index + 1, video_path)
                except Exception as error:
                    cog.logger.error("Failed to download video %d: %s", index + 1, error)
    return generated_videos


async def _create_video_response_embed(
    cog: "GeminiCog",
    video_params: VideoGenerationParameters,
    generated_videos: list[str],
    attachment: Attachment | None,
) -> tuple[Embed, list[File]]:
    """Create the embed and files for generated videos."""

    files: list[File] = []
    for index, video_path in enumerate(generated_videos):
        try:
            files.append(File(video_path, filename=f"generated_video_{index + 1}.mp4"))
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
    person_generation: str,
    attachment: Attachment | None,
    last_frame: Attachment | None,
    number_of_videos: int,
    duration_seconds: int | None,
    negative_prompt: str | None,
    enhance_prompt: bool | None,
) -> None:
    """Run the `/gemini video` command."""

    await ctx.defer()
    try:
        for current_attachment in (attachment, last_frame):
            if current_attachment:
                validation_error = attachments._validate_attachment_size(current_attachment)
                if validation_error:
                    await ctx.send_followup(embed=embeds.build_error_embed(validation_error))
                    return

        if last_frame and "veo-3.1" not in model:
            await ctx.send_followup(
                embed=embeds.build_error_embed(
                    "The `last_frame` parameter is only supported on Veo 3.1 models."
                )
            )
            return

        video_params = VideoGenerationParameters(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            negative_prompt=negative_prompt,
            number_of_videos=number_of_videos,
            duration_seconds=duration_seconds,
            enhance_prompt=enhance_prompt,
            has_last_frame=last_frame is not None,
        )

        generated_videos = await _generate_video_with_veo(
            cog,
            video_params,
            attachment,
            last_frame,
        )

        if generated_videos:
            num_videos = len(generated_videos)
            est_duration = video_params.duration_seconds or 8
            cost = calculate_video_cost(model, est_duration, num_videos)
            daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
            cog._log_cost(
                "video",
                ctx.author.id,
                model,
                cost,
                daily_cost,
                videos=num_videos,
                duration_seconds=est_duration,
            )

            embed, files = await _create_video_response_embed(
                cog,
                video_params=video_params,
                generated_videos=generated_videos,
                attachment=attachment,
            )
            response_embeds = [embed]
            if SHOW_COST_EMBEDS:
                pricing_desc = (
                    f"${cost:.2f} · {num_videos} video{'s' if num_videos != 1 else ''} "
                    f"× {est_duration}s · daily ${daily_cost:.2f}"
                )
                response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))

            await ctx.send_followup(embeds=response_embeds, files=files)
            return

        await ctx.send_followup(
            embed=Embed(
                title="No Videos Generated",
                description=(
                    "The model did not generate any videos. This may be due to resource "
                    "constraints or safety filters. Please try again with a different prompt "
                    "or parameters."
                ),
                color=Colour.orange(),
            )
        )
    except Exception as error:
        await cog._send_error_followup(ctx, error, "video")


__all__ = [
    "_create_video_response_embed",
    "_generate_video_with_veo",
    "video_command",
]
