"""Music generation helpers for the Gemini cog."""

import asyncio
import contextlib
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from discord import Attachment, Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types
from PIL import Image

from ...util import (
    DEFAULT_MUSIC_MODEL,
    LYRIA_3_MODELS,
    LYRIA_REALTIME_MODEL,
    WS_DRAIN_INTERVAL,
    MusicGenerationParameters,
    truncate_text,
)
from . import attachments, embeds, state
from .client import build_lyria_realtime_client
from .responses import MusicGenerationError, _get_response_content_parts

if TYPE_CHECKING:
    from .cog import GeminiCog


def _build_lyria3_prompt(music_params: MusicGenerationParameters) -> str:
    """Translate structured music controls into natural-language guidance for Lyria 3."""

    weighted_prompts = music_params.to_weighted_prompts()
    prompt_blocks = []
    for weighted_prompt in weighted_prompts:
        text = weighted_prompt["text"]
        weight = weighted_prompt["weight"]
        if weight == 1.0 or len(weighted_prompts) == 1:
            prompt_blocks.append(text)
        else:
            prompt_blocks.append(f"{text} (importance {weight})")

    production_notes: list[str] = []
    if music_params.model == "lyria-3-clip-preview":
        production_notes.append("Generate a 30-second music clip.")
    if music_params.bpm is not None:
        production_notes.append(f"Tempo: {music_params.bpm} BPM.")
    if music_params.scale is not None:
        production_notes.append(f"Musical key or scale: {music_params.scale.replace('_', ' ')}.")
    if music_params.density is not None:
        production_notes.append(f"Density: {music_params.density} on a 0 to 1 scale.")
    if music_params.brightness is not None:
        production_notes.append(f"Brightness: {music_params.brightness} on a 0 to 1 scale.")
    if music_params.guidance != 4.0:
        production_notes.append(
            f"Prompt adherence target: {music_params.guidance} on a 0 to 6 scale."
        )

    return (
        "\n".join(prompt_blocks)
        + "\n\nProduction notes:\n"
        + "\n".join(f"- {note}" for note in production_notes)
    )


def _music_file_suffix_for_mime_type(mime_type: str | None) -> str:
    """Map an audio MIME type to a Discord-friendly file extension."""

    if mime_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return "wav"
    if mime_type == "audio/opus":
        return "opus"
    if mime_type == "audio/ogg":
        return "ogg"
    if mime_type == "audio/alaw":
        return "alaw"
    if mime_type == "audio/mulaw":
        return "mulaw"
    return "mp3"


def _build_music_notes_file(notes: str | None) -> File | None:
    """Attach long music notes separately instead of overloading the embed."""

    if not notes or len(notes) <= 500:
        return None
    return File(BytesIO(notes.encode("utf-8")), filename="music_notes.txt")


def _validate_music_attachment(
    cog: "GeminiCog",
    model: str,
    attachment: Attachment,
) -> str | None:
    """Validate an attachment for use with music generation."""

    size_error = attachments._validate_attachment_size(attachment)
    if size_error:
        return size_error
    if model == LYRIA_REALTIME_MODEL:
        return (
            "Reference images are only supported for Lyria 3 Pro Preview and Lyria 3 Clip Preview."
        )

    mime_type = attachments._guess_attachment_mime_type(attachment)
    if not mime_type.startswith("image/"):
        return "Music reference attachments must be image files."
    return None


async def _build_lyria3_music_contents(
    cog: "GeminiCog",
    music_params: MusicGenerationParameters,
    attachment: Attachment | None = None,
) -> str | list[Any]:
    """Build generate_content contents for Lyria 3 requests."""

    prompt = _build_lyria3_prompt(music_params)
    if attachment is None:
        return prompt

    attachment_data = await attachments._fetch_attachment_bytes(cog, attachment)
    if attachment_data is None:
        raise MusicGenerationError(
            "Failed to read the uploaded image for music generation. Please try again."
        )

    try:
        image = Image.open(BytesIO(attachment_data))
        image.load()
    except Exception as error:
        raise MusicGenerationError(
            "The uploaded music reference attachment could not be processed as an image."
        ) from error
    return [prompt, image]


async def _generate_music_with_lyria3(
    cog: "GeminiCog",
    music_params: MusicGenerationParameters,
    attachment: Attachment | None = None,
) -> tuple[bytes | None, str | None, str | None]:
    """Generate music using a Lyria 3 model via generate_content."""

    try:
        response = await cog.client.aio.models.generate_content(
            model=music_params.model,
            contents=await _build_lyria3_music_contents(cog, music_params, attachment),
            config=types.GenerateContentConfig(response_modalities=["AUDIO", "TEXT"]),
        )

        text_parts: list[str] = []
        audio_data: bytes | None = None
        mime_type: str | None = None
        response_parts = _get_response_content_parts(response) or []
        for part in response_parts:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                audio_data = inline_data.data
                mime_type = getattr(inline_data, "mime_type", None)

        return audio_data, "\n\n".join(text_parts) if text_parts else None, mime_type
    except Exception as error:
        cog.logger.error("Error generating music with Lyria 3: %s", error)
        raise MusicGenerationError(f"Music generation failed: {error}") from error


async def _generate_music_with_lyria_realtime(
    cog: "GeminiCog",
    music_params: MusicGenerationParameters,
) -> bytes | None:
    """Generate music using Gemini's Lyria RealTime model."""

    try:
        music_client = build_lyria_realtime_client()
        audio_chunks: list[bytes] = []
        stop_receiving = False

        async def receive_audio(session: Any) -> None:
            cog.logger.info("Audio receiver task started.")
            try:
                async for message in session.receive():
                    if stop_receiving:
                        cog.logger.info("Stop signal received, breaking from audio receiver")
                        break
                    if hasattr(message, "server_content") and hasattr(
                        message.server_content,
                        "audio_chunks",
                    ):
                        audio_data = message.server_content.audio_chunks[0].data
                        if audio_data:
                            audio_chunks.append(audio_data)
                            cog.logger.debug(
                                "Received audio chunk, size: %d bytes", len(audio_data)
                            )
            except asyncio.CancelledError:
                cog.logger.info("Audio receiver task cancelled")
                raise
            except Exception as error:
                cog.logger.error("Error in audio receiver: %s", error)
            finally:
                cog.logger.info("Audio receiver task finished.")

        try:
            async with music_client.aio.live.music.connect(
                model="models/lyria-realtime-exp"
            ) as session:
                receiver_task = asyncio.create_task(receive_audio(session))
                try:
                    await session.set_weighted_prompts(
                        prompts=[
                            types.WeightedPrompt(
                                text=prompt_data["text"],
                                weight=prompt_data["weight"],
                            )
                            for prompt_data in music_params.to_weighted_prompts()
                        ]
                    )

                    config_dict = music_params.to_music_config()
                    if music_params.scale:
                        scale_enum = getattr(types.Scale, music_params.scale, None)
                        if scale_enum:
                            config_dict["scale"] = scale_enum
                    await session.set_music_generation_config(
                        config=types.LiveMusicGenerationConfig(**config_dict)
                    )

                    await session.play()
                    cog.logger.info(
                        "Waiting %d seconds for music generation",
                        music_params.duration,
                    )
                    await asyncio.sleep(music_params.duration)
                    cog.logger.info("Stopping music generation session")
                    await session.stop()
                    stop_receiving = True
                finally:
                    if not receiver_task.done():
                        cog.logger.info("Cancelling receiver task")
                        receiver_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.sleep(WS_DRAIN_INTERVAL)

            if audio_chunks:
                cog.logger.info("Successfully collected %d audio chunks", len(audio_chunks))
                return b"".join(audio_chunks)

            cog.logger.warning("No audio chunks received from Lyria RealTime")
            return None
        except Exception as websocket_error:
            if "404" in str(websocket_error):
                cog.logger.error("Lyria RealTime endpoint not found (404).")
                raise MusicGenerationError(
                    "Music generation is currently unavailable. This could be due to:\n"
                    "1. The Lyria RealTime model is not available in your region\n"
                    "2. Your account doesn't have access to the music generation API\n"
                    "3. The service is temporarily unavailable\n\n"
                    "Please check Google AI Studio or contact support for more information."
                ) from websocket_error
            if "401" in str(websocket_error) or "403" in str(websocket_error):
                cog.logger.error("Authentication/authorization error for Lyria RealTime")
                raise MusicGenerationError(
                    "Authentication error: Please check your API key permissions for music generation."
                ) from websocket_error

            cog.logger.error("WebSocket connection error: %s", websocket_error)
            raise MusicGenerationError(f"Connection error: {websocket_error}") from websocket_error
    except MusicGenerationError:
        raise
    except Exception as error:
        cog.logger.error("Error generating music with Lyria RealTime: %s", error)
        raise MusicGenerationError(f"Music generation failed: {error}") from error


async def music_command(
    cog: "GeminiCog",
    ctx: ApplicationContext,
    prompt: str,
    attachment: Attachment | None,
    model: str = DEFAULT_MUSIC_MODEL,
    duration: int = 30,
    bpm: int | None = None,
    scale: str | None = None,
    density: float | None = None,
    brightness: float | None = None,
    guidance: float = 4.0,
) -> None:
    """Run the `/gemini music` command."""

    await ctx.defer()
    try:
        if attachment:
            validation_error = _validate_music_attachment(cog, model, attachment)
            if validation_error:
                await ctx.send_followup(embed=embeds.build_error_embed(validation_error))
                return

        music_params = MusicGenerationParameters(
            prompts=[prompt],
            model=model,
            duration=duration,
            bpm=bpm,
            scale=scale,
            guidance=guidance,
            density=density,
            brightness=brightness,
        )

        text_response = None
        audio_mime_type = "audio/wav"
        if model in LYRIA_3_MODELS:
            audio_data, text_response, audio_mime_type = await _generate_music_with_lyria3(
                cog,
                music_params,
                attachment,
            )
        else:
            audio_data = await _generate_music_with_lyria_realtime(cog, music_params)

        daily_cost = state._track_daily_cost(cog, ctx.author.id, 0.0)
        log_details: dict[str, Any] = {}
        if model == LYRIA_REALTIME_MODEL:
            log_details["duration_seconds"] = duration
        cog._log_cost("music", ctx.author.id, model, 0.0, daily_cost, **log_details)

        if not audio_data:
            await ctx.send_followup(
                embed=Embed(
                    title="No Music Generated",
                    description=(
                        "The model did not generate any music. Please try again with a "
                        "different prompt or parameters."
                    ),
                    color=Colour.orange(),
                )
            )
            return

        suffix = (
            "wav"
            if model == LYRIA_REALTIME_MODEL
            else _music_file_suffix_for_mime_type(audio_mime_type)
        )
        audio_file_path = Path(f"music_output_{int(time.time())}.{suffix}")
        if model == LYRIA_REALTIME_MODEL:
            with wave.open(str(audio_file_path), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(audio_data)
        else:
            audio_file_path.write_bytes(audio_data)

        truncated_prompt = truncate_text(prompt, 2000)
        description = f"**Prompt:** {truncated_prompt}\n"
        description += f"**Model:** {model}\n"
        if attachment:
            description += "**Reference Image:** Attached\n"
        if model == "lyria-3-clip-preview":
            description += "**Mode:** Clip generation\n**Duration:** 30 seconds (fixed by model)\n"
        elif model == "lyria-3-pro-preview":
            description += "**Mode:** Song generation\n"
        else:
            description += (
                f"**Mode:** Real-time instrumental generation\n**Duration:** {duration} seconds\n"
            )
        if bpm is not None:
            description += f"**BPM:** {bpm}\n"
        if scale is not None:
            description += f"**Scale:** {scale.replace('_', ' ')}\n"
        if density is not None:
            description += f"**Density:** {density}\n"
        if brightness is not None:
            description += f"**Brightness:** {brightness}\n"
        description += f"**Guidance:** {guidance}\n"
        if model == LYRIA_REALTIME_MODEL:
            description += "**Format:** WAV (48kHz, 16-bit, Stereo)\n"
        else:
            description += f"**Format:** {suffix.upper()}\n"
            if text_response:
                description += f"**Lyrics / Notes:** {truncate_text(text_response, 500)}\n"
            if (
                any(value is not None for value in (bpm, scale, density, brightness))
                or guidance != 4.0
            ):
                description += (
                    "*Advanced controls were translated into prompt guidance for Lyria 3.*\n"
                )

        notes_file = _build_music_notes_file(text_response)
        files = [File(audio_file_path)]
        if notes_file is not None:
            files.append(notes_file)

        await ctx.send_followup(
            embed=Embed(
                title="Music Generation", description=description, color=embeds.GEMINI_BLUE
            ),
            files=files,
        )
        audio_file_path.unlink(missing_ok=True)
    except MusicGenerationError as error:
        cog.logger.error("Music generation error: %s", error, exc_info=True)
        description = str(error)
        if len(description) > 4000:
            description = description[:4000] + "\n\n... (error message truncated)"
        await ctx.send_followup(
            embed=Embed(
                title="Music Generation Error",
                description=description,
                color=Colour.orange(),
            )
        )
    except Exception as error:
        await cog._send_error_followup(ctx, error, "music")


__all__ = [
    "_build_lyria3_music_contents",
    "_build_lyria3_prompt",
    "_build_music_notes_file",
    "_generate_music_with_lyria3",
    "_generate_music_with_lyria_realtime",
    "_music_file_suffix_for_mime_type",
    "_validate_music_attachment",
    "music_command",
]
