"""Speech generation helpers for the Gemini cog."""

import wave
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from discord import Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types

from ...config.auth import SHOW_COST_EMBEDS
from ...cost_line import format_cost_line
from ...util import TTS_SPEECH_METADATA_MODELS, SpeechGenerationParameters, calculate_tts_cost
from . import embeds, state, usage
from .client import disable_afc
from .embed_delivery import send_embed_batches

if TYPE_CHECKING:
    from .cog import GeminiCog

# Headerless PCM from the pre-3.8 TTS models: 24 kHz, 16-bit, mono.
PCM_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1


def _build_speech_contents(tts_params: SpeechGenerationParameters) -> str | list[types.Content]:
    """Build the TTS request contents, placing the style prompt where the model reads it.

    Gemini 3.8 TTS models (`TTS_SPEECH_METADATA_MODELS`) take it as the part's
    `speech_metadata.style`, because they read the text as a verbatim transcript.
    Older models take it as a `"<style>: <text>"` prefix.
    """

    if not tts_params.style_prompt:
        return tts_params.input_text
    if tts_params.model in TTS_SPEECH_METADATA_MODELS:
        return [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=tts_params.input_text,
                        speech_metadata=types.SpeechMetadata(style=tts_params.style_prompt),
                    )
                ],
            )
        ]
    return f"{tts_params.style_prompt}: {tts_params.input_text}"


def _to_wav(audio_data: bytes) -> bytes:
    """Return a WAV file: RIFF data unchanged, headerless PCM wrapped in a WAV header.

    Gemini 3.8 TTS returns `audio/wav` that already has a RIFF header; wrapping it
    again would put a second header inside the audio data.
    """

    if audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE":
        return audio_data
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(audio_data)
    return buffer.getvalue()


def _describe_wav_format(wav_data: bytes) -> str:
    """Describe a WAV file's sample rate, sample width and channel count."""

    try:
        with wave.open(BytesIO(wav_data), "rb") as wav_file:
            rate = wav_file.getframerate()
            bits = wav_file.getsampwidth() * 8
            channels = wav_file.getnchannels()
    except (wave.Error, EOFError):
        return "WAV"
    channel_label = "Mono" if channels == 1 else f"{channels} channels"
    return f"WAV ({rate / 1000:g}kHz, {bits}-bit, {channel_label})"


async def _generate_speech_with_gemini(
    cog: "GeminiCog",
    tts_params: SpeechGenerationParameters,
) -> tuple[bytes | None, int, int]:
    """Generate speech using Gemini TTS models."""

    response = await cog.client.aio.models.generate_content(
        model=tts_params.model,
        contents=cast(Any, _build_speech_contents(tts_params)),
        config=types.GenerateContentConfig(
            **cast(Any, tts_params.to_dict()),
            automatic_function_calling=disable_afc(),
        ),
    )

    usage_counts = usage.extract_usage_counts(response)
    input_tokens = usage_counts.input_tokens
    output_tokens = usage_counts.output_tokens

    if (
        response.candidates
        and len(response.candidates) > 0
        and response.candidates[0].content
        and response.candidates[0].content.parts
    ):
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                return part.inline_data.data, input_tokens, output_tokens

    cog.logger.warning("No audio data found in Gemini TTS response")
    return None, input_tokens, output_tokens


async def tts_command(
    cog: "GeminiCog",
    ctx: ApplicationContext,
    input_text: str,
    model: str,
    voice_name: str,
    style_prompt: str | None,
) -> None:
    """Run the `/gemini-tools tts` command."""

    await ctx.defer()
    try:
        tts_params = SpeechGenerationParameters(
            input_text=input_text,
            model=model,
            voice_name=voice_name,
            multi_speaker=False,
            style_prompt=style_prompt,
        )

        audio_data, input_tokens, output_tokens = await _generate_speech_with_gemini(
            cog, tts_params
        )
        cost = calculate_tts_cost(model, input_tokens, output_tokens)
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "tts",
            ctx.author.id,
            model,
            cost,
            daily_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if not audio_data:
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="No Audio Generated",
                    description=(
                        "The model did not generate any audio. Please try again with different "
                        "text or parameters."
                    ),
                    color=Colour.orange(),
                ),
                logger=cog.logger,
            )
            return

        wav_data = _to_wav(audio_data)

        description = f"**Text:** {input_text[:500]}{'...' if len(input_text) > 500 else ''}\n"
        description += f"**Model:** {model}\n"
        description += f"**Voice:** {voice_name}\n"
        if style_prompt:
            description += f"**Style Instructions:** {style_prompt}\n"
        description += f"**Format:** {_describe_wav_format(wav_data)}\n"

        response_embeds = [
            Embed(
                title="Text-to-Speech Generation",
                description=description,
                color=embeds.GEMINI_BLUE,
            )
        ]
        if SHOW_COST_EMBEDS:
            pricing_desc = format_cost_line(
                cost, daily_cost, input_tokens=input_tokens, output_tokens=output_tokens
            )
            response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))

        await send_embed_batches(
            ctx.send_followup,
            embeds=response_embeds,
            file=File(BytesIO(wav_data), filename=f"tts_output_{voice_name}.wav"),
            logger=cog.logger,
        )
    except Exception as error:
        await cog._send_error_followup(ctx, error, "tts")


__all__ = ["_generate_speech_with_gemini", "tts_command"]
