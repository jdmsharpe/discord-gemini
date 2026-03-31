"""Speech generation helpers for the Gemini cog."""

import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from discord import Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types

from ...config.auth import SHOW_COST_EMBEDS
from ...util import SpeechGenerationParameters, calculate_tts_cost
from . import embeds, state, usage

if TYPE_CHECKING:
    from .cog import GeminiCog


async def _generate_speech_with_gemini(
    cog: "GeminiCog",
    tts_params: SpeechGenerationParameters,
) -> tuple[bytes | None, int, int]:
    """Generate speech using Gemini TTS models."""

    response = await cog.client.aio.models.generate_content(
        model=tts_params.model,
        contents=tts_params.input_text,
        config=types.GenerateContentConfig(**cast(Any, tts_params.to_dict())),
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
    """Run the `/gemini tts` command."""

    await ctx.defer()
    try:
        content_text = input_text if not style_prompt else f"{style_prompt}: {input_text}"
        tts_params = SpeechGenerationParameters(
            input_text=content_text,
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
            await ctx.send_followup(
                embed=Embed(
                    title="No Audio Generated",
                    description=(
                        "The model did not generate any audio. Please try again with different "
                        "text or parameters."
                    ),
                    color=Colour.orange(),
                )
            )
            return

        audio_file_path = Path(f"tts_output_{voice_name}.wav")
        with wave.open(str(audio_file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)

        description = f"**Text:** {input_text[:500]}{'...' if len(input_text) > 500 else ''}\n"
        description += f"**Model:** {model}\n"
        description += f"**Voice:** {voice_name}\n"
        if style_prompt:
            description += f"**Style Instructions:** {style_prompt}\n"
        description += "**Format:** WAV (24kHz, 16-bit, Mono)\n"

        response_embeds = [
            Embed(
                title="Text-to-Speech Generation",
                description=description,
                color=embeds.GEMINI_BLUE,
            )
        ]
        if SHOW_COST_EMBEDS:
            pricing_desc = (
                f"${cost:.4f} · {input_tokens:,} in / {output_tokens:,} out (audio) · "
                f"daily ${daily_cost:.2f}"
            )
            response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))

        await ctx.send_followup(embeds=response_embeds, file=File(audio_file_path))
        audio_file_path.unlink(missing_ok=True)
    except Exception as error:
        await cog._send_error_followup(ctx, error, "tts")


__all__ = ["_generate_speech_with_gemini", "tts_command"]
