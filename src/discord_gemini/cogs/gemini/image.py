"""Image generation helpers for the Gemini cog."""

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from discord import Attachment, Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types
from PIL import Image, UnidentifiedImageError

from ...config.auth import SHOW_COST_EMBEDS
from ...util import ImageGenerationParameters, calculate_image_cost, truncate_text
from . import attachments, embeds, responses, state, usage
from .client import disable_afc
from .embed_delivery import send_embed_batches

if TYPE_CHECKING:
    from .cog import GeminiCog

# `image_size` values each model accepts, lower-cased for comparison (the option
# values themselves are the API's canonical uppercase `1K`/`2K`/`4K` — see
# `IMAGE_SIZE_CHOICES`). Probed live 2026-08-28 with the bot's exact request shape:
# Flash Image renders `512` (704x384 at the model's own 11:6 when no aspect ratio
# is sent, 512x512 with `1:1`) and `4K` (5632x3072); Pro renders `4K`; Lite 400s on
# `512`, `2K` and `4K` and Pro on `512`, all with "Image size <size> is not
# supported for this model"; `0.5K` 400s everywhere ("Supported values are: 1K, 2K,
# 4K, 512, 512P, 512PX."). Gemini 2.5 accepts a `4K` field but still returns a
# 1024x1024 image, so only its truthful fixed-size `1K` option is allowed.
IMAGE_SUPPORTED_SIZES: dict[str, frozenset[str]] = {
    "gemini-2.5-flash-image": frozenset({"1k"}),
    "gemini-3.1-flash-image": frozenset({"512", "1k", "2k", "4k"}),
    "gemini-3.1-flash-lite-image": frozenset({"1k"}),
    "gemini-3-pro-image": frozenset({"1k", "2k", "4k"}),
}
_IMAGE_SIZE_DISPLAY_ORDER = ("512", "1k", "2k", "4k")

# Discord renders these inline as-is, so the API's original bytes are attached
# unchanged; anything else is re-encoded to PNG. Re-encoding the probe's 4K JPEGs to
# PNG produced 10.06-12.95 MB files, over Discord's 10 MB bot upload cap, while the
# originals were 5.5-7.1 MB (2026-08-28).
DISCORD_NATIVE_IMAGE_EXTENSIONS: dict[str, str] = {"image/png": "png", "image/jpeg": "jpg"}
# Sent whenever the caller leaves aspect_ratio unset, so the advertised default holds
# at the API boundary and not only through the slash-command parameter default.
DEFAULT_IMAGE_ASPECT_RATIO = "1:1"


@dataclass(frozen=True)
class GeneratedImage:
    """One image part of a generate_content response: the API's own bytes and MIME."""

    data: bytes
    mime_type: str


def _validate_image_size_request(image_params: ImageGenerationParameters) -> str | None:
    """Reject `image_size` values the chosen model rejects or silently ignores."""

    if not image_params.image_size:
        return None
    supported = IMAGE_SUPPORTED_SIZES.get(image_params.model)
    if supported is None or image_params.image_size.lower() in supported:
        return None
    supported_list = ", ".join(
        size.upper() for size in _IMAGE_SIZE_DISPLAY_ORDER if size in supported
    )
    return (
        f"Image size {image_params.image_size} is not supported for this model. "
        f"`{image_params.model}` supports {supported_list}; choose a supported size or "
        "another image model."
    )


async def _generate_image_with_gemini(
    cog: "GeminiCog",
    image_params: ImageGenerationParameters,
    attachment: Attachment | None,
) -> tuple[str | None, list[GeneratedImage], int, int]:
    """Generate images using Gemini models with generate_content.

    Returns the text part, the image parts, the input token count (tool-use prompt
    tokens included), and the number of Google Search queries (web plus image search)
    the response reports.
    """

    prompt = image_params.prompt

    if attachment:
        contents: str | list[str | Image.Image] = prompt
    else:
        contents = f"Create image: {prompt}"

    if attachment:
        image_data = await attachments._fetch_attachment_bytes(cog, attachment)
        if image_data:
            try:
                image = Image.open(BytesIO(image_data))
            except (UnidentifiedImageError, OSError, ValueError) as error:
                cog.logger.warning("Failed to open attachment for image generation: %s", error)
            else:
                contents = [prompt, image]

    config_kwargs: dict[str, Any] = {"response_modalities": ["TEXT", "IMAGE"]}
    if image_params.seed is not None:
        config_kwargs["seed"] = image_params.seed

    image_config_kwargs: dict[str, Any] = {}
    # Always on the wire, 1:1 included: with the field omitted the model picks its
    # own ratio (a 512 request came back 704x384, 11:6), so the advertised 1:1
    # default only holds when it is sent explicitly (probe 2026-08-28).
    image_config_kwargs["aspect_ratio"] = image_params.aspect_ratio or DEFAULT_IMAGE_ASPECT_RATIO
    if image_params.image_size:
        image_config_kwargs["image_size"] = image_params.image_size
    if image_config_kwargs:
        config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

    if image_params.google_image_search and image_params.model == "gemini-3.1-flash-image":
        config_kwargs["tools"] = [
            types.Tool(
                google_search=types.GoogleSearch(
                    search_types=types.SearchTypes(
                        web_search=types.WebSearch(),
                        image_search=types.ImageSearch(),
                    )
                )
            )
        ]

    config_kwargs["automatic_function_calling"] = disable_afc()

    gemini_response = await cog.client.aio.models.generate_content(
        model=image_params.model,
        contents=cast(Any, contents),
        config=types.GenerateContentConfig(**cast(Any, config_kwargs)),
    )

    usage_counts = usage.extract_usage_counts(gemini_response)
    # Tool-use prompt tokens are billed as input, as in chat.
    input_tokens = usage_counts.input_tokens + usage_counts.tool_use_prompt_tokens
    search_queries = responses.count_search_queries(gemini_response)

    text_response = None
    generated_images: list[GeneratedImage] = []
    if gemini_response.candidates and len(gemini_response.candidates) > 0:
        candidate = gemini_response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text is not None:
                    text_response = part.text
                elif (
                    hasattr(part, "inline_data")
                    and part.inline_data is not None
                    and part.inline_data.data
                ):
                    mime_type = (part.inline_data.mime_type or "").split(";")[0].strip().lower()
                    generated_images.append(GeneratedImage(part.inline_data.data, mime_type))

    return text_response, generated_images, input_tokens, search_queries


async def _create_image_response_embed(
    cog: "GeminiCog",
    image_params: ImageGenerationParameters,
    generated_images: list[GeneratedImage],
    attachment: Attachment | None,
    text_response: str | None = None,
) -> tuple[Embed, list[File]]:
    """Create the embed and file attachments for image generation results.

    PNG and JPEG parts are attached exactly as the API returned them (see
    `DISCORD_NATIVE_IMAGE_EXTENSIONS`); any other format is re-encoded to PNG.
    """

    files: list[File] = []
    for index, image in enumerate(generated_images):
        try:
            extension = DISCORD_NATIVE_IMAGE_EXTENSIONS.get(image.mime_type)
            if extension:
                # Attach untouched, but still decode-check so a truncated payload is
                # logged and skipped instead of reaching Discord as a broken file.
                with Image.open(BytesIO(image.data)) as native:
                    native.verify()
                payload = BytesIO(image.data)
            else:
                payload = BytesIO()
                with Image.open(BytesIO(image.data)) as decoded:
                    decoded.save(payload, format="PNG")
                payload.seek(0)
                extension = "png"
            files.append(File(payload, filename=f"generated_image_{index + 1}.{extension}"))
        except (OSError, ValueError) as error:
            cog.logger.error("Failed to save image %d: %s", index + 1, error)

    truncated_prompt = truncate_text(image_params.prompt, 2000)
    description = f"**Prompt:** {truncated_prompt}\n"
    description += f"**Model:** {image_params.model}\n"
    description += "**Mode:** Image Editing\n" if attachment else "**Mode:** Image Generation\n"
    description += f"**Number of Images:** {len(generated_images)}"

    if image_params.seed is not None:
        description += f"\n**Seed:** {image_params.seed}"
    if image_params.aspect_ratio != "1:1":
        description += f"\n**Aspect Ratio:** {image_params.aspect_ratio}"
    if image_params.image_size:
        description += f"\n**Image Size:** {image_params.image_size}"
    if image_params.google_image_search:
        description += "\n**Google Image Search:** Enabled"

    if text_response:
        description += f"\n\n**AI Response:** {truncate_text(text_response, 500)}"

    embed = Embed(
        title="Gemini Image Generation",
        description=description,
        color=embeds.GEMINI_BLUE,
    )
    if files:
        embed.set_image(url=f"attachment://{files[0].filename}")
    return embed, files


async def image_command(
    cog: "GeminiCog",
    ctx: ApplicationContext,
    prompt: str,
    model: str,
    aspect_ratio: str,
    attachment: Attachment | None,
    seed: int | None,
    image_size: str | None,
    google_image_search: bool | None,
) -> None:
    """Run the `/gemini-media image` command."""

    await ctx.defer()
    try:
        if attachment:
            validation_error = attachments._validate_attachment_size(attachment)
            if validation_error:
                await send_embed_batches(
                    ctx.send_followup,
                    embed=embeds.build_error_embed(validation_error),
                    logger=cog.logger,
                )
                return

        image_params = ImageGenerationParameters(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            seed=seed,
            image_size=image_size,
            google_image_search=bool(google_image_search),
        )

        validation_error = _validate_image_size_request(image_params)
        if validation_error:
            await send_embed_batches(
                ctx.send_followup,
                embed=embeds.build_error_embed(validation_error),
                logger=cog.logger,
            )
            return

        (
            text_response,
            generated_images,
            input_tokens,
            search_queries,
        ) = await _generate_image_with_gemini(cog, image_params, attachment)

        num_images = len(generated_images)
        cost = calculate_image_cost(
            model, num_images, input_tokens, image_size, google_search_queries=search_queries
        )
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "image",
            ctx.author.id,
            model,
            cost,
            daily_cost,
            images=num_images,
            input_tokens=input_tokens,
            google_search_queries=search_queries,
        )

        if generated_images:
            embed, files = await _create_image_response_embed(
                cog,
                image_params=image_params,
                generated_images=generated_images,
                attachment=attachment,
                text_response=text_response,
            )
            response_embeds = [embed]
            if SHOW_COST_EMBEDS:
                pricing_desc = f"${cost:.4f} · {num_images} image{'s' if num_images != 1 else ''}"
                if input_tokens:
                    pricing_desc += f" · {input_tokens:,} input tokens"
                if search_queries:
                    pricing_desc += f" · {embeds.format_search_queries(search_queries)}"
                pricing_desc += f" · daily ${daily_cost:.2f}"
                response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))
            await send_embed_batches(
                ctx.send_followup,
                embeds=response_embeds,
                files=files,
                logger=cog.logger,
            )
            return

        embed_description = "The model did not generate any images.\n"
        if text_response:
            embed_description += f"Text response: {truncate_text(text_response, 3800)}\n"
        else:
            embed_description += "Try asking explicitly for image generation (e.g., 'a red car').\n"

        response_embeds = [
            Embed(
                title="No Images Generated",
                description=embed_description,
                color=Colour.orange(),
            )
        ]
        if SHOW_COST_EMBEDS and cost > 0:
            pricing_desc = f"${cost:.4f} · 0 images"
            if input_tokens:
                pricing_desc += f" · {input_tokens:,} input tokens"
            if search_queries:
                pricing_desc += f" · {embeds.format_search_queries(search_queries)}"
            pricing_desc += f" · daily ${daily_cost:.2f}"
            response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))
        await send_embed_batches(ctx.send_followup, embeds=response_embeds, logger=cog.logger)
    except Exception as error:
        await cog._send_error_followup(ctx, error, "image")


__all__ = [
    "IMAGE_SUPPORTED_SIZES",
    "GeneratedImage",
    "_create_image_response_embed",
    "_generate_image_with_gemini",
    "_validate_image_size_request",
    "image_command",
]
