"""Image generation helpers for the Gemini cog."""

import asyncio
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from discord import Attachment, Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types
from PIL import Image, UnidentifiedImageError

from ...config.auth import SHOW_COST_EMBEDS
from ...cost_line import count_label, format_cost_line
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
# 1024x1024 image, so only its truthful fixed-size `1K` option is allowed. Nano
# Banana 2.1 400s on `512` with the same message and renders `1K`, `2K` and `4K`.
IMAGE_SUPPORTED_SIZES: dict[str, frozenset[str]] = {
    "gemini-nano-banana-2.1": frozenset({"1k", "2k", "4k"}),
    "gemini-2.5-flash-image": frozenset({"1k"}),
    "gemini-3.1-flash-image": frozenset({"512", "1k", "2k", "4k"}),
    "gemini-3.1-flash-lite-image": frozenset({"1k"}),
    "gemini-3-pro-image": frozenset({"1k", "2k", "4k"}),
}
_IMAGE_SIZE_DISPLAY_ORDER = ("512", "1k", "2k", "4k")

# `aspect_ratio` values each model accepts. Every model takes the five standard
# ratios; the tall and wide ratios `1:4`, `4:1`, `1:8` and `8:1` are accepted on
# Nano Banana 2.1, 3.1 Flash Image and 3.1 Flash Lite Image (at 1K they render `1:4`
# as 512x2064, `4:1` as 2064x512 and `1:8` as 352x2928; `8:1` renders as 1456x176 at
# 512, 2928x352 at 1K on Flash and Flash Lite, 5856x704 at 2K and 11712x1408 at 4K),
# and each is billed the image tokens of a square image of the same size. 3.0 Pro
# Image and 2.5 Flash Image return 400 "Aspect ratio X is not supported for this
# model" for each of the four.
_STANDARD_IMAGE_ASPECT_RATIOS = frozenset({"1:1", "3:4", "4:3", "9:16", "16:9"})
_WIDE_IMAGE_ASPECT_RATIOS = _STANDARD_IMAGE_ASPECT_RATIOS | {"1:4", "4:1", "1:8", "8:1"}
IMAGE_SUPPORTED_ASPECT_RATIOS: dict[str, frozenset[str]] = {
    "gemini-nano-banana-2.1": _WIDE_IMAGE_ASPECT_RATIOS,
    "gemini-2.5-flash-image": _STANDARD_IMAGE_ASPECT_RATIOS,
    "gemini-3.1-flash-image": _WIDE_IMAGE_ASPECT_RATIOS,
    "gemini-3.1-flash-lite-image": _WIDE_IMAGE_ASPECT_RATIOS,
    "gemini-3-pro-image": _STANDARD_IMAGE_ASPECT_RATIOS,
}
_IMAGE_ASPECT_RATIO_DISPLAY_ORDER = (
    "1:1",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
)

# Models that take the `google_image_search` option's Google Search tool (web plus
# image search); the option is ignored for every other model.
IMAGE_SEARCH_GROUNDING_MODELS = frozenset({"gemini-nano-banana-2.1", "gemini-3.1-flash-image"})

# Discord renders these inline as-is, so the API's original bytes are attached
# unchanged; anything else is re-encoded to PNG. Re-encoding the probe's 4K JPEGs to
# PNG produced 10.06-12.95 MB files, over Discord's 10 MB bot upload cap, while the
# originals were 5.5-7.1 MB (2026-08-28).
DISCORD_NATIVE_IMAGE_EXTENSIONS: dict[str, str] = {"image/png": "png", "image/jpeg": "jpg"}
# Discord's per-file upload limit for a bot in a guild without a raised limit and in
# DMs; `Guild.filesize_limit` gives the real value for the guild the command runs in.
DEFAULT_UPLOAD_LIMIT_BYTES = 10 * 1024 * 1024
# An image over the upload limit is re-encoded as JPEG at these qualities in turn,
# then at half the width and height, until it fits.
REENCODE_JPEG_QUALITIES = (90, 80, 70)
# Sent whenever the caller leaves aspect_ratio unset, so the advertised default holds
# at the API boundary and not only through the slash-command parameter default.
DEFAULT_IMAGE_ASPECT_RATIO = "1:1"


@dataclass(frozen=True)
class GeneratedImage:
    """One image part of a generate_content response: the API's own bytes and MIME."""

    data: bytes
    mime_type: str


@dataclass(frozen=True)
class PreparedImage:
    """An image file ready to attach.

    ``reencoded_quality`` is set when the image was re-encoded as JPEG to fit the
    upload limit; ``original_bytes`` and ``pixels`` then describe the change.
    """

    data: bytes
    extension: str
    reencoded_quality: int | None = None
    original_bytes: int = 0
    pixels: tuple[int, int] | None = None


def _format_megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _upload_limit_for(ctx: Any) -> int:
    """Return the upload limit for this reply.

    Discord sends the limit with each interaction (`attachment_size_limit`); the guild's
    `filesize_limit` and then `DEFAULT_UPLOAD_LIMIT_BYTES` are used when it is missing.
    """

    for limit in (
        getattr(getattr(ctx, "interaction", None), "attachment_size_limit", None),
        getattr(getattr(ctx, "guild", None), "filesize_limit", None),
    ):
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            return limit
    return DEFAULT_UPLOAD_LIMIT_BYTES


def _fit_to_upload_limit(data: bytes, upload_limit: int) -> PreparedImage:
    """Re-encode an image as JPEG, lowering the quality and then halving the
    dimensions, until it is no larger than ``upload_limit`` bytes.

    The image is converted to RGB because JPEG has no alpha channel, so a transparent
    image loses its transparency. Raises ``ValueError`` when even a 1-pixel image
    does not fit.
    """

    with Image.open(BytesIO(data)) as decoded:
        current = decoded.convert("RGB")
    while True:
        for quality in REENCODE_JPEG_QUALITIES:
            buffer = BytesIO()
            current.save(buffer, format="JPEG", quality=quality)
            if buffer.tell() <= upload_limit:
                return PreparedImage(
                    buffer.getvalue(),
                    "jpg",
                    reencoded_quality=quality,
                    original_bytes=len(data),
                    pixels=current.size,
                )
        width, height = current.size
        if width <= 1 and height <= 1:
            raise ValueError(
                f"image does not fit the {_format_megabytes(upload_limit)} upload limit"
            )
        current = current.resize(
            (max(width // 2, 1), max(height // 2, 1)), Image.Resampling.LANCZOS
        )


def _prepare_image_file(image: "GeneratedImage", upload_limit: int) -> PreparedImage:
    """Build the attachment bytes for one generated image.

    PNG and JPEG parts keep the API's original bytes (decode-checked, so a truncated
    payload raises instead of reaching Discord as a broken file); other formats are
    re-encoded to PNG. A result over ``upload_limit`` is re-encoded as JPEG to fit.
    Blocking Pillow work: run it with ``asyncio.to_thread``.
    """

    extension = DISCORD_NATIVE_IMAGE_EXTENSIONS.get(image.mime_type)
    if extension:
        with Image.open(BytesIO(image.data)) as native:
            native.verify()
        data = image.data
    else:
        buffer = BytesIO()
        with Image.open(BytesIO(image.data)) as decoded:
            decoded.save(buffer, format="PNG")
        data = buffer.getvalue()
        extension = "png"
    if len(data) <= upload_limit:
        return PreparedImage(data, extension)
    return _fit_to_upload_limit(data, upload_limit)


def _reencode_note(prepared: PreparedImage, upload_limit: int) -> str:
    pixels = f" at {prepared.pixels[0]}x{prepared.pixels[1]}" if prepared.pixels else ""
    return (
        f"Re-encoded to fit Discord's {_format_megabytes(upload_limit)} upload limit: "
        f"{_format_megabytes(prepared.original_bytes)} → "
        f"{_format_megabytes(len(prepared.data))} JPEG (quality "
        f"{prepared.reencoded_quality}){pixels}"
    )


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


def _validate_image_aspect_ratio_request(image_params: ImageGenerationParameters) -> str | None:
    """Reject `aspect_ratio` values the chosen model rejects."""

    aspect_ratio = image_params.aspect_ratio or DEFAULT_IMAGE_ASPECT_RATIO
    supported = IMAGE_SUPPORTED_ASPECT_RATIOS.get(image_params.model)
    if supported is None or aspect_ratio in supported:
        return None
    supported_list = ", ".join(
        ratio for ratio in _IMAGE_ASPECT_RATIO_DISPLAY_ORDER if ratio in supported
    )
    return (
        f"Aspect ratio {aspect_ratio} is not supported for this model. "
        f"`{image_params.model}` supports {supported_list}; choose a supported aspect "
        "ratio or another image model."
    )


@dataclass(frozen=True)
class ImageUsage:
    """Billable counts from one image response.

    ``image_tokens`` are billed through the per-image price. ``text_output_tokens``
    is the output billed at the model's text rate: thinking tokens plus non-image
    candidate tokens (``thinking_tokens`` is part of it). ``output_tokens`` is every
    billed output token, the sum of the two.
    """

    input_tokens: int = 0
    image_tokens: int = 0
    text_output_tokens: int = 0
    thinking_tokens: int = 0
    search_queries: int = 0

    @property
    def output_tokens(self) -> int:
        return self.image_tokens + self.text_output_tokens


def _image_modality_tokens(usage_metadata: Any) -> int | None:
    """IMAGE-modality candidate tokens, or None when the response has no breakdown
    (the field is missing or an empty list)."""

    details = getattr(usage_metadata, "candidates_tokens_details", None)
    if not isinstance(details, list) or not details:
        return None
    image_tokens = 0
    for detail in details:
        modality = getattr(detail, "modality", None)
        if getattr(modality, "value", modality) == "IMAGE":
            image_tokens += int(getattr(detail, "token_count", None) or 0)
    return image_tokens


def _extract_image_usage(response: Any) -> ImageUsage:
    """Split an image response's usage into the counts `calculate_image_cost` bills.

    Without a per-modality breakdown every candidate token is treated as an image
    token, so only the thinking tokens are billed at the text rate.
    """

    usage_counts = usage.extract_usage_counts(response)
    candidate_tokens = max(usage_counts.output_tokens, 0)
    image_tokens = _image_modality_tokens(getattr(response, "usage_metadata", None))
    if image_tokens is None:
        image_tokens = candidate_tokens
    image_tokens = min(image_tokens, candidate_tokens)
    return ImageUsage(
        # Tool-use prompt tokens are billed as input, as in chat.
        input_tokens=usage_counts.input_tokens + usage_counts.tool_use_prompt_tokens,
        image_tokens=image_tokens,
        text_output_tokens=candidate_tokens - image_tokens + usage_counts.thinking_tokens,
        thinking_tokens=usage_counts.thinking_tokens,
        search_queries=responses.count_search_queries(response),
    )


def _image_cost_line(
    cost: float, daily_cost: float, image_usage: ImageUsage, num_images: int
) -> str:
    """Build the image cost line. `out` is every billed output token (image,
    non-image candidate and thinking tokens), as in the other bots' cost lines."""

    details = [count_label(num_images, "image")]
    if image_usage.search_queries > 0:
        details.append(count_label(image_usage.search_queries, "search", "searches"))
    return format_cost_line(
        cost,
        daily_cost,
        input_tokens=image_usage.input_tokens or None,
        output_tokens=image_usage.output_tokens or None,
        thinking_tokens=image_usage.thinking_tokens,
        details=details,
    )


async def _generate_image_with_gemini(
    cog: "GeminiCog",
    image_params: ImageGenerationParameters,
    attachment: Attachment | None,
) -> tuple[str | None, list[GeneratedImage], ImageUsage]:
    """Generate images using Gemini models with generate_content.

    Returns the text part, the image parts, and the billable usage: input tokens
    (tool-use prompt tokens included), image and text/thinking output tokens, and the
    number of Google Search queries (web plus image search) the response reports.
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

    if image_params.google_image_search and image_params.model in IMAGE_SEARCH_GROUNDING_MODELS:
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

    image_usage = _extract_image_usage(gemini_response)

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

    return text_response, generated_images, image_usage


async def _create_image_response_embed(
    cog: "GeminiCog",
    image_params: ImageGenerationParameters,
    generated_images: list[GeneratedImage],
    attachment: Attachment | None,
    text_response: str | None = None,
    upload_limit: int = DEFAULT_UPLOAD_LIMIT_BYTES,
) -> tuple[Embed, list[File]]:
    """Create the embed and file attachments for image generation results.

    PNG and JPEG parts are attached exactly as the API returned them (see
    `DISCORD_NATIVE_IMAGE_EXTENSIONS`); any other format is re-encoded to PNG. An
    image larger than ``upload_limit`` bytes is re-encoded as JPEG to fit, and the
    embed says so.
    """

    files: list[File] = []
    reencode_notes: list[str] = []
    for index, image in enumerate(generated_images):
        try:
            prepared = await asyncio.to_thread(_prepare_image_file, image, upload_limit)
        except (OSError, ValueError) as error:
            cog.logger.error("Failed to save image %d: %s", index + 1, error)
            continue
        if prepared.reencoded_quality is not None:
            note = _reencode_note(prepared, upload_limit)
            reencode_notes.append(note)
            cog.logger.info("Image %d: %s", index + 1, note)
        files.append(
            File(
                BytesIO(prepared.data), filename=f"generated_image_{index + 1}.{prepared.extension}"
            )
        )

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
    if image_params.google_image_search and image_params.model in IMAGE_SEARCH_GROUNDING_MODELS:
        description += "\n**Google Image Search:** Enabled"
    for note in reencode_notes:
        description += f"\n**Delivery:** {note}"

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
        validation_error = validation_error or _validate_image_aspect_ratio_request(image_params)
        if validation_error:
            await send_embed_batches(
                ctx.send_followup,
                embed=embeds.build_error_embed(validation_error),
                logger=cog.logger,
            )
            return

        text_response, generated_images, image_usage = await _generate_image_with_gemini(
            cog, image_params, attachment
        )

        num_images = len(generated_images)
        cost = calculate_image_cost(
            model,
            num_images,
            image_usage.input_tokens,
            image_size,
            google_search_queries=image_usage.search_queries,
            text_output_tokens=image_usage.text_output_tokens,
        )
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "image",
            ctx.author.id,
            model,
            cost,
            daily_cost,
            images=num_images,
            input_tokens=image_usage.input_tokens,
            output_tokens=image_usage.output_tokens,
            text_output_tokens=image_usage.text_output_tokens,
            thinking_tokens=image_usage.thinking_tokens,
            google_search_queries=image_usage.search_queries,
        )

        if generated_images:
            embed, files = await _create_image_response_embed(
                cog,
                image_params=image_params,
                generated_images=generated_images,
                attachment=attachment,
                text_response=text_response,
                upload_limit=_upload_limit_for(ctx),
            )
            response_embeds = [embed]
            if SHOW_COST_EMBEDS:
                pricing_desc = _image_cost_line(cost, daily_cost, image_usage, num_images)
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
            pricing_desc = _image_cost_line(cost, daily_cost, image_usage, num_images)
            response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))
        await send_embed_batches(ctx.send_followup, embeds=response_embeds, logger=cog.logger)
    except Exception as error:
        await cog._send_error_followup(ctx, error, "image")


__all__ = [
    "DEFAULT_UPLOAD_LIMIT_BYTES",
    "IMAGE_SEARCH_GROUNDING_MODELS",
    "IMAGE_SUPPORTED_ASPECT_RATIOS",
    "IMAGE_SUPPORTED_SIZES",
    "GeneratedImage",
    "ImageUsage",
    "PreparedImage",
    "_create_image_response_embed",
    "_generate_image_with_gemini",
    "_validate_image_aspect_ratio_request",
    "_validate_image_size_request",
    "image_command",
]
