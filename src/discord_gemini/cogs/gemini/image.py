"""Image generation helpers for the Gemini cog."""

from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from discord import Attachment, Colour, Embed, File
from discord.commands import ApplicationContext
from google.genai import types
from PIL import Image, UnidentifiedImageError

from ...config.auth import SHOW_COST_EMBEDS
from ...util import ImageGenerationParameters, calculate_image_cost, truncate_text
from . import attachments, embeds, state, usage
from .embed_delivery import send_embed_batches

if TYPE_CHECKING:
    from .cog import GeminiCog


async def _generate_image_with_gemini(
    cog: "GeminiCog",
    image_params: ImageGenerationParameters,
    attachment: Attachment | None,
) -> tuple[str | None, list[Image.Image], int]:
    """Generate images using Gemini models with generate_content."""

    prompt = image_params.prompt
    number_of_images = image_params.number_of_images

    if attachment:
        contents: str | list[str | Image.Image] = prompt
    else:
        image_word = "image(s)" if number_of_images > 1 else "image"
        contents = f"Create {image_word}: {prompt}"

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
    if number_of_images and number_of_images > 1:
        config_kwargs["candidate_count"] = number_of_images
    elif image_params.seed is not None:
        config_kwargs["seed"] = image_params.seed

    if image_params.aspect_ratio != "1:1" or image_params.image_size:
        image_config_kwargs = {}
        if image_params.aspect_ratio != "1:1":
            image_config_kwargs["aspect_ratio"] = image_params.aspect_ratio
        if image_params.image_size:
            image_config_kwargs["image_size"] = image_params.image_size
        config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

    if image_params.google_image_search and image_params.model == "gemini-3.1-flash-image-preview":
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

    gemini_response = await cog.client.aio.models.generate_content(
        model=image_params.model,
        contents=cast(Any, contents),
        config=types.GenerateContentConfig(**cast(Any, config_kwargs)),
    )

    usage_counts = usage.extract_usage_counts(gemini_response)
    input_tokens = usage_counts.input_tokens

    text_response = None
    generated_images: list[Image.Image] = []
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
                    generated_images.append(Image.open(BytesIO(part.inline_data.data)))

    return text_response, generated_images, input_tokens


async def _generate_image_with_imagen(
    cog: "GeminiCog",
    image_params: ImageGenerationParameters,
) -> list[Image.Image]:
    """Generate images using Imagen models."""

    imagen_response = await cog.client.aio.models.generate_images(
        model=image_params.model,
        prompt=image_params.prompt,
        config=types.GenerateImagesConfig(**image_params.to_dict()),
    )

    generated_images: list[Image.Image] = []
    if hasattr(imagen_response, "generated_images") and imagen_response.generated_images:
        for generated_image in imagen_response.generated_images:
            if hasattr(generated_image, "image") and generated_image.image:
                image_obj = generated_image.image
                if hasattr(image_obj, "image_bytes") and image_obj.image_bytes:
                    try:
                        generated_images.append(Image.open(BytesIO(image_obj.image_bytes)))
                    except (UnidentifiedImageError, OSError, ValueError) as error:
                        cog.logger.error("Failed to convert image_bytes to PIL Image: %s", error)
                else:
                    cog.logger.error(
                        "Image object missing image_bytes attribute. Type: %s",
                        type(image_obj),
                    )
    return generated_images


async def _create_image_response_embed(
    cog: "GeminiCog",
    image_params: ImageGenerationParameters,
    generated_images: list[Image.Image],
    attachment: Attachment | None,
    text_response: str | None = None,
) -> tuple[Embed, list[File]]:
    """Create the embed and file attachments for image generation results."""

    is_gemini_model = image_params.model.startswith("gemini-")
    files: list[File] = []
    for index, image in enumerate(generated_images):
        try:
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            files.append(File(image_bytes, filename=f"generated_image_{index + 1}.png"))
        except (OSError, ValueError) as error:
            cog.logger.error("Failed to save image %d: %s", index + 1, error)

    truncated_prompt = truncate_text(image_params.prompt, 2000)
    description = f"**Prompt:** {truncated_prompt}\n"
    description += f"**Model:** {image_params.model}\n"
    description += "**Mode:** Image Editing\n" if attachment else "**Mode:** Image Generation\n"
    description += f"**Number of Images:** {len(generated_images)}"

    if is_gemini_model:
        if image_params.number_of_images > 1:
            description += f" (requested: {image_params.number_of_images})"
        if image_params.seed is not None:
            description += f"\n**Seed:** {image_params.seed}"
        if image_params.aspect_ratio != "1:1":
            description += f"\n**Aspect Ratio:** {image_params.aspect_ratio}"
        if image_params.image_size:
            description += f"\n**Image Size:** {image_params.image_size}"
        if image_params.google_image_search:
            description += "\n**Google Image Search:** Enabled"

        unsupported_params = []
        if image_params.negative_prompt:
            unsupported_params.append(f"negative_prompt: {image_params.negative_prompt}")
        if image_params.guidance_scale:
            unsupported_params.append(f"guidance_scale: {image_params.guidance_scale}")
        if image_params.person_generation != "allow_adult":
            unsupported_params.append(f"person_generation: {image_params.person_generation}")
        if unsupported_params:
            description += (
                "\n\n*Note: Advanced parameters not yet implemented for Gemini: "
                f"{', '.join(unsupported_params)}*"
            )
    else:
        if image_params.seed is not None:
            description += f"\n**Seed:** {image_params.seed}"
        if image_params.negative_prompt:
            description += f"\n**Negative Prompt:** {image_params.negative_prompt}"
        if image_params.guidance_scale:
            description += f"\n**Guidance Scale:** {image_params.guidance_scale}"
        if image_params.aspect_ratio != "1:1":
            description += f"\n**Aspect Ratio:** {image_params.aspect_ratio}"
        if image_params.person_generation != "allow_adult":
            description += f"\n**Person Generation:** {image_params.person_generation}"

    if text_response:
        description += f"\n\n**AI Response:** {truncate_text(text_response, 500)}"

    embed = Embed(
        title=f"{'Gemini' if is_gemini_model else 'Imagen'} Image Generation",
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
    number_of_images: int,
    aspect_ratio: str,
    person_generation: str,
    attachment: Attachment | None,
    negative_prompt: str | None,
    seed: int | None,
    guidance_scale: float | None,
    image_size: str | None,
    google_image_search: bool | None,
) -> None:
    """Run the `/gemini image` command."""

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
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            image_size=image_size,
            google_image_search=bool(google_image_search),
        )

        is_gemini_model = model.startswith("gemini-")
        text_response = None
        generated_images: list[Image.Image] = []
        input_tokens = 0

        if is_gemini_model:
            text_response, generated_images, input_tokens = await _generate_image_with_gemini(
                cog,
                image_params,
                attachment,
            )
        else:
            if attachment:
                await send_embed_batches(
                    ctx.send_followup,
                    embed=Embed(
                        title="Not Supported",
                        description=(
                            "Image editing is not supported with Imagen models. "
                            "Please use a Gemini model for image editing."
                        ),
                        color=Colour.orange(),
                    ),
                    logger=cog.logger,
                )
                return
            generated_images = await _generate_image_with_imagen(cog, image_params)

        num_images = len(generated_images)
        cost = calculate_image_cost(model, num_images, input_tokens, image_size)
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "image",
            ctx.author.id,
            model,
            cost,
            daily_cost,
            images=num_images,
            input_tokens=input_tokens,
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
        elif is_gemini_model:
            embed_description += "Try asking explicitly for image generation (e.g., 'a red car').\n"
        else:
            embed_description += (
                "Imagen models should generate images. Check your prompt or try different "
                "parameters.\n"
            )

        if is_gemini_model:
            unsupported_params = []
            if image_params.negative_prompt:
                unsupported_params.append("negative_prompt")
            if image_params.guidance_scale:
                unsupported_params.append("guidance_scale")
            if image_params.person_generation != "allow_adult":
                unsupported_params.append("person_generation")
            if unsupported_params:
                embed_description += (
                    "\n*Note: These parameters are not yet implemented for Gemini: "
                    f"{', '.join(unsupported_params)}*"
                )

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
            pricing_desc += f" · daily ${daily_cost:.2f}"
            response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))
        await send_embed_batches(ctx.send_followup, embeds=response_embeds, logger=cog.logger)
    except Exception as error:
        await cog._send_error_followup(ctx, error, "image")


__all__ = [
    "_create_image_response_embed",
    "_generate_image_with_gemini",
    "_generate_image_with_imagen",
    "image_command",
]
