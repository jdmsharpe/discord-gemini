from discord.ext import commands
from google import genai
from google.genai import types
from discord import (
    Attachment,
    Colour,
    Embed,
    File,
)
from discord.commands import (
    ApplicationContext,
    command,
    option,
    OptionChoice,
    slash_command,
)
from typing import Optional, Dict, List, Union, Any
from util import (
    ChatCompletionParameters,
    chunk_text,
    ImageGenerationParameters,
    VideoGenerationParameters,
)
import aiohttp
import asyncio
from button_view import ButtonView
import logging
from config.auth import GUILD_IDS, GEMINI_API_KEY
from dataclasses import dataclass
from PIL import Image
from io import BytesIO


@dataclass
class Conversation:
    """A dataclass to store conversation state."""

    params: ChatCompletionParameters
    history: List[Dict[str, Any]]


def append_response_embeds(embeds, response_text):
    # Ensure each chunk is no larger than 3500 characters to stay well under Discord's 6000 char limit
    # If response is extremely long (>20000 chars), truncate it to prevent too many embeds
    if len(response_text) > 20000:
        response_text = (
            response_text[:19500] + "\n\n... [Response truncated due to length]"
        )

    for index, chunk in enumerate(chunk_text(response_text, 3500), start=1):
        embeds.append(
            Embed(
                title="Response" + f" (Part {index})" if index > 1 else "Response",
                description=chunk,
                color=Colour.blue(),
            )
        )


class GeminiAPI(commands.Cog):
    def __init__(self, bot):
        """
        Initialize the GeminiAPI class.

        Args:
            bot: The bot instance.
        """
        self.bot = bot
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        logging.basicConfig(
            level=logging.DEBUG,  # Capture all levels of logs
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Dictionary to store conversation state for each converse interaction
        self.conversations: Dict[int, Conversation] = {}
        # Dictionary to map any message ID to the main conversation ID for tracking
        self.message_to_conversation_id: Dict[int, int] = {}
        # Dictionary to store UI views for each conversation
        self.views = {}

    async def handle_new_message_in_conversation(
        self, message, conversation_wrapper: Conversation
    ):
        """
        Handles a new message in an ongoing conversation.

        Args:
            message: The incoming Discord Message object.
            conversation_wrapper: The conversation object wrapper.
        """
        params = conversation_wrapper.params
        history = conversation_wrapper.history

        self.logger.info(
            f"Handling new message in conversation {params.conversation_id}."
        )
        typing_task = None
        embeds = []

        try:
            # Only respond to the user who started the conversation and if not paused.
            if message.author != params.conversation_starter or params.paused:
                return

            self.logger.debug(f"Starting typing indicator for followup message from {message.author}")
            typing_task = asyncio.create_task(self.keep_typing(message.channel))

            user_parts: List[Union[str, Dict]] = [{"text": message.content}]
            if message.attachments:
                for attachment in message.attachments:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                user_parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": attachment.content_type,
                                            "data": image_data,
                                        }
                                    }
                                )

            history.append({"role": "user", "parts": user_parts})

            self.logger.debug(f"Sending history to Gemini: {history}")

            config_args = {}
            if params.system_instruction:
                config_args["system_instruction"] = params.system_instruction
            if params.temperature is not None:
                config_args["temperature"] = params.temperature
            if params.top_p is not None:
                config_args["top_p"] = params.top_p

            # Convert history to the format expected by Gemini API
            contents = []
            for entry in history:
                if entry["role"] == "user":
                    contents.append({"role": "user", "parts": entry["parts"]})
                elif entry["role"] == "model":
                    contents.append({"role": "model", "parts": entry["parts"]})

            self.logger.debug(f"Sending contents to Gemini: {contents}")

            generation_config = (
                types.GenerateContentConfig(**config_args) if config_args else None
            )
            self.logger.debug(f"Sending contents to Gemini: {contents}")
            response = self.client.models.generate_content(
                model=params.model,
                contents=contents,
                config=generation_config,
            )
            response_text = response.text
            self.logger.debug(f"Received response from Gemini: {response_text}")

            # Stop typing indicator as soon as we have the response
            if typing_task:
                self.logger.debug(f"Stopping typing indicator for conversation {params.conversation_id}")
                typing_task.cancel()
                typing_task = None

            # Handle case where response text might be None
            if response_text is None:
                response_text = "No response generated by the model."
                self.logger.warning("Model returned None as response text")

            history.append({"role": "model", "parts": [{"text": response_text}]})

            append_response_embeds(embeds, response_text)

            view = self.views.get(message.author)
            main_conversation_id = conversation_wrapper.params.conversation_id

            # Ensure conversation_id is not None
            if main_conversation_id is None:
                self.logger.error("Conversation ID is None, cannot track message")
                return

            if embeds:
                # Send the first embed as a direct reply to the user's message
                try:
                    reply_message = await message.reply(embed=embeds[0], view=view)
                    self.message_to_conversation_id[reply_message.id] = (
                        main_conversation_id
                    )
                except Exception as embed_error:
                    # If embed fails due to size, try sending as plain text
                    self.logger.warning(f"Embed failed, sending as text: {embed_error}")
                    safe_response_text = response_text or "No response text available"
                    reply_message = await message.reply(
                        content=f"**Response:**\n{safe_response_text[:1900]}{'...' if len(safe_response_text) > 1900 else ''}",
                        view=view,
                    )
                    self.message_to_conversation_id[reply_message.id] = (
                        main_conversation_id
                    )

                # Send any remaining embeds as separate messages in the same channel
                for embed in embeds[1:]:
                    try:
                        followup_message = await message.channel.send(
                            embed=embed, view=view
                        )
                        self.message_to_conversation_id[followup_message.id] = (
                            main_conversation_id
                        )
                    except Exception as embed_error:
                        # If embed fails, send as plain text
                        self.logger.warning(f"Followup embed failed: {embed_error}")
                        followup_message = await message.channel.send(
                            content=f"**Response (continued):**\n{embed.description[:1900]}{'...' if len(embed.description) > 1900 else ''}",
                            view=view,
                        )
                        self.message_to_conversation_id[followup_message.id] = (
                            main_conversation_id
                        )

                self.logger.debug("Replied with generated response.")
            else:
                self.logger.warning("No embeds to send in the reply.")
                await message.reply(
                    content="An error occurred: No content to send.",
                    view=view,
                )

        except Exception as e:
            description = str(e)
            self.logger.error(
                f"Error in handle_new_message_in_conversation: {description}",
                exc_info=True,
            )
            # Truncate the description to fit within Discord's embed limits
            if len(description) > 4000:
                description = description[:4000] + "\n\n... (error message truncated)"
            await message.reply(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

        finally:
            # Cancel typing task if it exists
            if typing_task:
                typing_task.cancel()

    async def keep_typing(self, channel):
        """
        Coroutine to keep the typing indicator alive in a channel.

        Args:
            channel: The Discord channel object.
        """
        try:
            self.logger.debug(f"Starting typing indicator loop in channel {channel.id}")
            while True:
                async with channel.typing():
                    self.logger.debug(f"Sent typing indicator to channel {channel.id}")
                    await asyncio.sleep(5)  # Resend typing indicator every 5 seconds
        except asyncio.CancelledError:
            # Task was cancelled, stop typing
            self.logger.debug(f"Typing indicator cancelled for channel {channel.id}")
            raise

    # Added for debugging purposes
    @commands.Cog.listener()
    async def on_ready(self):
        """
        Event listener that runs when the bot is ready.
        Logs bot details and attempts to synchronize commands.
        """
        self.logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.owner_id})")
        self.logger.info(f"Attempting to sync commands for guilds: {GUILD_IDS}")
        try:
            await self.bot.sync_commands()
            self.logger.info("Commands synchronized successfully.")
        except Exception as e:
            self.logger.error(
                f"Error during command synchronization: {e}", exc_info=True
            )

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Event listener that runs when a message is sent.
        Generates a response using chat completion API when a new message from the conversation author is detected.

        Args:
            message: The incoming Discord Message object.
        """
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        self.logger.debug(f"Received message from {message.author} in channel {message.channel.id}: '{message.content}'")
        
        # Check for active conversations in this channel
        for conversation_wrapper in self.conversations.values():
            self.logger.debug(f"Checking conversation {conversation_wrapper.params.conversation_id} in channel {conversation_wrapper.params.channel_id}")
            # Skip conversations that are not in the same channel
            if message.channel.id != conversation_wrapper.params.channel_id:
                self.logger.debug(f"Channel mismatch: message in {message.channel.id}, conversation in {conversation_wrapper.params.channel_id}")
                continue

            # Skip if the message is not from the conversation starter
            if message.author != conversation_wrapper.params.conversation_starter:
                self.logger.debug(f"Author mismatch: message from {message.author}, conversation started by {conversation_wrapper.params.conversation_starter}")
                continue

            self.logger.info(f"Processing followup message for conversation {conversation_wrapper.params.conversation_id}")
            # Process the message for the matching conversation
            await self.handle_new_message_in_conversation(message, conversation_wrapper)
            break  # Stop looping once we've handled the message
        
        self.logger.debug("No matching conversations found for this message")

    @commands.Cog.listener()
    async def on_error(self, event, *args, **kwargs):
        """
        Event listener that runs when an error occurs.

        Args:
            event: The name of the event that raised the error.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.logger.error(f"Error in event {event}: {args} {kwargs}", exc_info=True)

    @command()
    async def check_permissions(self, ctx):
        """
        Checks and reports the bot's permissions in the current channel.
        """
        permissions = ctx.channel.permissions_for(ctx.guild.me)
        if permissions.read_messages and permissions.read_message_history:
            await ctx.send("Bot has permission to read messages and message history.")
        else:
            await ctx.send("Bot is missing necessary permissions in this channel.")

    @slash_command(
        name="converse",
        description="Starts a conversation with a model.",
        guild_ids=GUILD_IDS,
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "system_instruction",
        description="Additional instructions for the model. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "model",
        description="Choose from the following Gemini models. (default: gemini-2.5-flash)",
        required=False,
        choices=[
            OptionChoice(name="Gemini 2.5 Pro", value="gemini-2.5-pro"),
            OptionChoice(name="Gemini 2.5 Flash", value="gemini-2.5-flash"),
            OptionChoice(
                name="Gemini 2.5 Flash Lite Preview 06-17",
                value="gemini-2.5-flash-lite-preview-06-17",
            ),
            OptionChoice(name="Gemini 2.0 Flash", value="gemini-2.0-flash"),
            OptionChoice(name="Gemini 2.0 Flash Lite", value="gemini-2.0-flash-lite"),
            OptionChoice(name="Gemini 1.5 Flash", value="gemini-1.5-flash"),
            OptionChoice(name="Gemini 1.5 Flash 8B", value="gemini-1.5-flash-8b"),
            OptionChoice(name="Gemini 1.5 Pro", value="gemini-1.5-pro"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="Attachment to append to the prompt. Only images are supported at this time. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "frequency_penalty",
        description="(Advanced) Controls how much the model should repeat itself. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "presence_penalty",
        description="(Advanced) Controls how much the model should talk about the prompt. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "seed",
        description="(Advanced) Seed for deterministic outputs. (default: not set)",
        required=False,
        type=int,
    )
    @option(
        "temperature",
        description="(Advanced) Controls the randomness of the model. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "top_p",
        description="(Advanced) Nucleus sampling. (default: not set)",
        required=False,
        type=float,
    )
    async def converse(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "gemini-2.5-flash",
        system_instruction: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        attachment: Optional[Attachment] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """
        Creates a model response for the given chat conversation.
        """
        # Acknowledge the interaction immediately - reply can take some time
        await ctx.defer()
        typing_task = None

        for conv_wrapper in self.conversations.values():
            if (
                conv_wrapper.params.conversation_starter == ctx.author
                and conv_wrapper.params.channel_id == ctx.channel.id
            ):
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description="You already have an active conversation in this channel. Please finish it before starting a new one.",
                        color=Colour.red(),
                    )
                )
                return

        try:
            # Start typing and keep it alive until the response is ready
            typing_task = asyncio.create_task(self.keep_typing(ctx.channel))

            parts: List[Dict] = [{"text": prompt}]
            if attachment:
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": attachment.content_type,
                                        "data": image_data,
                                    }
                                }
                            )

            config_args = {}
            if system_instruction is not None:
                config_args["system_instruction"] = system_instruction
            if frequency_penalty is not None:
                config_args["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                config_args["presence_penalty"] = presence_penalty
            if seed is not None:
                config_args["seed"] = seed
            if temperature is not None:
                config_args["temperature"] = temperature
            if top_p is not None:
                config_args["top_p"] = top_p

            generation_config = (
                types.GenerateContentConfig(**config_args) if config_args else None
            )

            # Convert parts to the format expected by Gemini API
            formatted_parts = []
            for part in parts:
                if isinstance(part, dict):
                    if "text" in part:
                        formatted_parts.append(types.Part(text=part["text"]))
                    elif "inline_data" in part:
                        formatted_parts.append(
                            types.Part(inline_data=part["inline_data"])
                        )
                else:
                    # Assume it's a string or other supported type
                    formatted_parts.append(part)

            response = self.client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": formatted_parts}],
                config=generation_config,
            )
            response_text = response.text

            self.logger.debug(f"Received response from Gemini: {response_text}")

            # Update initial response description based on input parameters
            description = f"{prompt}\n"
            description += f"**Model:** {model}\n"
            description += f"**System Instruction:** {system_instruction}\n"
            description += (
                f"**Frequency Penalty:** {frequency_penalty}\n"
                if frequency_penalty
                else ""
            )
            description += (
                f"**Presence Penalty:** {presence_penalty}\n"
                if presence_penalty
                else ""
            )
            description += f"**Seed:** {seed}\n" if seed else ""
            description += f"**Temperature:** {temperature}\n" if temperature else ""
            description += f"**Nucleus Sampling:** {top_p}\n" if top_p else ""
            await ctx.send_followup(
                embed=Embed(
                    title="Prompt",
                    description=description,
                    color=Colour.green(),
                )
            )

            # Assemble the response
            embeds = []
            append_response_embeds(embeds, response_text)

            if not embeds:
                await ctx.send_followup("No response generated.")
                return

            # Send the first part of the response as a brand new message
            message = await ctx.channel.send(embed=embeds[0])
            main_conversation_id = message.id
            self.message_to_conversation_id[main_conversation_id] = main_conversation_id

            # Create the view with buttons and attach it to the new message
            view = ButtonView(
                cog=self,
                conversation_starter=ctx.author,
                conversation_id=main_conversation_id,
            )
            self.views[ctx.author] = view
            await message.edit(view=view)

            # Store the conversation details
            params = ChatCompletionParameters(
                model=model,
                system_instruction=system_instruction,
                conversation_starter=ctx.author,
                channel_id=ctx.channel.id,
                conversation_id=main_conversation_id,
                temperature=temperature,
                top_p=top_p,
            )
            history = [
                {"role": "user", "parts": parts},
                {"role": "model", "parts": [{"text": response_text}]},
            ]
            conversation_wrapper = Conversation(params=params, history=history)
            self.conversations[main_conversation_id] = conversation_wrapper

            # Send any remaining embeds as separate messages
            for embed in embeds[1:]:
                followup_message = await ctx.channel.send(embed=embed, view=view)
                self.message_to_conversation_id[followup_message.id] = (
                    main_conversation_id
                )

        except Exception as e:
            description = str(e)
            self.logger.error(
                f"Error in converse: {description}",
                exc_info=True,
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

        finally:
            if typing_task:
                typing_task.cancel()

    @slash_command(
        name="generate_image",
        description="Generates an image based on a prompt.",
        guild_ids=GUILD_IDS,
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "model",
        description="Choose between Gemini or Imagen models. (default: Gemini 2.0 Flash Preview Image Generation)",
        required=False,
        choices=[
            OptionChoice(
                name="Gemini 2.0 Flash Preview Image Generation",
                value="gemini-2.0-flash-preview-image-generation",
            ),
            OptionChoice(name="Imagen 3", value="imagen-3.0-generate-001"),
            OptionChoice(name="Imagen 4", value="imagen-4.0-generate-preview-06-06"),
            OptionChoice(
                name="Imagen 4 Ultra", value="imagen-4.0-ultra-generate-preview-06-06"
            ),
        ],
        type=str,
    )
    @option(
        "number_of_images",
        description="Number of images to generate (1-4). (default: 1)",
        required=False,
        type=int,
        min_value=1,
        max_value=4,
    )
    @option(
        "aspect_ratio",
        description="Aspect ratio of the generated image. (default: 1:1)",
        required=False,
        choices=[
            OptionChoice(name="Square (1:1)", value="1:1"),
            OptionChoice(name="Portrait (3:4)", value="3:4"),
            OptionChoice(name="Landscape (4:3)", value="4:3"),
            OptionChoice(name="Portrait (9:16)", value="9:16"),
            OptionChoice(name="Landscape (16:9)", value="16:9"),
        ],
        type=str,
    )
    @option(
        "person_generation",
        description="(Imagen only) Control generation of people in images. (default: allow_adult)",
        required=False,
        choices=[
            OptionChoice(name="Don't Allow", value="dont_allow"),
            OptionChoice(name="Allow Adults", value="allow_adult"),
            OptionChoice(name="Allow All", value="allow_all"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="(Gemini only) Image to edit. Upload an image for image editing tasks. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "negative_prompt",
        description="(Advanced) Description of what to discourage in the generated images. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "seed",
        description="(Advanced) Random seed for image generation. (default: not set)",
        required=False,
        type=int,
    )
    @option(
        "guidance_scale",
        description="(Advanced) Controls adherence to prompt. Ranges from 0 to 20. (default: not set)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=20.0,
    )
    async def generate_image(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "gemini-2.0-flash-preview-image-generation",
        number_of_images: int = 1,
        aspect_ratio: str = "1:1",
        person_generation: str = "allow_adult",
        attachment: Optional[Attachment] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ):
        """
        Generates images from a prompt using either Gemini or Imagen models.

        This function supports both Google's Gemini and Imagen image generation models,
        automatically selecting the appropriate API based on the model chosen:

        Gemini Models (via generate_content API):
        - Text-to-image generation with response_modalities=['TEXT', 'IMAGE']
        - Image editing capabilities when attachments are provided
        - Multiple image generation via candidate_count
        - Seed control for reproducible outputs
        - Returns both text and image responses

        Imagen Models (via generate_images API):
        - Advanced image generation with full parameter support
        - number_of_images, aspect_ratio, negative_prompt, seed, guidance_scale
        - person_generation controls for content safety
        - No image editing support (attachment not allowed)
        - Image-only responses

        The function uses a clean modular architecture with separate helper methods
        for each model type, ensuring maintainable and extensible code.

        Args:
            ctx: Discord application context
            prompt: Text description of the image to generate
            model: Model to use (Gemini or Imagen variants)
            number_of_images: Number of images to generate (1-4)
            aspect_ratio: Image dimensions (1:1, 3:4, 4:3, 9:16, 16:9)
            person_generation: Control people in images (Imagen only)
            attachment: Image to edit (Gemini only)
            negative_prompt: What to avoid in generation (Imagen only)
            seed: Random seed for reproducible results
            guidance_scale: Text prompt adherence control (Imagen only)

        Returns:
            Discord response with generated images and parameter information
        """
        await ctx.defer()
        try:
            # Create ImageGenerationParameters for clean parameter handling
            image_params = ImageGenerationParameters(
                prompt=prompt,
                model=model,
                number_of_images=number_of_images,
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                negative_prompt=negative_prompt,
                seed=seed,
                guidance_scale=guidance_scale,
            )

            # Check if this is a Gemini or Imagen model and use appropriate API
            is_gemini_model = model.startswith("gemini-")

            # Process response - handle both Gemini and Imagen response formats
            text_response = None
            generated_images = []

            if is_gemini_model:
                # Handle Gemini models using generate_content
                text_response, generated_images = (
                    await self._generate_image_with_gemini(
                        prompt, model, number_of_images, seed, attachment
                    )
                )
            else:
                # Handle Imagen models using generate_images
                if attachment:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Not Supported",
                            description="Image editing is not supported with Imagen models. Please use a Gemini model for image editing.",
                            color=Colour.orange(),
                        )
                    )
                    return

                generated_images = await self._generate_image_with_imagen(image_params)

            # Send response
            if generated_images:
                embed, files = await self._create_image_response_embed(
                    image_params=image_params,
                    generated_images=generated_images,
                    attachment=attachment,
                    text_response=text_response,
                )

                await ctx.send_followup(embed=embed, files=files)
            else:
                # No images generated, but maybe there's text (only for Gemini)
                embed_description = "The model did not generate any images.\n"
                if text_response:
                    embed_description += f"Text response: {text_response}\n"
                elif is_gemini_model:
                    embed_description += (
                        f"Try asking explicitly for image generation.\n"
                    )
                else:
                    embed_description += f"Imagen models should generate images. Check your prompt or try different parameters.\n"

                # Show unsupported parameters info based on model type
                if is_gemini_model:
                    unsupported_params = []
                    if image_params.negative_prompt:
                        unsupported_params.append("negative_prompt")
                    if image_params.guidance_scale:
                        unsupported_params.append("guidance_scale")
                    if image_params.aspect_ratio != "1:1":
                        unsupported_params.append("aspect_ratio")
                    if image_params.person_generation != "allow_adult":
                        unsupported_params.append("person_generation")

                    if unsupported_params:
                        embed_description += f"\n*Note: These parameters are not yet implemented for Gemini: {', '.join(unsupported_params)}*"

                await ctx.send_followup(
                    embed=Embed(
                        title="No Images Generated",
                        description=embed_description,
                        color=Colour.orange(),
                    )
                )

        except Exception as e:
            description = str(e)
            self.logger.error(
                f"Error in generate_image: {description}",
                exc_info=True,
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

    @slash_command(
        name="generate_video",
        description="Generates a video based on a prompt using Veo.",
        guild_ids=GUILD_IDS,
    )
    @option(
        "prompt", description="Prompt for video generation", required=True, type=str
    )
    @option(
        "model",
        description="Choose Veo model. (default: Veo 2.0 Generate)",
        required=False,
        choices=[
            OptionChoice(name="Veo 2.0 Generate", value="veo-2.0-generate-001"),
        ],
        type=str,
    )
    @option(
        "aspect_ratio",
        description="Aspect ratio of the generated video. (default: 16:9)",
        required=False,
        choices=[
            OptionChoice(name="Landscape (16:9)", value="16:9"),
            OptionChoice(name="Portrait (9:16)", value="9:16"),
        ],
        type=str,
    )
    @option(
        "person_generation",
        description="Control generation of people in videos. (default: allow_adult)",
        required=False,
        choices=[
            OptionChoice(name="Don't Allow", value="dont_allow"),
            OptionChoice(name="Allow Adults", value="allow_adult"),
            OptionChoice(name="Allow All", value="allow_all"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="Image to use as the first frame for the video. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "number_of_videos",
        description="Number of videos to generate (1-2). (default: 1)",
        required=False,
        type=int,
        min_value=1,
        max_value=2,
    )
    @option(
        "duration_seconds",
        description="Length of each output video in seconds (5-8). (default: not set)",
        required=False,
        type=int,
        min_value=5,
        max_value=8,
    )
    @option(
        "negative_prompt",
        description="(Advanced) Description of what to discourage. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "enhance_prompt",
        description="(Advanced) Enable or disable the prompt rewriter. (default: True)",
        required=False,
        type=bool,
    )
    async def generate_video(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "veo-2.0-generate-001",
        aspect_ratio: str = "16:9",
        person_generation: str = "allow_adult",
        attachment: Optional[Attachment] = None,
        number_of_videos: int = 1,
        duration_seconds: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        enhance_prompt: Optional[bool] = None,
    ):
        """
        Generates videos from a prompt using Veo models.

        This function uses Google's Veo video generation model to create videos based on text prompts
        and optionally starting from an image. The generation process is asynchronous and can take
        2-6 minutes to complete.

        Veo Features:
        - Text-to-video generation with detailed prompts
        - Image-to-video generation when attachments are provided
        - 5-8 second video duration at 720p resolution and 24fps
        - Support for landscape (16:9) and portrait (9:16) aspect ratios
        - Person generation controls for content safety
        - Advanced parameters like negative prompts and prompt enhancement

        The function handles the long-running operation by polling the API until completion,
        then downloads and sends the generated videos to Discord.

        Args:
            ctx: Discord application context
            prompt: Text description of the video to generate
            model: Veo model to use (currently only veo-2.0-generate-001)
            aspect_ratio: Video dimensions (16:9 or 9:16)
            person_generation: Control people in videos (dont_allow, allow_adult, allow_all)
            attachment: Optional image to use as first frame (image-to-video)
            number_of_videos: Number of videos to generate (1-2)
            duration_seconds: Video length in seconds (5-8)
            negative_prompt: What to avoid in generation
            enhance_prompt: Enable/disable prompt rewriter

        Returns:
            Discord response with generated videos and parameter information
        """
        await ctx.defer()
        try:
            # Create VideoGenerationParameters for clean parameter handling
            video_params = VideoGenerationParameters(
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                negative_prompt=negative_prompt,
                number_of_videos=number_of_videos,
                duration_seconds=duration_seconds,
                enhance_prompt=enhance_prompt,
            )

            # Generate videos using Veo
            generated_videos = await self._generate_video_with_veo(
                video_params, attachment
            )

            # Send response
            if generated_videos:
                embed, files = await self._create_video_response_embed(
                    video_params=video_params,
                    generated_videos=generated_videos,
                    attachment=attachment,
                )

                await ctx.send_followup(embed=embed, files=files)
            else:
                await ctx.send_followup(
                    embed=Embed(
                        title="No Videos Generated",
                        description="The model did not generate any videos. This may be due to resource constraints or safety filters. Please try again with a different prompt or parameters.",
                        color=Colour.orange(),
                    )
                )

        except Exception as e:
            description = str(e)
            self.logger.error(
                f"Error in generate_video: {description}",
                exc_info=True,
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

    async def _generate_image_with_gemini(
        self,
        prompt: str,
        model: str,
        number_of_images: int,
        seed: Optional[int],
        attachment: Optional[Attachment],
    ) -> tuple[Optional[str], List[Image.Image]]:
        """
        Generate images using Gemini models with generate_content API.

        Returns:
            tuple: (text_response, generated_images)
        """
        contents = prompt

        # Add attachment for image editing if provided
        if attachment:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        image = Image.open(BytesIO(image_data))
                        contents = [prompt, image]

        # Create the configuration for image generation
        generate_config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )

        # For additional parameters like seed and candidate count
        if number_of_images and number_of_images > 1:
            generate_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                candidate_count=number_of_images,
            )
        elif seed is not None:
            generate_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"], seed=seed
            )

        gemini_response = self.client.models.generate_content(
            model=model, contents=contents, config=generate_config
        )

        # Process Gemini response from generate_content
        text_response = None
        generated_images = []

        if gemini_response.candidates and len(gemini_response.candidates) > 0:
            candidate = gemini_response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text is not None:
                        text_response = part.text
                    elif hasattr(part, "inline_data") and part.inline_data is not None:
                        if part.inline_data.data:
                            image = Image.open(BytesIO(part.inline_data.data))
                            generated_images.append(image)

        return text_response, generated_images

    async def _generate_image_with_imagen(
        self, image_params: ImageGenerationParameters
    ) -> List[Image.Image]:
        """
        Generate images using Imagen models with generate_images API.

        Returns:
            List of generated PIL Images
        """
        imagen_response = self.client.models.generate_images(
            model=image_params.model,
            prompt=image_params.prompt,
            config=types.GenerateImagesConfig(**image_params.to_dict()),
        )

        # Process Imagen response from generate_images
        generated_images = []
        if (
            hasattr(imagen_response, "generated_images")
            and imagen_response.generated_images
        ):
            for generated_image in imagen_response.generated_images:
                if hasattr(generated_image, "image") and generated_image.image:
                    img = generated_image.image

                    # Extract image data using the canonical image_bytes attribute
                    if hasattr(img, "image_bytes") and img.image_bytes:
                        try:
                            pil_image = Image.open(BytesIO(img.image_bytes))
                            generated_images.append(pil_image)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to convert image_bytes to PIL Image: {e}"
                            )
                            continue
                    else:
                        self.logger.error(
                            f"Image object missing image_bytes attribute. Type: {type(img)}"
                        )
                        continue

        return generated_images

    async def _create_image_response_embed(
        self,
        image_params: ImageGenerationParameters,
        generated_images: List[Image.Image],
        attachment: Optional[Attachment],
        text_response: Optional[str] = None,
    ) -> tuple[Embed, List[File]]:
        """
        Create Discord embed and file attachments for image generation response.

        Returns:
            tuple: (embed, files)
        """
        is_gemini_model = image_params.model.startswith("gemini-")

        # Create files for Discord
        files = []
        for i, image in enumerate(generated_images):
            try:
                image_bytes = BytesIO()
                # All generated_images should be PIL Image objects at this point
                image.save(image_bytes, format="PNG")
                image_bytes.seek(0)
                files.append(File(image_bytes, filename=f"generated_image_{i+1}.png"))
            except Exception as e:
                self.logger.error(f"Failed to save image {i+1}: {e}")
                continue

        description = f"**Prompt:** {image_params.prompt}\n"
        description += f"**Model:** {image_params.model}\n"
        if attachment:
            description += f"**Mode:** Image Editing\n"
        else:
            description += f"**Mode:** Image Generation\n"
        description += f"**Number of Images:** {len(generated_images)}"

        # Show which parameters are currently supported
        if is_gemini_model:
            # For Gemini models, show what's supported
            if image_params.number_of_images > 1:
                description += f" (requested: {image_params.number_of_images})"
            if image_params.seed is not None:
                description += f"\n**Seed:** {image_params.seed}"

            # Note about unsupported parameters for Gemini
            unsupported_params = []
            if image_params.negative_prompt:
                unsupported_params.append(
                    f"negative_prompt: {image_params.negative_prompt}"
                )
            if image_params.guidance_scale:
                unsupported_params.append(
                    f"guidance_scale: {image_params.guidance_scale}"
                )
            if image_params.aspect_ratio != "1:1":
                unsupported_params.append(f"aspect_ratio: {image_params.aspect_ratio}")
            if image_params.person_generation != "allow_adult":
                unsupported_params.append(
                    f"person_generation: {image_params.person_generation}"
                )

            if unsupported_params:
                description += f"\n\n*Note: Advanced parameters not yet implemented for Gemini: {', '.join(unsupported_params)}*"
        else:
            # For Imagen models, show all supported parameters
            if image_params.seed is not None:
                description += f"\n**Seed:** {image_params.seed}"
            if image_params.negative_prompt:
                description += f"\n**Negative Prompt:** {image_params.negative_prompt}"
            if image_params.guidance_scale:
                description += f"\n**Guidance Scale:** {image_params.guidance_scale}"
            if image_params.aspect_ratio != "1:1":
                description += f"\n**Aspect Ratio:** {image_params.aspect_ratio}"
            if image_params.person_generation != "allow_adult":
                description += (
                    f"\n**Person Generation:** {image_params.person_generation}"
                )

        if text_response:
            # Truncate long text responses for the embed
            truncated_text = (
                text_response[:500] + "..."
                if len(text_response) > 500
                else text_response
            )
            description += f"\n\n**AI Response:** {truncated_text}"

        embed = Embed(
            title=f"Images Generated with {'Gemini' if is_gemini_model else 'Imagen'}",
            description=description,
            color=Colour.green(),
        )

        if files:
            embed.set_image(url=f"attachment://{files[0].filename}")

        return embed, files

    async def _generate_video_with_veo(
        self,
        video_params: VideoGenerationParameters,
        attachment: Optional[Attachment] = None,
    ) -> List[str]:
        """
        Generate videos using Veo models with generate_videos API.

        Returns:
            List of generated video file paths
        """
        import time

        # Prepare the generation call
        kwargs = {
            "model": video_params.model,
            "prompt": video_params.prompt,
            "config": types.GenerateVideosConfig(**video_params.to_dict()),
        }

        # Add image if provided for image-to-video generation
        if attachment:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        image = Image.open(BytesIO(image_data))
                        kwargs["image"] = image

        # Start the video generation operation
        operation = self.client.models.generate_videos(**kwargs)

        self.logger.info(f"Started video generation operation: {operation.name}")

        # Poll for completion (this can take 2-6 minutes)
        max_wait_time = 600  # 10 minutes timeout
        start_time = time.time()
        poll_interval = 20  # Poll every 20 seconds

        while not operation.done:
            if time.time() - start_time > max_wait_time:
                raise Exception("Video generation timed out after 10 minutes")

            await asyncio.sleep(poll_interval)
            operation = self.client.operations.get(operation)
            self.logger.debug(f"Operation status: {operation.done}")

        # Process completed operation
        generated_videos = []
        if hasattr(operation, "response") and operation.response:
            if (
                hasattr(operation.response, "generated_videos")
                and operation.response.generated_videos
            ):
                for i, generated_video in enumerate(
                    operation.response.generated_videos
                ):
                    if hasattr(generated_video, "video") and generated_video.video:
                        try:
                            # Download the video file
                            video_file = self.client.files.download(
                                file=generated_video.video
                            )

                            # Save to a temporary file path
                            video_path = f"temp_video_{i}.mp4"
                            generated_video.video.save(video_path)
                            generated_videos.append(video_path)

                            self.logger.info(f"Downloaded video {i+1}: {video_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to download video {i+1}: {e}")
                            continue

        return generated_videos

    async def _create_video_response_embed(
        self,
        video_params: VideoGenerationParameters,
        generated_videos: List[str],
        attachment: Optional[Attachment],
    ) -> tuple[Embed, List[File]]:
        """
        Create Discord embed and file attachments for video generation response.

        Returns:
            tuple: (embed, files)
        """
        # Create files for Discord
        files = []
        for i, video_path in enumerate(generated_videos):
            try:
                files.append(File(video_path, filename=f"generated_video_{i+1}.mp4"))
            except Exception as e:
                self.logger.error(f"Failed to create file for video {i+1}: {e}")
                continue

        description = f"**Prompt:** {video_params.prompt}\n"
        description += f"**Model:** {video_params.model}\n"
        if attachment:
            description += f"**Mode:** Image-to-Video\n"
        else:
            description += f"**Mode:** Text-to-Video\n"
        description += f"**Number of Videos:** {len(generated_videos)}"
        if video_params.number_of_videos and video_params.number_of_videos > len(
            generated_videos
        ):
            description += f" (requested: {video_params.number_of_videos})"

        # Show generation parameters
        if video_params.aspect_ratio:
            description += f"\n**Aspect Ratio:** {video_params.aspect_ratio}"
        if (
            video_params.person_generation
            and video_params.person_generation != "allow_adult"
        ):
            description += f"\n**Person Generation:** {video_params.person_generation}"
        if video_params.duration_seconds:
            description += f"\n**Duration:** {video_params.duration_seconds} seconds"
        if video_params.negative_prompt:
            description += f"\n**Negative Prompt:** {video_params.negative_prompt}"
        if video_params.enhance_prompt is not None:
            description += f"\n**Prompt Enhancement:** {'Enabled' if video_params.enhance_prompt else 'Disabled'}"

        embed = Embed(
            title="Videos Generated with Veo",
            description=description,
            color=Colour.green(),
        )

        # Note about video generation time and storage
        embed.add_field(
            name="ℹ️ Note",
            value="Video generation typically takes 2-6 minutes. Generated videos are stored on the server for 2 days.",
            inline=False,
        )

        return embed, files
