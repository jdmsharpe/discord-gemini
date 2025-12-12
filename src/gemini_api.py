# Standard library imports
import asyncio
import logging
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import aiohttp
from PIL import Image

# Discord imports
from discord import Attachment, Colour, Embed, File
from discord.commands import (
    ApplicationContext,
    OptionChoice,
    SlashCommandGroup,
    command,
    option,
)
from discord.ext import commands

# Google AI imports
from google import genai
from google.genai import types

# Local imports
from button_view import ButtonView
from config.auth import GEMINI_API_KEY, GUILD_IDS
from util import (
    ChatCompletionParameters,
    ImageGenerationParameters,
    MusicGenerationParameters,
    SpeechGenerationParameters,
    VideoGenerationParameters,
    chunk_text,
)


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
                color=Colour.dark_blue(),
            )
        )


class GeminiAPI(commands.Cog):
    # Slash command group for all Gemini commands: /gemini <subcommand>
    gemini = SlashCommandGroup("gemini", "Gemini AI commands", guild_ids=GUILD_IDS)

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
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session and not self._http_session.closed:
            return self._http_session
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession()
            return self._http_session

    async def _fetch_attachment_bytes(self, attachment: Attachment) -> Optional[bytes]:
        session = await self._get_http_session()
        try:
            async with session.get(attachment.url) as response:
                if response.status == 200:
                    return await response.read()
                self.logger.warning(
                    "Failed to fetch attachment %s: HTTP %s",
                    attachment.url,
                    response.status,
                )
        except aiohttp.ClientError as error:
            self.logger.warning(
                "Error fetching attachment %s: %s", attachment.url, error
            )
        return None

    def cog_unload(self):
        loop = getattr(self.bot, "loop", None)

        # Close HTTP session
        session = self._http_session
        if session and not session.closed:
            if loop and loop.is_running():
                loop.create_task(session.close())
            else:
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(session.close())
                finally:
                    new_loop.close()
        self._http_session = None

        # Close Gemini clients
        if loop and loop.is_running():
            loop.create_task(self.client.aio.aclose())
        self.client.close()

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

            self.logger.debug(
                f"Starting typing indicator for followup message from {message.author}"
            )
            typing_task = asyncio.create_task(self.keep_typing(message.channel))

            user_parts: List[Union[str, Dict]] = [{"text": message.content}]
            if message.attachments:
                for attachment in message.attachments:
                    image_data = await self._fetch_attachment_bytes(attachment)
                    if image_data is None:
                        continue
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
            response = await self.client.aio.models.generate_content(
                model=params.model,
                contents=contents,
                config=generation_config,
            )
            response_text = response.text
            self.logger.debug(f"Received response from Gemini: {response_text}")

            # Stop typing indicator as soon as we have the response
            if typing_task:
                self.logger.debug(
                    f"Stopping typing indicator for conversation {params.conversation_id}"
                )
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

        self.logger.debug(
            f"Received message from {message.author} in channel {message.channel.id}: '{message.content}'"
        )

        # Check for active conversations in this channel
        for conversation_wrapper in self.conversations.values():
            # Skip conversations that are not in the same channel
            if message.channel.id != conversation_wrapper.params.channel_id:
                continue

            # Skip if the message is not from the conversation starter
            if message.author != conversation_wrapper.params.conversation_starter:
                continue

            self.logger.info(
                f"Processing followup message for conversation {conversation_wrapper.params.conversation_id}"
            )
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

    @gemini.command(
        name="converse",
        description="Starts a conversation with a model.",
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
        description="Choose from the following Gemini models. (default: Gemini 3.0 Pro)",
        required=False,
        choices=[
            OptionChoice(name="Gemini 3.0 Pro", value="gemini-3-pro-preview"),
            OptionChoice(name="Gemini 2.5 Pro", value="gemini-2.5-pro"),
            OptionChoice(name="Gemini 2.5 Flash", value="gemini-2.5-flash"),
            OptionChoice(name="Gemini 2.5 Flash Lite", value="gemini-2.5-flash-lite"),
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
        description="Attachment to append to the prompt. (default: not set)",
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
        model: str = "gemini-3-pro-preview",
        system_instruction: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        attachment: Optional[Attachment] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """
        Creates a persistent conversation session with a Gemini model.

        Initiates an interactive conversation with context preservation across multiple exchanges.
        Supports multimodal inputs (text + images) and provides interactive UI controls for
        conversation management.

        Args:
            ctx: Discord application context
            prompt: Initial conversation prompt or question
            model: Gemini model variant (default: gemini-3-pro-preview)
            system_instruction: Optional behavioral guidelines for the AI
            frequency_penalty: Controls repetition reduction (experimental)
            presence_penalty: Controls topic focus (experimental)
            seed: Random seed for deterministic responses
            attachment: Optional image attachment for multimodal input
            temperature: Response creativity (0.0 conservative → 2.0 creative)
            top_p: Nucleus sampling threshold (0.0 focused → 1.0 diverse)

        Returns:
            Discord response with initial AI message and interactive conversation controls

        Note:
            Only one conversation per user per channel allowed. Conversations persist until
            explicitly ended or bot restarts. Follow-up messages automatically handled.
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
                image_data = await self._fetch_attachment_bytes(attachment)
                if image_data is not None:
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

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": formatted_parts}],
                config=generation_config,
            )
            response_text = response.text

            self.logger.debug(f"Received response from Gemini: {response_text}")

            # Update initial response description based on input parameters
            description = f"**Prompt:** {prompt}\n"
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

            # Assemble all embeds for a single message
            embeds = [
                Embed(
                    title="Conversation Started",
                    description=description,
                    color=Colour.green(),
                )
            ]
            append_response_embeds(embeds, response_text)

            if len(embeds) == 1:
                await ctx.send_followup("No response generated.")
                return

            # Create the view with buttons
            main_conversation_id = ctx.interaction.id
            view = ButtonView(
                cog=self,
                conversation_starter=ctx.author,
                conversation_id=main_conversation_id,
            )
            self.views[ctx.author] = view

            # Send all embeds as a single message with buttons
            message = await ctx.send_followup(embeds=embeds, view=view)
            self.message_to_conversation_id[message.id] = main_conversation_id

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

    @gemini.command(
        name="image",
        description="Generates an image based on a prompt.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "model",
        description="Choose between Gemini or Imagen models. (default: Gemini 3.0 Pro Image)",
        required=False,
        choices=[
            OptionChoice(
                name="Gemini 3.0 Pro Image", value="gemini-3-pro-image-preview"
            ),
            OptionChoice(name="Gemini 2.5 Flash Image", value="gemini-2.5-flash-image"),
            OptionChoice(name="Imagen 4", value="imagen-4.0-generate-001"),
            OptionChoice(name="Imagen 4 Ultra", value="imagen-4.0-ultra-generate-001"),
            OptionChoice(name="Imagen 4 Fast", value="imagen-4.0-fast-generate-001"),
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
        description="(Advanced) Controls adherence to prompt. Ranges from 0.0 to 20.0. (default: not set)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=20.0,
    )
    async def generate_image(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "gemini-3-pro-image-preview",
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
                    # Truncate text response to avoid exceeding Discord's 4096 char limit
                    # Reserve ~200 chars for the rest of the message
                    max_text_length = 3800
                    truncated_text = (
                        text_response[:max_text_length] + "..."
                        if len(text_response) > max_text_length
                        else text_response
                    )
                    embed_description += f"Text response: {truncated_text}\n"
                elif is_gemini_model:
                    embed_description += (
                        f"Try asking explicitly for image generation (e.g., 'Generate an image of...').\n"
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

    @gemini.command(
        name="video",
        description="Generates a video based on a prompt using Veo.",
    )
    @option(
        "prompt", description="Prompt for video generation", required=True, type=str
    )
    @option(
        "model",
        description="Choose Veo model for video generation. (default: Veo 3.1 Preview)",
        required=False,
        choices=[
            OptionChoice(name="Veo 3.1 Preview", value="veo-3.1-generate-preview"),
            OptionChoice(
                name="Veo 3.1 Fast Preview", value="veo-3.1-fast-generate-preview"
            ),
            OptionChoice(name="Veo 3", value="veo-3.0-generate-001"),
            OptionChoice(name="Veo 3 Fast", value="veo-3.0-fast-generate-001"),
            OptionChoice(name="Veo 2", value="veo-2.0-generate-001"),
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
        description="Number of videos to generate, from 1 to 2. (default: 1)",
        required=False,
        type=int,
        min_value=1,
        max_value=2,
    )
    @option(
        "duration_seconds",
        description="Length of each output video in seconds, from 5 to 8 seconds. (default: not set)",
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
        model: str = "veo-3.1-generate-preview",
        aspect_ratio: str = "16:9",
        person_generation: str = "allow_adult",
        attachment: Optional[Attachment] = None,
        number_of_videos: int = 1,
        duration_seconds: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        enhance_prompt: Optional[bool] = None,
    ):
        """
        Generates videos from a prompt using Veo models (2.0, 3.0, or 3.1).

        This function uses Google's Veo video generation models to create videos based on text prompts
        and optionally starting from an image. The generation process is asynchronous and can take
        2-6 minutes to complete.

        Veo Features:
        - Text-to-video generation with detailed prompts
        - Image-to-video generation when attachments are provided
        - 5-8 second video duration at 720p or 1080p resolution and 24fps
        - Support for landscape (16:9) and portrait (9:16) aspect ratios
        - Person generation controls for content safety
        - Advanced parameters like negative prompts and prompt enhancement
        - Veo 3.1: Native audio generation, video extension, reference image support

        The function handles the long-running operation by polling the API until completion,
        then downloads and sends the generated videos to Discord.

        Args:
            ctx: Discord application context
            prompt: Text description of the video to generate
            model: Veo model variant (2.0, 3.0, 3.1, or 3.1-fast)
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

    @gemini.command(
        name="tts",
        description="Generates lifelike audio from input text using Gemini text-to-speech.",
    )
    @option(
        "input_text",
        description="Text to convert to speech. Max 32k tokens.",
        required=True,
        type=str,
    )
    @option(
        "model",
        description="Choose Gemini text-to-speech model. (default: Gemini 2.5 Flash Preview TTS)",
        required=False,
        choices=[
            OptionChoice(
                name="Gemini 2.5 Flash Preview TTS",
                value="gemini-2.5-flash-preview-tts",
            ),
            OptionChoice(
                name="Gemini 2.5 Pro Preview TTS", value="gemini-2.5-pro-preview-tts"
            ),
        ],
        type=str,
    )
    @option(
        "voice_name",
        description="Voice to use for single-speaker text-to-speech. (default: Kore)",
        required=False,
        choices=[
            OptionChoice(name="Kore (Firm)", value="Kore"),
            OptionChoice(name="Puck (Upbeat)", value="Puck"),
            OptionChoice(name="Charon (Informative)", value="Charon"),
            OptionChoice(name="Zephyr (Bright)", value="Zephyr"),
            OptionChoice(name="Fenrir (Excitable)", value="Fenrir"),
            OptionChoice(name="Leda (Youthful)", value="Leda"),
            OptionChoice(name="Orus (Firm)", value="Orus"),
            OptionChoice(name="Aoede (Breezy)", value="Aoede"),
            OptionChoice(name="Callirrhoe (Easy-going)", value="Callirrhoe"),
            OptionChoice(name="Autonoe (Bright)", value="Autonoe"),
            OptionChoice(name="Enceladus (Breathy)", value="Enceladus"),
            OptionChoice(name="Iapetus (Clear)", value="Iapetus"),
            OptionChoice(name="Umbriel (Easy-going)", value="Umbriel"),
            OptionChoice(name="Algieba (Smooth)", value="Algieba"),
            OptionChoice(name="Despina (Smooth)", value="Despina"),
            OptionChoice(name="Erinome (Clear)", value="Erinome"),
            OptionChoice(name="Algenib (Gravelly)", value="Algenib"),
            OptionChoice(name="Rasalgethi (Informative)", value="Rasalgethi"),
            OptionChoice(name="Laomedeia (Upbeat)", value="Laomedeia"),
            OptionChoice(name="Achernar (Soft)", value="Achernar"),
            OptionChoice(name="Alnilam (Firm)", value="Alnilam"),
            OptionChoice(name="Schedar (Even)", value="Schedar"),
            OptionChoice(name="Gacrux (Mature)", value="Gacrux"),
            OptionChoice(name="Achird (Friendly)", value="Achird"),
            OptionChoice(name="Sulafat (Warm)", value="Sulafat"),
        ],
        type=str,
    )
    @option(
        "style_prompt",
        description="Natural language instructions to control voice style, tone, accent, pace. (default: not set)",
        required=False,
        type=str,
    )
    async def text_to_speech(
        self,
        ctx: ApplicationContext,
        input_text: str,
        model: str = "gemini-2.5-flash-preview-tts",
        voice_name: str = "Kore",
        style_prompt: Optional[str] = None,
    ):
        """
        Generates audio from input text using Gemini's native text-to-speech capabilities.

        This function uses Google's Gemini TTS models to convert text into lifelike audio.
        The TTS capability supports natural language control over style, accent, pace, and tone.

        Features:
        - Single-speaker TTS with 30 prebuilt voice options
        - Natural language style control through prompts
        - Support for 24 languages with automatic detection
        - High-quality audio output in WAV format
        - Up to 32k token context window

        Args:
            ctx: Discord application context
            input_text: Text to convert to speech (max 32k tokens)
            model: Gemini TTS model to use
            voice_name: Voice option for speech generation
            style_prompt: Natural language instructions for voice control

        Returns:
            Discord response with generated audio file and parameter information
        """
        await ctx.defer()

        try:
            # Prepare the text input with optional style guidance
            content_text = input_text
            if style_prompt:
                content_text = f"{style_prompt}: {input_text}"

            # Create TTS parameters
            tts_params = SpeechGenerationParameters(
                input_text=content_text,
                model=model,
                voice_name=voice_name,
                multi_speaker=False,
                style_prompt=style_prompt,
            )

            # Generate audio using Gemini TTS
            audio_data = await self._generate_speech_with_gemini(tts_params)

            if audio_data:
                # Create temporary audio file
                audio_file_path = Path(f"tts_output_{voice_name}.wav")

                # Write audio data to WAV file
                with wave.open(str(audio_file_path), "wb") as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(24000)  # 24kHz sample rate
                    wf.writeframes(audio_data)

                # Create response embed
                description = f"**Text:** {input_text[:500]}{'...' if len(input_text) > 500 else ''}\n"
                description += f"**Model:** {model}\n"
                description += f"**Voice:** {voice_name}\n"
                if style_prompt:
                    description += f"**Style Instructions:** {style_prompt}\n"
                description += f"**Format:** WAV (24kHz, 16-bit, Mono)\n"

                embed = Embed(
                    title="Text-to-Speech Generation",
                    description=description,
                    color=Colour.dark_blue(),
                )

                # Send the audio file
                await ctx.send_followup(embed=embed, file=File(audio_file_path))

                # Clean up the temporary file
                audio_file_path.unlink(missing_ok=True)
            else:
                await ctx.send_followup(
                    embed=Embed(
                        title="No Audio Generated",
                        description="The model did not generate any audio. Please try again with different text or parameters.",
                        color=Colour.orange(),
                    )
                )

        except Exception as e:
            description = str(e)
            self.logger.error(
                f"Error in text_to_speech: {description}",
                exc_info=True,
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

    @gemini.command(
        name="music",
        description="Generate instrumental music using Gemini's Lyria RealTime model.",
    )
    @option(
        "prompt",
        description="Musical prompt describing genre, mood, instruments, or style.",
        required=True,
        type=str,
    )
    @option(
        "duration",
        description="Duration of music to generate in seconds. Max is 120. (default: 30)",
        required=False,
        type=int,
        min_value=5,
        max_value=120,
    )
    @option(
        "bpm",
        description="Beats per minute from 60 to 200. Leave empty for model to decide.",
        required=False,
        type=int,
        min_value=60,
        max_value=200,
    )
    @option(
        "scale",
        description="Scale/key for the music.",
        required=False,
        choices=[
            OptionChoice(name="C Major / A Minor", value="C_MAJOR_A_MINOR"),
            OptionChoice(name="D♭ Major / B♭ Minor", value="D_FLAT_MAJOR_B_FLAT_MINOR"),
            OptionChoice(name="D Major / B Minor", value="D_MAJOR_B_MINOR"),
            OptionChoice(name="E♭ Major / C Minor", value="E_FLAT_MAJOR_C_MINOR"),
            OptionChoice(name="E Major / C# Minor", value="E_MAJOR_D_FLAT_MINOR"),
            OptionChoice(name="F Major / D Minor", value="F_MAJOR_D_MINOR"),
            OptionChoice(name="G♭ Major / E♭ Minor", value="G_FLAT_MAJOR_E_FLAT_MINOR"),
            OptionChoice(name="G Major / E Minor", value="G_MAJOR_E_MINOR"),
            OptionChoice(name="A♭ Major / F Minor", value="A_FLAT_MAJOR_F_MINOR"),
            OptionChoice(name="A Major / F# Minor", value="A_MAJOR_G_FLAT_MINOR"),
            OptionChoice(name="B♭ Major / G Minor", value="B_FLAT_MAJOR_G_MINOR"),
            OptionChoice(name="B Major / G# Minor", value="B_MAJOR_A_FLAT_MINOR"),
        ],
        type=str,
    )
    @option(
        "density",
        description="(Advanced) Musical density from 0.0 top 1.0. Lower is sparser, Higher is busier. (default: not set)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=1.0,
    )
    @option(
        "brightness",
        description="(Advanced) Tonal brightness from 0.0 to 1.0. Higher is brighter sound. (default: not set)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=1.0,
    )
    @option(
        "guidance",
        description="(Advanced) Prompt adherence from 0.0 to 6.0. Higher is more faithful to prompt. (default: 4.0)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=6.0,
    )
    async def generate_music(
        self,
        ctx: ApplicationContext,
        prompt: str,
        duration: int = 30,
        bpm: Optional[int] = None,
        scale: Optional[str] = None,
        density: Optional[float] = None,
        brightness: Optional[float] = None,
        guidance: float = 4.0,
    ):
        """
        Generate instrumental music using Gemini's Lyria RealTime model.

        This function uses Google's state-of-the-art real-time streaming music generation
        model to create instrumental music based on text prompts. The model supports
        real-time control over various musical parameters.

        Features:
        - Real-time streaming music generation
        - Support for various genres, instruments, and moods
        - Controllable parameters: BPM, scale, density, brightness
        - High-quality stereo audio output (48kHz, 16-bit)
        - Instrumental music only

        Args:
            ctx: Discord application context
            prompt: Musical prompt describing desired style/characteristics
            duration: Length of music to generate in seconds
            bpm: Beats per minute (optional, model decides if not set)
            scale: Musical scale/key for generation
            density: Musical density (sparseness vs business)
            brightness: Tonal brightness (frequency emphasis)
            guidance: How strictly to follow the prompt

        Returns:
            Discord response with generated music file and parameter information
        """
        await ctx.defer()

        try:
            # Create music generation parameters
            music_params = MusicGenerationParameters(
                prompts=[prompt],
                duration=duration,
                bpm=bpm,
                scale=scale,
                guidance=guidance,
                density=density,
                brightness=brightness,
            )

            # Generate music using Lyria RealTime
            audio_data = await self._generate_music_with_lyria(music_params)

            if audio_data:
                # Create temporary audio file
                audio_file_path = Path(f"music_output_{int(time.time())}.wav")

                # Write audio data to WAV file (stereo, 48kHz, 16-bit)
                with wave.open(str(audio_file_path), "wb") as wf:
                    wf.setnchannels(2)  # Stereo
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(48000)  # 48kHz sample rate
                    wf.writeframes(audio_data)

                # Create response embed
                description = f"**Prompt:** {prompt}\n"
                description += f"**Model:** Lyria RealTime\n"
                description += f"**Duration:** {duration} seconds\n"
                if bpm is not None:
                    description += f"**BPM:** {bpm}\n"
                if scale is not None:
                    description += f"**Scale:** {scale.replace('_', ' ')}\n"
                if density is not None:
                    description += f"**Density:** {density}\n"
                if brightness is not None:
                    description += f"**Brightness:** {brightness}\n"
                description += f"**Guidance:** {guidance}\n"
                description += f"**Format:** WAV (48kHz, 16-bit, Stereo)\n"

                embed = Embed(
                    title="Music Generation",
                    description=description,
                    color=Colour.dark_blue(),
                )

                # Send the audio file
                await ctx.send_followup(embed=embed, file=File(audio_file_path))

                # Clean up the temporary file
                audio_file_path.unlink(missing_ok=True)
            else:
                await ctx.send_followup(
                    embed=Embed(
                        title="No Music Generated",
                        description="The model did not generate any music. Please try again with a different prompt or parameters.",
                        color=Colour.orange(),
                    )
                )

        except Exception as e:
            description = str(e)
            self.logger.error(
                f"Error in generate_music: {description}",
                exc_info=True,
            )

            # Provide more helpful error messages for common issues
            if "Music generation is currently unavailable" in description:
                embed_title = "Music Generation Unavailable"
                embed_color = Colour.orange()
            elif "Authentication error" in description:
                embed_title = "Authentication Error"
                embed_color = Colour.red()
            elif "404" in description:
                embed_title = "Service Not Available"
                embed_color = Colour.orange()
                description = (
                    "The Lyria RealTime music generation service is not available. "
                    "This feature may not be enabled for your account or region. "
                    "Please check Google AI Studio for availability."
                )
            else:
                embed_title = "Music Generation Error"
                embed_color = Colour.red()

            await ctx.send_followup(
                embed=Embed(
                    title=embed_title, description=description, color=embed_color
                )
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
        # Explicitly request image generation to reduce text-only responses
        if attachment:
            # For image editing, keep the prompt as-is for more natural edits
            contents = prompt
        else:
            # For generation, explicitly request an image
            contents = f"Generate an image: {prompt}"

        # Add attachment for image editing if provided
        if attachment:
            image_data = await self._fetch_attachment_bytes(attachment)
            if image_data:
                try:
                    image = Image.open(BytesIO(image_data))
                except Exception as error:
                    self.logger.warning(
                        "Failed to open attachment for image generation: %s", error
                    )
                else:
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

        gemini_response = await self.client.aio.models.generate_content(
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
        imagen_response = await self.client.aio.models.generate_images(
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
            title=f"{'Gemini' if is_gemini_model else 'Imagen'} Image Generation",
            description=description,
            color=Colour.dark_blue(),
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

        # Prepare the generation call
        kwargs = {
            "model": video_params.model,
            "prompt": video_params.prompt,
            "config": types.GenerateVideosConfig(**video_params.to_dict()),
        }

        # Add image if provided for image-to-video generation
        if attachment:
            image_data = await self._fetch_attachment_bytes(attachment)
            if image_data:
                try:
                    image = Image.open(BytesIO(image_data))
                except Exception as error:
                    self.logger.warning(
                        "Failed to open attachment for video generation: %s", error
                    )
                else:
                    kwargs["image"] = image

        # Start the video generation operation
        operation = await self.client.aio.models.generate_videos(**kwargs)

        self.logger.info(f"Started video generation operation: {operation.name}")

        # Poll for completion (this can take 2-6 minutes)
        max_wait_time = 600  # 10 minutes timeout
        start_time = time.time()
        poll_interval = 20  # Poll every 20 seconds

        while not operation.done:
            if time.time() - start_time > max_wait_time:
                raise Exception("Video generation timed out after 10 minutes")

            await asyncio.sleep(poll_interval)
            operation = await self.client.aio.operations.get(operation)
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
                            # Note: async files.download doesn't support Video objects,
                            # so we use the sync version via to_thread
                            await asyncio.to_thread(
                                self.client.files.download,
                                file=generated_video.video,
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
            title="Video Generation",
            description=description,
            color=Colour.dark_blue(),
        )

        return embed, files

    async def _generate_music_with_lyria(
        self, music_params: MusicGenerationParameters
    ) -> Optional[bytes]:
        """
        Generate music using Gemini's Lyria RealTime model.

        Args:
            music_params: Music generation parameters

        Returns:
            Raw audio data as bytes, or None if generation failed
        """
        try:
            # Create a client specifically for music generation with proper API version
            music_client = genai.Client(
                api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"}
            )

            # Collect audio chunks
            audio_chunks = []
            stop_receiving = False

            async def receive_audio(session):
                """Background task to process incoming audio."""
                self.logger.info("Audio receiver task started.")
                try:
                    # Iterate through the async generator to receive messages with timeout
                    async for message in session.receive():
                        if stop_receiving:
                            self.logger.info(
                                "Stop signal received, breaking from audio receiver"
                            )
                            break

                        if hasattr(message, "server_content") and hasattr(
                            message.server_content, "audio_chunks"
                        ):
                            audio_data = message.server_content.audio_chunks[0].data
                            if audio_data:
                                audio_chunks.append(audio_data)
                                self.logger.debug(
                                    f"Received audio chunk, size: {len(audio_data)} bytes"
                                )
                except asyncio.CancelledError:
                    self.logger.info("Audio receiver task cancelled")
                    raise
                except Exception as e:
                    # This might catch errors if the connection is closed unexpectedly
                    self.logger.error(f"Error in audio receiver: {e}")
                finally:
                    self.logger.info("Audio receiver task finished.")

            # Connect to Lyria RealTime using WebSocket with proper structure
            try:
                async with music_client.aio.live.music.connect(
                    model="models/lyria-realtime-exp"
                ) as session:
                    # Set up task to receive server messages
                    receiver_task = asyncio.create_task(receive_audio(session))

                    try:
                        # Send initial prompts and config following the official pattern
                        await session.set_weighted_prompts(
                            prompts=[
                                types.WeightedPrompt(
                                    text=prompt_data["text"],
                                    weight=prompt_data["weight"],
                                )
                                for prompt_data in music_params.to_weighted_prompts()
                            ]
                        )

                        # Set music generation configuration
                        config_dict = music_params.to_music_config()

                        # Convert scale string to enum if provided
                        if music_params.scale:
                            scale_enum = getattr(types.Scale, music_params.scale, None)
                            if scale_enum:
                                config_dict["scale"] = scale_enum

                        await session.set_music_generation_config(
                            config=types.LiveMusicGenerationConfig(**config_dict)
                        )

                        # Start streaming music
                        await session.play()

                        # Wait for the specified duration to collect audio
                        self.logger.info(
                            f"Waiting {music_params.duration} seconds for music generation"
                        )
                        await asyncio.sleep(music_params.duration)

                        # Stop the session
                        self.logger.info("Stopping music generation session")
                        await session.stop()

                        # Signal the receiver to stop and wait a bit for final chunks
                        self.logger.info("Signaling receiver to stop")
                        stop_receiving = True

                    finally:
                        # Cancel the receiver task if it's still running
                        if not receiver_task.done():
                            self.logger.info("Cancelling receiver task")
                            receiver_task.cancel()

                        # Give the task a moment to finish cancellation
                        try:
                            await asyncio.sleep(0.1)  # Brief pause for cleanup
                        except Exception:
                            pass

                # Combine all audio chunks
                if audio_chunks:
                    self.logger.info(
                        f"Successfully collected {len(audio_chunks)} audio chunks"
                    )
                    return b"".join(audio_chunks)
                else:
                    self.logger.warning("No audio chunks received from Lyria RealTime")
                    return None

            except Exception as websocket_error:
                # Handle specific WebSocket connection errors
                if "404" in str(websocket_error):
                    self.logger.error("Lyria RealTime endpoint not found (404).")
                    raise Exception(
                        "Music generation is currently unavailable. This could be due to:\n"
                        "1. The Lyria RealTime model is not available in your region\n"
                        "2. Your account doesn't have access to the music generation API\n"
                        "3. The service is temporarily unavailable\n\n"
                        "Please check Google AI Studio or contact support for more information."
                    )
                elif "401" in str(websocket_error) or "403" in str(websocket_error):
                    self.logger.error(
                        "Authentication/authorization error for Lyria RealTime"
                    )
                    raise Exception(
                        "Authentication error: Please check your API key permissions for music generation."
                    )
                else:
                    self.logger.error(f"WebSocket connection error: {websocket_error}")
                    raise Exception(f"Connection error: {websocket_error}")

        except Exception as e:
            if "Music generation is currently unavailable" in str(
                e
            ) or "Authentication error" in str(e):
                # Re-raise our custom error messages
                raise
            else:
                self.logger.error(f"Error generating music with Lyria RealTime: {e}")
                raise Exception(f"Music generation failed: {e}")

    async def _generate_speech_with_gemini(
        self, tts_params: SpeechGenerationParameters
    ) -> Optional[bytes]:
        """
        Generate speech using Gemini TTS models.

        Args:
            tts_params: TTS parameters containing input text, model, voice, etc.

        Returns:
            Raw audio data as bytes, or None if generation failed
        """
        try:
            # Generate speech using Gemini TTS
            response = await self.client.aio.models.generate_content(
                model=tts_params.model,
                contents=tts_params.input_text,
                config=types.GenerateContentConfig(**tts_params.to_dict()),
            )

            # Extract audio data from response
            if (
                response.candidates
                and len(response.candidates) > 0
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        if part.inline_data.data:
                            return part.inline_data.data

            self.logger.warning("No audio data found in Gemini TTS response")
            return None

        except Exception as e:
            self.logger.error(f"Error generating speech with Gemini: {e}")
            raise
