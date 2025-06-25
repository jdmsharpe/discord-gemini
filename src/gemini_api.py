from discord.ext import commands
from google import genai
from google.genai import types
from discord import (
    ApplicationContext,
    Attachment,
    Colour,
    Embed,
)
from discord.commands import command, option, OptionChoice, slash_command
from typing import Optional, Dict, List, Union, Any
from util import (
    ChatCompletionParameters,
    chunk_text,
)
import aiohttp
import asyncio
from button_view import ButtonView
import logging
from config.auth import GUILD_IDS, GEMINI_API_KEY
from dataclasses import dataclass


@dataclass
class Conversation:
    """A dataclass to store conversation state."""

    params: ChatCompletionParameters
    history: List[Dict[str, Any]]


def append_response_embeds(embeds, response_text):
    # Ensure each chunk is no larger than 4096 characters (max Discord embed description length)
    for index, chunk in enumerate(chunk_text(response_text), start=1):
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

            typing_task = asyncio.create_task(self.keep_typing(message.channel))

            user_parts: List[Union[str, Dict]] = [message.content]
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

            generation_config = (
                types.GenerateContentConfig(**config_args) if config_args else None
            )

            response = self.client.models.generate_content(
                model=params.model,
                contents=history,
                config=generation_config,
            )
            response_text = response.text
            self.logger.debug(f"Received response from Gemini: {response_text}")

            history.append({"role": "model", "parts": [{"text": response_text}]})

            append_response_embeds(embeds, response_text)

            view = self.views.get(message.author)
            main_conversation_id = conversation_wrapper.params.conversation_id

            if embeds:
                # Send the first embed as a direct reply to the user's message
                reply_message = await message.reply(embed=embeds[0], view=view)
                self.message_to_conversation_id[reply_message.id] = main_conversation_id

                # Send any remaining embeds as separate messages in the same channel
                for embed in embeds[1:]:
                    followup_message = await message.channel.send(
                        embed=embed, view=view
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
            await message.reply(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

        finally:
            if typing_task:
                typing_task.cancel()

    async def keep_typing(self, channel):
        while True:
            async with channel.typing():
                await asyncio.sleep(5)  # Resend typing indicator every 5 seconds

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

        # Use the new mapping to find the conversation
        if (
            message.reference
            and message.reference.message_id in self.message_to_conversation_id
        ):
            main_conversation_id = self.message_to_conversation_id[
                message.reference.message_id
            ]
            if main_conversation_id in self.conversations:
                conversation_wrapper = self.conversations[main_conversation_id]
                await self.handle_new_message_in_conversation(
                    message, conversation_wrapper
                )

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
        "temperature",
        description="A value between 0.0 and 1.0. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "top_p",
        description="A value between 0.0 and 1.0. (default: not set)",
        required=False,
        type=float,
    )
    async def converse(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "gemini-2.5-flash",
        system_instruction: Optional[str] = None,
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

            parts: List[Union[str, Dict]] = [prompt]
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
            if system_instruction:
                config_args["system_instruction"] = system_instruction
            if temperature is not None:
                config_args["temperature"] = temperature
            if top_p is not None:
                config_args["top_p"] = top_p

            generation_config = (
                types.GenerateContentConfig(**config_args) if config_args else None
            )

            response = self.client.models.generate_content(
                model=model,
                contents=parts,
                config=generation_config,
            )
            response_text = response.text

            self.logger.debug(f"Received response from Gemini: {response_text}")

            # Update initial response description based on input parameters
            description = ""
            description += f"**Prompt:** {prompt}\n"
            description += f"**Model:** {model}\n"
            description += f"**System Instruction:** {system_instruction}\n"
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
