from typing import Union, TYPE_CHECKING, cast

from discord import (
    ButtonStyle,
    Interaction,
    Member,
    TextChannel,
    User,
)
from discord.ui import button, Button, View
import logging

if TYPE_CHECKING:
    from gemini_api import GeminiAPI


class ButtonView(View):
    def __init__(
        self,
        cog: "GeminiAPI",
        conversation_starter: Union[Member, User],
        conversation_id: int,
    ):
        """
        Initialize the ButtonView class.
        """
        super().__init__(timeout=None)
        self.cog = cog
        self.conversation_starter = conversation_starter
        self.conversation_id = conversation_id

    @button(emoji="🔄", style=ButtonStyle.green)
    async def regenerate_button(self, _: Button, interaction: Interaction):
        """
        Regenerate the last response for the current conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        logging.info("Regenerate button clicked.")
        removed_entries = []

        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to regenerate the response.", ephemeral=True
                )
                return

            conversation = self.cog.conversations.get(self.conversation_id)
            if conversation is None:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
                return

            await interaction.response.defer(ephemeral=True)

            if len(conversation.history) < 2:
                await interaction.followup.send(
                    "Not enough history to regenerate yet.", ephemeral=True
                )
                return

            removed_entries = conversation.history[-2:]
            del conversation.history[-2:]

            channel = interaction.channel
            if not hasattr(channel, "history"):
                conversation.history.extend(removed_entries)
                await interaction.followup.send(
                    "Couldn't find the message to regenerate.", ephemeral=True
                )
                return

            # Type narrowing: hasattr check above ensures channel has history()
            text_channel = cast(TextChannel, channel)
            messages = [message async for message in text_channel.history(limit=10)]
            user_message = next(
                (
                    message
                    for message in messages
                    if message.author == self.conversation_starter
                ),
                None,
            )

            if user_message is None:
                conversation.history.extend(removed_entries)
                await interaction.followup.send(
                    "Couldn't find the message to regenerate.", ephemeral=True
                )
                return

            await self.cog.handle_new_message_in_conversation(
                user_message, conversation
            )
            await interaction.followup.send(
                "Response regenerated.", ephemeral=True, delete_after=3
            )
        except Exception as error:
            logging.error(
                f"Error in regenerate_button: {error}",
                exc_info=True,
            )

            if removed_entries:
                conversation = self.cog.conversations.get(self.conversation_id)
                if conversation is not None:
                    conversation.history.extend(removed_entries)

            if interaction.response.is_done():
                await interaction.followup.send(
                    "An error occurred while regenerating the response.", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "An error occurred while regenerating the response.", ephemeral=True
                )

    @button(emoji="⏯️", style=ButtonStyle.gray)
    async def play_pause_button(self, button: Button, interaction: Interaction):
        """
        Pause or resume the conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        # Check if the interaction user is the one who started the conversation
        if interaction.user != self.conversation_starter:
            await interaction.response.send_message(
                "You are not allowed to pause the conversation.", ephemeral=True
            )
            return

        # Toggle the paused state
        if self.conversation_id in self.cog.conversations:
            conversation = self.cog.conversations[self.conversation_id]
            conversation.params.paused = not conversation.params.paused
            status = "paused" if conversation.params.paused else "resumed"
            await interaction.response.send_message(
                f"Conversation {status}. Press again to toggle.",
                ephemeral=True,
                delete_after=3,
            )
        else:
            await interaction.response.send_message(
                "No active conversation found.", ephemeral=True
            )

    @button(emoji="⏹️", style=ButtonStyle.blurple)
    async def stop_button(self, button: Button, interaction: Interaction):
        """
        End the conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        # Check if the interaction user is the one who started the conversation
        if interaction.user != self.conversation_starter:
            await interaction.response.send_message(
                "You are not allowed to end this conversation.", ephemeral=True
            )
            return

        # End the conversation
        if self.conversation_id in self.cog.conversations:
            del self.cog.conversations[self.conversation_id]
            button.disabled = True
            await interaction.response.send_message(
                "Conversation ended.", ephemeral=True, delete_after=3
            )
        else:
            await interaction.response.send_message(
                "No active conversation found.", ephemeral=True
            )
