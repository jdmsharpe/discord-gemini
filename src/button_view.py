from discord import (
    ButtonStyle,
    Interaction,
    TextChannel,
)
from discord.ui import button, Button, View
from typing import cast
import logging


class ButtonView(View):
    def __init__(self, cog, conversation_starter, conversation_id):
        """
        Initialize the ButtonView class.
        """
        super().__init__(timeout=None)
        self.cog = cog
        self.conversation_starter = conversation_starter
        self.conversation_id = conversation_id

    @button(emoji="🔄", style=ButtonStyle.green)
    async def regenerate_button(self, interaction: Interaction, _: Button):
        """
        Regenerate the last response for the current conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        logging.info("Regenerate button clicked.")
        try:
            # Check if the interaction user is the one who started the conversation
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to regenerate the response.", ephemeral=True
                )
                return

            if self.conversation_id in self.cog.conversations:
                # Modify the conversation history and regenerate the response
                conversation = self.cog.conversations[self.conversation_id]

                conversation.history.pop()  # Remove the last assistant message
                conversation.history.pop()  # Remove the last user message

                # For now, get the last user message from the channel history
                if hasattr(interaction.channel, 'history'):
                    channel = cast(TextChannel, interaction.channel)
                    messages = [msg async for msg in channel.history(limit=2)]
                    user_message = messages[1]
                else:
                    # Fallback if channel doesn't support history
                    await interaction.response.send_message(
                        "Cannot regenerate in this type of channel.", ephemeral=True
                    )
                    return

                await self.cog.handle_new_message_in_conversation(
                    user_message, conversation
                )
                await interaction.response.send_message(
                    "Response regenerated.", ephemeral=True, delete_after=3
                )
            else:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
        except Exception as e:
            logging.error(f"Error in regenerate_button: {str(e)}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "An error occurred while regenerating the response.", ephemeral=True
                )
            else:
                await interaction.followup.send(
                    "An error occurred while regenerating the response.", ephemeral=True
                )

    @button(emoji="⏯️", style=ButtonStyle.gray)
    async def play_pause_button(self, interaction: Interaction, _: Button):
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
    async def stop_button(self, interaction: Interaction, _: Button):
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
            await interaction.response.send_message(
                "Conversation ended.", ephemeral=True, delete_after=3
            )
        else:
            await interaction.response.send_message(
                "No active conversation found.", ephemeral=True
            )
