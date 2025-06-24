from discord import (
    ButtonStyle,
    Interaction,
)
from discord.ui import button, Button, View
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
    async def regenerate_button(self, _: Button, interaction: Interaction):
        # ... (logging and user check)
        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to regenerate the response.", ephemeral=True
                )
                return

            if self.conversation_id in self.cog.conversations:
                # For now, get the last user message from the channel history
                messages = await interaction.channel.history(limit=2).flatten()
                user_message = messages[1]

                await self.cog.handle_new_message_in_conversation(
                    user_message, self.cog.conversations[self.conversation_id]
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
            await interaction.followup.send(
                "An error occurred while regenerating the response.", ephemeral=True
            )

    @button(emoji="⏹️", style=ButtonStyle.blurple)
    async def stop_button(self, button: Button, interaction: Interaction):
        # ... (user check)
        if interaction.user != self.conversation_starter:
            await interaction.response.send_message(
                "You are not allowed to end the conversation.", ephemeral=True
            )
            return

        # Remove the conversation from the histories
        if self.conversation_id in self.cog.conversations:
            del self.cog.conversations[self.conversation_id]
            await interaction.response.send_message(
                "Conversation ended.", ephemeral=True, delete_after=3
            )
        else:
            await interaction.response.send_message(
                "No active conversation found to end.", ephemeral=True
            )
