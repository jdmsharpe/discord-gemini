import asyncio
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
    async def regenerate_button(self, _: Button, interaction: Interaction):
        """
        Regenerate the last response for the current conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        logging.info("Regenerate button clicked.")

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

        async def run_regeneration():
            removed_entries = []
            try:
                if len(conversation.history) < 2:
                    await interaction.followup.send(
                        "Not enough history to regenerate yet.", ephemeral=True
                    )
                    return

                if not hasattr(interaction.channel, "history"):
                    await interaction.followup.send(
                        "Cannot regenerate in this type of channel.", ephemeral=True
                    )
                    return

                channel = cast(TextChannel, interaction.channel)
                user_message = None
                async for msg in channel.history(limit=10):
                    if msg.author == self.conversation_starter:
                        user_message = msg
                        break

                if user_message is None:
                    await interaction.followup.send(
                        "Could not find the previous message to regenerate.",
                        ephemeral=True,
                    )
                    return

                try:
                    removed_entries.append(conversation.history.pop())
                    removed_entries.append(conversation.history.pop())
                except IndexError:
                    if removed_entries:
                        for entry in reversed(removed_entries):
                            conversation.history.append(entry)
                        removed_entries.clear()
                    await interaction.followup.send(
                        "Conversation history is too short to regenerate.",
                        ephemeral=True,
                    )
                    return

                removed_entries.reverse()

                try:
                    await self.cog.handle_new_message_in_conversation(
                        user_message, conversation
                    )
                except Exception:
                    conversation.history.extend(removed_entries)
                    removed_entries.clear()
                    raise

                await interaction.followup.send(
                    "Response regenerated.", ephemeral=True, delete_after=3
                )
            except Exception:
                logging.exception("Error in regenerate_button background task")
                if removed_entries:
                    conversation.history.extend(removed_entries)
                await interaction.followup.send(
                    "An error occurred while regenerating the response.", ephemeral=True
                )

        asyncio.create_task(run_regeneration())

    @button(emoji="⏯️", style=ButtonStyle.gray)
    async def play_pause_button(self, _: Button, interaction: Interaction):
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
    async def stop_button(self, _: Button, interaction: Interaction):
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
