import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, cast

from discord import (
    ButtonStyle,
    Interaction,
    Member,
    SelectOption,
    TextChannel,
    User,
)
from discord.ui import Button, Select, View, button

from ...util import resolve_tool_name
from .tool_registry import get_tool_registry, iter_tool_registry


async def _send_interaction_error(interaction: Interaction, context: str, error: Exception) -> None:
    """Log an error and send the user a safe ephemeral message."""
    logging.error(f"Error in {context}: {error}", exc_info=True)
    msg = f"An error occurred while {context}."
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)


async def _build_view_on_running_loop(view: View, *, timeout: float | None) -> None:
    View.__init__(view, timeout=timeout)


def _initialize_view(view: View, *, timeout: float | None) -> ConcurrentFuture[bool] | None:
    """Build a discord View even when tests construct it outside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_build_view_on_running_loop(view, timeout=timeout))
        finally:
            loop.close()
        return ConcurrentFuture()
    else:
        View.__init__(view, timeout=timeout)
        return None


class ButtonView(View):
    def __init__(
        self,
        *,
        conversation_starter: Member | User,
        conversation_id: int,
        initial_tools: list[dict[str, Any]] | None = None,
        custom_functions_enabled: bool = False,
        get_conversation: Callable[[int], Any | None],
        on_regenerate: Callable[[Any, Any], Awaitable[None]],
        on_stop: Callable[[int, Member | User], Awaitable[None]],
        on_tools_changed: Callable[[list[str], bool, Any], tuple[set[str], str | None]],
    ):
        """
        Initialize the ButtonView class.
        """
        self._offline_stopped = _initialize_view(self, timeout=None)
        self.conversation_starter = conversation_starter
        self.conversation_id = conversation_id
        self._get_conversation = get_conversation
        self._on_regenerate = on_regenerate
        self._on_stop = on_stop
        self._on_tools_changed = on_tools_changed
        self._add_tool_select(initial_tools or [], custom_functions_enabled)

    async def wait(self) -> bool:
        """Support wait() even when the view was constructed outside a running loop."""
        if self._offline_stopped is not None:
            return await asyncio.wrap_future(self._offline_stopped)
        return await super().wait()

    def _add_tool_select(
        self,
        initial_tools: list[dict[str, Any]],
        custom_functions_enabled: bool = False,
    ) -> None:
        selected_tools = {
            name for tool in initial_tools if (name := resolve_tool_name(tool)) is not None
        }
        options = [
            SelectOption(
                label=tool.label,
                value=tool.canonical_id,
                description=tool.description,
                default=(
                    custom_functions_enabled
                    if tool.canonical_id == "custom_functions"
                    else tool.canonical_id in selected_tools
                ),
            )
            for tool in iter_tool_registry()
        ]
        tool_select = Select(
            placeholder="Toggle conversation tools",
            min_values=0,
            max_values=len(options),
            row=1,
            options=options,
        )

        async def _tool_callback(interaction: Interaction) -> None:
            await self.tool_select_callback(interaction, tool_select)

        tool_select.callback = _tool_callback
        self.add_item(tool_select)

    async def tool_select_callback(self, interaction: Interaction, tool_select: Select) -> None:
        """Toggle tool availability for this conversation."""
        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to change tools for this conversation.",
                    ephemeral=True,
                )
                return

            conversation = self._get_conversation(self.conversation_id)
            if conversation is None:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
                return

            tool_registry = get_tool_registry()
            selected_values = [
                value
                for value in tool_select.values
                if value in tool_registry and value != "custom_functions"
            ]
            custom_functions_selected = "custom_functions" in tool_select.values

            active_names, result_message = self._on_tools_changed(
                selected_values, custom_functions_selected, conversation
            )

            # Error case: no tools activated and message is an error (not a status)
            if not active_names and result_message and "cannot be combined" in result_message:
                await interaction.response.send_message(result_message, ephemeral=True)
                return

            # Update Select dropdown defaults
            for child in self.children:
                if isinstance(child, Select):
                    for option in child.options:
                        option.default = option.value in active_names
                    break

            message = result_message or (
                "Tools disabled for this conversation."
                if not active_names
                else f"Tools updated: {', '.join(sorted(active_names))}."
            )
            await interaction.response.send_message(message, ephemeral=True, delete_after=3)
        except Exception as e:
            await _send_interaction_error(interaction, "updating tools", e)

    @button(emoji="🔄", style=ButtonStyle.green, row=0)
    async def regenerate_button(self, _: Button, interaction: Interaction):
        """
        Regenerate the last response for the current conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        logging.info("Regenerate button clicked.")
        removed_entries: list[Any] = []

        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to regenerate the response.", ephemeral=True
                )
                return

            conversation = self._get_conversation(self.conversation_id)
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
                (message for message in messages if message.author == self.conversation_starter),
                None,
            )

            if user_message is None:
                conversation.history.extend(removed_entries)
                await interaction.followup.send(
                    "Couldn't find the message to regenerate.", ephemeral=True
                )
                return

            await self._on_regenerate(user_message, conversation)
            await interaction.followup.send("Response regenerated.", ephemeral=True, delete_after=3)
        except Exception as error:
            logging.error(
                f"Error in regenerate_button: {error}",
                exc_info=True,
            )

            if removed_entries:
                conversation = self._get_conversation(self.conversation_id)
                if conversation is not None:
                    conversation.history.extend(removed_entries)

            if interaction.response.is_done():
                await interaction.followup.send(
                    "An error occurred while regenerating the response.",
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    "An error occurred while regenerating the response.",
                    ephemeral=True,
                )

    @button(emoji="⏯️", style=ButtonStyle.gray, row=0)
    async def play_pause_button(self, button: Button, interaction: Interaction):
        """
        Pause or resume the conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to pause the conversation.", ephemeral=True
                )
                return

            conversation = self._get_conversation(self.conversation_id)
            if conversation is not None:
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
        except Exception as e:
            await _send_interaction_error(interaction, "toggling pause", e)

    @button(emoji="⏹️", style=ButtonStyle.blurple, row=0)
    async def stop_button(self, button: Button, interaction: Interaction):
        """
        End the conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to end this conversation.", ephemeral=True
                )
                return

            conversation = self._get_conversation(self.conversation_id)
            if conversation is not None:
                await self._on_stop(self.conversation_id, self.conversation_starter)
                await interaction.response.send_message(
                    "Conversation ended.", ephemeral=True, delete_after=3
                )
            else:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
        except Exception as e:
            await _send_interaction_error(interaction, "ending the conversation", e)
