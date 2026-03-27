import logging
from copy import deepcopy
from typing import Any, Protocol, cast

from discord import (
    ButtonStyle,
    Interaction,
    Member,
    SelectOption,
    TextChannel,
    User,
)
from discord.ui import Button, Select, View, button

from util import (
    AVAILABLE_TOOLS,
    check_mutually_exclusive_tools,
    filter_file_search_incompatible_tools,
    filter_supported_tools_for_model,
    resolve_tool_name,
)


class ConversationHost(Protocol):
    """Protocol defining the interface ButtonView needs from its host (the cog).

    This decouples ButtonView from the concrete GeminiAPI class, following
    the Dependency Inversion Principle.  Any object satisfying this protocol
    can serve as the host.
    """

    def get_conversation(self, conversation_id: int) -> Any:
        """Return the Conversation for *conversation_id*, or ``None``."""
        ...

    def enrich_file_search_tools(self, tools: list[dict[str, Any]]) -> str | None:
        """Inject file search store IDs.  Return an error string, or ``None``."""
        ...

    async def handle_new_message_in_conversation(self, message: Any, conversation: Any) -> None:
        """Re-run generation for the given message in *conversation*."""
        ...

    async def end_conversation(self, conversation_id: int, user: Member | User) -> None:
        """Fully tear down a conversation (cache, files, state)."""
        ...


class ButtonView(View):
    def __init__(
        self,
        host: ConversationHost,
        conversation_starter: Member | User,
        conversation_id: int,
        initial_tools: list[dict[str, Any]] | None = None,
        custom_functions_enabled: bool = False,
    ):
        """
        Initialize the ButtonView class.
        """
        super().__init__(timeout=None)
        self.host = host
        self.conversation_starter = conversation_starter
        self.conversation_id = conversation_id
        self._add_tool_select(initial_tools or [], custom_functions_enabled)

    def _add_tool_select(
        self,
        initial_tools: list[dict[str, Any]],
        custom_functions_enabled: bool = False,
    ) -> None:
        selected_tools = {
            name for tool in initial_tools if (name := resolve_tool_name(tool)) is not None
        }
        tool_select = Select(
            placeholder="Toggle conversation tools",
            min_values=0,
            max_values=6,
            row=1,
            options=[
                SelectOption(
                    label="Google Search",
                    value="google_search",
                    description="Ground answers with live web results.",
                    default="google_search" in selected_tools,
                ),
                SelectOption(
                    label="Code Execution",
                    value="code_execution",
                    description="Run Python code for calculations.",
                    default="code_execution" in selected_tools,
                ),
                SelectOption(
                    label="Google Maps",
                    value="google_maps",
                    description="Ground answers with Maps place data.",
                    default="google_maps" in selected_tools,
                ),
                SelectOption(
                    label="URL Context",
                    value="url_context",
                    description="Retrieve and analyze provided URLs.",
                    default="url_context" in selected_tools,
                ),
                SelectOption(
                    label="File Search",
                    value="file_search",
                    description="Search over uploaded document stores.",
                    default="file_search" in selected_tools,
                ),
                SelectOption(
                    label="Custom Functions",
                    value="custom_functions",
                    description="Call Python tools (time, dice, etc.).",
                    default=custom_functions_enabled,
                ),
            ],
        )
        tool_select.callback = self.tool_select_callback
        self.add_item(tool_select)

    async def tool_select_callback(self, interaction: Interaction):
        """
        Toggle tool availability for this conversation.
        """
        if interaction.user != self.conversation_starter:
            await interaction.response.send_message(
                "You are not allowed to change tools for this conversation.",
                ephemeral=True,
            )
            return

        conversation = self.host.get_conversation(self.conversation_id)
        if conversation is None:
            await interaction.response.send_message("No active conversation found.", ephemeral=True)
            return

        selected_values: list[str] = []
        custom_functions_selected = False
        if isinstance(interaction.data, dict):
            raw_values = interaction.data.get("values", [])
            if isinstance(raw_values, list):
                custom_functions_selected = "custom_functions" in raw_values
                selected_values = [value for value in raw_values if value in AVAILABLE_TOOLS]

        exclusive_error = check_mutually_exclusive_tools(set(selected_values))
        if exclusive_error:
            await interaction.response.send_message(exclusive_error, ephemeral=True)
            return

        requested_tools = [deepcopy(AVAILABLE_TOOLS[tool_name]) for tool_name in selected_values]
        supported_tools, unsupported_tools = filter_supported_tools_for_model(
            conversation.params.model, requested_tools
        )
        supported_tools, incompatible_tools = filter_file_search_incompatible_tools(supported_tools)

        # Enrich file_search tools with store IDs via the host
        enrich_error = self.host.enrich_file_search_tools(supported_tools)
        if enrich_error:
            await interaction.response.send_message(enrich_error, ephemeral=True)
            return

        conversation.params.tools = supported_tools
        conversation.params.custom_functions_enabled = custom_functions_selected
        selected_tool_names = {
            tool_name
            for tool_config in supported_tools
            if (tool_name := resolve_tool_name(tool_config)) is not None
        }
        if custom_functions_selected:
            selected_tool_names.add("custom_functions")

        for child in self.children:
            if isinstance(child, Select):
                for option in child.options:
                    option.default = option.value in selected_tool_names
                break

        if selected_tool_names:
            tool_names = ", ".join(sorted(selected_tool_names))
            message = f"Tools updated: {tool_names}."
        else:
            message = "Tools disabled for this conversation."
        if unsupported_tools:
            unsupported_text = ", ".join(sorted(set(unsupported_tools)))
            message += f" Skipped for model `{conversation.params.model}`: {unsupported_text}."
        if incompatible_tools:
            incompatible_text = ", ".join(sorted(set(incompatible_tools)))
            message += f" Disabled (incompatible with file_search): {incompatible_text}."

        await interaction.response.send_message(message, ephemeral=True, delete_after=3)

    @button(emoji="🔄", style=ButtonStyle.green, row=0)
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

            conversation = self.host.get_conversation(self.conversation_id)
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

            await self.host.handle_new_message_in_conversation(user_message, conversation)
            await interaction.followup.send("Response regenerated.", ephemeral=True, delete_after=3)
        except Exception as error:
            logging.error(
                f"Error in regenerate_button: {error}",
                exc_info=True,
            )

            if removed_entries:
                conversation = self.host.get_conversation(self.conversation_id)
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

    @button(emoji="⏯️", style=ButtonStyle.gray, row=0)
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
        conversation = self.host.get_conversation(self.conversation_id)
        if conversation is not None:
            conversation.params.paused = not conversation.params.paused
            status = "paused" if conversation.params.paused else "resumed"
            await interaction.response.send_message(
                f"Conversation {status}. Press again to toggle.",
                ephemeral=True,
                delete_after=3,
            )
        else:
            await interaction.response.send_message("No active conversation found.", ephemeral=True)

    @button(emoji="⏹️", style=ButtonStyle.blurple, row=0)
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

        # End the conversation via the host
        conversation = self.host.get_conversation(self.conversation_id)
        if conversation is not None:
            await self.host.end_conversation(self.conversation_id, self.conversation_starter)
            await interaction.response.send_message(
                "Conversation ended.", ephemeral=True, delete_after=3
            )
        else:
            await interaction.response.send_message("No active conversation found.", ephemeral=True)
