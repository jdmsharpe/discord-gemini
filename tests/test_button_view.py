from unittest.mock import AsyncMock, MagicMock

import pytest
from discord.ui import Select

from discord_gemini.util import TOOL_GOOGLE_SEARCH


def _make_view(
    conversation_starter=None,
    conversation_id=None,
    initial_tools=None,
    custom_functions_enabled=False,
    get_conversation=None,
    on_tools_changed=None,
    on_stop=None,
):
    from discord_gemini.cogs.gemini.views import ButtonView

    return ButtonView(
        conversation_starter=conversation_starter or MagicMock(),
        conversation_id=conversation_id or 987654321,
        initial_tools=initial_tools,
        custom_functions_enabled=custom_functions_enabled,
        get_conversation=get_conversation or MagicMock(return_value=None),
        on_regenerate=AsyncMock(),
        on_stop=on_stop or AsyncMock(),
        on_tools_changed=on_tools_changed or MagicMock(return_value=(set(), None)),
    )


class TestButtonView:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.conversation_starter = MagicMock()
        self.conversation_starter.id = 123456789
        self.conversation_id = 987654321
        self.get_conversation = MagicMock(return_value=None)
        self.on_regenerate = AsyncMock()
        self.on_stop = AsyncMock()
        self.on_tools_changed = MagicMock(return_value=(set(), None))

        self.view = _make_view(
            conversation_starter=self.conversation_starter,
            conversation_id=self.conversation_id,
            get_conversation=self.get_conversation,
            on_tools_changed=self.on_tools_changed,
            on_stop=self.on_stop,
        )

    async def test_init(self):
        """Test that ButtonView initializes correctly."""
        assert self.view.conversation_starter == self.conversation_starter
        assert self.view.conversation_id == self.conversation_id
        tool_selects = [item for item in self.view.children if isinstance(item, Select)]
        assert len(tool_selects) == 1
        assert tool_selects[0].max_values == 6

    async def test_init_with_initial_tools(self):
        """Test that initial_tools marks matching select options as default."""
        view = _make_view(
            conversation_starter=self.conversation_starter,
            initial_tools=[TOOL_GOOGLE_SEARCH],
        )
        tool_select = next(item for item in view.children if isinstance(item, Select))
        defaults = {option.value: option.default for option in tool_select.options}
        assert defaults["google_search"] is True
        assert defaults["code_execution"] is False
        assert defaults["google_maps"] is False
        assert defaults["url_context"] is False
        assert defaults["file_search"] is False
        assert defaults["custom_functions"] is False

    async def test_init_with_custom_functions_enabled(self):
        """Test that custom_functions_enabled marks the option as default."""
        view = _make_view(
            conversation_starter=self.conversation_starter,
            custom_functions_enabled=True,
        )
        tool_select = next(item for item in view.children if isinstance(item, Select))
        defaults = {option.value: option.default for option in tool_select.options}
        assert defaults["custom_functions"] is True
        assert defaults["google_search"] is False

    async def test_tool_select_callback_calls_on_tools_changed(self):
        """Test that tool selection calls on_tools_changed with correct args."""
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []
        self.get_conversation.return_value = conversation

        active = {"google_search", "code_execution"}
        self.on_tools_changed.return_value = (
            active,
            "Tools updated: code_execution, google_search.",
        )

        mock_select = MagicMock()
        mock_select.values = ["google_search", "code_execution"]

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await self.view.tool_select_callback(interaction, mock_select)

        self.on_tools_changed.assert_called_once_with(
            ["google_search", "code_execution"], False, conversation
        )
        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "Tools updated" in call_args[0]

    async def test_tool_select_callback_custom_functions_toggle(self):
        """Test that selecting custom_functions passes the flag."""
        conversation = MagicMock()
        conversation.params = MagicMock()
        self.get_conversation.return_value = conversation

        active = {"custom_functions"}
        self.on_tools_changed.return_value = (active, "Tools updated: custom_functions.")

        mock_select = MagicMock()
        mock_select.values = ["custom_functions"]

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await self.view.tool_select_callback(interaction, mock_select)

        # Verify custom_functions_selected was True
        call_args = self.on_tools_changed.call_args
        assert call_args.args[1] is True  # custom_functions_selected

    async def test_tool_select_callback_updates_defaults(self):
        """Test that Select defaults get updated after tool change."""
        conversation = MagicMock()
        self.get_conversation.return_value = conversation

        active = {"google_search"}
        self.on_tools_changed.return_value = (active, "Tools updated: google_search.")

        # Get real select for verifying defaults; use mock for .values
        real_select = next(item for item in self.view.children if isinstance(item, Select))
        mock_select = MagicMock()
        mock_select.values = ["google_search"]

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await self.view.tool_select_callback(interaction, mock_select)

        # Verify Select defaults match active tools on real widget
        defaults = {option.value: option.default for option in real_select.options}
        assert defaults["google_search"] is True
        assert defaults["code_execution"] is False
        assert defaults["custom_functions"] is False

    async def test_tool_select_callback_error_from_callback(self):
        """Test that error from on_tools_changed is shown to user."""
        conversation = MagicMock()
        self.get_conversation.return_value = conversation
        self.on_tools_changed.return_value = (
            set(),
            "google_search and google_maps cannot be combined.",
        )

        mock_select = MagicMock()
        mock_select.values = ["google_search", "google_maps"]

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.tool_select_callback(interaction, mock_select)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "cannot be combined" in call_args[0]

    async def test_tool_select_callback_wrong_user(self):
        """Test that tool selection rejects non-conversation starters."""
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.tool_select_callback(interaction, mock_select)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "not allowed" in call_args[0]

    async def test_regenerate_button_wrong_user(self):
        """Test that regenerate button rejects non-conversation starters."""
        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.regenerate_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "not allowed" in call_args[0]

    async def test_regenerate_button_no_conversation(self):
        """Test that regenerate button handles missing conversation."""
        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.regenerate_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "No active conversation" in call_args[0]

    async def test_play_pause_button_wrong_user(self):
        """Test that play/pause button rejects non-conversation starters."""
        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.play_pause_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "not allowed" in call_args[0]

    async def test_play_pause_button_no_conversation(self):
        """Test that play/pause button handles missing conversation."""
        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.play_pause_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "No active conversation" in call_args[0]

    async def test_play_pause_button_toggles_pause(self):
        """Test that play/pause button toggles the paused state."""
        mock_conversation = MagicMock()
        mock_conversation.params = MagicMock()
        mock_conversation.params.paused = False
        self.get_conversation.return_value = mock_conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await self.view.play_pause_button.callback(interaction)
        assert mock_conversation.params.paused is True
        call_args = interaction.response.send_message.call_args.args
        assert "paused" in call_args[0]

        interaction.response.send_message.reset_mock()
        await self.view.play_pause_button.callback(interaction)
        assert mock_conversation.params.paused is False
        call_args = interaction.response.send_message.call_args.args
        assert "resumed" in call_args[0]

    async def test_stop_button_wrong_user(self):
        """Test that stop button rejects non-conversation starters."""
        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.stop_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "not allowed" in call_args[0]

    async def test_stop_button_no_conversation(self):
        """Test that stop button handles missing conversation."""
        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.stop_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_args = interaction.response.send_message.call_args.args
        assert "No active conversation" in call_args[0]

    async def test_stop_button_ends_conversation(self):
        """Test that stop button properly ends the conversation."""
        mock_conversation = MagicMock()
        self.get_conversation.return_value = mock_conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await self.view.stop_button.callback(interaction)

        self.on_stop.assert_awaited_once_with(self.conversation_id, self.conversation_starter)
        call_args = interaction.response.send_message.call_args.args
        assert "Conversation ended" in call_args[0]
