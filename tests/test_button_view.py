import unittest
from unittest.mock import AsyncMock, MagicMock

from discord.ui import Select

from util import (
    AVAILABLE_TOOLS,
    TOOL_GOOGLE_SEARCH,
)


class TestButtonView(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Import here after the environment is set up
        from button_view import ButtonView

        # Build a mock that satisfies the ConversationHost protocol
        self.host = MagicMock()
        self.host.get_conversation = MagicMock(return_value=None)
        self.host.enrich_file_search_tools = MagicMock(return_value=None)
        self.host.handle_new_message_in_conversation = AsyncMock()
        self.host.end_conversation = AsyncMock()

        self.conversation_starter = MagicMock()
        self.conversation_starter.id = 123456789

        self.conversation_id = 987654321

        self.view = ButtonView(self.host, self.conversation_starter, self.conversation_id)

    async def test_init(self):
        """Test that ButtonView initializes correctly."""
        self.assertEqual(self.view.host, self.host)
        self.assertEqual(self.view.conversation_starter, self.conversation_starter)
        self.assertEqual(self.view.conversation_id, self.conversation_id)
        tool_selects = [item for item in self.view.children if isinstance(item, Select)]
        self.assertEqual(len(tool_selects), 1)
        self.assertEqual(tool_selects[0].max_values, 6)

    async def test_init_with_initial_tools(self):
        """Test that initial_tools marks matching select options as default."""
        from button_view import ButtonView

        view = ButtonView(
            self.host,
            self.conversation_starter,
            self.conversation_id,
            initial_tools=[TOOL_GOOGLE_SEARCH],
        )
        tool_select = next(item for item in view.children if isinstance(item, Select))
        defaults = {option.value: option.default for option in tool_select.options}
        self.assertTrue(defaults["google_search"])
        self.assertFalse(defaults["code_execution"])
        self.assertFalse(defaults["google_maps"])
        self.assertFalse(defaults["url_context"])
        self.assertFalse(defaults["file_search"])
        self.assertFalse(defaults["custom_functions"])

    async def test_init_with_custom_functions_enabled(self):
        """Test that custom_functions_enabled marks the option as default."""
        from button_view import ButtonView

        view = ButtonView(
            self.host,
            self.conversation_starter,
            self.conversation_id,
            custom_functions_enabled=True,
        )
        tool_select = next(item for item in view.children if isinstance(item, Select))
        defaults = {option.value: option.default for option in tool_select.options}
        self.assertTrue(defaults["custom_functions"])
        self.assertFalse(defaults["google_search"])

    async def test_tool_select_callback_custom_functions_toggle(self):
        """Test that selecting custom_functions sets the flag on params."""
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []
        conversation.params.model = "gemini-2.5-flash"
        conversation.params.custom_functions_enabled = False
        self.host.get_conversation.return_value = conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.data = {"values": ["custom_functions"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.tool_select_callback(interaction)

        self.assertTrue(conversation.params.custom_functions_enabled)
        call_args = interaction.response.send_message.call_args.args
        call_kwargs = interaction.response.send_message.call_args.kwargs
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("custom_functions", message)

    async def test_tool_select_callback_updates_tools(self):
        """Test that tool selection updates conversation params tools."""
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []
        conversation.params.model = "gemini-2.5-flash"
        self.host.get_conversation.return_value = conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.data = {"values": ["google_search", "code_execution"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.tool_select_callback(interaction)

        self.assertEqual(
            conversation.params.tools,
            [
                AVAILABLE_TOOLS["google_search"],
                AVAILABLE_TOOLS["code_execution"],
            ],
        )
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("Tools updated", message)
        self.assertTrue(call_kwargs.get("ephemeral", False))

    async def test_tool_select_callback_filters_incompatible_tools(self):
        """Test that incompatible tools are skipped for the selected model."""
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []
        conversation.params.model = "gemini-2.0-flash-lite"
        self.host.get_conversation.return_value = conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.data = {"values": ["google_maps", "url_context"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.tool_select_callback(interaction)

        self.assertEqual(conversation.params.tools, [])
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("Skipped", message)
        self.assertIn("google_maps", message)

    async def test_tool_select_callback_rejects_mutually_exclusive_tools(self):
        """Test that selecting google_search and google_maps together is rejected."""
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []
        conversation.params.model = "gemini-2.5-pro"
        self.host.get_conversation.return_value = conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.data = {"values": ["google_search", "google_maps"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.tool_select_callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("google_search", message)
        self.assertIn("google_maps", message)
        self.assertIn("cannot be combined", message)
        self.assertTrue(call_kwargs.get("ephemeral", False))
        # Tools should NOT have been updated
        self.assertEqual(conversation.params.tools, [])

    async def test_tool_select_callback_wrong_user(self):
        """Test that tool selection rejects non-conversation starters."""
        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.data = {"values": ["google_search"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.tool_select_callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("not allowed", message)
        self.assertTrue(call_kwargs.get("ephemeral", False))

    async def test_regenerate_button_wrong_user(self):
        """Test that regenerate button rejects non-conversation starters."""
        interaction = MagicMock()
        interaction.user = MagicMock()  # Different user
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        # The callback is already bound to self, pass only interaction
        await self.view.regenerate_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        # Get the message from the call
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("not allowed", message)
        self.assertTrue(call_kwargs.get("ephemeral", False))

    async def test_regenerate_button_no_conversation(self):
        """Test that regenerate button handles missing conversation."""
        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.defer = AsyncMock()

        # host.get_conversation returns None by default
        await self.view.regenerate_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("No active conversation", message)

    async def test_play_pause_button_wrong_user(self):
        """Test that play/pause button rejects non-conversation starters."""
        interaction = MagicMock()
        interaction.user = MagicMock()  # Different user
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.play_pause_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("not allowed", message)
        self.assertTrue(call_kwargs.get("ephemeral", False))

    async def test_play_pause_button_no_conversation(self):
        """Test that play/pause button handles missing conversation."""
        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        # host.get_conversation returns None by default
        await self.view.play_pause_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("No active conversation", message)

    async def test_play_pause_button_toggles_pause(self):
        """Test that play/pause button toggles the paused state."""
        # Create a mock conversation with params
        mock_conversation = MagicMock()
        mock_conversation.params = MagicMock()
        mock_conversation.params.paused = False

        self.host.get_conversation.return_value = mock_conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        # First toggle - should pause
        await self.view.play_pause_button.callback(interaction)

        self.assertTrue(mock_conversation.params.paused)
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("paused", message)

        # Second toggle - should resume
        interaction.response.send_message.reset_mock()
        await self.view.play_pause_button.callback(interaction)

        self.assertFalse(mock_conversation.params.paused)
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("resumed", message)

    async def test_stop_button_wrong_user(self):
        """Test that stop button rejects non-conversation starters."""
        interaction = MagicMock()
        interaction.user = MagicMock()  # Different user
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.stop_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("not allowed", message)
        self.assertTrue(call_kwargs.get("ephemeral", False))

    async def test_stop_button_no_conversation(self):
        """Test that stop button handles missing conversation."""
        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        # host.get_conversation returns None by default
        await self.view.stop_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("No active conversation", message)

    async def test_stop_button_ends_conversation(self):
        """Test that stop button properly ends the conversation."""
        mock_conversation = MagicMock()
        self.host.get_conversation.return_value = mock_conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.stop_button.callback(interaction)

        # Verify end_conversation was called with the right args
        self.host.end_conversation.assert_awaited_once_with(
            self.conversation_id, self.conversation_starter
        )

        # Verify response was sent
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("Conversation ended", message)


if __name__ == "__main__":
    unittest.main()
