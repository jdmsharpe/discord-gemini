import unittest
from unittest.mock import AsyncMock, MagicMock


class TestButtonView(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Import here after the environment is set up
        from button_view import ButtonView

        self.cog = MagicMock()
        self.cog.conversations = {}
        self.cog.handle_new_message_in_conversation = AsyncMock()

        self.conversation_starter = MagicMock()
        self.conversation_starter.id = 123456789

        self.conversation_id = 987654321

        self.view = ButtonView(
            self.cog, self.conversation_starter, self.conversation_id
        )

    async def test_init(self):
        """Test that ButtonView initializes correctly."""
        self.assertEqual(self.view.cog, self.cog)
        self.assertEqual(self.view.conversation_starter, self.conversation_starter)
        self.assertEqual(self.view.conversation_id, self.conversation_id)

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

        # No conversation in cog.conversations
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

        # No conversation in cog.conversations
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

        self.cog.conversations[self.conversation_id] = mock_conversation

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

        # No conversation in cog.conversations
        await self.view.stop_button.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("No active conversation", message)

    async def test_stop_button_ends_conversation(self):
        """Test that stop button properly ends the conversation."""
        mock_conversation = MagicMock()
        self.cog.conversations[self.conversation_id] = mock_conversation

        interaction = MagicMock()
        interaction.user = self.conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await self.view.stop_button.callback(interaction)

        # Verify conversation was deleted
        self.assertNotIn(self.conversation_id, self.cog.conversations)

        # Verify response was sent
        call_kwargs = interaction.response.send_message.call_args.kwargs
        call_args = interaction.response.send_message.call_args.args
        message = call_args[0] if call_args else call_kwargs.get("content", "")
        self.assertIn("Conversation ended", message)


if __name__ == "__main__":
    unittest.main()
