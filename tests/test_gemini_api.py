import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from discord import Bot, Embed, Intents


class TestGeminiAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock the auth module before importing GeminiAPI
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key", GUILD_IDS=[123456789]
                )
            },
        )
        self.auth_patcher.start()

        # Mock the genai Client
        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        # Now import GeminiAPI after mocking
        from gemini_api import GeminiAPI, Conversation, append_response_embeds

        self.GeminiAPI = GeminiAPI
        self.Conversation = Conversation
        self.append_response_embeds = append_response_embeds

        # Setting up the bot with the GeminiAPI cog
        intents = Intents.default()
        intents.presences = False
        intents.members = True
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)
        self.bot.add_cog(self.cog)
        self.bot.owner_id = 1234567890

        # Mock bot commands
        self.bot.sync_commands = AsyncMock()

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    async def test_cog_init(self):
        """Test that GeminiAPI cog initializes correctly."""
        self.assertEqual(self.cog.bot, self.bot)
        self.assertEqual(self.cog.conversations, {})
        self.assertEqual(self.cog.message_to_conversation_id, {})
        self.assertEqual(self.cog.views, {})
        self.assertIsNone(self.cog._http_session)

    async def test_on_ready(self):
        """Test that on_ready syncs commands."""
        await self.cog.on_ready()
        self.bot.sync_commands.assert_called_once()

    async def test_append_response_embeds_short(self):
        """Test append_response_embeds with short text."""
        embeds = []
        self.append_response_embeds(embeds, "Hello, World!")
        self.assertEqual(len(embeds), 1)
        self.assertEqual(embeds[0].description, "Hello, World!")
        self.assertEqual(embeds[0].title, "Response")

    async def test_append_response_embeds_long(self):
        """Test append_response_embeds with long text that needs chunking."""
        embeds = []
        long_text = "A" * 7000  # Over the 3500 chunk limit
        self.append_response_embeds(embeds, long_text)
        self.assertEqual(len(embeds), 2)
        self.assertEqual(embeds[0].title, "Response")
        self.assertEqual(embeds[1].title, "Response (Part 2)")

    async def test_append_response_embeds_very_long(self):
        """Test append_response_embeds truncates very long text."""
        embeds = []
        very_long_text = "B" * 25000  # Over the 20000 truncation limit
        self.append_response_embeds(embeds, very_long_text)
        # Should truncate to ~19500 chars plus truncation message
        total_length = sum(len(e.description) for e in embeds)
        self.assertLess(total_length, 21000)  # Should be well under 25000

    async def test_get_http_session_creates_session(self):
        """Test that _get_http_session creates a new session."""
        self.assertIsNone(self.cog._http_session)
        session = await self.cog._get_http_session()
        self.assertIsNotNone(session)
        self.assertEqual(self.cog._http_session, session)
        # Clean up
        await session.close()

    async def test_get_http_session_reuses_session(self):
        """Test that _get_http_session reuses existing session."""
        session1 = await self.cog._get_http_session()
        session2 = await self.cog._get_http_session()
        self.assertEqual(session1, session2)
        # Clean up
        await session1.close()

    async def test_conversation_dataclass(self):
        """Test the Conversation dataclass."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-pro-preview")
        history = [{"role": "user", "parts": [{"text": "Hello"}]}]
        conversation = self.Conversation(params=params, history=history)

        self.assertEqual(conversation.params, params)
        self.assertEqual(conversation.history, history)

    async def test_on_message_ignores_bot(self):
        """Test that on_message ignores messages from the bot itself."""
        message = MagicMock()
        message.author = self.bot.user

        # Should return early without processing
        await self.cog.on_message(message)

        # No conversations should be affected
        self.assertEqual(len(self.cog.conversations), 0)

    async def test_on_message_no_matching_conversation(self):
        """Test that on_message handles no matching conversation gracefully."""
        message = MagicMock()
        message.author = MagicMock()
        message.author.id = 999
        message.channel = MagicMock()
        message.channel.id = 888
        message.content = "Hello"

        # Should complete without errors
        await self.cog.on_message(message)

    async def test_cog_unload_closes_session(self):
        """Test that cog_unload closes the HTTP session."""
        # Create a session first
        session = await self.cog._get_http_session()
        self.assertFalse(session.closed)

        # Unload the cog
        self.cog.cog_unload()

        # Session should be closed or set to None
        self.assertIsNone(self.cog._http_session)


class TestGeminiAPIHelpers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock the auth module before importing GeminiAPI
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key", GUILD_IDS=[123456789]
                )
            },
        )
        self.auth_patcher.start()

        # Mock the genai Client
        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        from gemini_api import GeminiAPI

        intents = Intents.default()
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    async def test_fetch_attachment_bytes_success(self):
        """Test successful attachment download."""
        mock_session = MagicMock()
        mock_session.closed = False  # Ensure the mock session is not considered closed
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"image data")

        # Create a proper async context manager for session.get()
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context)

        self.cog._http_session = mock_session

        attachment = MagicMock()
        attachment.url = "https://example.com/image.png"

        result = await self.cog._fetch_attachment_bytes(attachment)
        self.assertEqual(result, b"image data")

    async def test_fetch_attachment_bytes_failure(self):
        """Test failed attachment download returns None."""
        mock_session = MagicMock()
        mock_session.closed = False  # Ensure the mock session is not considered closed
        mock_response = AsyncMock()
        mock_response.status = 404

        # Create a proper async context manager for session.get()
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context)

        self.cog._http_session = mock_session

        attachment = MagicMock()
        attachment.url = "https://example.com/not-found.png"

        result = await self.cog._fetch_attachment_bytes(attachment)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
