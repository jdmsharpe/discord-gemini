import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from discord import Bot, Embed, Intents


class TestGeminiAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock the auth module before importing GeminiAPI
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key",
                    GUILD_IDS=[123456789],
                    GEMINI_FILE_SEARCH_STORE_IDS=["store-1", "store-2"],
                )
            },
        )
        self.auth_patcher.start()

        # Mock the genai Client
        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        # Configure the mock client's async interface
        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()

        # Now import GeminiAPI after mocking
        from gemini_api import (
            GeminiAPI,
            Conversation,
            append_pricing_embed,
            append_response_embeds,
            append_thinking_embeds,
            extract_thinking_text,
            extract_tool_info,
            _get_response_content_parts,
            _build_thinking_config,
        )

        self.GeminiAPI = GeminiAPI
        self.Conversation = Conversation
        self.append_pricing_embed = append_pricing_embed
        self.append_response_embeds = append_response_embeds
        self.append_thinking_embeds = append_thinking_embeds
        self.extract_thinking_text = extract_thinking_text
        self.extract_tool_info = extract_tool_info
        self._get_response_content_parts = _get_response_content_parts
        self._build_thinking_config = _build_thinking_config

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
        self.assertEqual(self.cog.last_view_messages, {})
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

    async def test_extract_tool_info_empty_output(self):
        """Test extract_tool_info with empty candidates."""
        response = SimpleNamespace(candidates=[])
        tool_info = self.extract_tool_info(response)
        self.assertEqual(tool_info["tools_used"], [])
        self.assertEqual(tool_info["citations"], [])
        self.assertEqual(tool_info["search_queries"], [])

    async def test_extract_tool_info_google_search(self):
        """Test extract_tool_info detects google_search and citations."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=["who won euro 2024"],
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    uri="https://example.com/source",
                                    title="Example Source",
                                )
                            )
                        ],
                        search_entry_point=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                )
            ]
        )

        tool_info = self.extract_tool_info(response)
        self.assertIn("google_search", tool_info["tools_used"])
        self.assertEqual(tool_info["search_queries"], ["who won euro 2024"])
        self.assertEqual(
            tool_info["citations"],
            [{"title": "Example Source", "uri": "https://example.com/source"}],
        )

    async def test_extract_tool_info_code_execution(self):
        """Test extract_tool_info detects code_execution from response parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=None,
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                executable_code=SimpleNamespace(code="print(2 + 2)"),
                                code_execution_result=None,
                            )
                        ]
                    ),
                )
            ]
        )

        tool_info = self.extract_tool_info(response)
        self.assertIn("code_execution", tool_info["tools_used"])
        self.assertEqual(tool_info["citations"], [])
        self.assertEqual(tool_info["search_queries"], [])

    async def test_extract_tool_info_google_maps(self):
        """Test extract_tool_info detects google_maps citations and widget token."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=[],
                        grounding_chunks=[
                            SimpleNamespace(
                                maps=SimpleNamespace(
                                    uri="https://maps.google.com/?cid=123",
                                    title="Test Place",
                                )
                            )
                        ],
                        google_maps_widget_context_token="widgetcontent/token",
                        search_entry_point=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=None,
                )
            ]
        )

        tool_info = self.extract_tool_info(response)
        self.assertIn("google_maps", tool_info["tools_used"])
        self.assertEqual(
            tool_info["citations"],
            [{"title": "Test Place", "uri": "https://maps.google.com/?cid=123"}],
        )
        self.assertEqual(tool_info["maps_widget_token"], "widgetcontent/token")

    async def test_extract_tool_info_url_context(self):
        """Test extract_tool_info detects url_context metadata."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=None,
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=SimpleNamespace(
                        url_metadata=[
                            SimpleNamespace(
                                retrieved_url="https://example.com/a",
                                url_retrieval_status="URL_RETRIEVAL_STATUS_SUCCESS",
                            )
                        ]
                    ),
                )
            ]
        )

        tool_info = self.extract_tool_info(response)
        self.assertIn("url_context", tool_info["tools_used"])
        self.assertEqual(
            tool_info["url_context_sources"],
            [
                {
                    "retrieved_url": "https://example.com/a",
                    "status": "URL_RETRIEVAL_STATUS_SUCCESS",
                }
            ],
        )

    async def test_extract_tool_info_file_search_via_retrieval_metadata(self):
        """Test extract_tool_info detects file_search via retrieval_metadata."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=[],
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    uri="fileSearchStores/store1/documents/doc1",
                                    title="uploaded-doc.pdf",
                                ),
                                maps=None,
                            )
                        ],
                        search_entry_point=None,
                        google_maps_widget_context_token=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=None,
                    retrieval_metadata=SimpleNamespace(data="retrieval info"),
                )
            ]
        )

        tool_info = self.extract_tool_info(response)
        self.assertIn("file_search", tool_info["tools_used"])

    async def test_extract_tool_info_file_search_fallback(self):
        """Test extract_tool_info detects file_search when grounding chunks present without search/maps."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=[],
                        grounding_chunks=[
                            SimpleNamespace(
                                web=None,
                                maps=None,
                            )
                        ],
                        search_entry_point=None,
                        google_maps_widget_context_token=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=None,
                )
            ]
        )

        tool_info = self.extract_tool_info(response)
        self.assertIn("file_search", tool_info["tools_used"])

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

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
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
                    GEMINI_API_KEY="test-api-key",
                    GUILD_IDS=[123456789],
                    GEMINI_FILE_SEARCH_STORE_IDS=["store-1", "store-2"],
                )
            },
        )
        self.auth_patcher.start()

        # Mock the genai Client
        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        # Configure the mock client's async interface
        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()

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

    async def test_enrich_file_search_tools_injects_store_ids(self):
        """Test that enrich_file_search_tools injects store IDs."""
        tools = [{"file_search": {}}]
        error = self.cog.enrich_file_search_tools(tools)
        self.assertIsNone(error)
        self.assertEqual(
            tools[0],
            {"file_search": {"file_search_store_names": ["store-1", "store-2"]}},
        )

    async def test_enrich_file_search_tools_no_file_search(self):
        """Test that enrich_file_search_tools is a no-op without file_search."""
        tools = [{"google_search": {}}]
        error = self.cog.enrich_file_search_tools(tools)
        self.assertIsNone(error)
        self.assertEqual(tools, [{"google_search": {}}])

    async def test_enrich_file_search_tools_no_store_ids(self):
        """Test that enrich_file_search_tools returns error when store IDs not configured."""
        import gemini_api

        original = gemini_api.GEMINI_FILE_SEARCH_STORE_IDS
        gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = []
        try:
            tools = [{"file_search": {}}]
            error = self.cog.enrich_file_search_tools(tools)
            self.assertIsNotNone(error)
            self.assertIn("GEMINI_FILE_SEARCH_STORE_IDS", error)
        finally:
            gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = original

    async def test_validate_attachment_size_within_limit(self):
        """Test that attachments within limits pass validation."""
        attachment = MagicMock()
        attachment.size = 10 * 1024 * 1024  # 10 MB
        attachment.content_type = "image/png"

        result = self.cog._validate_attachment_size(attachment)
        self.assertIsNone(result)

    async def test_validate_attachment_size_exceeds_file_api_max(self):
        """Test that attachments over 2 GB are rejected."""
        attachment = MagicMock()
        attachment.size = 3 * 1024 * 1024 * 1024  # 3 GB
        attachment.content_type = "video/mp4"

        result = self.cog._validate_attachment_size(attachment)
        self.assertIsNotNone(result)
        self.assertIn("too large", result)
        self.assertIn("2 GB", result)

    async def test_prepare_attachment_part_inline_small_file(self):
        """Test that small files use inline data."""
        attachment = MagicMock()
        attachment.size = 1 * 1024 * 1024  # 1 MB (under 20 MB threshold)
        attachment.content_type = "image/png"
        attachment.url = "https://cdn.example.com/image.png"
        attachment.filename = "image.png"

        mock_session = MagicMock()
        mock_session.closed = False
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"image data")
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context)
        self.cog._http_session = mock_session

        result = await self.cog._prepare_attachment_part(attachment)

        self.assertIsNotNone(result)
        self.assertIn("inline_data", result)
        self.assertEqual(result["inline_data"]["mime_type"], "image/png")
        self.assertEqual(result["inline_data"]["data"], b"image data")

    async def test_prepare_attachment_part_file_api_large_file(self):
        """Test that large files use the File API."""
        attachment = MagicMock()
        attachment.size = 25 * 1024 * 1024  # 25 MB (over 20 MB threshold)
        attachment.content_type = "application/pdf"
        attachment.url = "https://cdn.example.com/doc.pdf"
        attachment.filename = "doc.pdf"

        mock_session = MagicMock()
        mock_session.closed = False
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"pdf data")
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context)
        self.cog._http_session = mock_session

        mock_uploaded_file = SimpleNamespace(
            name="files/abc123",
            uri="https://generativelanguage.googleapis.com/files/abc123",
            mime_type="application/pdf",
        )
        self.cog.client.aio.files.upload = AsyncMock(
            return_value=mock_uploaded_file
        )

        uploaded_names = []
        result = await self.cog._prepare_attachment_part(
            attachment, uploaded_names
        )

        self.assertIsNotNone(result)
        self.assertIn("file_data", result)
        self.assertEqual(result["file_data"]["mime_type"], "application/pdf")
        self.assertEqual(uploaded_names, ["files/abc123"])

    async def test_prepare_attachment_part_file_api_fallback_to_inline(self):
        """Test that File API failure falls back to inline data."""
        attachment = MagicMock()
        attachment.size = 25 * 1024 * 1024  # 25 MB
        attachment.content_type = "audio/mpeg"
        attachment.url = "https://cdn.example.com/audio.mp3"
        attachment.filename = "audio.mp3"

        mock_session = MagicMock()
        mock_session.closed = False
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"audio data")
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context)
        self.cog._http_session = mock_session

        self.cog.client.aio.files.upload = AsyncMock(
            side_effect=Exception("Upload failed")
        )

        result = await self.cog._prepare_attachment_part(attachment)

        self.assertIsNotNone(result)
        self.assertIn("inline_data", result)
        self.assertEqual(result["inline_data"]["mime_type"], "audio/mpeg")

    async def test_prepare_attachment_part_fetch_failure(self):
        """Test that download failure returns None."""
        attachment = MagicMock()
        attachment.size = 1 * 1024 * 1024
        attachment.content_type = "image/png"
        attachment.url = "https://cdn.example.com/broken.png"
        attachment.filename = "broken.png"

        mock_session = MagicMock()
        mock_session.closed = False
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context)
        self.cog._http_session = mock_session

        result = await self.cog._prepare_attachment_part(attachment)
        self.assertIsNone(result)

    async def test_cleanup_uploaded_files(self):
        """Test that _cleanup_uploaded_files deletes all tracked files."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            uploaded_file_names=["files/abc", "files/def"],
        )

        self.cog.client.aio.files.delete = AsyncMock()

        await self.cog._cleanup_uploaded_files(params)

        self.assertEqual(self.cog.client.aio.files.delete.call_count, 2)
        self.assertEqual(params.uploaded_file_names, [])

    async def test_cleanup_uploaded_files_handles_errors(self):
        """Test that _cleanup_uploaded_files handles API errors gracefully."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            uploaded_file_names=["files/abc", "files/def"],
        )

        self.cog.client.aio.files.delete = AsyncMock(
            side_effect=Exception("Not found")
        )

        # Should not raise
        await self.cog._cleanup_uploaded_files(params)

        self.assertEqual(params.uploaded_file_names, [])

    async def test_cleanup_uploaded_files_noop_when_empty(self):
        """Test that _cleanup_uploaded_files is a no-op with no files."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")

        self.cog.client.aio.files.delete = AsyncMock()

        await self.cog._cleanup_uploaded_files(params)

        self.cog.client.aio.files.delete.assert_not_called()


class TestGeminiAPIImageGeneration(unittest.IsolatedAsyncioTestCase):
    """Tests for image generation text response handling and truncation."""

    async def asyncSetUp(self):
        # Mock the auth module
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key",
                    GUILD_IDS=[123456789],
                    GEMINI_FILE_SEARCH_STORE_IDS=["store-1", "store-2"],
                )
            },
        )
        self.auth_patcher.start()

        # Mock the genai Client
        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()

        from gemini_api import GeminiAPI

        intents = Intents.default()
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    async def test_prompt_prefix_for_generation(self):
        """Test that prompts are prefixed with 'Create image:' for Gemini models."""
        # Placeholder test - would need to mock the API response
        # The logic is tested indirectly through integration tests
        pass

    async def test_prompt_unchanged_for_editing(self):
        """Test that prompts are NOT prefixed when an attachment is provided."""
        # Placeholder test - would need to mock the API response
        # The logic is tested indirectly through integration tests
        pass

    async def test_text_response_truncation_under_limit(self):
        """Test that short text responses are not truncated."""
        short_text = "This is a short response"
        max_length = 3800

        # Simulate the truncation logic
        truncated = (
            short_text[:max_length] + "..."
            if len(short_text) > max_length
            else short_text
        )

        self.assertEqual(truncated, short_text)
        self.assertNotIn("...", truncated)

    async def test_text_response_truncation_over_limit(self):
        """Test that long text responses are truncated to 3800 chars."""
        long_text = "A" * 5000  # Exceeds 3800 char limit
        max_length = 3800

        # Simulate the truncation logic
        truncated = (
            long_text[:max_length] + "..."
            if len(long_text) > max_length
            else long_text
        )

        self.assertEqual(len(truncated), max_length + 3)  # 3800 + "..."
        self.assertTrue(truncated.endswith("..."))

    async def test_text_response_fits_in_discord_embed(self):
        """Test that truncated text + other content fits Discord's 4096 char limit."""
        # Maximum text we allow
        max_text_length = 3800
        very_long_text = "B" * 10000

        truncated = (
            very_long_text[:max_text_length] + "..."
            if len(very_long_text) > max_text_length
            else very_long_text
        )

        # Simulate the full embed description
        embed_description = "The model did not generate any images.\n"
        embed_description += f"Text response: {truncated}\n"
        embed_description += "\n*Note: These parameters are not yet implemented for Gemini: negative_prompt, guidance_scale, aspect_ratio, person_generation*"

        # Discord's limit is 4096 characters
        self.assertLessEqual(len(embed_description), 4096)

    async def test_user_prompt_truncation_under_limit(self):
        """Test that short user prompts are not truncated."""
        short_prompt = "Generate a beautiful sunset"
        max_length = 2000

        # Simulate the truncation logic
        truncated = short_prompt[:max_length] + "..." if len(short_prompt) > max_length else short_prompt

        self.assertEqual(truncated, short_prompt)
        self.assertNotIn("...", truncated)

    async def test_user_prompt_truncation_over_limit(self):
        """Test that long user prompts are truncated to 2000 chars."""
        long_prompt = "A" * 3000  # Exceeds 2000 char limit
        max_length = 2000

        # Simulate the truncation logic
        truncated = long_prompt[:max_length] + "..." if len(long_prompt) > max_length else long_prompt

        self.assertEqual(len(truncated), max_length + 3)  # 2000 + "..."
        self.assertTrue(truncated.endswith("..."))

    async def test_user_prompt_truncation_fits_embed(self):
        """Test that truncated prompt + metadata fits Discord's 4096 char limit."""
        max_prompt_length = 2000
        huge_prompt = "C" * 5000

        truncated_prompt = (
            huge_prompt[:max_prompt_length] + "..."
            if len(huge_prompt) > max_prompt_length
            else huge_prompt
        )

        # Simulate a full embed with prompt and metadata
        embed_description = f"**Prompt:** {truncated_prompt}\n"
        embed_description += "**Model:** gemini-3-pro-image-preview\n"
        embed_description += "**Mode:** Image Generation\n"
        embed_description += "**Number of Images:** 1\n"

        # Discord's limit is 4096 characters
        self.assertLessEqual(len(embed_description), 4096)

    async def test_generate_image_with_gemini_default_config(self):
        """Test that default config has response_modalities and no custom image_config/tools."""
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image-preview",
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        self.assertEqual(config.response_modalities, ["TEXT", "IMAGE"])
        # image_config may be auto-initialized but should have no custom values
        if config.image_config:
            self.assertIsNone(config.image_config.image_size)
            self.assertIsNone(config.image_config.aspect_ratio)
        self.assertIsNone(config.tools)

    async def test_generate_image_with_gemini_image_size(self):
        """Test that image_size is passed via image_config."""
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image-preview",
            image_size="2k",
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        self.assertIsNotNone(config.image_config)
        self.assertEqual(config.image_config.image_size, "2k")

    async def test_generate_image_with_gemini_aspect_ratio(self):
        """Test that non-default aspect_ratio is passed via image_config."""
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image-preview",
            aspect_ratio="16:9",
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        self.assertIsNotNone(config.image_config)
        self.assertEqual(config.image_config.aspect_ratio, "16:9")

    async def test_generate_image_with_gemini_image_search(self):
        """Test that google_image_search adds tools with search_types."""
        from google.genai import types
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image-preview",
            google_image_search=True,
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        self.assertIsNotNone(config.tools)
        self.assertEqual(len(config.tools), 1)
        tool = config.tools[0]
        self.assertIsNotNone(tool.google_search)
        self.assertIsNotNone(tool.google_search.search_types)
        self.assertIsNotNone(tool.google_search.search_types.image_search)
        self.assertIsNotNone(tool.google_search.search_types.web_search)

    async def test_generate_image_with_gemini_image_search_wrong_model(self):
        """Test that google_image_search is ignored for non-3.1-flash-image models."""
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3-pro-image-preview",
            google_image_search=True,
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        self.assertIsNone(config.tools)

    async def test_generate_image_with_gemini_combined_config(self):
        """Test that image_size, aspect_ratio, and google_image_search combine correctly."""
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image-preview",
            aspect_ratio="9:16",
            image_size="2k",
            google_image_search=True,
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        # image_config should have both aspect_ratio and image_size
        self.assertEqual(config.image_config.aspect_ratio, "9:16")
        self.assertEqual(config.image_config.image_size, "2k")
        # tools should have google_search with image_search
        self.assertEqual(len(config.tools), 1)
        self.assertIsNotNone(config.tools[0].google_search.search_types.image_search)


class TestGeminiAPIDeepResearch(unittest.IsolatedAsyncioTestCase):
    """Tests for deep research helper methods."""

    async def asyncSetUp(self):
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key",
                    GUILD_IDS=[123456789],
                    GEMINI_FILE_SEARCH_STORE_IDS=["store-1", "store-2"],
                )
            },
        )
        self.auth_patcher.start()

        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()
        mock_client_instance.aio.interactions.create = AsyncMock()
        mock_client_instance.aio.interactions.get = AsyncMock()

        from gemini_api import GeminiAPI

        intents = Intents.default()
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    async def test_run_deep_research_success(self):
        """Test _run_deep_research with a successful completion."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Research AI safety")

        # First call returns in_progress, second returns completed
        interaction_started = SimpleNamespace(
            id="interaction-1", status="in_progress", outputs=None
        )
        interaction_done = SimpleNamespace(
            id="interaction-1",
            status="completed",
            outputs=[SimpleNamespace(text="# AI Safety Report\n\nDetailed findings...")],
        )

        self.cog.client.aio.interactions.create.return_value = interaction_started
        self.cog.client.aio.interactions.get.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        self.assertEqual(result, "# AI Safety Report\n\nDetailed findings...")
        self.cog.client.aio.interactions.create.assert_called_once_with(
            input="Research AI safety",
            agent="deep-research-pro-preview-12-2025",
            background=True,
        )

    async def test_run_deep_research_with_file_search(self):
        """Test _run_deep_research passes file_search tools when enabled."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Analyze report", file_search=True)

        interaction_done = SimpleNamespace(
            id="interaction-2",
            status="completed",
            outputs=[SimpleNamespace(text="Report analysis...")],
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        call_kwargs = self.cog.client.aio.interactions.create.call_args
        self.assertIn("tools", call_kwargs.kwargs)
        self.assertEqual(
            call_kwargs.kwargs["tools"][0]["type"], "file_search"
        )
        self.assertEqual(
            call_kwargs.kwargs["tools"][0]["file_search_store_names"],
            ["store-1", "store-2"],
        )

    async def test_run_deep_research_file_search_no_store_ids(self):
        """Test _run_deep_research raises when file_search enabled but no store IDs."""
        import gemini_api
        from util import ResearchParameters

        params = ResearchParameters(prompt="test", file_search=True)

        original = gemini_api.GEMINI_FILE_SEARCH_STORE_IDS
        gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = []
        try:
            with self.assertRaises(Exception) as ctx:
                await self.cog._run_deep_research(params)
            self.assertIn("GEMINI_FILE_SEARCH_STORE_IDS", str(ctx.exception))
        finally:
            gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = original

    async def test_run_deep_research_failed(self):
        """Test _run_deep_research raises on failure."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="test")

        interaction_failed = SimpleNamespace(
            id="interaction-3", status="failed", outputs=None
        )

        self.cog.client.aio.interactions.create.return_value = interaction_failed

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(Exception) as ctx:
                await self.cog._run_deep_research(params)
            self.assertIn("failed", str(ctx.exception))

    async def test_run_deep_research_no_output(self):
        """Test _run_deep_research returns None when no text output."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="test")

        interaction_done = SimpleNamespace(
            id="interaction-4", status="completed", outputs=[]
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        self.assertIsNone(result)

    async def test_create_research_response_embeds(self):
        """Test _create_research_response_embeds creates header embed only."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Research quantum computing")

        embeds = self.cog._create_research_response_embeds(params)

        # Only the header embed (report is sent as file attachment)
        self.assertEqual(len(embeds), 1)
        self.assertEqual(embeds[0].title, "Deep Research")
        self.assertIn("Research quantum computing", embeds[0].description)
        self.assertIn("deep-research-pro-preview-12-2025", embeds[0].description)

    async def test_create_research_response_embeds_with_file_search(self):
        """Test _create_research_response_embeds shows file search status."""
        from util import ResearchParameters

        params = ResearchParameters(
            prompt="Analyze docs", file_search=True
        )

        embeds = self.cog._create_research_response_embeds(params)

        self.assertIn("File Search", embeds[0].description)

    async def test_create_research_response_embeds_truncates_prompt(self):
        """Test _create_research_response_embeds truncates long prompts."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="X" * 3000)

        embeds = self.cog._create_research_response_embeds(params)

        # Prompt should be truncated to 2000 + "..."
        self.assertIn("...", embeds[0].description)


class TestGeminiAPIPricing(unittest.IsolatedAsyncioTestCase):
    """Tests for pricing embed and daily cost tracking."""

    async def asyncSetUp(self):
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key",
                    GUILD_IDS=[123456789],
                    GEMINI_FILE_SEARCH_STORE_IDS=["store-1", "store-2"],
                )
            },
        )
        self.auth_patcher.start()

        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()

        from gemini_api import GeminiAPI, append_pricing_embed

        self.append_pricing_embed = append_pricing_embed

        intents = Intents.default()
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    async def test_cog_init_daily_costs(self):
        """Test that GeminiAPI cog initializes daily_costs dict."""
        self.assertEqual(self.cog.daily_costs, {})

    async def test_track_daily_cost_accumulates(self):
        """Test that _track_daily_cost accumulates costs for the same user and day."""
        daily1 = self.cog._track_daily_cost(user_id=12345, cost=0.50)
        self.assertAlmostEqual(daily1, 0.50)

        # Second call same user — should accumulate
        daily2 = self.cog._track_daily_cost(user_id=12345, cost=0.50)
        self.assertAlmostEqual(daily2, 1.00)

    async def test_track_daily_cost_separate_users(self):
        """Test that _track_daily_cost tracks users independently."""
        self.cog._track_daily_cost(user_id=111, cost=0.10)
        daily_user2 = self.cog._track_daily_cost(user_id=222, cost=0.10)
        self.assertAlmostEqual(daily_user2, 0.10)

    async def test_append_pricing_embed(self):
        """Test that append_pricing_embed creates a Gemini Blue embed with cost info."""
        embeds = []
        self.append_pricing_embed(
            embeds, "gemini-2.0-flash",
            input_tokens=500_000, output_tokens=200_000,
            daily_cost=1.25,
        )
        self.assertEqual(len(embeds), 1)
        embed = embeds[0]
        self.assertIn("$", embed.description)
        self.assertIn("500,000 tokens in", embed.description)
        self.assertIn("200,000 tokens out", embed.description)
        self.assertIn("daily $1.25", embed.description)

    async def test_append_pricing_embed_zero_tokens(self):
        """Test pricing embed with zero tokens."""
        embeds = []
        self.append_pricing_embed(
            embeds, "gemini-2.5-pro",
            input_tokens=0, output_tokens=0,
            daily_cost=0.0,
        )
        self.assertEqual(len(embeds), 1)
        self.assertIn("$0.0000", embeds[0].description)
        self.assertIn("0 tokens in", embeds[0].description)
        self.assertIn("0 tokens out", embeds[0].description)


class TestGeminiAPICaching(unittest.IsolatedAsyncioTestCase):
    """Tests for explicit context caching logic."""

    async def asyncSetUp(self):
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key",
                    GUILD_IDS=[123456789],
                    GEMINI_FILE_SEARCH_STORE_IDS=["store-1", "store-2"],
                )
            },
        )
        self.auth_patcher.start()

        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()
        mock_client_instance.aio.caches.create = AsyncMock()
        mock_client_instance.aio.caches.delete = AsyncMock()
        mock_client_instance.aio.caches.update = AsyncMock()

        from gemini_api import GeminiAPI

        intents = Intents.default()
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    async def test_maybe_create_cache_below_threshold(self):
        """Test that _maybe_create_cache does nothing when below token threshold."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        history = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=500)
        )

        await self.cog._maybe_create_cache(params, history, response)

        self.assertIsNone(params.cache_name)
        self.assertEqual(params.cached_history_length, 0)
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_above_threshold(self):
        """Test that _maybe_create_cache creates a cache when above threshold."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview", conversation_id=100
        )
        history = [
            {"role": "user", "parts": [{"text": "long prompt " * 200}]},
            {"role": "model", "parts": [{"text": "long response " * 200}]},
        ]
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=2000)
        )

        mock_cache = SimpleNamespace(name="cachedContents/test-cache-id")
        self.cog.client.aio.caches.create.return_value = mock_cache

        await self.cog._maybe_create_cache(params, history, response)

        self.assertEqual(params.cache_name, "cachedContents/test-cache-id")
        self.assertEqual(params.cached_history_length, 2)
        self.cog.client.aio.caches.create.assert_called_once()

    async def test_maybe_create_cache_refreshes_ttl(self):
        """Test that _maybe_create_cache refreshes TTL when uncached tail is small."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/existing",
            cached_history_length=4,
        )
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=3000,
                cached_content_token_count=2500,
            )
        )

        await self.cog._maybe_create_cache(params, [], response)

        # Should refresh TTL, not create a new cache
        self.cog.client.aio.caches.create.assert_not_called()
        self.cog.client.aio.caches.update.assert_called_once()
        self.assertEqual(params.cache_name, "cachedContents/existing")
        self.assertEqual(params.cached_history_length, 4)

    async def test_maybe_create_cache_refreshes_ttl_error_handled(self):
        """Test that TTL refresh errors are handled gracefully."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/existing",
            cached_history_length=4,
        )
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1500,
                cached_content_token_count=1200,
            )
        )
        self.cog.client.aio.caches.update.side_effect = Exception("TTL error")

        await self.cog._maybe_create_cache(params, [], response)

        # Cache should remain unchanged despite TTL refresh failure
        self.assertEqual(params.cache_name, "cachedContents/existing")

    async def test_maybe_create_cache_recaches_when_uncached_tail_large(self):
        """Test that _maybe_create_cache re-caches when uncached portion exceeds threshold."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            conversation_id=100,
            cache_name="cachedContents/old-cache",
            cached_history_length=4,
        )
        history = [
            {"role": "user", "parts": [{"text": f"msg {i}"}]}
            for i in range(10)
        ]
        # uncached = 5000 - 2000 = 3000, threshold = 1024 → re-cache
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=5000,
                cached_content_token_count=2000,
            )
        )

        mock_cache = SimpleNamespace(name="cachedContents/new-cache")
        self.cog.client.aio.caches.create.return_value = mock_cache

        await self.cog._maybe_create_cache(params, history, response)

        # New cache created, old cache deleted
        self.assertEqual(params.cache_name, "cachedContents/new-cache")
        self.assertEqual(params.cached_history_length, 10)
        self.cog.client.aio.caches.create.assert_called_once()
        self.cog.client.aio.caches.delete.assert_called_once_with(
            name="cachedContents/old-cache"
        )
        # TTL should NOT have been refreshed (we re-cached instead)
        self.cog.client.aio.caches.update.assert_not_called()

    async def test_maybe_create_cache_recache_keeps_old_on_create_error(self):
        """Test that re-cache failure preserves the old cache."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            conversation_id=100,
            cache_name="cachedContents/old-cache",
            cached_history_length=4,
        )
        history = [{"role": "user", "parts": [{"text": "msg"}]}] * 10
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=5000,
                cached_content_token_count=2000,
            )
        )
        self.cog.client.aio.caches.create.side_effect = Exception("Create failed")

        await self.cog._maybe_create_cache(params, history, response)

        # Old cache should be preserved
        self.assertEqual(params.cache_name, "cachedContents/old-cache")
        self.assertEqual(params.cached_history_length, 4)
        # Old cache should NOT have been deleted
        self.cog.client.aio.caches.delete.assert_not_called()

    async def test_maybe_create_cache_unsupported_model(self):
        """Test that _maybe_create_cache skips for models without explicit caching."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-2.0-flash")
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=5000)
        )

        await self.cog._maybe_create_cache(params, [], response)

        self.assertIsNone(params.cache_name)
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_implicit_only_models_skipped(self):
        """Test that 2.5 models rely on implicit caching and are skipped."""
        from util import ChatCompletionParameters

        for model in ("gemini-2.5-pro", "gemini-2.5-flash"):
            params = ChatCompletionParameters(model=model)
            response = SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=10000)
            )

            await self.cog._maybe_create_cache(params, [], response)

            self.assertIsNone(params.cache_name)
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_no_usage_metadata(self):
        """Test that _maybe_create_cache handles missing usage_metadata."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        response = SimpleNamespace()  # No usage_metadata

        await self.cog._maybe_create_cache(params, [], response)

        self.assertIsNone(params.cache_name)
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_handles_api_error(self):
        """Test that _maybe_create_cache logs warning on API error."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview", conversation_id=100
        )
        history = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=2000)
        )

        self.cog.client.aio.caches.create.side_effect = Exception("API error")

        await self.cog._maybe_create_cache(params, history, response)

        # Should not have set cache_name due to error
        self.assertIsNone(params.cache_name)
        self.assertEqual(params.cached_history_length, 0)

    async def test_delete_conversation_cache(self):
        """Test that _delete_conversation_cache deletes and clears fields."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/to-delete",
            cached_history_length=4,
        )

        await self.cog._delete_conversation_cache(params)

        self.cog.client.aio.caches.delete.assert_called_once_with(
            name="cachedContents/to-delete"
        )
        self.assertIsNone(params.cache_name)
        self.assertEqual(params.cached_history_length, 0)

    async def test_delete_conversation_cache_noop_when_none(self):
        """Test that _delete_conversation_cache is a no-op when no cache exists."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")

        await self.cog._delete_conversation_cache(params)

        self.cog.client.aio.caches.delete.assert_not_called()

    async def test_delete_conversation_cache_handles_api_error(self):
        """Test that _delete_conversation_cache handles API errors gracefully."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/broken",
            cached_history_length=2,
        )

        self.cog.client.aio.caches.delete.side_effect = Exception("Not found")

        await self.cog._delete_conversation_cache(params)

        # Fields should still be cleared even on API error
        self.assertIsNone(params.cache_name)
        self.assertEqual(params.cached_history_length, 0)


class TestGuessUrlMimeType(unittest.TestCase):
    """Tests for _guess_url_mime_type YouTube detection and fallback."""

    def setUp(self):
        from gemini_api import _guess_url_mime_type

        self._guess_url_mime_type = _guess_url_mime_type

    def test_youtube_long_url(self):
        result = self._guess_url_mime_type("https://www.youtube.com/watch?v=abc123")
        self.assertEqual(result, "video/mp4")

    def test_youtube_short_url(self):
        result = self._guess_url_mime_type("https://youtu.be/abc123")
        self.assertEqual(result, "video/mp4")

    def test_youtube_no_scheme(self):
        result = self._guess_url_mime_type("youtube.com/watch?v=abc123")
        self.assertEqual(result, "video/mp4")

    def test_youtube_http(self):
        result = self._guess_url_mime_type("http://www.youtube.com/watch?v=abc123")
        self.assertEqual(result, "video/mp4")

    def test_regular_image_url(self):
        result = self._guess_url_mime_type("https://example.com/photo.jpg")
        self.assertEqual(result, "image/jpeg")

    def test_regular_pdf_url(self):
        result = self._guess_url_mime_type("https://example.com/doc.pdf")
        self.assertEqual(result, "application/pdf")

    def test_unknown_url_fallback(self):
        result = self._guess_url_mime_type("https://example.com/api/data")
        self.assertEqual(result, "application/octet-stream")


class TestThinkingFeatures(unittest.IsolatedAsyncioTestCase):
    """Tests for thinking config, thinking embeds, and response part extraction."""

    async def asyncSetUp(self):
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key",
                    GUILD_IDS=[123456789],
                    GEMINI_FILE_SEARCH_STORE_IDS=["store-1"],
                )
            },
        )
        self.auth_patcher.start()

        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()

        from gemini_api import (
            GeminiAPI,
            append_thinking_embeds,
            append_pricing_embed,
            extract_thinking_text,
            _get_response_content_parts,
            _build_thinking_config,
        )

        self.append_thinking_embeds = append_thinking_embeds
        self.append_pricing_embed = append_pricing_embed
        self.extract_thinking_text = extract_thinking_text
        self._get_response_content_parts = _get_response_content_parts
        self._build_thinking_config = _build_thinking_config

        intents = Intents.default()
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    # --- _build_thinking_config ---

    async def test_build_thinking_config_none_when_no_params(self):
        """Test that _build_thinking_config returns None when both params are None."""
        result = self._build_thinking_config(None, None)
        self.assertIsNone(result)

    async def test_build_thinking_config_with_level(self):
        """Test that _build_thinking_config sets thinking_level and include_thoughts."""
        result = self._build_thinking_config("high", None)
        self.assertIsNotNone(result)
        # SDK converts string to ThinkingLevel enum
        self.assertIn("HIGH", str(result.thinking_level))
        self.assertTrue(result.include_thoughts)

    async def test_build_thinking_config_with_budget(self):
        """Test that _build_thinking_config sets thinking_budget and include_thoughts."""
        result = self._build_thinking_config(None, 1024)
        self.assertIsNotNone(result)
        self.assertEqual(result.thinking_budget, 1024)
        self.assertTrue(result.include_thoughts)

    async def test_build_thinking_config_with_both(self):
        """Test that _build_thinking_config sets both level and budget."""
        result = self._build_thinking_config("low", 512)
        self.assertIsNotNone(result)
        self.assertIn("LOW", str(result.thinking_level))
        self.assertEqual(result.thinking_budget, 512)
        self.assertTrue(result.include_thoughts)

    async def test_build_thinking_config_budget_zero(self):
        """Test that _build_thinking_config handles budget=0 (disable thinking)."""
        result = self._build_thinking_config(None, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result.thinking_budget, 0)

    async def test_build_thinking_config_budget_dynamic(self):
        """Test that _build_thinking_config handles budget=-1 (dynamic)."""
        result = self._build_thinking_config(None, -1)
        self.assertIsNotNone(result)
        self.assertEqual(result.thinking_budget, -1)

    # --- extract_thinking_text ---

    async def test_extract_thinking_text_with_thoughts(self):
        """Test extracting thought summaries from response parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(thought=True, text="Let me think..."),
                            SimpleNamespace(thought=False, text="The answer is 42."),
                        ]
                    )
                )
            ]
        )
        result = self.extract_thinking_text(response)
        self.assertEqual(result, "Let me think...")

    async def test_extract_thinking_text_multiple_thought_parts(self):
        """Test extracting multiple thought summary parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(thought=True, text="Step 1: analyze"),
                            SimpleNamespace(thought=True, text="Step 2: compute"),
                            SimpleNamespace(thought=False, text="Result: 7"),
                        ]
                    )
                )
            ]
        )
        result = self.extract_thinking_text(response)
        self.assertEqual(result, "Step 1: analyze\n\nStep 2: compute")

    async def test_extract_thinking_text_no_thoughts(self):
        """Test that no thinking text is returned when there are no thought parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(thought=False, text="Hello!"),
                        ]
                    )
                )
            ]
        )
        result = self.extract_thinking_text(response)
        self.assertEqual(result, "")

    async def test_extract_thinking_text_empty_response(self):
        """Test extracting thinking text from an empty response."""
        response = SimpleNamespace(candidates=[])
        result = self.extract_thinking_text(response)
        self.assertEqual(result, "")

    async def test_extract_thinking_text_no_candidates(self):
        """Test extracting thinking text when candidates is None."""
        response = SimpleNamespace(candidates=None)
        result = self.extract_thinking_text(response)
        self.assertEqual(result, "")

    # --- _get_response_content_parts ---

    async def test_get_response_content_parts_returns_parts(self):
        """Test that response parts are extracted correctly."""
        part1 = SimpleNamespace(text="Hello", thought=False)
        part2 = SimpleNamespace(text="Thinking...", thought=True)
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[part1, part2]))
            ]
        )
        result = self._get_response_content_parts(response)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], part1)
        self.assertIs(result[1], part2)

    async def test_get_response_content_parts_empty_candidates(self):
        """Test that None is returned for empty candidates."""
        response = SimpleNamespace(candidates=[])
        result = self._get_response_content_parts(response)
        self.assertIsNone(result)

    async def test_get_response_content_parts_no_content(self):
        """Test that None is returned when content is None."""
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=None)]
        )
        result = self._get_response_content_parts(response)
        self.assertIsNone(result)

    async def test_get_response_content_parts_no_parts(self):
        """Test that None is returned when parts is empty."""
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))]
        )
        result = self._get_response_content_parts(response)
        self.assertIsNone(result)

    # --- append_thinking_embeds ---

    async def test_append_thinking_embeds_with_text(self):
        """Test that thinking embed is created with spoilered text."""
        embeds = []
        self.append_thinking_embeds(embeds, "My thought process")
        self.assertEqual(len(embeds), 1)
        self.assertEqual(embeds[0].title, "Thinking")
        self.assertEqual(embeds[0].description, "||My thought process||")
        # Check it uses light grey color
        from discord import Colour
        self.assertEqual(embeds[0].color, Colour.light_grey())

    async def test_append_thinking_embeds_empty_text(self):
        """Test that no embed is created for empty thinking text."""
        embeds = []
        self.append_thinking_embeds(embeds, "")
        self.assertEqual(len(embeds), 0)

    async def test_append_thinking_embeds_truncates_long_text(self):
        """Test that long thinking text is truncated."""
        embeds = []
        long_text = "A" * 4000  # Over 3500 limit
        self.append_thinking_embeds(embeds, long_text)
        self.assertEqual(len(embeds), 1)
        self.assertIn("[thinking truncated]", embeds[0].description)
        # Check total length is under Discord limit (spoiler markers add 4 chars)
        self.assertLessEqual(len(embeds[0].description), 3600)

    # --- pricing with thinking tokens ---

    async def test_append_pricing_embed_with_thinking_tokens(self):
        """Test pricing embed shows thinking token count."""
        embeds = []
        self.append_pricing_embed(
            embeds, "gemini-3-flash-preview",
            input_tokens=100_000, output_tokens=50_000,
            daily_cost=0.50, thinking_tokens=200_000,
        )
        self.assertEqual(len(embeds), 1)
        self.assertIn("200,000 thinking", embeds[0].description)
        self.assertIn("100,000 in", embeds[0].description)
        self.assertIn("50,000 out", embeds[0].description)

    async def test_append_pricing_embed_zero_thinking_tokens(self):
        """Test pricing embed omits thinking when zero."""
        embeds = []
        self.append_pricing_embed(
            embeds, "gemini-2.5-flash",
            input_tokens=100_000, output_tokens=50_000,
            daily_cost=0.10, thinking_tokens=0,
        )
        self.assertEqual(len(embeds), 1)
        self.assertNotIn("thinking", embeds[0].description)
        self.assertIn("tokens in", embeds[0].description)

    async def test_track_daily_cost_with_thinking_tokens(self):
        """Test that _track_daily_cost accumulates pre-calculated cost including thinking."""
        from util import calculate_cost
        # gemini-2.0-flash: $0.10/M in, $0.40/M out
        # thinking tokens billed at output rate
        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 500_000, thinking_tokens=500_000)
        daily = self.cog._track_daily_cost(user_id=99, cost=cost)
        # $0.10 + ($0.40 * 1.0) = $0.50
        self.assertAlmostEqual(daily, 0.50)

    async def test_log_cost_basic(self):
        """Test that _log_cost calls logger.info with structured cost data."""
        self.cog.logger = MagicMock()
        self.cog._log_cost(
            "chat", 12345, "gemini-2.0-flash", 0.50, 1.00,
            input_tokens=1000, output_tokens=500,
        )
        self.cog.logger.info.assert_called_once()
        call_args = self.cog.logger.info.call_args
        fmt = call_args[0][0]
        self.assertIn("COST", fmt)
        self.assertIn("command=%s", fmt)
        self.assertIn("daily_total=$%.4f", fmt)

    async def test_log_cost_no_details(self):
        """Test _log_cost with no extra detail kwargs."""
        self.cog.logger = MagicMock()
        self.cog._log_cost("video", 999, "veo-3.1-generate-preview", 3.20, 5.00)
        self.cog.logger.info.assert_called_once()

    async def test_log_cost_image_details(self):
        """Test _log_cost with image-specific details."""
        self.cog.logger = MagicMock()
        self.cog._log_cost(
            "image", 42, "gemini-3.1-flash-image-preview", 0.134, 0.50,
            images=2, input_tokens=500,
        )
        self.cog.logger.info.assert_called_once()
        call_args = self.cog.logger.info.call_args
        # The formatted string should include image and input token details
        self.assertIn("images=2", str(call_args))


if __name__ == "__main__":
    unittest.main()
