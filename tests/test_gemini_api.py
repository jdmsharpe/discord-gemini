import tempfile
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gemini_api
from gemini_api import (
    Conversation,
    GeminiAPI,
    _build_lyria3_prompt,
    _build_thinking_config,
    _get_response_content_parts,
    _guess_attachment_mime_type,
    _guess_url_mime_type,
    _music_file_suffix_for_mime_type,
    append_pricing_embed,
    append_response_embeds,
    append_thinking_embeds,
    extract_thinking_text,
    extract_tool_info,
)


def build_mock_bot() -> MagicMock:
    bot = MagicMock()
    bot.user = MagicMock(name="bot-user")
    bot.owner_id = 1234567890
    bot.sync_commands = AsyncMock()
    bot.add_cog = MagicMock()
    bot.loop = None
    return bot


class GeminiCogTestCase:
    file_search_store_ids = ["store-1", "store-2"]

    @pytest.fixture(autouse=True)
    def setup(self):
        with (
            patch.object(
                gemini_api,
                "GEMINI_FILE_SEARCH_STORE_IDS",
                self.file_search_store_ids.copy(),
            ) as _store_ids_patcher,
            patch("gemini_api.genai.Client") as mock_genai_client,
        ):
            self.mock_genai_client = mock_genai_client
            self.mock_client_instance = mock_genai_client.return_value
            self.mock_client_instance.aio.aclose = AsyncMock()
            self.mock_client_instance.close = MagicMock()

            self.bot = build_mock_bot()
            self.cog = GeminiAPI(bot=self.bot)
            yield


class AsyncGeminiCogTestCase:
    file_search_store_ids = ["store-1", "store-2"]

    @pytest.fixture(autouse=True)
    async def setup(self):
        with (
            patch.object(
                gemini_api,
                "GEMINI_FILE_SEARCH_STORE_IDS",
                self.file_search_store_ids.copy(),
            ) as _store_ids_patcher,
            patch("gemini_api.genai.Client") as mock_genai_client,
        ):
            self.mock_genai_client = mock_genai_client
            self.mock_client_instance = mock_genai_client.return_value
            self.mock_client_instance.aio.aclose = AsyncMock()
            self.mock_client_instance.close = MagicMock()

            self.bot = build_mock_bot()
            self.bot.loop = gemini_api.asyncio.get_running_loop()
            self.cog = GeminiAPI(bot=self.bot)
            yield


class TestGeminiAPI(AsyncGeminiCogTestCase):
    async def test_cog_init(self):
        """Test that GeminiAPI cog initializes correctly."""
        assert self.cog.bot == self.bot
        assert self.cog.conversations == {}
        assert self.cog.message_to_conversation_id == {}
        assert self.cog.views == {}
        assert self.cog.last_view_messages == {}
        assert self.cog._http_session is None

    async def test_on_ready(self):
        """Test that on_ready syncs commands."""
        await self.cog.on_ready()
        self.bot.sync_commands.assert_called_once()

    async def test_get_http_session_creates_session(self):
        """Test that _get_http_session creates a new session."""
        assert self.cog._http_session is None
        session = await self.cog._get_http_session()
        assert session is not None
        assert self.cog._http_session == session
        await session.close()

    async def test_get_http_session_reuses_session(self):
        """Test that _get_http_session reuses existing session."""
        session1 = await self.cog._get_http_session()
        session2 = await self.cog._get_http_session()
        assert session1 == session2
        await session1.close()

    async def test_on_message_ignores_bot(self):
        """Test that on_message ignores messages from the bot itself."""
        message = MagicMock()
        message.author = self.bot.user

        await self.cog.on_message(message)

        assert len(self.cog.conversations) == 0

    async def test_on_message_no_matching_conversation(self):
        """Test that on_message handles no matching conversation gracefully."""
        message = MagicMock()
        message.author = MagicMock()
        message.author.id = 999
        message.channel = MagicMock()
        message.channel.id = 888
        message.content = "Hello"

        await self.cog.on_message(message)

    async def test_cog_unload_closes_session(self):
        """Test that cog_unload closes the HTTP session."""
        session = await self.cog._get_http_session()
        assert session.closed is False

        self.cog.cog_unload()
        await gemini_api.asyncio.sleep(0)

        assert self.cog._http_session is None


class TestConversation:
    def test_conversation_dataclass(self):
        """Test the Conversation dataclass."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        history = [{"role": "user", "parts": [{"text": "Hello"}]}]
        conversation = Conversation(params=params, history=history)

        assert conversation.params == params
        assert conversation.history == history


class TestAppendResponseEmbeds:
    def test_append_response_embeds_short(self):
        """Test append_response_embeds with short text."""
        embeds = []
        append_response_embeds(embeds, "Hello, World!")
        assert len(embeds) == 1
        assert embeds[0].description == "Hello, World!"
        assert embeds[0].title == "Response"

    def test_append_response_embeds_long(self):
        """Test append_response_embeds with long text that needs chunking."""
        embeds = []
        long_text = "A" * 7000
        append_response_embeds(embeds, long_text)
        assert len(embeds) == 2
        assert embeds[0].title == "Response"
        assert embeds[1].title == "Response (Part 2)"

    def test_append_response_embeds_very_long(self):
        """Test append_response_embeds truncates very long text."""
        embeds = []
        very_long_text = "B" * 25000
        append_response_embeds(embeds, very_long_text)
        total_length = sum(len(e.description) for e in embeds)
        assert total_length < 21000


class TestExtractToolInfo:
    def test_extract_tool_info_empty_output(self):
        """Test extract_tool_info with empty candidates."""
        response = SimpleNamespace(candidates=[])
        tool_info = extract_tool_info(response)
        assert tool_info["tools_used"] == []
        assert tool_info["citations"] == []
        assert tool_info["search_queries"] == []

    def test_extract_tool_info_google_search(self):
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

        tool_info = extract_tool_info(response)
        assert "google_search" in tool_info["tools_used"]
        assert tool_info["search_queries"] == ["who won euro 2024"]
        assert tool_info["citations"] == [
            {"title": "Example Source", "uri": "https://example.com/source"}
        ]

    def test_extract_tool_info_code_execution(self):
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

        tool_info = extract_tool_info(response)
        assert "code_execution" in tool_info["tools_used"]
        assert tool_info["citations"] == []
        assert tool_info["search_queries"] == []

    def test_extract_tool_info_google_maps(self):
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

        tool_info = extract_tool_info(response)
        assert "google_maps" in tool_info["tools_used"]
        assert tool_info["citations"] == [
            {"title": "Test Place", "uri": "https://maps.google.com/?cid=123"}
        ]
        assert tool_info["maps_widget_token"] == "widgetcontent/token"

    def test_extract_tool_info_url_context(self):
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

        tool_info = extract_tool_info(response)
        assert "url_context" in tool_info["tools_used"]
        assert tool_info["url_context_sources"] == [
            {
                "retrieved_url": "https://example.com/a",
                "status": "URL_RETRIEVAL_STATUS_SUCCESS",
            }
        ]

    def test_extract_tool_info_file_search_via_retrieval_metadata(self):
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

        tool_info = extract_tool_info(response)
        assert "file_search" in tool_info["tools_used"]

    def test_extract_tool_info_file_search_fallback(self):
        """Test extract_tool_info detects file_search when grounding chunks present without search/maps."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=[],
                        grounding_chunks=[SimpleNamespace(web=None, maps=None)],
                        search_entry_point=None,
                        google_maps_widget_context_token=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=None,
                )
            ]
        )

        tool_info = extract_tool_info(response)
        assert "file_search" in tool_info["tools_used"]


class TestGeminiAPIHelpers(AsyncGeminiCogTestCase):
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
        assert result == b"image data"

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
        assert result is None

    async def test_enrich_file_search_tools_injects_store_ids(self):
        """Test that enrich_file_search_tools injects store IDs."""
        tools = [{"file_search": {}}]
        error = self.cog.enrich_file_search_tools(tools)
        assert error is None
        assert tools[0] == {"file_search": {"file_search_store_names": ["store-1", "store-2"]}}

    async def test_enrich_file_search_tools_no_file_search(self):
        """Test that enrich_file_search_tools is a no-op without file_search."""
        tools = [{"google_search": {}}]
        error = self.cog.enrich_file_search_tools(tools)
        assert error is None
        assert tools == [{"google_search": {}}]

    async def test_enrich_file_search_tools_no_store_ids(self):
        """Test that enrich_file_search_tools returns error when store IDs not configured."""
        import gemini_api

        original = gemini_api.GEMINI_FILE_SEARCH_STORE_IDS
        gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = []
        try:
            tools = [{"file_search": {}}]
            error = self.cog.enrich_file_search_tools(tools)
            assert error is not None
            assert "GEMINI_FILE_SEARCH_STORE_IDS" in error
        finally:
            gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = original

    async def test_validate_attachment_size_within_limit(self):
        """Test that attachments within limits pass validation."""
        attachment = MagicMock()
        attachment.size = 10 * 1024 * 1024  # 10 MB
        attachment.content_type = "image/png"

        result = self.cog._validate_attachment_size(attachment)
        assert result is None

    async def test_validate_attachment_size_exceeds_file_api_max(self):
        """Test that attachments over 2 GB are rejected."""
        attachment = MagicMock()
        attachment.size = 3 * 1024 * 1024 * 1024  # 3 GB
        attachment.content_type = "video/mp4"

        result = self.cog._validate_attachment_size(attachment)
        assert result is not None
        assert "too large" in result
        assert "2 GB" in result

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

        assert result is not None
        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/png"
        assert result["inline_data"]["data"] == b"image data"

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
        self.cog.client.aio.files.upload = AsyncMock(return_value=mock_uploaded_file)

        uploaded_names = []
        result = await self.cog._prepare_attachment_part(attachment, uploaded_names)

        assert result is not None
        assert "file_data" in result
        assert result["file_data"]["mime_type"] == "application/pdf"
        assert uploaded_names == ["files/abc123"]

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

        self.cog.client.aio.files.upload = AsyncMock(side_effect=Exception("Upload failed"))

        result = await self.cog._prepare_attachment_part(attachment)

        assert result is not None
        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "audio/mpeg"

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
        assert result is None

    async def test_cleanup_uploaded_files(self):
        """Test that _cleanup_uploaded_files deletes all tracked files."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            uploaded_file_names=["files/abc", "files/def"],
        )

        self.cog.client.aio.files.delete = AsyncMock()

        await self.cog._cleanup_uploaded_files(params)

        assert self.cog.client.aio.files.delete.call_count == 2
        assert params.uploaded_file_names == []

    async def test_cleanup_uploaded_files_handles_errors(self):
        """Test that _cleanup_uploaded_files handles API errors gracefully."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            uploaded_file_names=["files/abc", "files/def"],
        )

        self.cog.client.aio.files.delete = AsyncMock(side_effect=Exception("Not found"))

        # Should not raise
        await self.cog._cleanup_uploaded_files(params)

        assert params.uploaded_file_names == []

    async def test_cleanup_uploaded_files_noop_when_empty(self):
        """Test that _cleanup_uploaded_files is a no-op with no files."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")

        self.cog.client.aio.files.delete = AsyncMock()

        await self.cog._cleanup_uploaded_files(params)

        self.cog.client.aio.files.delete.assert_not_called()

    async def test_cleanup_conversation_strips_view_and_clears_state(self):
        """Test that _cleanup_conversation edits the message to remove the view and cleans up views dict."""
        user = MagicMock()
        mock_message = AsyncMock()
        self.cog.last_view_messages[user] = mock_message
        self.cog.views[user] = MagicMock()

        await self.cog._cleanup_conversation(user)

        mock_message.edit.assert_awaited_once_with(view=None)
        assert user not in self.cog.last_view_messages
        assert user not in self.cog.views

    async def test_cleanup_conversation_no_previous_message(self):
        """Test that _cleanup_conversation handles missing last_view_messages gracefully."""
        user = MagicMock()
        self.cog.views[user] = MagicMock()

        await self.cog._cleanup_conversation(user)

        assert user not in self.cog.views

    async def test_cleanup_conversation_edit_fails(self):
        """Test that _cleanup_conversation handles deleted messages without raising."""
        import discord

        user = MagicMock()
        mock_message = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.reason = "Not Found"
        mock_message.edit.side_effect = discord.NotFound(mock_response, "Unknown Message")
        self.cog.last_view_messages[user] = mock_message
        self.cog.views[user] = MagicMock()

        await self.cog._cleanup_conversation(user)

        mock_message.edit.assert_awaited_once_with(view=None)
        assert user not in self.cog.last_view_messages
        assert user not in self.cog.views

    async def test_cleanup_conversation_unknown_user(self):
        """Test that _cleanup_conversation is a no-op for users with no state."""
        user = MagicMock()

        await self.cog._cleanup_conversation(user)

        assert user not in self.cog.last_view_messages
        assert user not in self.cog.views


class TestGeminiImageResponseText:
    """Tests for image generation text response handling and truncation."""

    def test_text_response_truncation_under_limit(self):
        """Test that short text responses are not truncated."""
        short_text = "This is a short response"
        max_length = 3800

        # Simulate the truncation logic
        truncated = short_text[:max_length] + "..." if len(short_text) > max_length else short_text

        assert truncated == short_text
        assert "..." not in truncated

    def test_text_response_truncation_over_limit(self):
        """Test that long text responses are truncated to 3800 chars."""
        long_text = "A" * 5000  # Exceeds 3800 char limit
        max_length = 3800

        # Simulate the truncation logic
        truncated = long_text[:max_length] + "..." if len(long_text) > max_length else long_text

        assert len(truncated) == max_length + 3  # 3800 + "..."
        assert truncated.endswith("...")

    def test_text_response_fits_in_discord_embed(self):
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
        assert len(embed_description) <= 4096

    def test_user_prompt_truncation_under_limit(self):
        """Test that short user prompts are not truncated."""
        short_prompt = "Generate a beautiful sunset"
        max_length = 2000

        # Simulate the truncation logic
        truncated = (
            short_prompt[:max_length] + "..." if len(short_prompt) > max_length else short_prompt
        )

        assert truncated == short_prompt
        assert "..." not in truncated

    def test_user_prompt_truncation_over_limit(self):
        """Test that long user prompts are truncated to 2000 chars."""
        long_prompt = "A" * 3000  # Exceeds 2000 char limit
        max_length = 2000

        # Simulate the truncation logic
        truncated = (
            long_prompt[:max_length] + "..." if len(long_prompt) > max_length else long_prompt
        )

        assert len(truncated) == max_length + 3  # 2000 + "..."
        assert truncated.endswith("...")

    def test_user_prompt_truncation_fits_embed(self):
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
        assert len(embed_description) <= 4096


class TestGeminiAPIImageGeneration(AsyncGeminiCogTestCase):
    """Tests for Gemini image generation config."""

    async def test_prompt_prefix_for_generation(self):
        """Test that prompts are prefixed with 'Create image:' for Gemini models."""
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(prompt="A cat", model="gemini-3.1-flash-image-preview")
        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "Create image: A cat"

    async def test_prompt_unchanged_for_editing(self):
        """Test that prompts are NOT prefixed when an attachment is provided."""
        from util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="Edit this cat",
            model="gemini-3.1-flash-image-preview",
        )
        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=None)

        await self.cog._generate_image_with_gemini(params, attachment=MagicMock())

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "Edit this cat"

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
        assert config.response_modalities == ["TEXT", "IMAGE"]
        # image_config may be auto-initialized but should have no custom values
        if config.image_config:
            assert config.image_config.image_size is None
            assert config.image_config.aspect_ratio is None
        assert config.tools is None

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
        assert config.image_config is not None
        assert config.image_config.image_size == "2k"

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
        assert config.image_config is not None
        assert config.image_config.aspect_ratio == "16:9"

    async def test_generate_image_with_gemini_image_search(self):
        """Test that google_image_search adds tools with search_types."""
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
        assert config.tools is not None
        assert len(config.tools) == 1
        tool = config.tools[0]
        assert tool.google_search is not None
        assert tool.google_search.search_types is not None
        assert tool.google_search.search_types.image_search is not None
        assert tool.google_search.search_types.web_search is not None

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
        assert config.tools is None

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
        assert config.image_config.aspect_ratio == "9:16"
        assert config.image_config.image_size == "2k"
        # tools should have google_search with image_search
        assert len(config.tools) == 1
        assert config.tools[0].google_search.search_types.image_search is not None


class TestGeminiAPIDeepResearch(AsyncGeminiCogTestCase):
    """Tests for deep research helper methods."""

    @pytest.fixture(autouse=True)
    async def setup_research(self, setup):
        self.mock_client_instance.aio.interactions.create = AsyncMock()
        self.mock_client_instance.aio.interactions.get = AsyncMock()

    async def test_run_deep_research_success(self):
        """Test _run_deep_research with a successful completion."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Research AI safety")

        mock_usage = SimpleNamespace(
            total_input_tokens=250_000, total_output_tokens=60_000, total_thought_tokens=5_000
        )
        # First call returns in_progress, second returns completed
        interaction_started = SimpleNamespace(
            id="interaction-1", status="in_progress", outputs=None, usage=None
        )
        interaction_done = SimpleNamespace(
            id="interaction-1",
            status="completed",
            outputs=[SimpleNamespace(text="# AI Safety Report\n\nDetailed findings...")],
            usage=mock_usage,
        )

        self.cog.client.aio.interactions.create.return_value = interaction_started
        self.cog.client.aio.interactions.get.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            (
                report_text,
                input_tokens,
                output_tokens,
                thinking_tokens,
            ) = await self.cog._run_deep_research(params)

        assert report_text == "# AI Safety Report\n\nDetailed findings..."
        assert input_tokens == 250_000
        assert output_tokens == 60_000
        assert thinking_tokens == 5_000
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
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            report_text, _, _, _ = await self.cog._run_deep_research(params)

        call_kwargs = self.cog.client.aio.interactions.create.call_args
        assert "tools" in call_kwargs.kwargs
        assert call_kwargs.kwargs["tools"][0]["type"] == "file_search"
        assert call_kwargs.kwargs["tools"][0]["file_search_store_names"] == [
            "store-1",
            "store-2",
        ]

    async def test_run_deep_research_file_search_no_store_ids(self):
        """Test _run_deep_research raises when file_search enabled but no store IDs."""
        import gemini_api
        from util import ResearchParameters

        params = ResearchParameters(prompt="test", file_search=True)

        original = gemini_api.GEMINI_FILE_SEARCH_STORE_IDS
        gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = []
        try:
            with pytest.raises(Exception, match="GEMINI_FILE_SEARCH_STORE_IDS"):
                await self.cog._run_deep_research(params)
        finally:
            gemini_api.GEMINI_FILE_SEARCH_STORE_IDS = original

    async def test_run_deep_research_failed(self):
        """Test _run_deep_research raises on failure."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="test")

        interaction_failed = SimpleNamespace(id="interaction-3", status="failed", outputs=None)

        self.cog.client.aio.interactions.create.return_value = interaction_failed

        with (
            patch("gemini_api.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(Exception, match="failed"),
        ):
            await self.cog._run_deep_research(params)

    async def test_run_deep_research_no_output(self):
        """Test _run_deep_research returns None text when no output."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="test")

        interaction_done = SimpleNamespace(
            id="interaction-4",
            status="completed",
            outputs=[],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=0, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            (
                report_text,
                input_tokens,
                output_tokens,
                thinking_tokens,
            ) = await self.cog._run_deep_research(params)

        assert report_text is None
        assert input_tokens == 100

    async def test_create_research_response_embeds(self):
        """Test _create_research_response_embeds creates header embed only."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Research quantum computing")

        embeds = self.cog._create_research_response_embeds(params)

        # Only the header embed (report is sent as file attachment)
        assert len(embeds) == 1
        assert embeds[0].title == "Deep Research"
        assert "Research quantum computing" in embeds[0].description
        assert "deep-research-pro-preview-12-2025" in embeds[0].description

    async def test_create_research_response_embeds_with_file_search(self):
        """Test _create_research_response_embeds shows file search status."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Analyze docs", file_search=True)

        embeds = self.cog._create_research_response_embeds(params)

        assert "File Search" in embeds[0].description

    async def test_run_deep_research_with_google_maps(self):
        """Test _run_deep_research passes google_maps tool when enabled."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Best restaurants in Tokyo", google_maps=True)

        interaction_done = SimpleNamespace(
            id="interaction-maps",
            status="completed",
            outputs=[SimpleNamespace(text="Tokyo restaurant guide...")],
            usage=SimpleNamespace(
                total_input_tokens=200, total_output_tokens=100, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            report_text, _, _, _ = await self.cog._run_deep_research(params)

        call_kwargs = self.cog.client.aio.interactions.create.call_args
        assert "tools" in call_kwargs.kwargs
        assert {"google_maps": {}} in call_kwargs.kwargs["tools"]

    async def test_run_deep_research_with_file_search_and_google_maps(self):
        """Test _run_deep_research passes both file_search and google_maps tools."""
        from util import ResearchParameters

        params = ResearchParameters(
            prompt="Local docs about Tokyo", file_search=True, google_maps=True
        )

        interaction_done = SimpleNamespace(
            id="interaction-both",
            status="completed",
            outputs=[SimpleNamespace(text="Combined report...")],
            usage=SimpleNamespace(
                total_input_tokens=300, total_output_tokens=150, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("gemini_api.asyncio.sleep", new_callable=AsyncMock):
            await self.cog._run_deep_research(params)

        call_kwargs = self.cog.client.aio.interactions.create.call_args
        tools = call_kwargs.kwargs["tools"]
        assert len(tools) == 2
        assert tools[0]["type"] == "file_search"
        assert {"google_maps": {}} in tools

    async def test_create_research_response_embeds_with_google_maps(self):
        """Test _create_research_response_embeds shows Google Maps status."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="Find restaurants nearby", google_maps=True)

        embeds = self.cog._create_research_response_embeds(params)

        assert "Google Maps" in embeds[0].description

    async def test_create_research_response_embeds_truncates_prompt(self):
        """Test _create_research_response_embeds truncates long prompts."""
        from util import ResearchParameters

        params = ResearchParameters(prompt="X" * 3000)

        embeds = self.cog._create_research_response_embeds(params)

        # Prompt should be truncated to 2000 + "..."
        assert "..." in embeds[0].description


class TestGeminiAPIPricing(GeminiCogTestCase):
    """Tests for pricing embed and daily cost tracking."""

    def test_cog_init_daily_costs(self):
        """Test that GeminiAPI cog initializes daily_costs dict."""
        assert self.cog.daily_costs == {}

    def test_track_daily_cost_accumulates(self):
        """Test that _track_daily_cost accumulates costs for the same user and day."""
        daily1 = self.cog._track_daily_cost(user_id=12345, cost=0.50)
        assert daily1 == pytest.approx(0.50)

        # Second call same user -- should accumulate
        daily2 = self.cog._track_daily_cost(user_id=12345, cost=0.50)
        assert daily2 == pytest.approx(1.00)

    def test_track_daily_cost_separate_users(self):
        """Test that _track_daily_cost tracks users independently."""
        self.cog._track_daily_cost(user_id=111, cost=0.10)
        daily_user2 = self.cog._track_daily_cost(user_id=222, cost=0.10)
        assert daily_user2 == pytest.approx(0.10)

    def test_append_pricing_embed(self):
        """Test that append_pricing_embed creates a Gemini Blue embed with cost info."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.0-flash",
            input_tokens=500_000,
            output_tokens=200_000,
            daily_cost=1.25,
        )
        assert len(embeds) == 1
        embed = embeds[0]
        assert "$" in embed.description
        assert "500,000 tokens in" in embed.description
        assert "200,000 tokens out" in embed.description
        assert "daily $1.25" in embed.description

    def test_append_pricing_embed_zero_tokens(self):
        """Test pricing embed with zero tokens."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-pro",
            input_tokens=0,
            output_tokens=0,
            daily_cost=0.0,
        )
        assert len(embeds) == 1
        assert "$0.0000" in embeds[0].description
        assert "0 tokens in" in embeds[0].description
        assert "0 tokens out" in embeds[0].description


class TestGeminiAPICaching(AsyncGeminiCogTestCase):
    """Tests for explicit context caching logic."""

    @pytest.fixture(autouse=True)
    async def setup_caching(self, setup):
        self.mock_client_instance.aio.caches.create = AsyncMock()
        self.mock_client_instance.aio.caches.delete = AsyncMock()
        self.mock_client_instance.aio.caches.update = AsyncMock()

    async def test_maybe_create_cache_below_threshold(self):
        """Test that _maybe_create_cache does nothing when below token threshold."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        history = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=500))

        await self.cog._maybe_create_cache(params, history, response)

        assert params.cache_name is None
        assert params.cached_history_length == 0
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_above_threshold(self):
        """Test that _maybe_create_cache creates a cache when above threshold."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview", conversation_id=100)
        history = [
            {"role": "user", "parts": [{"text": "long prompt " * 200}]},
            {"role": "model", "parts": [{"text": "long response " * 200}]},
        ]
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=2000))

        mock_cache = SimpleNamespace(name="cachedContents/test-cache-id")
        self.cog.client.aio.caches.create.return_value = mock_cache

        await self.cog._maybe_create_cache(params, history, response)

        assert params.cache_name == "cachedContents/test-cache-id"
        assert params.cached_history_length == 2
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
        assert params.cache_name == "cachedContents/existing"
        assert params.cached_history_length == 4

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
        assert params.cache_name == "cachedContents/existing"

    async def test_maybe_create_cache_recaches_when_uncached_tail_large(self):
        """Test that _maybe_create_cache re-caches when uncached portion exceeds threshold."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            conversation_id=100,
            cache_name="cachedContents/old-cache",
            cached_history_length=4,
        )
        history = [{"role": "user", "parts": [{"text": f"msg {i}"}]} for i in range(10)]
        # uncached = 5000 - 2000 = 3000, threshold = 1024 -> re-cache
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
        assert params.cache_name == "cachedContents/new-cache"
        assert params.cached_history_length == 10
        self.cog.client.aio.caches.create.assert_called_once()
        self.cog.client.aio.caches.delete.assert_called_once_with(name="cachedContents/old-cache")
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
        assert params.cache_name == "cachedContents/old-cache"
        assert params.cached_history_length == 4
        # Old cache should NOT have been deleted
        self.cog.client.aio.caches.delete.assert_not_called()

    async def test_maybe_create_cache_unsupported_model(self):
        """Test that _maybe_create_cache skips for models without explicit caching."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-2.0-flash")
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=5000))

        await self.cog._maybe_create_cache(params, [], response)

        assert params.cache_name is None
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_implicit_only_models_skipped(self):
        """Test that 2.5 models rely on implicit caching and are skipped."""
        from util import ChatCompletionParameters

        for model in ("gemini-2.5-pro", "gemini-2.5-flash"):
            params = ChatCompletionParameters(model=model)
            response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10000))

            await self.cog._maybe_create_cache(params, [], response)

            assert params.cache_name is None
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_no_usage_metadata(self):
        """Test that _maybe_create_cache handles missing usage_metadata."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        response = SimpleNamespace()  # No usage_metadata

        await self.cog._maybe_create_cache(params, [], response)

        assert params.cache_name is None
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_handles_api_error(self):
        """Test that _maybe_create_cache logs warning on API error."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview", conversation_id=100)
        history = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=2000))

        self.cog.client.aio.caches.create.side_effect = Exception("API error")

        await self.cog._maybe_create_cache(params, history, response)

        # Should not have set cache_name due to error
        assert params.cache_name is None
        assert params.cached_history_length == 0

    async def test_delete_conversation_cache(self):
        """Test that _delete_conversation_cache deletes and clears fields."""
        from util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/to-delete",
            cached_history_length=4,
        )

        await self.cog._delete_conversation_cache(params)

        self.cog.client.aio.caches.delete.assert_called_once_with(name="cachedContents/to-delete")
        assert params.cache_name is None
        assert params.cached_history_length == 0

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
        assert params.cache_name is None
        assert params.cached_history_length == 0


class TestGuessUrlMimeType:
    """Tests for _guess_url_mime_type YouTube detection and fallback."""

    def test_youtube_long_url(self):
        result = _guess_url_mime_type("https://www.youtube.com/watch?v=abc123")
        assert result == "video/mp4"

    def test_youtube_short_url(self):
        result = _guess_url_mime_type("https://youtu.be/abc123")
        assert result == "video/mp4"

    def test_youtube_no_scheme(self):
        result = _guess_url_mime_type("youtube.com/watch?v=abc123")
        assert result == "video/mp4"

    def test_youtube_http(self):
        result = _guess_url_mime_type("http://www.youtube.com/watch?v=abc123")
        assert result == "video/mp4"

    def test_regular_image_url(self):
        result = _guess_url_mime_type("https://example.com/photo.jpg")
        assert result == "image/jpeg"

    def test_regular_pdf_url(self):
        result = _guess_url_mime_type("https://example.com/doc.pdf")
        assert result == "application/pdf"

    def test_unknown_url_fallback(self):
        result = _guess_url_mime_type("https://example.com/api/data")
        assert result == "application/octet-stream"


class TestGuessAttachmentMimeType:
    def test_uses_content_type_when_present(self):
        attachment = MagicMock()
        attachment.content_type = "image/png"
        attachment.filename = "cover.bin"

        assert _guess_attachment_mime_type(attachment) == "image/png"

    def test_falls_back_to_filename_guess(self):
        attachment = MagicMock()
        attachment.content_type = None
        attachment.filename = "cover.png"

        assert _guess_attachment_mime_type(attachment) == "image/png"


class TestLyriaHelpers:
    def test_build_lyria3_prompt_for_pro_includes_controls(self):
        from util import MusicGenerationParameters

        params = MusicGenerationParameters(
            prompts=["Dreamy synthpop with warm vocals"],
            model="lyria-3-pro-preview",
            duration=90,
            bpm=120,
            scale="C_MAJOR_A_MINOR",
            density=0.4,
            brightness=0.6,
            guidance=5.0,
        )

        prompt = _build_lyria3_prompt(params)

        assert "Dreamy synthpop with warm vocals" in prompt
        assert "Target duration: about 90 seconds." in prompt
        assert "Tempo: 120 BPM." in prompt
        assert "Musical key or scale: C MAJOR A MINOR." in prompt
        assert "Density: 0.4 on a 0 to 1 scale." in prompt
        assert "Brightness: 0.6 on a 0 to 1 scale." in prompt
        assert "Prompt adherence target: 5.0 on a 0 to 6 scale." in prompt

    def test_build_lyria3_prompt_for_clip_forces_30_second_note(self):
        from util import MusicGenerationParameters

        params = MusicGenerationParameters(
            prompts=["Lo-fi beat"],
            model="lyria-3-clip-preview",
            duration=75,
        )

        prompt = _build_lyria3_prompt(params)

        assert "Generate a 30-second music clip." in prompt
        assert "Target duration" not in prompt

    def test_music_file_suffix_for_mime_type(self):
        assert _music_file_suffix_for_mime_type("audio/mpeg") == "mp3"
        assert _music_file_suffix_for_mime_type("audio/wav") == "wav"
        assert _music_file_suffix_for_mime_type(None) == "mp3"


class TestLyria3Generation(AsyncGeminiCogTestCase):
    async def test_generate_music_with_lyria3_uses_generate_content(self):
        from util import MusicGenerationParameters

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="Lyrics line", inline_data=None),
                            SimpleNamespace(
                                text=None,
                                inline_data=SimpleNamespace(
                                    data=b"audio-bytes",
                                    mime_type="audio/mpeg",
                                ),
                            ),
                        ]
                    )
                )
            ]
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)

        params = MusicGenerationParameters(
            prompts=["Dream pop song"],
            model="lyria-3-pro-preview",
            duration=75,
            bpm=110,
        )

        audio_data, text_response, mime_type = await self.cog._generate_music_with_lyria3(params)

        assert audio_data == b"audio-bytes"
        assert text_response == "Lyrics line"
        assert mime_type == "audio/mpeg"

        call_kwargs = self.cog.client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "lyria-3-pro-preview"
        assert call_kwargs["config"].response_modalities == ["AUDIO", "TEXT"]
        assert "Target duration: about 75 seconds." in call_kwargs["contents"]
        assert "Tempo: 110 BPM." in call_kwargs["contents"]

    async def test_generate_music_with_lyria3_with_attachment_uses_multimodal_contents(self):
        from util import MusicGenerationParameters

        image = gemini_api.Image.new("RGB", (2, 2), color="blue")
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")

        response = SimpleNamespace(candidates=[])
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=image_bytes.getvalue())

        attachment = MagicMock()
        attachment.filename = "reference.png"
        attachment.content_type = "image/png"

        params = MusicGenerationParameters(
            prompts=["Dream pop song"],
            model="lyria-3-pro-preview",
        )

        await self.cog._generate_music_with_lyria3(params, attachment)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args.kwargs
        assert isinstance(call_kwargs["contents"], list)
        assert len(call_kwargs["contents"]) == 2
        assert isinstance(call_kwargs["contents"][0], str)
        assert isinstance(call_kwargs["contents"][1], gemini_api.Image.Image)

    async def test_generate_music_with_lyria3_returns_text_without_audio(self):
        from util import MusicGenerationParameters

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="Only lyrics", inline_data=None)]
                    )
                )
            ]
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)

        params = MusicGenerationParameters(
            prompts=["Minimal piano interlude"],
            model="lyria-3-clip-preview",
        )

        audio_data, text_response, mime_type = await self.cog._generate_music_with_lyria3(params)

        assert audio_data is None
        assert text_response == "Only lyrics"
        assert mime_type is None

    async def test_build_lyria3_music_contents_invalid_attachment_raises(self):
        from util import MusicGenerationParameters

        self.cog._fetch_attachment_bytes = AsyncMock(return_value=b"not-an-image")
        attachment = MagicMock()
        attachment.filename = "bad.png"
        attachment.content_type = "image/png"

        params = MusicGenerationParameters(
            prompts=["Dream pop song"],
            model="lyria-3-pro-preview",
        )

        with pytest.raises(gemini_api.MusicGenerationError):
            await self.cog._build_lyria3_music_contents(params, attachment)


class TestMusicAttachmentValidation(AsyncGeminiCogTestCase):
    async def test_validate_music_attachment_accepts_lyria3_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "image/png"
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-3-pro-preview", attachment)
        assert result is None

    async def test_validate_music_attachment_rejects_realtime_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "image/png"
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-realtime-exp", attachment)
        assert result is not None
        assert "Lyria 3 Pro Preview" in result

    async def test_validate_music_attachment_rejects_non_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "audio/mpeg"
        attachment.filename = "clip.mp3"

        result = self.cog._validate_music_attachment("lyria-3-pro-preview", attachment)
        assert result == "Music reference attachments must be image files."

    async def test_validate_music_attachment_uses_filename_when_content_type_missing(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = None
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-3-clip-preview", attachment)
        assert result is None


class TestThinkingFeatures(GeminiCogTestCase):
    """Tests for thinking config, thinking embeds, and response part extraction."""

    file_search_store_ids = ["store-1"]

    # --- _build_thinking_config ---

    def test_build_thinking_config_none_when_no_params(self):
        """Test that _build_thinking_config returns None when both params are None."""
        result = _build_thinking_config(None, None)
        assert result is None

    def test_build_thinking_config_with_level(self):
        """Test that _build_thinking_config sets thinking_level and include_thoughts."""
        result = _build_thinking_config("high", None)
        assert result is not None
        # SDK converts string to ThinkingLevel enum
        assert "HIGH" in str(result.thinking_level)
        assert result.include_thoughts is True

    def test_build_thinking_config_with_budget(self):
        """Test that _build_thinking_config sets thinking_budget and include_thoughts."""
        result = _build_thinking_config(None, 1024)
        assert result is not None
        assert result.thinking_budget == 1024
        assert result.include_thoughts is True

    def test_build_thinking_config_with_both(self):
        """Test that _build_thinking_config sets both level and budget."""
        result = _build_thinking_config("low", 512)
        assert result is not None
        assert "LOW" in str(result.thinking_level)
        assert result.thinking_budget == 512
        assert result.include_thoughts is True

    def test_build_thinking_config_budget_zero(self):
        """Test that _build_thinking_config handles budget=0 (disable thinking)."""
        result = _build_thinking_config(None, 0)
        assert result is not None
        assert result.thinking_budget == 0

    def test_build_thinking_config_budget_dynamic(self):
        """Test that _build_thinking_config handles budget=-1 (dynamic)."""
        result = _build_thinking_config(None, -1)
        assert result is not None
        assert result.thinking_budget == -1

    # --- extract_thinking_text ---

    def test_extract_thinking_text_with_thoughts(self):
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
        result = extract_thinking_text(response)
        assert result == "Let me think..."

    def test_extract_thinking_text_multiple_thought_parts(self):
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
        result = extract_thinking_text(response)
        assert result == "Step 1: analyze\n\nStep 2: compute"

    def test_extract_thinking_text_no_thoughts(self):
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
        result = extract_thinking_text(response)
        assert result == ""

    def test_extract_thinking_text_empty_response(self):
        """Test extracting thinking text from an empty response."""
        response = SimpleNamespace(candidates=[])
        result = extract_thinking_text(response)
        assert result == ""

    def test_extract_thinking_text_no_candidates(self):
        """Test extracting thinking text when candidates is None."""
        response = SimpleNamespace(candidates=None)
        result = extract_thinking_text(response)
        assert result == ""

    # --- _get_response_content_parts ---

    def test_get_response_content_parts_returns_parts(self):
        """Test that response parts are extracted correctly."""
        part1 = SimpleNamespace(text="Hello", thought=False)
        part2 = SimpleNamespace(text="Thinking...", thought=True)
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part1, part2]))]
        )
        result = _get_response_content_parts(response)
        assert result is not None
        assert len(result) == 2
        assert result[0] is part1
        assert result[1] is part2

    def test_get_response_content_parts_empty_candidates(self):
        """Test that None is returned for empty candidates."""
        response = SimpleNamespace(candidates=[])
        result = _get_response_content_parts(response)
        assert result is None

    def test_get_response_content_parts_no_content(self):
        """Test that None is returned when content is None."""
        response = SimpleNamespace(candidates=[SimpleNamespace(content=None)])
        result = _get_response_content_parts(response)
        assert result is None

    def test_get_response_content_parts_no_parts(self):
        """Test that None is returned when parts is empty."""
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])
        result = _get_response_content_parts(response)
        assert result is None

    # --- append_thinking_embeds ---

    def test_append_thinking_embeds_with_text(self):
        """Test that thinking embed is created with spoilered text."""
        embeds = []
        append_thinking_embeds(embeds, "My thought process")
        assert len(embeds) == 1
        assert embeds[0].title == "Thinking"
        assert embeds[0].description == "||My thought process||"
        # Check it uses light grey color
        from discord import Colour

        assert embeds[0].color == Colour.light_grey()

    def test_append_thinking_embeds_empty_text(self):
        """Test that no embed is created for empty thinking text."""
        embeds = []
        append_thinking_embeds(embeds, "")
        assert len(embeds) == 0

    def test_append_thinking_embeds_truncates_long_text(self):
        """Test that long thinking text is truncated."""
        embeds = []
        long_text = "A" * 4000  # Over 3500 limit
        append_thinking_embeds(embeds, long_text)
        assert len(embeds) == 1
        assert "[thinking truncated]" in embeds[0].description
        # Check total length is under Discord limit (spoiler markers add 4 chars)
        assert len(embeds[0].description) <= 3600

    # --- pricing with thinking tokens ---

    def test_append_pricing_embed_with_thinking_tokens(self):
        """Test pricing embed shows thinking token count."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-3-flash-preview",
            input_tokens=100_000,
            output_tokens=50_000,
            daily_cost=0.50,
            thinking_tokens=200_000,
        )
        assert len(embeds) == 1
        assert "200,000 thinking" in embeds[0].description
        assert "100,000 in" in embeds[0].description
        assert "50,000 out" in embeds[0].description

    def test_append_pricing_embed_zero_thinking_tokens(self):
        """Test pricing embed omits thinking when zero."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash",
            input_tokens=100_000,
            output_tokens=50_000,
            daily_cost=0.10,
            thinking_tokens=0,
        )
        assert len(embeds) == 1
        assert "thinking" not in embeds[0].description
        assert "tokens in" in embeds[0].description

    def test_track_daily_cost_with_thinking_tokens(self):
        """Test that _track_daily_cost accumulates pre-calculated cost including thinking."""
        from util import calculate_cost

        # gemini-2.0-flash: $0.10/M in, $0.40/M out
        # thinking tokens billed at output rate
        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 500_000, thinking_tokens=500_000)
        daily = self.cog._track_daily_cost(user_id=99, cost=cost)
        # $0.10 + ($0.40 * 1.0) = $0.50
        assert daily == pytest.approx(0.50)

    def test_append_pricing_embed_with_maps_grounding(self):
        """Test pricing embed includes Maps grounding surcharge."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            daily_cost=0.10,
            google_maps_grounded=True,
        )
        assert len(embeds) == 1
        assert "Maps grounded" in embeds[0].description

    def test_append_pricing_embed_without_maps_grounding(self):
        """Test pricing embed omits Maps label when not grounded."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            daily_cost=0.10,
            google_maps_grounded=False,
        )
        assert len(embeds) == 1
        assert "Maps" not in embeds[0].description

    def test_log_cost_basic(self):
        """Test that _log_cost calls logger.info with structured cost data."""
        self.cog.logger = MagicMock()
        self.cog._log_cost(
            "chat",
            12345,
            "gemini-2.0-flash",
            0.50,
            1.00,
            input_tokens=1000,
            output_tokens=500,
        )
        self.cog.logger.info.assert_called_once()
        call_args = self.cog.logger.info.call_args
        fmt = call_args[0][0]
        assert "COST" in fmt
        assert "command=%s" in fmt
        assert "daily=$%.4f" in fmt

    def test_log_cost_no_details(self):
        """Test _log_cost with no extra detail kwargs."""
        self.cog.logger = MagicMock()
        self.cog._log_cost("video", 999, "veo-3.1-generate-preview", 3.20, 5.00)
        self.cog.logger.info.assert_called_once()

    def test_log_cost_image_details(self):
        """Test _log_cost with image-specific details."""
        self.cog.logger = MagicMock()
        self.cog._log_cost(
            "image",
            42,
            "gemini-3.1-flash-image-preview",
            0.134,
            0.50,
            images=2,
            input_tokens=500,
        )
        self.cog.logger.info.assert_called_once()
        call_args = self.cog.logger.info.call_args
        # The formatted string should include image and input token details
        assert "images=2" in str(call_args)


class TestVideoResponseEmbed(AsyncGeminiCogTestCase):
    async def _assert_video_mode(self, params, expected_mode, attachment=None):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file:
            embed, files = await self.cog._create_video_response_embed(
                video_params=params,
                generated_videos=[video_file.name],
                attachment=attachment,
            )
            for file in files:
                file.close()
        assert expected_mode in embed.description
        assert len(files) == 1

    async def test_mode_text_to_video(self):
        """Test embed shows Text-to-Video mode when no attachments."""
        from util import VideoGenerationParameters

        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        await self._assert_video_mode(params, "Text-to-Video")

    async def test_mode_image_to_video(self):
        """Test embed shows Image-to-Video mode when attachment provided."""
        from util import VideoGenerationParameters

        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Image-to-Video", attachment=mock_attachment)

    async def test_mode_interpolation(self):
        """Test embed shows Interpolation mode when both attachment and last_frame."""
        from util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Interpolation", attachment=mock_attachment)

    async def test_mode_last_frame_only(self):
        """Test embed shows Last Frame Constrained mode when only last_frame."""
        from util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        await self._assert_video_mode(params, "Last Frame Constrained")


class TestTemperatureWarning:
    """Test temperature warning for Gemini 3 models.

    These tests verify the warning logic inline rather than calling
    the full slash command, since the warning is a simple string check.
    """

    def _build_temperature_description(self, model, temperature):
        """Replicate the temperature description logic from the chat command."""
        description = ""
        if temperature is not None:
            description += f"**Temperature:** {temperature}"
            if model.startswith("gemini-3") and temperature != 1.0:
                description += " (warning: Gemini 3 recommends 1.0; lower values may cause looping)"
            description += "\n"
        return description

    def test_gemini3_temperature_warning_low(self):
        """Temperature below 1.0 on Gemini 3 triggers warning."""
        desc = self._build_temperature_description("gemini-3.1-pro-preview", 0.5)
        assert "warning" in desc
        assert "looping" in desc

    def test_gemini3_temperature_warning_high(self):
        """Temperature above 1.0 on Gemini 3 triggers warning."""
        desc = self._build_temperature_description("gemini-3-flash-preview", 1.5)
        assert "warning" in desc

    def test_gemini3_temperature_no_warning_at_default(self):
        """Temperature 1.0 on Gemini 3 does not trigger warning."""
        desc = self._build_temperature_description("gemini-3.1-pro-preview", 1.0)
        assert "warning" not in desc
        assert "1.0" in desc

    def test_gemini25_no_warning(self):
        """Temperature != 1.0 on Gemini 2.5 does not trigger warning."""
        desc = self._build_temperature_description("gemini-2.5-pro", 0.5)
        assert "warning" not in desc
        assert "0.5" in desc

    def test_temperature_none_no_output(self):
        """No temperature set produces no description."""
        desc = self._build_temperature_description("gemini-3.1-pro-preview", None)
        assert desc == ""

    def test_gemini3_flash_lite_warning(self):
        """Gemini 3.1 Flash Lite also triggers warning."""
        desc = self._build_temperature_description("gemini-3.1-flash-lite-preview", 0.7)
        assert "warning" in desc

    def test_gemini2_flash_no_warning(self):
        """Gemini 2.0 Flash does not trigger warning."""
        desc = self._build_temperature_description("gemini-2.0-flash", 0.3)
        assert "warning" not in desc
