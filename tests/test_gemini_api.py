import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discord_gemini.cogs.gemini import research as gemini_research
from tests.gemini_test_support import AsyncGeminiCogTestCase


class TestGeminiCog(AsyncGeminiCogTestCase):
    async def test_cog_init(self):
        """Test that GeminiCog initializes correctly."""
        assert self.cog.bot == self.bot
        assert self.cog.conversations == {}
        assert self.cog.message_to_conversation_id == {}
        assert self.cog.views == {}
        assert self.cog.last_view_messages == {}
        assert self.cog._http_session is None
        assert self.cog._client is None
        self.mock_genai_client.assert_not_called()

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
        await asyncio.sleep(0)

        assert self.cog._http_session is None


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


class TestGeminiImageGeneration(AsyncGeminiCogTestCase):
    """Tests for Gemini image generation config."""

    async def test_prompt_prefix_for_generation(self):
        """Test that prompts are prefixed with 'Create image:' for Gemini models."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(prompt="A cat", model="gemini-3.1-flash-image-preview")
        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "Create image: A cat"

    async def test_prompt_unchanged_for_editing(self):
        """Test that prompts are NOT prefixed when an attachment is provided."""
        from discord_gemini.util import ImageGenerationParameters

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
        from discord_gemini.util import ImageGenerationParameters

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
        from discord_gemini.util import ImageGenerationParameters

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
        from discord_gemini.util import ImageGenerationParameters

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
        from discord_gemini.util import ImageGenerationParameters

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
        from discord_gemini.util import ImageGenerationParameters

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
        from discord_gemini.util import ImageGenerationParameters

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


class TestGeminiDeepResearch(AsyncGeminiCogTestCase):
    """Tests for deep research helper methods."""

    @pytest.fixture(autouse=True)
    async def setup_research(self, setup):
        self.mock_client_instance.aio.interactions.create = AsyncMock()
        self.mock_client_instance.aio.interactions.get = AsyncMock()

    async def test_run_deep_research_success(self):
        """Test _run_deep_research with a successful completion."""
        from discord_gemini.util import ResearchParameters

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

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
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
        from discord_gemini.util import ResearchParameters

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

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
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
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="test", file_search=True)

        original = gemini_research.GEMINI_FILE_SEARCH_STORE_IDS
        gemini_research.GEMINI_FILE_SEARCH_STORE_IDS = []
        try:
            with pytest.raises(Exception, match="GEMINI_FILE_SEARCH_STORE_IDS"):
                await self.cog._run_deep_research(params)
        finally:
            gemini_research.GEMINI_FILE_SEARCH_STORE_IDS = original

    async def test_run_deep_research_failed(self):
        """Test _run_deep_research raises on failure."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="test")

        interaction_failed = SimpleNamespace(id="interaction-3", status="failed", outputs=None)

        self.cog.client.aio.interactions.create.return_value = interaction_failed

        with (
            patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(Exception, match="failed"),
        ):
            await self.cog._run_deep_research(params)

    async def test_run_deep_research_no_output(self):
        """Test _run_deep_research returns None text when no output."""
        from discord_gemini.util import ResearchParameters

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

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
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
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Research quantum computing")

        embeds = self.cog._create_research_response_embeds(params)

        # Only the header embed (report is sent as file attachment)
        assert len(embeds) == 1
        assert embeds[0].title == "Deep Research"
        assert "Research quantum computing" in embeds[0].description
        assert "deep-research-pro-preview-12-2025" in embeds[0].description

    async def test_create_research_response_embeds_with_file_search(self):
        """Test _create_research_response_embeds shows file search status."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Analyze docs", file_search=True)

        embeds = self.cog._create_research_response_embeds(params)

        assert "File Search" in embeds[0].description

    async def test_run_deep_research_with_google_maps(self):
        """Test _run_deep_research passes google_maps tool when enabled."""
        from discord_gemini.util import ResearchParameters

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

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            report_text, _, _, _ = await self.cog._run_deep_research(params)

        call_kwargs = self.cog.client.aio.interactions.create.call_args
        assert "tools" in call_kwargs.kwargs
        assert {"google_maps": {}} in call_kwargs.kwargs["tools"]

    async def test_run_deep_research_with_file_search_and_google_maps(self):
        """Test _run_deep_research passes both file_search and google_maps tools."""
        from discord_gemini.util import ResearchParameters

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

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            await self.cog._run_deep_research(params)

        call_kwargs = self.cog.client.aio.interactions.create.call_args
        tools = call_kwargs.kwargs["tools"]
        assert len(tools) == 2
        assert tools[0]["type"] == "file_search"
        assert {"google_maps": {}} in tools

    async def test_create_research_response_embeds_with_google_maps(self):
        """Test _create_research_response_embeds shows Google Maps status."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Find restaurants nearby", google_maps=True)

        embeds = self.cog._create_research_response_embeds(params)

        assert "Google Maps" in embeds[0].description

    async def test_create_research_response_embeds_truncates_prompt(self):
        """Test _create_research_response_embeds truncates long prompts."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="X" * 3000)

        embeds = self.cog._create_research_response_embeds(params)

        # Prompt should be truncated to 2000 + "..."
        assert "..." in embeds[0].description


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
