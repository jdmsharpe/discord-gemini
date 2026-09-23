import asyncio
import inspect
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discord_gemini.cogs.gemini import research as gemini_research
from discord_gemini.cogs.gemini.cog import GeminiCog
from discord_gemini.cogs.gemini.command_options import (
    CHAT_MODEL_CHOICES,
    IMAGE_MODEL_CHOICES,
    MUSIC_MODEL_CHOICES,
    PERSON_GENERATION_CHOICES,
    RESEARCH_THINKING_SUMMARY_CHOICES,
    THINKING_LEVEL_CHOICES,
    TTS_MODEL_CHOICES,
    TTS_VOICE_CHOICES,
    VIDEO_MODEL_CHOICES,
    VIDEO_RESOLUTION_CHOICES,
)
from tests.support import AsyncGeminiCogTestCase, build_mock_bot


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


def test_cog_init_does_not_configure_root_logger():
    """GeminiCog initialization should not mutate global logging config."""
    with patch.object(logging, "basicConfig") as mock_basic_config:
        GeminiCog(bot=build_mock_bot())

    mock_basic_config.assert_not_called()


def test_default_chat_model_is_first_choice():
    """gemini-3.8-flash is the default chat model and is shown first in the picker."""
    assert CHAT_MODEL_CHOICES[0].value == "gemini-3.8-flash"


def test_chat_command_default_model_param():
    """The /gemini chat `model` parameter defaults to gemini-3.8-flash."""
    signature = inspect.signature(GeminiCog.chat.callback)
    assert signature.parameters["model"].default == "gemini-3.8-flash"


class TestThinkingValidation:
    """Thinking configurations the API rejects must fail before the request."""

    @staticmethod
    def _validate(model, level=None, budget=None):
        from discord_gemini.cogs.gemini.responses import _validate_thinking_request

        return _validate_thinking_request(model, level, budget)

    def test_accepts_a_level_alone(self):
        assert self._validate("gemini-3.7-flash", level="low") is None

    def test_accepts_a_budget_alone(self):
        assert self._validate("gemini-3.7-flash", budget=1024) is None

    def test_accepts_nothing_set(self):
        assert self._validate("gemini-3.7-flash") is None

    def test_rejects_level_and_budget_together(self):
        """`400 You can only set only one of thinking budget and thinking level`."""
        error = self._validate("gemini-3.7-flash", level="low", budget=1024)
        assert error is not None
        assert "cannot be combined" in error

    def test_level_and_budget_rejected_on_every_model(self):
        for model in ("gemini-3.6-flash", "gemini-2.5-flash", "gemini-3.1-pro-preview"):
            assert self._validate(model, level="high", budget=512) is not None

    def test_rejects_minimal_on_models_that_do_not_support_it(self):
        """3.8 Flash, 3.7 Flash and 3.1 Pro 400 on MINIMAL while the menu offers it."""
        for model in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.1-pro-preview"):
            error = self._validate(model, level="minimal")
            assert error is not None, model
            assert "Minimal" in error

    def test_allows_minimal_where_it_is_supported(self):
        assert self._validate("gemini-3.6-flash", level="minimal") is None

    def test_rejects_every_level_on_gemini_2_5(self):
        """Gemini 2.5 400s on ANY level: 'Thinking level is not supported for this model'."""
        for model, level in (
            ("gemini-2.5-pro", "low"),
            ("gemini-2.5-flash", "minimal"),
            ("gemini-2.5-flash-lite", "high"),
        ):
            error = self._validate(model, level=level)
            assert error is not None, (model, level)
            assert "thinking_budget" in error

    def test_accepts_a_budget_on_gemini_2_5(self):
        """Budgets are the 2.5 control; the level guard must not swallow them."""
        assert self._validate("gemini-2.5-flash", budget=2048) is None
        assert self._validate("gemini-2.5-pro", budget=2048) is None

    def test_thinking_level_unsupported_set_is_exactly_the_2_5_family(self):
        from discord_gemini.util import THINKING_LEVEL_UNSUPPORTED_MODELS

        assert {
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        } == THINKING_LEVEL_UNSUPPORTED_MODELS
        # Every 2.5 model in the picker is gated, and no Gemini 3 model is.
        for choice in CHAT_MODEL_CHOICES:
            gated = choice.value in THINKING_LEVEL_UNSUPPORTED_MODELS
            assert gated == choice.value.startswith("gemini-2.5"), choice.value

    def test_default_chat_model_rejects_minimal(self):
        """The default is the model users hit without choosing, so pin it directly."""
        from discord_gemini.util import MINIMAL_THINKING_UNSUPPORTED_MODELS

        signature = inspect.signature(GeminiCog.chat.callback)
        assert signature.parameters["model"].default in MINIMAL_THINKING_UNSUPPORTED_MODELS

    def test_minimal_is_still_offered_in_the_menu(self):
        """If Minimal ever leaves the menu, this guard becomes dead code."""
        assert any(choice.value == "minimal" for choice in THINKING_LEVEL_CHOICES)


class TestImageSizeValidation:
    """Per-model `image_size` support, from live requests probed 2026-08-28.

    `gemini-3.1-flash-lite-image` generates 1K only; 512 and 4K are Flash Image,
    4K also Pro; 512 on Pro is a 400. Gemini 2.5 silently renders 1K for a 4K
    request, so the bot refuses sizes that would be rejected or ignored.
    """

    @staticmethod
    def _params(model="gemini-3.1-flash-lite-image", **kwargs):
        from discord_gemini.util import ImageGenerationParameters

        return ImageGenerationParameters(prompt="x", model=model, **kwargs)

    def test_accepts_lite_without_image_size(self):
        from discord_gemini.cogs.gemini.image import _validate_image_size_request

        assert _validate_image_size_request(self._params()) is None

    def test_accepts_lite_at_1k_either_case(self):
        from discord_gemini.cogs.gemini.image import _validate_image_size_request

        assert _validate_image_size_request(self._params(image_size="1k")) is None
        assert _validate_image_size_request(self._params(image_size="1K")) is None

    def test_rejects_lite_at_2k(self):
        from discord_gemini.cogs.gemini.image import _validate_image_size_request

        error = _validate_image_size_request(self._params(image_size="2K"))
        assert error is not None
        assert error.startswith("Image size 2K is not supported for this model")
        assert "1K" in error

    @pytest.mark.parametrize(
        ("model", "size"),
        [
            ("gemini-3.1-flash-lite-image", "4K"),
            ("gemini-3.1-flash-lite-image", "512"),
            ("gemini-3-pro-image", "512"),
        ],
    )
    def test_rejects_sizes_the_api_400s_on_verbatim(self, model, size):
        from discord_gemini.cogs.gemini.image import _validate_image_size_request

        error = _validate_image_size_request(self._params(model=model, image_size=size))
        assert error is not None
        assert error.startswith(f"Image size {size} is not supported for this model")

    @pytest.mark.parametrize(
        ("model", "size"),
        [
            ("gemini-3.1-flash-image", "512"),
            ("gemini-3.1-flash-image", "4K"),
            ("gemini-3-pro-image", "4K"),
            ("gemini-3-pro-image", "2K"),
        ],
    )
    def test_accepts_sizes_the_api_rendered(self, model, size):
        from discord_gemini.cogs.gemini.image import _validate_image_size_request

        assert _validate_image_size_request(self._params(model=model, image_size=size)) is None

    def test_supported_sizes_table_matches_the_probes(self):
        from discord_gemini.cogs.gemini.image import IMAGE_SUPPORTED_SIZES

        assert {
            "gemini-2.5-flash-image": {"1k"},
            "gemini-3.1-flash-image": {"512", "1k", "2k", "4k"},
            "gemini-3.1-flash-lite-image": {"1k"},
            "gemini-3-pro-image": {"1k", "2k", "4k"},
        } == IMAGE_SUPPORTED_SIZES

    def test_gemini_2_5_only_accepts_its_fixed_1k_size(self):
        from discord_gemini.cogs.gemini.image import _validate_image_size_request

        params = self._params(model="gemini-2.5-flash-image", image_size="1K")
        assert _validate_image_size_request(params) is None
        for size in ("512", "2K", "4K"):
            params = self._params(model="gemini-2.5-flash-image", image_size=size)
            error = _validate_image_size_request(params)
            assert error is not None
            assert "supports 1K" in error

    def test_does_not_constrain_unknown_future_image_models(self):
        from discord_gemini.cogs.gemini.image import _validate_image_size_request

        params = self._params(model="gemini-future-image", image_size="4K")
        assert _validate_image_size_request(params) is None

    def test_choice_values_are_the_canonical_uppercase_spelling(self):
        """Lowercase `2k` was metered at the 1K tier (1,120 vs 1,680 IMAGE tokens,
        probe 2026-08-28); the documented spelling is uppercase, so send that."""
        from discord_gemini.cogs.gemini.command_options import IMAGE_SIZE_CHOICES

        assert [choice.value for choice in IMAGE_SIZE_CHOICES] == ["512", "1K", "2K", "4K"]


def test_critical_choice_values_present():
    assert any(choice.value == "gemini-3.6-flash" for choice in CHAT_MODEL_CHOICES)
    assert any(choice.value == "gemini-3.5-flash" for choice in CHAT_MODEL_CHOICES)
    assert any(choice.value == "gemini-3.5-flash-lite" for choice in CHAT_MODEL_CHOICES)
    assert any(choice.value == "gemini-3.1-pro-preview" for choice in CHAT_MODEL_CHOICES)
    assert any(choice.value == "gemini-3.1-flash-image" for choice in IMAGE_MODEL_CHOICES)
    assert any(choice.value == "gemini-3.1-flash-lite-image" for choice in IMAGE_MODEL_CHOICES)
    assert any(choice.value == "veo-3.1-lite-generate-preview" for choice in VIDEO_MODEL_CHOICES)
    assert any(choice.value == "veo-3.1-generate-preview" for choice in VIDEO_MODEL_CHOICES)
    assert any(choice.value == "gemini-2.5-flash-preview-tts" for choice in TTS_MODEL_CHOICES)
    assert any(choice.value == "Kore" for choice in TTS_VOICE_CHOICES)
    assert any(choice.value == "lyria-3-clip-preview" for choice in MUSIC_MODEL_CHOICES)


def _serialize_command_group_payload(group):
    return {
        "name": group.name,
        "description": group.description,
        "options": [
            {
                "name": command.name,
                "description": command.description,
                "options": [
                    option.to_dict() for option in command.options if option.input_type is not None
                ],
                "type": 1,
                "nsfw": False,
            }
            for command in group.subcommands
        ],
        "nsfw": False,
    }


def test_registered_command_groups_fit_discord_size_limit():
    """Discord rejects any single top-level command payload over 8000 bytes."""

    cog = GeminiCog(bot=build_mock_bot())

    commands_by_name = {command.name: command for command in cog.get_commands()}

    assert set(commands_by_name) == {"gemini", "gemini-media", "gemini-tools"}
    assert [command.name for command in commands_by_name["gemini"].subcommands] == [
        "check_permissions",
        "chat",
    ]
    assert [command.name for command in commands_by_name["gemini-media"].subcommands] == [
        "image",
        "video",
    ]
    assert [command.name for command in commands_by_name["gemini-tools"].subcommands] == [
        "tts",
        "music",
        "research",
    ]

    payload_sizes = {
        name: len(
            json.dumps(
                _serialize_command_group_payload(command),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for name, command in commands_by_name.items()
    }

    assert payload_sizes["gemini"] < 8000
    assert payload_sizes["gemini-media"] < 8000
    assert payload_sizes["gemini-tools"] < 8000


def test_image_command_offers_no_image_count_option():
    """`candidate_count=2` returns 400 "Multiple candidates is not enabled for this
    model" on every model in IMAGE_MODEL_CHOICES, so `/gemini-media image` has no
    option that requests more than one image."""
    cog = GeminiCog(bot=build_mock_bot())
    media = next(command for command in cog.get_commands() if command.name == "gemini-media")
    image = next(command for command in media.subcommands if command.name == "image")

    registered = [option.name for option in image.options if option.input_type is not None]
    assert registered == [
        "prompt",
        "model",
        "aspect_ratio",
        "attachment",
        "seed",
        "image_size",
        "google_image_search",
    ]


def test_thinking_level_choice_set():
    values = {choice.value for choice in THINKING_LEVEL_CHOICES}
    assert values == {"minimal", "low", "medium", "high"}


def test_person_generation_choice_set():
    values = {choice.value for choice in PERSON_GENERATION_CHOICES}
    assert values == {"dont_allow", "allow_adult", "allow_all"}


def test_research_thinking_summary_choice_set():
    values = {choice.value for choice in RESEARCH_THINKING_SUMMARY_CHOICES}
    assert values == {"auto", "none"}


def test_video_resolution_choice_set():
    values = {choice.value for choice in VIDEO_RESOLUTION_CHOICES}
    assert values == {"720p", "1080p", "4k"}


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
        embed_description += "**Model:** gemini-3-pro-image\n"
        embed_description += "**Mode:** Image Generation\n"
        embed_description += "**Number of Images:** 1\n"

        # Discord's limit is 4096 characters
        assert len(embed_description) <= 4096


class TestGeminiImageGeneration(AsyncGeminiCogTestCase):
    """Tests for Gemini image generation config."""

    async def test_prompt_prefix_for_generation(self):
        """Test that prompts are prefixed with 'Create image:' for Gemini models."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(prompt="A cat", model="gemini-3.1-flash-image")
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
            model="gemini-3.1-flash-image",
        )
        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=None)

        await self.cog._generate_image_with_gemini(params, attachment=MagicMock())

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "Edit this cat"

    async def test_generate_image_with_gemini_default_config(self):
        """The command default (aspect_ratio 1:1, no size) sends ONLY the aspect ratio.

        The ratio is always on the wire, 1:1 included: omitted, the model picked its
        own 11:6 (704x384 for a 512 request) while an explicit 1:1 returned 512x512
        (probe 2026-08-28), so the advertised `(default: 1:1)` was not real before.
        """
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.response_modalities == ["TEXT", "IMAGE"]
        assert config.image_config is not None
        assert config.image_config.aspect_ratio == "1:1"
        assert config.image_config.image_size is None
        assert config.tools is None

    async def test_generate_image_with_gemini_sends_seed_and_one_candidate(self):
        """A set seed always reaches the config, and no request asks for more than one
        candidate: every image model returns 400 "Multiple candidates is not enabled for
        this model" for `candidate_count=2`."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(prompt="A cat", model="gemini-3.1-flash-image", seed=7)

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        config = self.cog.client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.seed == 7
        assert config.candidate_count is None

    async def test_generate_image_with_gemini_image_size(self):
        """Test that image_size is passed via image_config, in the canonical spelling."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image",
            image_size="2K",
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.image_config is not None
        assert config.image_config.image_size == "2K"

    async def test_generate_image_with_gemini_aspect_ratio(self):
        """Test that non-default aspect_ratio is passed via image_config."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image",
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

    async def test_generate_image_with_gemini_sends_default_aspect_ratio_when_unset(self):
        """A caller that leaves aspect_ratio unset still sends the 1:1 default on the wire."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image",
        )

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.image_config is not None
        assert config.image_config.aspect_ratio == "1:1"

    async def test_generate_image_with_gemini_image_search(self):
        """Test that google_image_search adds tools with search_types."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A cat",
            model="gemini-3.1-flash-image",
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
            model="gemini-3-pro-image",
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
            model="gemini-3.1-flash-image",
            aspect_ratio="9:16",
            image_size="2K",
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
        assert config.image_config.image_size == "2K"
        # tools should have google_search with image_search
        assert len(config.tools) == 1
        assert config.tools[0].google_search.search_types.image_search is not None

    async def test_generate_image_returns_the_api_bytes_and_mime(self):
        """Image parts come back as the API's own bytes + MIME, not decoded PIL images."""
        from discord_gemini.cogs.gemini.image import GeneratedImage
        from discord_gemini.util import ImageGenerationParameters

        jpeg = _encoded_image("JPEG")
        part = SimpleNamespace(
            text=None,
            inline_data=SimpleNamespace(data=jpeg, mime_type="image/jpeg; charset=binary"),
        )
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
            usage_metadata=SimpleNamespace(prompt_token_count=11),
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)

        text, images, input_tokens, search_queries = await self.cog._generate_image_with_gemini(
            ImageGenerationParameters(prompt="A cat", model="gemini-3.1-flash-image"),
            attachment=None,
        )

        assert text is None
        assert images == [GeneratedImage(jpeg, "image/jpeg")]
        assert input_tokens == 11
        assert search_queries == 0

    async def test_generate_image_bills_tool_use_prompt_tokens_as_input(self):
        """`tool_use_prompt_token_count` is added to the input count, as in chat. A
        gemini-3.1-flash-image request with the Google Search tool reported 18 prompt
        tokens and no tool-use count (2026-09-22), so that shape bills 18."""
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="A red panda", model="gemini-3.1-flash-image", google_image_search=True
        )

        async def input_tokens_for(usage_metadata):
            response = SimpleNamespace(candidates=[], usage_metadata=usage_metadata)
            self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)
            _, _, input_tokens, _ = await self.cog._generate_image_with_gemini(
                params, attachment=None
            )
            return input_tokens

        live = SimpleNamespace(prompt_token_count=18, tool_use_prompt_token_count=None)
        with_tool_use = SimpleNamespace(prompt_token_count=18, tool_use_prompt_token_count=40)
        assert await input_tokens_for(live) == 18
        assert await input_tokens_for(with_tool_use) == 58

    async def test_image_command_bills_and_labels_search_queries(self):
        """Google Image Search grounding bills each web or image search query the
        response reports at the Gemini 3 rate ($0.014), on top of the image price."""
        part = SimpleNamespace(
            text=None,
            inline_data=SimpleNamespace(data=_encoded_image("PNG"), mime_type="image/png"),
        )
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[part]),
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=["red panda habitat"],
                        image_search_queries=["red panda", "red panda eating bamboo"],
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=0),
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()

        with (
            patch("discord_gemini.cogs.gemini.image.SHOW_COST_EMBEDS", True),
            patch.object(self.cog, "_log_cost") as log_cost,
        ):
            await self.cog.image.callback(
                self.cog,
                ctx=ctx,
                prompt="A red panda",
                image_size="1K",
                google_image_search=True,
            )

        send_kwargs = ctx.send_followup.call_args.kwargs
        for file in send_kwargs.get("files", []):
            file.close()
        # One 1K image @ $0.067 + 3 search queries @ $0.014
        assert log_cost.call_args.args[3] == pytest.approx(0.109)
        assert log_cost.call_args.kwargs["google_search_queries"] == 3
        footer = send_kwargs["embeds"][-1].description
        assert footer.startswith("$0.1090 · 1 image · ")
        assert "3 search queries" in footer


def _encoded_image(fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(buf, format=fmt)
    return buf.getvalue()


class TestGeminiImageDelivery(AsyncGeminiCogTestCase):
    """Attach the API's original PNG/JPEG bytes; re-encode anything else to PNG.

    Re-encoding the probe's 4K JPEGs to PNG produced 10.06-12.95 MB files, over
    Discord's 10 MB bot upload cap, while the originals were 5.5-7.1 MB (2026-08-28).
    """

    async def _deliver(self, image):
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(
            prompt="x", model="gemini-3.1-flash-image", image_size="4K"
        )
        return await self.cog._create_image_response_embed(
            image_params=params, generated_images=[image], attachment=None
        )

    @pytest.mark.parametrize(
        ("fmt", "mime", "extension"),
        [("JPEG", "image/jpeg", "jpg"), ("PNG", "image/png", "png")],
    )
    async def test_discord_native_formats_are_attached_unchanged(self, fmt, mime, extension):
        from discord_gemini.cogs.gemini.image import GeneratedImage

        data = _encoded_image(fmt)
        embed, files = await self._deliver(GeneratedImage(data, mime))
        try:
            assert [file.filename for file in files] == [f"generated_image_1.{extension}"]
            assert files[0].fp.read() == data
            assert embed.image.url == f"attachment://generated_image_1.{extension}"
        finally:
            for file in files:
                file.close()

    async def test_other_formats_are_reencoded_to_png(self):
        from PIL import Image as PILImage

        from discord_gemini.cogs.gemini.image import GeneratedImage

        data = _encoded_image("WEBP")
        embed, files = await self._deliver(GeneratedImage(data, "image/webp"))
        try:
            assert [file.filename for file in files] == ["generated_image_1.png"]
            with PILImage.open(files[0].fp) as decoded:
                assert decoded.format == "PNG"
                assert decoded.size == (8, 8)
            assert embed.image.url == "attachment://generated_image_1.png"
        finally:
            for file in files:
                file.close()

    async def test_undecodable_bytes_are_skipped_not_fatal(self):
        from discord_gemini.cogs.gemini.image import GeneratedImage

        embed, files = await self._deliver(GeneratedImage(b"not an image", "image/webp"))
        assert files == []
        assert "image" not in embed.to_dict()


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
            total_input_tokens=250_000,
            total_output_tokens=60_000,
            total_thought_tokens=5_000,
            total_cached_tokens=40_000,
        )
        # First call returns in_progress, second returns completed
        interaction_started = SimpleNamespace(
            id="interaction-1", status="in_progress", steps=None, usage=None
        )
        interaction_done = SimpleNamespace(
            id="interaction-1",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(
                            type="text", text="# AI Safety Report\n\nDetailed findings..."
                        )
                    ],
                )
            ],
            usage=mock_usage,
        )

        self.cog.client.aio.interactions.create.return_value = interaction_started
        self.cog.client.aio.interactions.get.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        assert result.report_text == "# AI Safety Report\n\nDetailed findings..."
        assert result.input_tokens == 250_000
        assert result.output_tokens == 60_000
        assert result.thinking_tokens == 5_000
        assert result.cached_tokens == 40_000
        self.cog.client.aio.interactions.create.assert_called_once_with(
            input="Research AI safety",
            agent="deep-research-preview-04-2026",
            background=True,
        )

    async def test_run_deep_research_with_thinking_summaries(self):
        """Test _run_deep_research passes agent_config when thinking summaries are requested."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Research AI safety", thinking_summaries="none")

        interaction_done = SimpleNamespace(
            id="interaction-thinking",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="Concise report")],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        assert result.report_text == "Concise report"
        self.cog.client.aio.interactions.create.assert_called_once_with(
            input="Research AI safety",
            agent="deep-research-preview-04-2026",
            agent_config={
                "type": "deep-research",
                "thinking_summaries": "none",
            },
            background=True,
        )

    async def test_run_deep_research_with_file_search(self):
        """Test _run_deep_research passes file_search tools when enabled."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Analyze report", file_search=True)

        interaction_done = SimpleNamespace(
            id="interaction-2",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="Report analysis...")],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            await self.cog._run_deep_research(params)

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

        interaction_failed = SimpleNamespace(id="interaction-3", status="failed", steps=None)

        self.cog.client.aio.interactions.create.return_value = interaction_failed

        with (
            patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(Exception, match="failed"),
        ):
            await self.cog._run_deep_research(params)

    async def test_run_deep_research_budget_exceeded(self):
        """Test _run_deep_research raises immediately on the budget_exceeded terminal status."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="test")

        interaction_budget = SimpleNamespace(
            id="interaction-budget", status="budget_exceeded", steps=None
        )

        self.cog.client.aio.interactions.create.return_value = interaction_budget

        with (
            patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(Exception, match="budget exceeded"),
        ):
            await self.cog._run_deep_research(params)

    async def test_run_deep_research_requires_action(self):
        """Test _run_deep_research raises a clear message when the agent requires confirmation."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="test", collaborative_planning=True)

        interaction_requires_action = SimpleNamespace(
            id="interaction-5",
            status="requires_action",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(type="text", text="1. Search the topic\n2. Review findings")
                    ],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=20, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_requires_action

        with pytest.raises(Exception, match="requires confirmation"):
            await self.cog._run_deep_research(params)

    async def test_run_deep_research_no_output(self):
        """Test _run_deep_research returns None text when no output."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="test")

        interaction_done = SimpleNamespace(
            id="interaction-4",
            status="completed",
            steps=[],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=0, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        assert result.report_text is None
        assert result.input_tokens == 100

    async def test_run_deep_research_aggregates_multi_step_output(self):
        """Test the report body is concatenated across multiple model_output steps.

        Deep research streams body and closing sources as separate steps in the
        v1beta `steps` timeline; returning only the last step would silently drop
        the body and leave just the trailing citations footer.
        """
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Research test")

        interaction_done = SimpleNamespace(
            id="multi-step",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="# Report Body\n\nFindings...\n")],
                ),
                SimpleNamespace(type="thought", content=[]),
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(type="text", text="**Sources:**\n1. [arxiv.org](url)\n")
                    ],
                ),
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        assert result.report_text is not None
        assert "# Report Body" in result.report_text
        assert "Findings..." in result.report_text
        assert "**Sources:**" in result.report_text
        assert "[arxiv.org]" in result.report_text

    async def test_run_deep_research_prefers_output_text(self):
        """Test the SDK's Interaction.output_text is used for the report when present.

        SDK 2.3.0+ exposes a fixed `output_text` that already aggregates the trailing
        model output; we prefer it over the manual steps walker while keeping the
        annotation collection (which still walks the steps) intact.
        """
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Research test")

        annotation = SimpleNamespace(
            type="url_citation", url="https://example.com", title="Example"
        )
        interaction_done = SimpleNamespace(
            id="output-text",
            status="completed",
            output_text="# SDK-aggregated report\n\nFull body.",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(
                            type="text",
                            text="ignored manual body",
                            annotations=[annotation],
                        )
                    ],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        assert result.report_text == "# SDK-aggregated report\n\nFull body."
        # Annotation collection still walks the steps regardless of output_text.
        assert result.annotations == [annotation]

    async def test_run_deep_research_falls_back_when_output_text_empty(self):
        """Test the manual steps walker is used when output_text is empty/missing."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Research test")

        interaction_done = SimpleNamespace(
            id="empty-output-text",
            status="completed",
            output_text="",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="manual fallback body")],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            result = await self.cog._run_deep_research(params)

        assert result.report_text == "manual fallback body"

    async def test_create_research_response_embeds(self):
        """Test _create_research_response_embeds creates header embed only."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Research quantum computing")

        embeds = self.cog._create_research_response_embeds(params, elapsed=83)

        # Only the header embed (report is sent as file attachment)
        assert len(embeds) == 1
        assert embeds[0].title == "Deep Research"
        assert "Research quantum computing" in embeds[0].description
        assert "deep-research-preview-04-2026" in embeds[0].description
        # OpenAI-style Tools + Time fields
        assert "**Tools:** google_search" in embeds[0].description
        assert "**Time:** 1m 23s" in embeds[0].description

    async def test_create_research_response_embeds_with_file_search(self):
        """Test _create_research_response_embeds shows file search status."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Analyze docs", file_search=True)

        embeds = self.cog._create_research_response_embeds(params)

        assert "file_search" in embeds[0].description

    async def test_run_deep_research_with_google_maps(self):
        """Test _run_deep_research passes google_maps tool when enabled."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Best restaurants in Tokyo", google_maps=True)

        interaction_done = SimpleNamespace(
            id="interaction-maps",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="Tokyo restaurant guide...")],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=200, total_output_tokens=100, total_thought_tokens=0
            ),
        )

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock):
            await self.cog._run_deep_research(params)

        call_kwargs = self.cog.client.aio.interactions.create.call_args
        # Interactions tools are keyed on `type`; google-genai rejects the
        # generate_content shape `{"google_maps": {}}` before any request is sent.
        assert call_kwargs.kwargs["tools"] == [{"type": "google_maps"}]

    async def test_run_deep_research_with_file_search_and_google_maps(self):
        """Test _run_deep_research passes both file_search and google_maps tools."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(
            prompt="Local docs about Tokyo", file_search=True, google_maps=True
        )

        interaction_done = SimpleNamespace(
            id="interaction-both",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="Combined report...")],
                )
            ],
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
        assert tools[1] == {"type": "google_maps"}
        # The same client-side validation interactions.create applies, run offline: it
        # raises "Tool: expected object with 'type' field" for `{"google_maps": {}}`.
        from google.genai import interactions
        from pydantic import TypeAdapter

        TypeAdapter(list[interactions.Tool]).validate_python(tools)

    async def test_create_research_response_embeds_with_google_maps(self):
        """Test _create_research_response_embeds shows Google Maps status."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="Find restaurants nearby", google_maps=True)

        embeds = self.cog._create_research_response_embeds(params)

        assert "google_maps" in embeds[0].description

    async def test_create_research_response_embeds_truncates_prompt(self):
        """Test _create_research_response_embeds truncates long prompts."""
        from discord_gemini.util import ResearchParameters

        params = ResearchParameters(prompt="X" * 3000)

        embeds = self.cog._create_research_response_embeds(params)

        # Prompt should be truncated to 2000 + "..."
        assert "..." in embeds[0].description


class TestResearchThinkingExtraction:
    """Tests for thought-summary extraction from the interaction timeline."""

    def test_collects_text_from_thought_steps_only(self):
        interaction = SimpleNamespace(
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="report body")],
                ),
                SimpleNamespace(
                    type="thought",
                    content=[
                        SimpleNamespace(type="text", text="First I will plan."),
                        SimpleNamespace(type="text", text="Then I will search."),
                    ],
                ),
            ]
        )
        thinking = gemini_research._extract_interaction_thinking(interaction)
        assert thinking == "First I will plan.\n\nThen I will search."
        assert "report body" not in thinking

    def test_returns_empty_without_thought_steps(self):
        interaction = SimpleNamespace(
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="body")],
                )
            ]
        )
        assert gemini_research._extract_interaction_thinking(interaction) == ""

    def test_handles_missing_steps(self):
        assert gemini_research._extract_interaction_thinking(SimpleNamespace(steps=None)) == ""


class TestResearchSourceLabeling:
    """Pure-function tests for the research source-label helpers."""

    def test_has_model_sources_footer_detects_heading(self):
        text = "# Report\n\nFindings.\n\n**Sources:**\n1. [arxiv.org](https://x)\n"
        assert gemini_research._has_model_sources_footer(text) is True

    def test_has_model_sources_footer_is_case_insensitive(self):
        assert gemini_research._has_model_sources_footer("**sources:**\n1. [a](b)\n") is True

    def test_has_model_sources_footer_returns_false_without_heading(self):
        assert gemini_research._has_model_sources_footer("just some prose") is False
        assert gemini_research._has_model_sources_footer(None) is False
        assert gemini_research._has_model_sources_footer("") is False

    def test_model_footer_url_titles_parses_numbered_entries(self):
        text = (
            "Body content here.\n\n"
            "**Sources:**\n"
            "1. [arxiv.org](https://example/1)\n"
            "2. [iclr.cc](https://example/2)\n"
            "3. [aclanthology.org](https://example/3)\n"
        )
        titles = gemini_research._model_footer_url_titles(text)
        assert titles == {
            "https://example/1": "arxiv.org",
            "https://example/2": "iclr.cc",
            "https://example/3": "aclanthology.org",
        }

    def test_model_footer_url_titles_empty_when_no_heading(self):
        assert gemini_research._model_footer_url_titles("no footer here") == {}
        assert gemini_research._model_footer_url_titles(None) == {}

    def test_hostname_label_extracts_host(self):
        assert gemini_research._hostname_label("https://example.com/path?q=1") == "example.com"
        assert gemini_research._hostname_label("https://sub.domain.co.uk/x") == "sub.domain.co.uk"

    def test_hostname_label_falls_back_to_untitled(self):
        assert gemini_research._hostname_label(None) == "(untitled)"
        assert gemini_research._hostname_label("") == "(untitled)"
        assert gemini_research._hostname_label("not a url") == "(untitled)"

    def test_url_citation_label_prefers_api_title(self):
        annotation = SimpleNamespace(
            type="url_citation", title="The Real Title", url="https://example.com"
        )
        assert (
            gemini_research._url_citation_label(annotation, {"https://example.com": "ignored"})
            == "The Real Title"
        )

    def test_url_citation_label_uses_footer_when_title_missing(self):
        annotation = SimpleNamespace(type="url_citation", title=None, url="https://redirect/abc")
        footer = {"https://redirect/abc": "arxiv.org"}
        assert gemini_research._url_citation_label(annotation, footer) == "arxiv.org"

    def test_url_citation_label_falls_back_to_hostname(self):
        annotation = SimpleNamespace(type="url_citation", title=None, url="https://example.org/x")
        assert gemini_research._url_citation_label(annotation, {}) == "example.org"

    def test_build_citations_embed_uses_footer_titles_for_unlabeled_urls(self):
        annotations = [
            SimpleNamespace(type="url_citation", title=None, url="https://redirect/1"),
            SimpleNamespace(type="url_citation", title=None, url="https://redirect/2"),
        ]
        report_text = (
            "**Sources:**\n1. [arxiv.org](https://redirect/1)\n2. [iclr.cc](https://redirect/2)\n"
        )
        embed = gemini_research._build_citations_embed(annotations, report_text)
        assert embed is not None
        assert "arxiv.org" in embed.description
        assert "iclr.cc" in embed.description
        assert "(untitled)" not in embed.description

    def test_build_citations_embed_falls_back_to_hostname_without_footer(self):
        annotations = [
            SimpleNamespace(type="url_citation", title=None, url="https://example.com/path"),
        ]
        embed = gemini_research._build_citations_embed(annotations, report_text=None)
        assert embed is not None
        assert "example.com" in embed.description
        assert "(untitled)" not in embed.description


class TestResearchReportAssembly(AsyncGeminiCogTestCase):
    """Integration tests for `research_command` report-body assembly (1A gating)."""

    @pytest.fixture(autouse=True)
    async def setup_research(self, setup):
        self.mock_client_instance.aio.interactions.create = AsyncMock()
        self.mock_client_instance.aio.interactions.get = AsyncMock()

    async def _run_with_interaction(self, interaction_done, **command_kwargs):
        captured: dict[str, Any] = {}

        async def capture_send_embed_batches(send, **kwargs):
            file_obj = kwargs.get("file")
            if file_obj is not None:
                captured["report_text"] = file_obj.fp.getvalue().decode("utf-8")
            # The flow sends a green status embed first, then edits it in place to the
            # final header, so the returned message must expose an awaitable `.edit`.
            status_msg = MagicMock()
            status_msg.edit = AsyncMock()
            return status_msg

        ctx = MagicMock()
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        ctx.author.id = 42

        self.cog.client.aio.interactions.create.return_value = interaction_done

        with (
            patch("discord_gemini.cogs.gemini.research.asyncio.sleep", new_callable=AsyncMock),
            patch(
                "discord_gemini.cogs.gemini.research.send_embed_batches",
                side_effect=capture_send_embed_batches,
            ),
        ):
            await gemini_research.research_command(self.cog, ctx, "Test prompt", **command_kwargs)

        return captured["report_text"]

    async def test_research_bills_cached_tokens_through_calculate_cost(self):
        """The Interactions API reports cache hits as `total_cached_tokens`, a subset of
        `total_input_tokens` (a repeated 9,804-token prompt reported 4,082 cached,
        2026-09-22). At gemini-3.1-pro-preview's rates the 5,722 uncached tokens cost
        $2.00/M and the 4,082 cached tokens $0.20/M: $0.0122604, not the $0.019608 the
        full input rate would bill."""
        interaction_done = SimpleNamespace(
            id="cached",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(type="text", text="# Report\n\nBody.", annotations=[])
                    ],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=9804,
                total_output_tokens=0,
                total_thought_tokens=0,
                total_cached_tokens=4082,
            ),
        )

        with patch.object(self.cog, "_log_cost") as log_cost:
            await self._run_with_interaction(interaction_done)

        log_cost.assert_called_once()
        assert log_cost.call_args.args[3] == pytest.approx(0.0122604)
        assert log_cost.call_args.kwargs["input_tokens"] == 9804
        assert log_cost.call_args.kwargs["cached_tokens"] == 4082

    async def test_research_bills_maps_grounding_surcharge_for_research_model(self):
        """A run the API reports as Maps-grounded must bill the research model's
        Maps surcharge ONCE PER GROUNDED PROMPT (Google bills each one, and the
        Interactions API reports the count); a run without Maps grounding must not.
        Mirrors chat, which bills on actual grounding use, not on the tool being
        enabled."""
        # Resolve through util so the expected rate is the one calculate_cost consults,
        # regardless of pricing-module reloads elsewhere in the suite.
        from discord_gemini.util import maps_grounding_cost_for_model

        def interaction_with(grounding_tool_count):
            return SimpleNamespace(
                id="maps-billing",
                status="completed",
                steps=[
                    SimpleNamespace(
                        type="model_output",
                        content=[
                            SimpleNamespace(type="text", text="# Report\n\nBody.", annotations=[])
                        ],
                    )
                ],
                usage=SimpleNamespace(
                    total_input_tokens=1_000_000,
                    total_output_tokens=0,
                    total_thought_tokens=0,
                    grounding_tool_count=grounding_tool_count,
                ),
            )

        async def billed(interaction_done, **command_kwargs):
            with patch.object(self.cog, "_log_cost") as log_cost:
                await self._run_with_interaction(interaction_done, **command_kwargs)
            log_cost.assert_called_once()
            return log_cost.call_args.args[3], log_cost.call_args.kwargs["google_maps_grounded"]

        maps_off_cost, maps_off_grounded = await billed(interaction_with(None))
        maps_once_cost, maps_once_grounded = await billed(
            interaction_with([SimpleNamespace(type="google_maps", count=1)]), google_maps=True
        )
        maps_twice_cost, maps_twice_grounded = await billed(
            interaction_with([SimpleNamespace(type="google_maps", count=2)]), google_maps=True
        )
        # Enabled but never invoked: the API reports no Maps grounding, so no surcharge
        # (the three Google Search queries are billed on their own, at $0.014 each).
        unused_cost, unused_grounded = await billed(
            interaction_with([SimpleNamespace(type="google_search", count=3)]), google_maps=True
        )

        surcharge = maps_grounding_cost_for_model("gemini-3.1-pro-preview")
        assert surcharge > 0
        assert maps_off_grounded is False
        assert maps_once_grounded is True
        assert maps_once_cost == pytest.approx(maps_off_cost + surcharge)
        assert maps_twice_grounded is True
        assert maps_twice_cost == pytest.approx(maps_off_cost + 2 * surcharge)
        assert unused_grounded is False
        assert unused_cost == pytest.approx(maps_off_cost + 3 * 0.014)

    async def test_research_bills_each_google_search_query(self):
        """Deep research always grounds on Google Search, and Gemini 3.x bills each
        search query at $14 / 1K: the research model's 1M input tokens cost $2.00 and the
        five searches the Interactions API reports add $0.07. Two Maps calls add their
        own $0.014 each on top."""

        def interaction_with(grounding_tool_count):
            return SimpleNamespace(
                id="search-billing",
                status="completed",
                steps=[
                    SimpleNamespace(
                        type="model_output",
                        content=[
                            SimpleNamespace(type="text", text="# Report\n\nBody.", annotations=[])
                        ],
                    )
                ],
                usage=SimpleNamespace(
                    total_input_tokens=1_000_000,
                    total_output_tokens=0,
                    total_thought_tokens=0,
                    grounding_tool_count=grounding_tool_count,
                ),
            )

        async def billed(interaction_done, **command_kwargs):
            with patch.object(self.cog, "_log_cost") as log_cost:
                await self._run_with_interaction(interaction_done, **command_kwargs)
            log_cost.assert_called_once()
            return log_cost.call_args.args[3], log_cost.call_args.kwargs

        no_search_cost, no_search_log = await billed(interaction_with(None))
        search_cost, search_log = await billed(
            interaction_with([SimpleNamespace(type="google_search", count=5)])
        )
        both_cost, both_log = await billed(
            interaction_with(
                [
                    SimpleNamespace(type="google_search", count=5),
                    SimpleNamespace(type="google_maps", count=2),
                ]
            ),
            google_maps=True,
        )
        # `search_query_count`, when the entry carries it, is the billed query count.
        query_count_cost, query_count_log = await billed(
            interaction_with([SimpleNamespace(type="google_search", count=2, search_query_count=6)])
        )

        assert no_search_cost == pytest.approx(2.00)
        assert no_search_log["google_search_queries"] == 0
        assert search_cost == pytest.approx(2.07)
        assert search_log["google_search_queries"] == 5
        assert both_cost == pytest.approx(2.098)
        assert both_log["google_search_queries"] == 5
        assert query_count_cost == pytest.approx(2.00 + 6 * 0.014)
        assert query_count_log["google_search_queries"] == 6
        assert query_count_log["grounding_tool_counts"] == {"google_search": 6}

    async def test_research_bills_tool_use_tokens_as_input(self):
        """Interactions reports tool-use prompt tokens in `total_tool_use_tokens`, apart
        from `total_input_tokens`, and research bills them at the input rate as chat
        does: 400K input + 600K tool-use tokens at gemini-3.1-pro-preview's $2.00/M is
        $2.00, plus 100K output at $12.00/M is $3.20."""
        interaction_done = SimpleNamespace(
            id="tool-use-billing",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(type="text", text="# Report\n\nBody.", annotations=[])
                    ],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=400_000,
                total_tool_use_tokens=600_000,
                total_output_tokens=100_000,
                total_thought_tokens=0,
            ),
        )

        with patch.object(self.cog, "_log_cost") as log_cost:
            await self._run_with_interaction(interaction_done)

        log_cost.assert_called_once()
        assert log_cost.call_args.args[3] == pytest.approx(3.20)
        assert log_cost.call_args.kwargs["input_tokens"] == 1_000_000

    async def test_report_body_skips_appended_sources_when_model_footer_present(self):
        """When the model emits its own `**Sources:**` footer, the wrapper-appended
        `## Sources` block must not duplicate it in the file."""
        annotation = SimpleNamespace(type="url_citation", title=None, url="https://redirect/1")
        interaction_done = SimpleNamespace(
            id="footer-present",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(
                            type="text",
                            text=(
                                "# Report Body\n\nDetailed findings.\n\n"
                                "**Sources:**\n1. [arxiv.org](https://redirect/1)\n"
                            ),
                            annotations=[annotation],
                        )
                    ],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        report_text = await self._run_with_interaction(interaction_done)
        assert "# Report Body" in report_text
        assert "**Sources:**" in report_text
        # The wrapper-appended block uses `## Sources`, which must not appear
        # alongside the model's own `**Sources:**` heading.
        assert "## Sources" not in report_text

    async def test_report_body_appends_wrapper_sources_when_model_omits_footer(self):
        """Without a model footer, the wrapper still appends its own `## Sources`
        block as a safety net, with hostname-based labels."""
        annotation = SimpleNamespace(type="url_citation", title=None, url="https://example.com/x")
        interaction_done = SimpleNamespace(
            id="no-footer",
            status="completed",
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(
                            type="text",
                            text="# Report Body\n\nFindings without a footer.\n",
                            annotations=[annotation],
                        )
                    ],
                )
            ],
            usage=SimpleNamespace(
                total_input_tokens=100, total_output_tokens=50, total_thought_tokens=0
            ),
        )

        report_text = await self._run_with_interaction(interaction_done)
        assert "# Report Body" in report_text
        assert "## Sources" in report_text
        # Hostname fallback should kick in for the (titleless) annotation.
        assert "example.com" in report_text
        assert "(untitled)" not in report_text


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
        desc = self._build_temperature_description("gemini-3.1-flash-lite", 0.7)
        assert "warning" in desc

    def test_gemini25_flash_no_warning(self):
        """Gemini 2.5 Flash does not trigger warning."""
        desc = self._build_temperature_description("gemini-2.5-flash", 0.3)
        assert "warning" not in desc
