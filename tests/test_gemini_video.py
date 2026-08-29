import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discord_gemini.cogs.gemini.cog import GeminiCog
from discord_gemini.cogs.gemini.command_options import VIDEO_MODEL_CHOICES
from discord_gemini.cogs.gemini.video import (
    DEFAULT_OMNI_VIDEO_MODEL,
    OMNI_VIDEO_MODELS,
    _build_veo_image,
    _generate_video_with_omni,
    _generate_video_with_veo,
    _validate_omni_video_request,
    _validate_video_request,
)
from discord_gemini.util import (
    VIDEO_TOKEN_PRICING,
    VideoGenerationParameters,
    calculate_omni_video_cost,
)
from tests.support import AsyncGeminiCogTestCase, build_mock_bot


def _encoded_image(fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(buf, format=fmt)
    return buf.getvalue()


class TestBuildVeoImage:
    """Veo image inputs must carry raw bytes + a mime type.

    A `PIL.Image` silently validates into an all-None `types.Image`, which the API
    rejects with "should contain both bytesBase64Encoded and mimeType".
    """

    @staticmethod
    def _attachment(content_type):
        attachment = MagicMock()
        attachment.content_type = content_type
        return attachment

    @pytest.mark.parametrize(
        ("fmt", "content_type", "expected_mime"),
        [
            ("WEBP", "image/webp", "image/webp"),
            ("PNG", "image/png", "image/png"),
            ("JPEG", "image/jpeg; charset=binary", "image/jpeg"),
        ],
    )
    def test_uses_attachment_content_type(self, fmt, content_type, expected_mime):
        data = _encoded_image(fmt)
        image = _build_veo_image(data, self._attachment(content_type))
        assert image.image_bytes == data
        assert image.mime_type == expected_mime

    def test_falls_back_to_detected_format(self):
        data = _encoded_image("WEBP")
        image = _build_veo_image(data, self._attachment(None))
        assert image.mime_type == "image/webp"
        assert image.image_bytes == data

    def test_ignores_non_image_content_type(self):
        data = _encoded_image("PNG")
        image = _build_veo_image(data, self._attachment("application/octet-stream"))
        assert image.mime_type == "image/png"

    def test_never_serializes_to_an_empty_struct(self):
        """The regression: the API 400s when image_bytes/mime_type are absent."""
        image = _build_veo_image(_encoded_image("WEBP"), self._attachment("image/webp"))
        payload = image.model_dump(exclude_none=True)
        assert payload, "types.Image serialized to {} — the API rejects this"
        assert "image_bytes" in payload
        assert "mime_type" in payload


class TestVideoResponseEmbed(AsyncGeminiCogTestCase):
    async def _assert_video_mode(self, params, expected_mode, attachment=None):
        embed, files = await self.cog._create_video_response_embed(
            video_params=params,
            generated_videos=[b"fake-video-bytes"],
            attachment=attachment,
        )
        for file in files:
            file.close()
        assert expected_mode in embed.description
        assert len(files) == 1

    async def test_mode_text_to_video(self):
        """Test embed shows Text-to-Video mode when no attachments."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        await self._assert_video_mode(params, "Text-to-Video")

    async def test_mode_image_to_video(self):
        """Test embed shows Image-to-Video mode when attachment provided."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Image-to-Video", attachment=mock_attachment)

    async def test_mode_interpolation(self):
        """Test embed shows Interpolation mode when both attachment and last_frame."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Interpolation", attachment=mock_attachment)

    async def test_mode_last_frame_only(self):
        """Test embed shows Last Frame Constrained mode when only last_frame."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        await self._assert_video_mode(params, "Last Frame Constrained")

    async def test_embed_includes_resolution(self):
        """Test embed includes the selected output resolution."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-lite-generate-preview",
            resolution="1080p",
        )
        embed, files = await self.cog._create_video_response_embed(
            video_params=params,
            generated_videos=[b"fake-video-bytes"],
            attachment=None,
        )
        for file in files:
            file.close()

        assert "**Resolution:** 1080p" in embed.description


class TestVideoValidation:
    def test_rejects_multiple_videos(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="16:9",
            resolution=None,
            number_of_videos=2,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert "number_of_videos" in error

    def test_rejects_4k_on_lite(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="16:9",
            resolution="4k",
            number_of_videos=1,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert "`4k`" in error

    def test_rejects_non_8s_for_1080p(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="9:16",
            resolution="1080p",
            number_of_videos=1,
            duration_seconds=6,
            has_last_frame=False,
        )
        assert error == "The `1080p` resolution only supports 8 second videos."

    def test_rejects_interpolation_without_8s(self):
        error = _validate_video_request(
            model="veo-3.1-generate-preview",
            aspect_ratio="16:9",
            resolution=None,
            number_of_videos=1,
            duration_seconds=6,
            has_last_frame=True,
        )
        assert "last_frame" in error

    def test_accepts_lite_1080p_portrait_at_8s(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="9:16",
            resolution="1080p",
            number_of_videos=1,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert error is None


class TestVeoGenerateVideosSource(AsyncGeminiCogTestCase):
    """google-genai 2.14.0 deprecated the prompt/image arguments in favour of `source`."""

    @staticmethod
    def _attachment():
        attachment = MagicMock()
        attachment.content_type = "image/png"
        return attachment

    async def _call_veo(self, attachment=None, last_frame_attachment=None):
        from discord_gemini.util import VideoGenerationParameters

        self.cog.client.aio.models.generate_videos = AsyncMock(
            return_value=SimpleNamespace(done=True, name="operations/1", response=None)
        )
        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        await _generate_video_with_veo(self.cog, params, attachment, last_frame_attachment)
        return self.cog.client.aio.models.generate_videos.call_args.kwargs

    async def test_prompt_is_passed_through_source(self):
        kwargs = await self._call_veo()

        assert "prompt" not in kwargs
        assert "image" not in kwargs
        assert kwargs["source"].prompt == "A sunset"
        assert kwargs["source"].image is None

    async def test_image_is_passed_through_source(self):
        data = _encoded_image("PNG")
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=data)

        kwargs = await self._call_veo(attachment=self._attachment())

        assert "image" not in kwargs
        assert kwargs["source"].image.image_bytes == data
        assert kwargs["source"].image.mime_type == "image/png"

    async def test_last_frame_stays_on_the_config(self):
        """`last_frame` is a GenerateVideosConfig field, not a source field."""
        data = _encoded_image("PNG")
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=data)

        kwargs = await self._call_veo(last_frame_attachment=self._attachment())

        assert kwargs["config"].last_frame.image_bytes == data
        # `hasattr` is always False on a pydantic model for an undeclared field,
        # so assert on the declared field set instead — that is what would
        # actually change if last_frame ever migrated onto the source.
        assert "last_frame" not in type(kwargs["source"]).model_fields


def _fake_omni_interaction(
    uri: str | None = "https://x/v1beta/files/abc123:download?alt=media",
    video_tokens: int = 57920,
):
    modality = MagicMock()
    modality.modality = "video"
    modality.tokens = video_tokens
    usage = MagicMock()
    usage.output_tokens_by_modality = [modality]
    interaction = MagicMock()
    interaction.status = "completed"
    interaction.usage = usage
    if uri is None:
        interaction.output_video = None
    else:
        output_video = MagicMock()
        output_video.uri = uri
        output_video.mime_type = "video/mp4"
        interaction.output_video = output_video
    return interaction


class TestOmniVideoGeneration(AsyncGeminiCogTestCase):
    async def test_downloads_video_and_returns_tokens(self):
        from discord_gemini.util import VideoGenerationParameters

        self.cog.client.aio.interactions.create = AsyncMock(return_value=_fake_omni_interaction())
        self.cog.client.files.download = MagicMock(return_value=b"mp4-bytes")

        params = VideoGenerationParameters(
            prompt="A red ball rolls across a table",
            model=DEFAULT_OMNI_VIDEO_MODEL,
            aspect_ratio="16:9",
        )
        videos, tokens = await _generate_video_with_omni(self.cog, params)

        assert videos == [b"mp4-bytes"]
        assert tokens == 57920
        _, kwargs = self.cog.client.aio.interactions.create.call_args
        assert kwargs["model"] == DEFAULT_OMNI_VIDEO_MODEL
        assert kwargs["input"] == "A red ball rolls across a table"
        assert kwargs["response_format"] == {
            "type": "video",
            "aspect_ratio": "16:9",
            "delivery": "uri",
        }
        # File name is parsed from the URI, not the raw URI.
        _, dl_kwargs = self.cog.client.files.download.call_args
        assert dl_kwargs["file"] == "files/abc123"

    async def test_missing_uri_returns_no_bytes(self):
        from discord_gemini.util import VideoGenerationParameters

        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(uri=None)
        )
        self.cog.client.files.download = MagicMock(return_value=b"unused")

        params = VideoGenerationParameters(prompt="x", model=DEFAULT_OMNI_VIDEO_MODEL)
        videos, tokens = await _generate_video_with_omni(self.cog, params)

        assert videos == []
        assert tokens == 57920
        self.cog.client.files.download.assert_not_called()

    def test_omni_is_default_and_first_choice(self):
        """gemini-omni-1.1-flash (GA 2026-08-27) is the default and heads the picker."""
        assert DEFAULT_OMNI_VIDEO_MODEL == "gemini-omni-1.1-flash"
        assert VIDEO_MODEL_CHOICES[0].value == DEFAULT_OMNI_VIDEO_MODEL
        default = inspect.signature(self.cog.video.callback).parameters["model"].default
        assert default == DEFAULT_OMNI_VIDEO_MODEL

    def test_legacy_preview_stays_selectable_until_shutdown(self):
        """gemini-omni-flash-preview shuts down 2026-09-30; keep it in the menu until then."""
        values = [choice.value for choice in VIDEO_MODEL_CHOICES]
        assert "gemini-omni-flash-preview" in values
        assert values.index("gemini-omni-flash-preview") > values.index(DEFAULT_OMNI_VIDEO_MODEL)


class TestOmniVideoModels:
    """Both the GA id and the legacy preview id must route through the Omni path."""

    def test_both_ids_are_omni(self):
        assert {"gemini-omni-1.1-flash", "gemini-omni-flash-preview"} == OMNI_VIDEO_MODELS
        assert DEFAULT_OMNI_VIDEO_MODEL in OMNI_VIDEO_MODELS
        for veo in ("veo-3.1-generate-preview", "veo-3.1-lite-generate-preview"):
            assert veo not in OMNI_VIDEO_MODELS

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    def test_each_omni_id_is_priced_at_17_50_per_million(self, model):
        # The unknown-model fallback happens to be 17.50 too, so pin the explicit
        # row's presence as well as its rate: a dropped row must fail here.
        assert model in VIDEO_TOKEN_PRICING
        assert VIDEO_TOKEN_PRICING[model] == 17.50
        assert calculate_omni_video_cost(model, 1_000_000) == pytest.approx(17.50)

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    async def test_video_command_routes_each_omni_id_to_the_interactions_path(self, model):
        """`is_omni` must key off the set, not a single id, or the legacy id would hit Veo."""
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        with patch("discord_gemini.cogs.gemini.client.build_gemini_client"):
            cog = GeminiCog(bot=bot)
        cog._send_error_followup = AsyncMock()

        with (
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([], 0)),
            ) as omni,
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_veo",
                AsyncMock(return_value=[]),
            ) as veo,
        ):
            await cog.video.callback(cog, ctx=ctx, prompt="a red ball", model=model)

        cog._send_error_followup.assert_not_awaited()
        omni.assert_awaited_once()
        assert omni.await_args.args[1].model == model
        veo.assert_not_awaited()

    async def test_video_command_routes_veo_to_generate_videos(self):
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        with patch("discord_gemini.cogs.gemini.client.build_gemini_client"):
            cog = GeminiCog(bot=bot)
        cog._send_error_followup = AsyncMock()

        with (
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([], 0)),
            ) as omni,
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_veo",
                AsyncMock(return_value=[]),
            ) as veo,
        ):
            await cog.video.callback(
                cog, ctx=ctx, prompt="a red ball", model="veo-3.1-generate-preview"
            )

        cog._send_error_followup.assert_not_awaited()
        veo.assert_awaited_once()
        omni.assert_not_awaited()

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    def test_each_omni_id_rejects_veo_only_options(self, model):
        params = VideoGenerationParameters(
            prompt="x", model=model, resolution="1080p", duration_seconds=8
        )
        error = _validate_omni_video_request(params, None, None)
        assert error and "resolution" in error and "duration" in error


class TestOmniVideoValidation:
    def _params(self, **kwargs):
        from discord_gemini.util import VideoGenerationParameters

        base = {"prompt": "x", "model": DEFAULT_OMNI_VIDEO_MODEL, "aspect_ratio": "16:9"}
        base.update(kwargs)
        return VideoGenerationParameters(**base)

    def test_accepts_bare_text_to_video(self):
        assert _validate_omni_video_request(self._params(), None, None) is None

    def test_rejects_resolution(self):
        error = _validate_omni_video_request(self._params(resolution="1080p"), None, None)
        assert error and "resolution" in error

    def test_rejects_duration(self):
        error = _validate_omni_video_request(self._params(duration_seconds=8), None, None)
        assert error and "duration" in error

    def test_rejects_multiple_videos(self):
        error = _validate_omni_video_request(self._params(number_of_videos=2), None, None)
        assert error and "number_of_videos" in error

    def test_rejects_image_attachment(self):
        error = _validate_omni_video_request(self._params(), MagicMock(), None)
        assert error and "attachment" in error

    def test_rejects_last_frame(self):
        error = _validate_omni_video_request(self._params(has_last_frame=True), None, MagicMock())
        assert error and "last_frame" in error


class TestOmniVideoCost:
    def test_cost_is_exact_token_based(self):
        from discord_gemini.util import calculate_omni_video_cost

        cost = calculate_omni_video_cost(DEFAULT_OMNI_VIDEO_MODEL, 57920)
        assert cost == pytest.approx(57920 * 17.5 / 1_000_000)
