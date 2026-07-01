import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from discord_gemini.cogs.gemini.command_options import VIDEO_MODEL_CHOICES
from discord_gemini.cogs.gemini.video import (
    OMNI_VIDEO_MODEL,
    _generate_video_with_omni,
    _validate_omni_video_request,
    _validate_video_request,
)
from tests.support import AsyncGeminiCogTestCase


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
            prompt="A red ball rolls across a table", model=OMNI_VIDEO_MODEL, aspect_ratio="16:9"
        )
        videos, tokens = await _generate_video_with_omni(self.cog, params)

        assert videos == [b"mp4-bytes"]
        assert tokens == 57920
        _, kwargs = self.cog.client.aio.interactions.create.call_args
        assert kwargs["model"] == OMNI_VIDEO_MODEL
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

        params = VideoGenerationParameters(prompt="x", model=OMNI_VIDEO_MODEL)
        videos, tokens = await _generate_video_with_omni(self.cog, params)

        assert videos == []
        assert tokens == 57920
        self.cog.client.files.download.assert_not_called()

    def test_omni_is_default_and_first_choice(self):
        assert VIDEO_MODEL_CHOICES[0].value == OMNI_VIDEO_MODEL
        default = inspect.signature(self.cog.video.callback).parameters["model"].default
        assert default == OMNI_VIDEO_MODEL


class TestOmniVideoValidation:
    def _params(self, **kwargs):
        from discord_gemini.util import VideoGenerationParameters

        base = {"prompt": "x", "model": OMNI_VIDEO_MODEL, "aspect_ratio": "16:9"}
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
        error = _validate_omni_video_request(
            self._params(has_last_frame=True), None, MagicMock()
        )
        assert error and "last_frame" in error


class TestOmniVideoCost:
    def test_cost_is_exact_token_based(self):
        from discord_gemini.util import calculate_omni_video_cost

        cost = calculate_omni_video_cost(OMNI_VIDEO_MODEL, 57920)
        assert cost == pytest.approx(57920 * 17.5 / 1_000_000)
