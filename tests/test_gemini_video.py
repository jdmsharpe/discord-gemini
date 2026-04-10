import os
import tempfile
from unittest.mock import MagicMock

from discord_gemini.cogs.gemini.video import _validate_video_request

from tests.support import AsyncGeminiCogTestCase


class TestVideoResponseEmbed(AsyncGeminiCogTestCase):
    async def _assert_video_mode(self, params, expected_mode, attachment=None):
        video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # noqa: SIM115
        try:
            video_file.close()
            embed, files = await self.cog._create_video_response_embed(
                video_params=params,
                generated_videos=[video_file.name],
                attachment=attachment,
            )
            for file in files:
                file.close()
        finally:
            os.unlink(video_file.name)
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

        video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # noqa: SIM115
        try:
            video_file.close()
            params = VideoGenerationParameters(
                prompt="A sunset",
                model="veo-3.1-lite-generate-preview",
                resolution="1080p",
            )
            embed, files = await self.cog._create_video_response_embed(
                video_params=params,
                generated_videos=[video_file.name],
                attachment=None,
            )
            for file in files:
                file.close()
        finally:
            os.unlink(video_file.name)

        assert "**Resolution:** 1080p" in embed.description


class TestVideoValidation:
    def test_rejects_multiple_videos_outside_veo_2(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="16:9",
            resolution=None,
            number_of_videos=2,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert "number_of_videos" in error

    def test_rejects_resolution_on_veo_2(self):
        error = _validate_video_request(
            model="veo-2.0-generate-001",
            aspect_ratio="16:9",
            resolution="720p",
            number_of_videos=1,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert error == "The `resolution` parameter is not supported on Veo 2."

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

    def test_rejects_1080p_portrait_on_veo_3(self):
        error = _validate_video_request(
            model="veo-3.0-fast-generate-001",
            aspect_ratio="9:16",
            resolution="1080p",
            number_of_videos=1,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert error == "Veo 3 and Veo 3 Fast only support `1080p` with a `16:9` aspect ratio."

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
