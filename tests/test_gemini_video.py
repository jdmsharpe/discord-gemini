import os
import tempfile
from unittest.mock import MagicMock

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
