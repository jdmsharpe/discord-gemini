from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from google.genai.errors import APIError

from discord_gemini.cogs.gemini.attachments import (
    _guess_attachment_mime_type,
    _guess_url_mime_type,
)
from tests.support import AsyncGeminiCogTestCase


class TestGeminiAttachmentHelpers(AsyncGeminiCogTestCase):
    async def test_fetch_attachment_bytes_success(self):
        """Test successful attachment download."""
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

        attachment = MagicMock()
        attachment.url = "https://example.com/image.png"

        result = await self.cog._fetch_attachment_bytes(attachment)
        assert result == b"image data"

    async def test_fetch_attachment_bytes_failure(self):
        """Test failed attachment download returns None."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context)

        self.cog._http_session = mock_session

        attachment = MagicMock()
        attachment.url = "https://example.com/not-found.png"

        result = await self.cog._fetch_attachment_bytes(attachment)
        assert result is None

    async def test_validate_attachment_size_within_limit(self):
        """Test that attachments within limits pass validation."""
        attachment = MagicMock()
        attachment.size = 10 * 1024 * 1024
        attachment.content_type = "image/png"

        result = self.cog._validate_attachment_size(attachment)
        assert result is None

    async def test_validate_attachment_size_exceeds_file_api_max(self):
        """Test that attachments over 2 GB are rejected."""
        attachment = MagicMock()
        attachment.size = 3 * 1024 * 1024 * 1024
        attachment.content_type = "video/mp4"

        result = self.cog._validate_attachment_size(attachment)
        assert result is not None
        assert "too large" in result
        assert "2 GB" in result

    async def test_prepare_attachment_part_inline_small_file(self):
        """Test that small files use inline data."""
        attachment = MagicMock()
        attachment.size = 1 * 1024 * 1024
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
        attachment.size = 25 * 1024 * 1024
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
        attachment.size = 25 * 1024 * 1024
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

        self.cog.client.aio.files.upload = AsyncMock(side_effect=APIError(500, {}))

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
        from discord_gemini.util import ChatCompletionParameters

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
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            uploaded_file_names=["files/abc", "files/def"],
        )

        self.cog.client.aio.files.delete = AsyncMock(side_effect=APIError(404, {}))

        await self.cog._cleanup_uploaded_files(params)

        assert params.uploaded_file_names == []

    async def test_cleanup_uploaded_files_noop_when_empty(self):
        """Test that _cleanup_uploaded_files is a no-op with no files."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")

        self.cog.client.aio.files.delete = AsyncMock()

        await self.cog._cleanup_uploaded_files(params)

        self.cog.client.aio.files.delete.assert_not_called()


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
