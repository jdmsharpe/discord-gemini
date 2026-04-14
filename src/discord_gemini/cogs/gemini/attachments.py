"""Attachment helpers for Gemini chat, image, music, and video flows."""

import mimetypes
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
from discord import Attachment
from google.genai.errors import APIError

from ...util import ATTACHMENT_FILE_API_MAX_SIZE, ATTACHMENT_FILE_API_THRESHOLD

if TYPE_CHECKING:
    from .cog import GeminiCog

_YOUTUBE_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:youtube\.com/|youtu\.be/)", re.IGNORECASE)
FILE_API_UPLOAD_EXCEPTIONS = (APIError, aiohttp.ClientError, OSError, TimeoutError, ValueError)
FILE_API_DELETE_EXCEPTIONS = (APIError, aiohttp.ClientError, TimeoutError)
_GENERIC_MIME_TYPES = {"application/octet-stream"}
_SUPPORTED_AUDIO_EXTENSION_MIME_TYPES = {
    ".opus": "audio/opus",
    ".alaw": "audio/alaw",
    ".mulaw": "audio/mulaw",
}


def _guess_filename_mime_type(name: str) -> str | None:
    """Guess a MIME type from a filename, including SDK-supported audio extensions."""

    suffix = Path(name).suffix.lower()
    if suffix in _SUPPORTED_AUDIO_EXTENSION_MIME_TYPES:
        return _SUPPORTED_AUDIO_EXTENSION_MIME_TYPES[suffix]

    mime_type, _ = mimetypes.guess_type(name)
    return mime_type


def _normalize_mime_type_for_name(mime_type: str | None, name: str) -> str:
    """Prefer filename-derived MIME types for supported formats and generic metadata."""

    guessed_mime_type = _guess_filename_mime_type(name)
    if (
        guessed_mime_type is not None
        and guessed_mime_type in _SUPPORTED_AUDIO_EXTENSION_MIME_TYPES.values()
    ):
        return guessed_mime_type
    if mime_type and mime_type not in _GENERIC_MIME_TYPES:
        return mime_type
    return guessed_mime_type if guessed_mime_type is not None else "application/octet-stream"


def _guess_url_mime_type(url: str) -> str:
    """Guess MIME type for a URL, with special handling for YouTube URLs."""

    if _YOUTUBE_URL_RE.match(url):
        return "video/mp4"
    parsed_url = urlparse(url)
    return _normalize_mime_type_for_name(None, parsed_url.path or url)


def _guess_attachment_mime_type(attachment: Attachment) -> str:
    """Guess MIME type for a Discord attachment from metadata and filename."""

    return _normalize_mime_type_for_name(attachment.content_type, attachment.filename)


async def _get_http_session(cog: "GeminiCog") -> aiohttp.ClientSession:
    """Reuse or lazily create the shared aiohttp session."""

    if cog._http_session and not cog._http_session.closed:
        return cog._http_session

    async with cog._session_lock:
        if cog._http_session is None or cog._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=15)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10, ttl_dns_cache=300)
            cog._http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return cog._http_session


async def _fetch_attachment_bytes(cog: "GeminiCog", attachment: Attachment) -> bytes | None:
    """Download an attachment via the shared HTTP session."""

    override = cog.__dict__.get("_fetch_attachment_bytes")
    if override is not None and override is not _fetch_attachment_bytes:
        return await override(attachment)

    session = await _get_http_session(cog)
    try:
        async with session.get(attachment.url) as response:
            if response.status == 200:
                return await response.read()
            cog.logger.warning(
                "Failed to fetch attachment %s: HTTP %s",
                attachment.url,
                response.status,
            )
    except aiohttp.ClientError as error:
        cog.logger.warning("Error fetching attachment %s: %s", attachment.url, error)
    return None


def _validate_attachment_size(attachment: Attachment) -> str | None:
    """Validate an attachment's size against Gemini API limits."""

    if attachment.size > ATTACHMENT_FILE_API_MAX_SIZE:
        size_mb = attachment.size / (1024 * 1024)
        return f"Attachment is too large ({size_mb:.1f} MB). Maximum file size is 2 GB."
    return None


async def _upload_attachment_to_file_api(
    cog: "GeminiCog",
    data: bytes,
    filename: str,
    mime_type: str,
) -> Any | None:
    """Upload attachment bytes to the Gemini File API."""

    temp_path = Path(f"temp_upload_{filename}")
    try:
        temp_path.write_bytes(data)
        uploaded_file = await cog.client.aio.files.upload(
            file=str(temp_path),
            config={"mime_type": mime_type},
        )
        cog.logger.info(
            "Uploaded %s to File API: %s (%d bytes)",
            filename,
            uploaded_file.name,
            len(data),
        )
        return uploaded_file
    except FILE_API_UPLOAD_EXCEPTIONS as error:
        cog.logger.warning("Failed to upload %s to File API: %s", filename, error)
        return None
    finally:
        temp_path.unlink(missing_ok=True)


async def _prepare_attachment_part(
    cog: "GeminiCog",
    attachment: Attachment,
    uploaded_file_names: list[str] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Prepare an attachment as either inline data or File API data."""

    content_type = _guess_attachment_mime_type(attachment)
    use_file_api = attachment.size > ATTACHMENT_FILE_API_THRESHOLD

    attachment_data = await _fetch_attachment_bytes(cog, attachment)
    if attachment_data is None:
        return None

    if use_file_api:
        uploaded_file = await _upload_attachment_to_file_api(
            cog, attachment_data, attachment.filename, content_type
        )
        if uploaded_file:
            if uploaded_file_names is not None:
                uploaded_file_names.append(uploaded_file.name)
            return {
                "file_data": {
                    "file_uri": uploaded_file.uri,
                    "mime_type": uploaded_file.mime_type,
                }
            }
        cog.logger.warning("File API upload failed, falling back to inline data")

    return {
        "inline_data": {
            "mime_type": content_type,
            "data": attachment_data,
        }
    }


async def _cleanup_uploaded_files(cog: "GeminiCog", params: Any) -> None:
    """Delete files uploaded to the File API for a conversation."""

    for file_name in params.uploaded_file_names:
        try:
            await cog.client.aio.files.delete(name=file_name)
            cog.logger.info("Deleted uploaded file %s", file_name)
        except FILE_API_DELETE_EXCEPTIONS as error:
            cog.logger.warning("Failed to delete uploaded file %s: %s", file_name, error)
    params.uploaded_file_names.clear()


__all__ = [
    "_cleanup_uploaded_files",
    "_fetch_attachment_bytes",
    "_get_http_session",
    "_guess_attachment_mime_type",
    "_guess_url_mime_type",
    "_prepare_attachment_part",
    "_upload_attachment_to_file_api",
    "_validate_attachment_size",
]
