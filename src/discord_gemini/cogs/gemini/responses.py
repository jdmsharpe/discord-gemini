"""Custom exception hierarchy for the Discord Gemini bot.

Using specific exception types instead of bare ``Exception`` makes catch
blocks precise: you can handle a cache failure differently from a file
upload failure without resorting to fragile string matching.
"""


class GeminiBotError(Exception):
    """Base exception for all bot-specific errors."""


class APICallError(GeminiBotError):
    """An error originating from a Google Gemini API call."""


class CacheError(GeminiBotError):
    """Failed to create, refresh, or delete an explicit context cache."""


class FileUploadError(GeminiBotError):
    """Failed to upload or clean up a file via the Gemini File API."""


class ValidationError(GeminiBotError):
    """A user-supplied parameter or combination of parameters is invalid."""


class MusicGenerationError(GeminiBotError):
    """An error specific to the Lyria RealTime music generation pipeline."""
