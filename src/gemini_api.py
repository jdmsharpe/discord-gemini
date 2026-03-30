"""Compatibility shim for the legacy gemini_api module."""

from warnings import warn

from discord_gemini import Conversation, GeminiAPI

warn(
    "gemini_api is deprecated; import from discord_gemini instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GeminiAPI", "Conversation"]
