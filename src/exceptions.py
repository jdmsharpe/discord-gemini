"""Legacy exceptions shim."""

from warnings import warn

from discord_gemini.cogs.gemini.responses import (
    APICallError,
    MusicGenerationError,
    ValidationError,
)

warn(
    "exceptions is deprecated; import from discord_gemini.cogs.gemini.responses instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["APICallError", "MusicGenerationError", "ValidationError"]
