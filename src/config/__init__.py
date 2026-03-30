"""Legacy config package shim."""

from warnings import warn

from .auth import (
    BOT_TOKEN,
    GEMINI_API_KEY,
    GEMINI_FILE_SEARCH_STORE_IDS,
    GUILD_IDS,
    SHOW_COST_EMBEDS,
)

warn(
    "config is deprecated; import from discord_gemini.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BOT_TOKEN",
    "GUILD_IDS",
    "GEMINI_API_KEY",
    "GEMINI_FILE_SEARCH_STORE_IDS",
    "SHOW_COST_EMBEDS",
]
