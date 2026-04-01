"""Public namespace for the discord-gemini package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cogs.gemini.cog import GeminiCog
    from .cogs.gemini.models import Conversation

__all__ = ["GeminiCog", "Conversation"]


def __getattr__(name: str):
    if name == "GeminiCog":
        from .cogs.gemini import GeminiCog

        return GeminiCog
    if name == "Conversation":
        from .cogs.gemini import Conversation

        return Conversation
    raise AttributeError(name)
