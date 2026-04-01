"""Gemini cog package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cog import GeminiCog
    from .models import Conversation

__all__ = ["GeminiCog", "Conversation"]


def __getattr__(name: str):
    if name == "GeminiCog":
        from .cog import GeminiCog

        return GeminiCog
    if name == "Conversation":
        from .models import Conversation

        return Conversation
    raise AttributeError(name)
