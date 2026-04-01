"""Gemini cog package."""

__all__ = ["GeminiCog", "Conversation"]  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    if name == "GeminiCog":
        from .cog import GeminiCog

        return GeminiCog
    if name == "Conversation":
        from .models import Conversation

        return Conversation
    raise AttributeError(name)
