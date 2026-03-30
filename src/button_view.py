"""Legacy ButtonView shim."""

from warnings import warn

from discord_gemini.cogs.gemini.views import ButtonView

warn(
    "button_view is deprecated; import from discord_gemini.cogs.gemini.views instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ButtonView"]
