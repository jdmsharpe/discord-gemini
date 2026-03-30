# ruff: noqa: F403

"""Legacy util shim."""

from warnings import warn

from discord_gemini.util import *

warn(
    "util is deprecated; import from discord_gemini.util instead.",
    DeprecationWarning,
    stacklevel=2,
)
