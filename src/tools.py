"""Legacy tools shim."""

from warnings import warn

from discord_gemini.cogs.gemini.tooling import (
    ToolEntry,
    execute_tool_call,
    get_registered_tools,
    get_tool_callables,
    tool,
)

warn(
    "tools is deprecated; import from discord_gemini.cogs.gemini.tooling instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["tool", "get_registered_tools", "get_tool_callables", "execute_tool_call", "ToolEntry"]
