"""Custom function tool registry for Gemini API tool calling.

Tools registered with the @tool decorator become available to Gemini models
during chat conversations when "Custom Functions" is enabled. The model can
call these functions and receive their return values as context.

Usage:
    @tool
    async def get_current_time(timezone: str = "UTC") -> str:
        '''Get the current date and time.

        Args:
            timezone: IANA timezone name (e.g., "America/New_York").
        '''
        ...
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ...config.auth import GEMINI_FILE_SEARCH_STORE_IDS
from ...util import (
    AVAILABLE_TOOLS,
    check_mutually_exclusive_tools,
    filter_file_search_incompatible_tools,
    filter_supported_tools_for_model,
    resolve_tool_name,
    validate_builtin_custom_tool_combination,
)

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: dict[str, "ToolEntry"] = {}


@dataclass
class ToolEntry:
    """A registered custom tool function."""

    func: Callable
    name: str
    description: str
    is_async: bool


def tool(func: Callable) -> Callable:
    """Decorator to register a Python function as a Gemini-callable tool.

    The function's name, docstring, and type hints are used by the SDK
    to auto-generate the JSON schema for the function declaration.
    """
    entry = ToolEntry(
        func=func,
        name=func.__name__,
        description=inspect.cleandoc(func.__doc__) if func.__doc__ else "",
        is_async=inspect.iscoroutinefunction(func),
    )
    _TOOL_REGISTRY[func.__name__] = entry
    return func


def get_registered_tools() -> dict[str, ToolEntry]:
    """Return a copy of the current tool registry."""
    return dict(_TOOL_REGISTRY)


def get_tool_callables() -> list[Callable]:
    """Return the list of registered tool callables for the SDK.

    These can be passed directly in the `tools` list alongside
    dict-based server-side tools. The google-genai SDK auto-generates
    JSON schema from type hints via FunctionDeclaration.from_callable().
    """
    return [entry.func for entry in _TOOL_REGISTRY.values()]


async def execute_tool_call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a registered tool by name with the given arguments.

    Returns a dict suitable for FunctionResponse output:
      {"result": ...} on success, {"error": ...} on failure.
    """
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        if entry.is_async:
            result = await entry.func(**(args or {}))
        else:
            result = await asyncio.to_thread(entry.func, **(args or {}))
    except Exception as e:
        logger.warning("Tool %s raised %s: %s", name, type(e).__name__, e)
        return {"error": f"{type(e).__name__}: {e}"}

    # Ensure the result is JSON-serializable
    if isinstance(result, str):
        return {"result": result}
    try:
        json.dumps(result)
        return {"result": result}
    except (TypeError, ValueError):
        return {"result": str(result)}


def clear_registry() -> None:
    """Clear all registered tools. Intended for testing."""
    _TOOL_REGISTRY.clear()


def enrich_file_search_tools(tools: list[dict[str, Any]]) -> str | None:
    """Inject file-search store IDs into file_search configs."""

    for index, tool in enumerate(tools):
        if "file_search" not in tool:
            continue
        if not GEMINI_FILE_SEARCH_STORE_IDS:
            return "File Search requires GEMINI_FILE_SEARCH_STORE_IDS to be set in your .env file."
        tools[index] = {
            "file_search": {
                "file_search_store_names": GEMINI_FILE_SEARCH_STORE_IDS.copy(),
            }
        }
    return None


def _resolve_tools_for_view(
    cog: Any,
    selected_values: list[str],
    custom_functions_selected: bool,
    conversation: Any,
) -> tuple[set[str], str | None]:
    """Resolve ButtonView tool selection into supported runtime tool config."""

    exclusive_error = check_mutually_exclusive_tools(set(selected_values))
    if exclusive_error:
        return set(), exclusive_error

    requested_tools = [deepcopy(AVAILABLE_TOOLS[name]) for name in selected_values]
    supported_tools, unsupported_tools = filter_supported_tools_for_model(
        conversation.params.model, requested_tools
    )
    supported_tools, incompatible_tools = filter_file_search_incompatible_tools(supported_tools)

    enrich_error = enrich_file_search_tools(supported_tools)
    if enrich_error:
        return set(), enrich_error

    combination_error = validate_builtin_custom_tool_combination(
        conversation.params.model,
        supported_tools,
        custom_functions_selected,
    )
    if combination_error:
        return set(), combination_error

    conversation.params.tools = supported_tools
    conversation.params.custom_functions_enabled = custom_functions_selected

    active_names: set[str] = {
        name for tool in supported_tools if (name := resolve_tool_name(tool)) is not None
    }
    if custom_functions_selected:
        active_names.add("custom_functions")

    parts: list[str] = []
    if active_names:
        parts.append(f"Tools updated: {', '.join(sorted(active_names))}.")
    else:
        parts.append("Tools disabled for this conversation.")
    if unsupported_tools:
        unsupported_text = ", ".join(sorted(set(unsupported_tools)))
        parts.append(f"Skipped for model `{conversation.params.model}`: {unsupported_text}.")
    if incompatible_tools:
        incompatible_text = ", ".join(sorted(set(incompatible_tools)))
        parts.append(f"Disabled (incompatible with file_search): {incompatible_text}.")

    return active_names, " ".join(parts)


# ---------------------------------------------------------------------------
# Starter tool definitions
# ---------------------------------------------------------------------------


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: IANA timezone name (e.g., "America/New_York", "Europe/London", "Asia/Tokyo").
    """
    try:
        tz = ZoneInfo(timezone)
    except KeyError:
        return f"Unknown timezone: {timezone}"
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


@tool
def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll one or more dice and return the results.

    Args:
        sides: Number of sides on each die (default 6).
        count: Number of dice to roll (default 1, max 100).
    """
    import random

    count = min(max(count, 1), 100)
    sides = max(sides, 2)
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    if count == 1:
        return f"Rolled a d{sides}: {total}"
    return f"Rolled {count}d{sides}: {rolls} (total: {total})"


__all__ = [
    "ToolEntry",
    "_resolve_tools_for_view",
    "clear_registry",
    "enrich_file_search_tools",
    "execute_tool_call",
    "get_registered_tools",
    "get_tool_callables",
    "tool",
]
