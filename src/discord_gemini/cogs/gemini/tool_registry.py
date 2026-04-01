"""Centralized tool metadata and runtime config builders for Gemini chat."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ToolRuntimeBuilder = Callable[[], dict[str, Any] | None]


@dataclass(frozen=True)
class ToolMetadata:
    """Metadata describing one selectable tool in the app."""

    canonical_id: str
    label: str
    description: str
    default_enabled: bool = False
    model_allowlist: frozenset[str] | None = None
    mutually_exclusive_with: frozenset[str] = frozenset()
    file_search_incompatible: bool = False
    runtime_builder: ToolRuntimeBuilder | None = None


_TOOL_REGISTRY: dict[str, ToolMetadata] = {
    "google_search": ToolMetadata(
        canonical_id="google_search",
        label="Google Search",
        description="Ground answers with live web results.",
        runtime_builder=lambda: {"google_search": {}},
        file_search_incompatible=True,
        mutually_exclusive_with=frozenset({"google_maps"}),
    ),
    "code_execution": ToolMetadata(
        canonical_id="code_execution",
        label="Code Execution",
        description="Run Python code for calculations.",
        runtime_builder=lambda: {"code_execution": {}},
    ),
    "google_maps": ToolMetadata(
        canonical_id="google_maps",
        label="Google Maps",
        description="Ground answers with Maps place data.",
        model_allowlist=frozenset(
            {
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-lite-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
            }
        ),
        runtime_builder=lambda: {"google_maps": {}},
        file_search_incompatible=True,
        mutually_exclusive_with=frozenset({"google_search"}),
    ),
    "url_context": ToolMetadata(
        canonical_id="url_context",
        label="URL Context",
        description="Retrieve and analyze provided URLs.",
        model_allowlist=frozenset(
            {
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-lite-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
            }
        ),
        runtime_builder=lambda: {"url_context": {}},
        file_search_incompatible=True,
    ),
    "file_search": ToolMetadata(
        canonical_id="file_search",
        label="File Search",
        description="Search over uploaded document stores.",
        model_allowlist=frozenset(
            {
                "gemini-3.1-pro-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
            }
        ),
        runtime_builder=lambda: {"file_search": {}},
    ),
    "custom_functions": ToolMetadata(
        canonical_id="custom_functions",
        label="Custom Functions",
        description="Call Python tools (time, dice, etc.).",
        runtime_builder=None,
    ),
}


def get_tool_registry() -> dict[str, ToolMetadata]:
    """Return the canonical tool registry indexed by tool id."""

    return _TOOL_REGISTRY


def get_tool_metadata(tool_id: str) -> ToolMetadata | None:
    """Look up one tool by canonical id."""

    return _TOOL_REGISTRY.get(tool_id)


def iter_tool_registry(include_custom_functions: bool = True) -> list[ToolMetadata]:
    """Return ordered tool metadata for UI and selection logic."""

    return [
        meta
        for meta in _TOOL_REGISTRY.values()
        if include_custom_functions or meta.canonical_id != "custom_functions"
    ]


def build_runtime_tool_config(tool_id: str) -> dict[str, Any] | None:
    """Build runtime Gemini SDK payload for the given tool id."""

    meta = _TOOL_REGISTRY.get(tool_id)
    if meta is None or meta.runtime_builder is None:
        return None
    return meta.runtime_builder()
