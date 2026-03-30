"""Shared Gemini cog model types."""

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from ...util import ChatCompletionParameters


@dataclass
class Conversation:
    """Conversation state tracked across follow-up messages."""

    params: ChatCompletionParameters
    history: list[dict[str, Any]]


class CitationInfo(TypedDict):
    title: str
    uri: str


class UrlContextInfo(TypedDict):
    retrieved_url: str
    status: str


class ToolInfo(TypedDict):
    tools_used: list[str]
    citations: list[CitationInfo]
    search_queries: list[str]
    url_context_sources: list[UrlContextInfo]
    maps_widget_token: str | None


class PermissionAwareChannel(Protocol):
    def permissions_for(self, member: Any) -> Any: ...


__all__ = [
    "CitationInfo",
    "Conversation",
    "PermissionAwareChannel",
    "ToolInfo",
    "UrlContextInfo",
]
