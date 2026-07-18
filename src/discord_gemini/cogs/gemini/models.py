"""Shared Gemini cog model types."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, TypedDict

from ...util import ChatCompletionParameters


@dataclass
class Conversation:
    """Conversation state tracked across follow-up messages."""

    params: ChatCompletionParameters
    history: list[dict[str, Any]]
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)


class CitationInfo(TypedDict):
    title: str
    uri: str


class UrlContextInfo(TypedDict):
    retrieved_url: str
    display_name: str
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
