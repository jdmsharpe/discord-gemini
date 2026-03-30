"""Response parsing helpers and Gemini-specific exceptions."""

from typing import Any

from google.genai import types

from .models import CitationInfo, ToolInfo, UrlContextInfo


class GeminiBotError(Exception):
    """Base exception for Gemini bot errors."""


class APICallError(GeminiBotError):
    """Raised when an API call to Gemini fails."""


class CacheError(GeminiBotError):
    """Raised when cache operations fail."""


class FileUploadError(GeminiBotError):
    """Raised when file upload fails."""


class ValidationError(GeminiBotError):
    """Raised when input validation fails."""


class MusicGenerationError(GeminiBotError):
    """Raised when music generation fails."""


def extract_thinking_text(response: Any) -> str:
    """Extract thought summary text from a Gemini response."""

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return ""

    thinking_parts = []
    for part in parts:
        if getattr(part, "thought", False) and getattr(part, "text", None):
            thinking_parts.append(part.text)
    return "\n\n".join(thinking_parts)


def _get_response_content_parts(response: Any) -> list[Any] | None:
    """Get raw content parts for history storage."""

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    if content is None:
        return None
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    return list(parts)


def _build_thinking_config(
    thinking_level: str | None,
    thinking_budget: int | None,
) -> types.ThinkingConfig | None:
    """Build a ThinkingConfig from user parameters."""

    if thinking_level is None and thinking_budget is None:
        return None

    kwargs: dict[str, Any] = {"include_thoughts": True}
    if thinking_level is not None:
        kwargs["thinking_level"] = thinking_level
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    return types.ThinkingConfig(**kwargs)


def extract_tool_info(response: Any) -> ToolInfo:
    """Extract tool usage and citation data from a Gemini response."""

    tool_info: ToolInfo = {
        "tools_used": [],
        "citations": [],
        "search_queries": [],
        "url_context_sources": [],
        "maps_widget_token": None,
    }
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return tool_info

    candidate = candidates[0]
    grounding_metadata = getattr(candidate, "grounding_metadata", None)

    search_used = False
    maps_used = False
    if grounding_metadata is not None:
        web_search_queries = getattr(grounding_metadata, "web_search_queries", None)
        if web_search_queries:
            tool_info["search_queries"] = [
                query for query in web_search_queries if isinstance(query, str) and query
            ]

        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
        seen_uris: set[str] = set()
        citations: list[CitationInfo] = []
        for chunk in grounding_chunks:
            web_chunk = getattr(chunk, "web", None)
            maps_chunk = getattr(chunk, "maps", None)

            if web_chunk is not None:
                uri = getattr(web_chunk, "uri", None)
                if uri and uri not in seen_uris:
                    title = getattr(web_chunk, "title", None) or uri
                    citations.append({"title": title, "uri": uri})
                    seen_uris.add(uri)
                    search_used = True

            if maps_chunk is not None:
                uri = getattr(maps_chunk, "uri", None)
                if uri and uri not in seen_uris:
                    title = getattr(maps_chunk, "title", None) or uri
                    citations.append({"title": title, "uri": uri})
                    seen_uris.add(uri)
                maps_used = True

        if citations:
            tool_info["citations"] = citations

        if getattr(grounding_metadata, "search_entry_point", None) is not None:
            search_used = True

        maps_widget_token = getattr(grounding_metadata, "google_maps_widget_context_token", None)
        if maps_widget_token:
            tool_info["maps_widget_token"] = str(maps_widget_token)
            maps_used = True

        if tool_info["search_queries"]:
            search_used = True

    if search_used:
        tool_info["tools_used"].append("google_search")
    if maps_used:
        tool_info["tools_used"].append("google_maps")

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if parts:
        code_execution_used = any(
            getattr(part, "executable_code", None) or getattr(part, "code_execution_result", None)
            for part in parts
        )
        if code_execution_used:
            tool_info["tools_used"].append("code_execution")

    url_context_metadata = getattr(candidate, "url_context_metadata", None)
    if url_context_metadata is not None:
        url_metadata_entries = getattr(url_context_metadata, "url_metadata", None) or []
        parsed_sources: list[UrlContextInfo] = []
        for entry in url_metadata_entries:
            retrieved_url = getattr(entry, "retrieved_url", None)
            if not retrieved_url:
                continue
            status = getattr(entry, "url_retrieval_status", None)
            parsed_sources.append(
                {
                    "retrieved_url": str(retrieved_url),
                    "status": str(status) if status is not None else "UNKNOWN",
                }
            )

        if parsed_sources:
            tool_info["url_context_sources"] = parsed_sources
            tool_info["tools_used"].append("url_context")

    retrieval_metadata = getattr(candidate, "retrieval_metadata", None)
    if retrieval_metadata is not None:
        tool_info["tools_used"].append("file_search")
    elif grounding_metadata is not None and not search_used and not maps_used:
        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
        if grounding_chunks:
            tool_info["tools_used"].append("file_search")

    return tool_info


__all__ = [
    "APICallError",
    "CacheError",
    "FileUploadError",
    "GeminiBotError",
    "MusicGenerationError",
    "ValidationError",
    "_build_thinking_config",
    "_get_response_content_parts",
    "extract_thinking_text",
    "extract_tool_info",
]
