"""Embed builders for Gemini responses."""

from discord import Colour, Embed

from ...cost_line import count_label, format_cost_line
from ...util import calculate_cost, chunk_text, truncate_text
from .models import ToolInfo

GEMINI_BLUE = Colour(0x4285F4)
ERROR_TRUNCATION_SUFFIX = "\n\n... (error message truncated)"


def fit_markdown_sections(
    sections: list[tuple[str | None, list[str]]],
    max_length: int = 4000,
) -> str:
    """Fit complete Markdown entries without slicing through links."""

    rendered_sections: list[str] = []
    for heading, entries in sections:
        accepted: list[str] = []
        for entry in entries:
            body = "\n".join([*accepted, entry])
            rendered = f"{heading}\n{body}" if heading else body
            candidate = "\n\n".join([*rendered_sections, rendered])
            if len(candidate) > max_length:
                break
            accepted.append(entry)
        if accepted:
            body = "\n".join(accepted)
            rendered_sections.append(f"{heading}\n{body}" if heading else body)
    return "\n\n".join(rendered_sections)


def build_error_embed(description: str) -> Embed:
    """Create a red error embed."""

    return Embed(title="Error", description=description, color=Colour.red())


def error_to_user_description(error: BaseException | str, max_length: int = 4000) -> str:
    """Normalize an error into safe embed description text."""

    description = error if isinstance(error, str) else str(error)
    if not description:
        return "An unexpected error occurred."
    if len(description) <= max_length:
        return description
    if max_length <= len(ERROR_TRUNCATION_SUFFIX):
        return ERROR_TRUNCATION_SUFFIX[:max_length]
    return description[: max_length - len(ERROR_TRUNCATION_SUFFIX)] + ERROR_TRUNCATION_SUFFIX


def append_response_embeds(embeds: list[Embed], response_text: str) -> None:
    """Append response chunks while respecting Discord embed limits."""

    for index, chunk in enumerate(chunk_text(response_text, 3500), start=1):
        title = "Response" if index == 1 else f"Response (Part {index})"
        embeds.append(Embed(title=title, description=chunk, color=GEMINI_BLUE))


def append_thinking_embeds(embeds: list[Embed], thinking_text: str) -> None:
    """Append a spoilered thinking summary."""

    if not thinking_text:
        return

    if len(thinking_text) > 3500:
        thinking_text = thinking_text[:3450] + "\n\n... [thinking truncated]"

    embeds.append(
        Embed(
            title="Thinking",
            description=f"||{thinking_text}||",
            color=Colour.light_grey(),
        )
    )


def append_sources_embed(embeds: list[Embed], tool_info: ToolInfo) -> None:
    """Append a compact sources embed for grounded responses."""

    citations = tool_info["citations"]
    url_context_sources = tool_info["url_context_sources"]
    if (not citations and not url_context_sources) or len(embeds) >= 10:
        return

    source_lines: list[str] = []
    seen_urls: set[str] = set()
    source_index = 1
    for citation in citations[:8]:
        safe_title = truncate_text(citation["title"], 120)
        source_lines.append(f"{source_index}. [{safe_title}]({citation['uri']})")
        seen_urls.add(citation["uri"])
        source_index += 1

    for source in url_context_sources[:6]:
        url = source["retrieved_url"]
        if url in seen_urls:
            continue
        safe_title = truncate_text(source["display_name"], 120)
        source_lines.append(f"{source_index}. [{safe_title}]({url})")
        seen_urls.add(url)
        source_index += 1

    sections: list[tuple[str | None, list[str]]] = [(None, source_lines)]
    queries = tool_info["search_queries"]
    if queries:
        query_preview = truncate_text(", ".join(queries[:3]), 500)
        sections.append((None, [f"**Queries:** {query_preview}"]))
    if tool_info["maps_widget_token"]:
        sections.append((None, ["**Maps Widget:** `google_maps_widget_context_token` returned."]))

    description = fit_markdown_sections(sections)
    if description:
        embeds.append(Embed(title="Sources", description=description, color=GEMINI_BLUE))


def append_pricing_embed(
    embeds: list[Embed],
    model: str,
    input_tokens: int,
    output_tokens: int,
    daily_cost: float,
    thinking_tokens: int = 0,
    google_maps_grounded: bool = False,
    cached_tokens: int = 0,
    google_search_queries: int = 0,
    google_search_grounded: bool | int = False,
) -> None:
    """Append the one-line pricing embed.

    ``output_tokens`` excludes thinking tokens (Gemini reports them apart), so the
    line's output count is their sum. ``cached_tokens`` is already part of
    ``input_tokens``.
    """

    cost = calculate_cost(
        model,
        input_tokens,
        output_tokens,
        thinking_tokens,
        google_maps_grounded,
        cached_tokens=cached_tokens,
        google_search_queries=google_search_queries,
        google_search_grounded=google_search_grounded,
    )
    details: list[str] = []
    if google_search_queries > 0:
        details.append(count_label(google_search_queries, "search", "searches"))
    if google_maps_grounded:
        details.append("maps grounded")
    line = format_cost_line(
        cost,
        daily_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens + thinking_tokens,
        cached_tokens=cached_tokens,
        thinking_tokens=thinking_tokens,
        details=details,
    )
    embeds.append(Embed(description=line, color=GEMINI_BLUE))


__all__ = [
    "GEMINI_BLUE",
    "append_pricing_embed",
    "append_response_embeds",
    "append_sources_embed",
    "append_thinking_embeds",
    "build_error_embed",
    "error_to_user_description",
    "fit_markdown_sections",
]
