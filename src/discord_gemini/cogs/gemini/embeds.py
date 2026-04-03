"""Embed builders for Gemini responses."""

from discord import Colour, Embed

from ...util import calculate_cost, chunk_text, truncate_text
from .models import ToolInfo

GEMINI_BLUE = Colour(0x4285F4)
ERROR_TRUNCATION_SUFFIX = "\n\n... (error message truncated)"


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

    used = sum(len(embed.title or "") + len(embed.description or "") for embed in embeds)
    available = max(500, 5500 - used)
    if len(response_text) > available:
        response_text = response_text[: available - 40] + "\n\n... [Response truncated]"

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
    for index, citation in enumerate(citations[:8], start=1):
        safe_title = truncate_text(citation["title"], 120)
        source_lines.append(f"{index}. [{safe_title}]({citation['uri']})")

    if url_context_sources:
        if source_lines:
            source_lines.append("")
        source_lines.append("**URL Context**")
        for source in url_context_sources[:6]:
            safe_url = truncate_text(source["retrieved_url"], 200)
            source_lines.append(f"- {safe_url} ({source['status']})")

    description = "\n".join(source_lines)
    queries = tool_info["search_queries"]
    if queries:
        query_preview = truncate_text(", ".join(queries[:3]), 500)
        description += f"\n\n**Queries:** {query_preview}"
    if tool_info["maps_widget_token"]:
        description += "\n\n**Maps Widget:** `google_maps_widget_context_token` returned."

    embeds.append(Embed(title="Sources", description=description, color=GEMINI_BLUE))


def append_pricing_embed(
    embeds: list[Embed],
    model: str,
    input_tokens: int,
    output_tokens: int,
    daily_cost: float,
    thinking_tokens: int = 0,
    google_maps_grounded: bool = False,
) -> None:
    """Append the compact pricing footer embed."""

    cost = calculate_cost(model, input_tokens, output_tokens, thinking_tokens, google_maps_grounded)
    parts = [f"${cost:.4f}"]
    if thinking_tokens > 0:
        parts.append(f"{input_tokens:,} in / {output_tokens:,} out / {thinking_tokens:,} thinking")
    else:
        parts.append(f"{input_tokens:,} tokens in / {output_tokens:,} tokens out")
    if google_maps_grounded:
        parts.append("Maps grounded")
    parts.append(f"daily ${daily_cost:.2f}")
    embeds.append(Embed(description=" · ".join(parts), color=GEMINI_BLUE))


__all__ = [
    "GEMINI_BLUE",
    "error_to_user_description",
    "append_pricing_embed",
    "append_response_embeds",
    "append_sources_embed",
    "append_thinking_embeds",
    "build_error_embed",
]
