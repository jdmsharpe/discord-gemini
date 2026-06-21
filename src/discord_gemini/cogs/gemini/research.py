"""Deep research helpers for the Gemini cog."""

import asyncio
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from discord import Colour, Embed, File
from discord.commands import ApplicationContext

from ...config.auth import GEMINI_FILE_SEARCH_STORE_IDS, SHOW_COST_EMBEDS
from ...util import ResearchParameters, calculate_cost, truncate_text
from . import embeds, state, usage
from .embed_delivery import send_embed_batches
from .responses import APICallError, ValidationError

_SOURCES_HEADING_RE = re.compile(r"\*\*Sources:\*\*", re.IGNORECASE)
_SOURCES_LINE_RE = re.compile(r"^\s*\d+\.\s*\[([^\]]+)\]\(([^)]+)\)\s*$", re.MULTILINE)


@dataclass
class _ResearchResult:
    """Internal carrier for everything we extract from a research interaction."""

    report_text: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    annotations: list[Any] = field(default_factory=list)
    grounding_tool_counts: dict[str, int] = field(default_factory=dict)


if TYPE_CHECKING:
    from .cog import GeminiCog


def _build_deep_research_agent_config(
    research_params: ResearchParameters,
) -> dict[str, Any] | None:
    """Build an agent_config payload for deep research when advanced options are set."""

    agent_config: dict[str, Any] = {"type": "deep-research"}
    has_custom_config = False

    if research_params.collaborative_planning:
        agent_config["collaborative_planning"] = True
        has_custom_config = True
    if research_params.thinking_summaries is not None:
        agent_config["thinking_summaries"] = research_params.thinking_summaries
        has_custom_config = True
    if research_params.visualization is not None:
        agent_config["visualization"] = research_params.visualization
        has_custom_config = True

    return agent_config if has_custom_config else None


def _extract_interaction_text(interaction: Any) -> str | None:
    """Return the report text, preferring the SDK's ``Interaction.output_text``.

    SDK 2.3.0 added ``Interaction.output_text`` and 2.4.0 fixed it to handle turns
    that don't end with text — exactly the multi-step deep-research case this used to
    work around. We use it when it yields content and otherwise fall back to manually
    walking every text content item across all ``model_output`` steps (the original
    behavior), since deep research splits the report across multiple steps in the
    v1beta ``steps`` timeline and collecting only the last step drops the body.
    """

    output_text = getattr(interaction, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for content in getattr(step, "content", None) or []:
            if getattr(content, "type", None) != "text":
                continue
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks) if chunks else None


def _extract_interaction_annotations(interaction: Any) -> list[Any]:
    """Collect annotations from every text content item across model_output steps.

    Annotations live on each text content item inside model_output.content[]. They
    are a discriminated union (URLCitation | FileCitation | PlaceCitation) keyed on
    `.type`. We return them in document order so any future footnote-style rendering
    can use byte indices into the joined report text.
    """

    collected: list[Any] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for content in getattr(step, "content", None) or []:
            if getattr(content, "type", None) != "text":
                continue
            collected.extend(getattr(content, "annotations", None) or [])
    return collected


def _has_model_sources_footer(report_text: str | None) -> bool:
    """Return True when the model already emitted its own `**Sources:**` block."""

    return bool(report_text and _SOURCES_HEADING_RE.search(report_text))


def _model_footer_url_titles(report_text: str | None) -> dict[str, str]:
    """Parse `{url: label}` pairs out of the model's closing `**Sources:**` footer.

    The model labels each entry with a paper title or domain (e.g. `arxiv.org`),
    which is far more useful than the raw `URLCitation.title=None` we get back
    from grounding-redirect URLs. Returns an empty dict when no footer is found.
    """

    if not report_text:
        return {}
    match = _SOURCES_HEADING_RE.search(report_text)
    if not match:
        return {}
    tail = report_text[match.end() :]
    return {url: label for label, url in _SOURCES_LINE_RE.findall(tail)}


def _hostname_label(url: str | None) -> str:
    """Best-effort hostname extraction, falling back to `(untitled)`."""

    if not url:
        return "(untitled)"
    try:
        host = urlparse(url).hostname
    except ValueError:
        return "(untitled)"
    return host or "(untitled)"


def _url_citation_label(annotation: Any, footer_titles: dict[str, str]) -> str:
    """Best label for a `url_citation`: API title → model footer label → hostname.

    Gemini leaves `URLCitation.title` empty for grounding-redirect URLs, so we
    fall back to the labels the model itself emitted in its closing footer (when
    available), and finally to the URL's hostname.
    """

    title = getattr(annotation, "title", None)
    if title:
        return str(title)
    url = getattr(annotation, "url", None)
    if url and url in footer_titles:
        return footer_titles[url]
    return _hostname_label(url)


def _format_citations_section(annotations: list[Any]) -> str:
    """Render annotations into a markdown section appended to the research report.

    Each annotation has a `.type` of "url_citation", "file_citation", or
    "place_citation". For each kind, fall back gracefully when individual fields are
    missing — annotations are loosely typed and any field can be None.

    Returns an empty string when no annotations are present.
    """

    if not annotations:
        return ""

    url_lines: list[str] = []
    file_lines: list[str] = []
    place_lines: list[str] = []

    for annotation in annotations:
        kind = getattr(annotation, "type", None)
        if kind == "url_citation":
            title = _url_citation_label(annotation, {})
            url = getattr(annotation, "url", None)
            url_lines.append(f"- [{title}]({url})" if url else f"- {title}")
        elif kind == "file_citation":
            file_name = getattr(annotation, "file_name", None) or "(unnamed file)"
            page = getattr(annotation, "page_number", None)
            uri = getattr(annotation, "document_uri", None)
            label = f"{file_name}"
            if page is not None:
                label += f" (p. {page})"
            file_lines.append(f"- [{label}]({uri})" if uri else f"- {label}")
        elif kind == "place_citation":
            name = getattr(annotation, "name", None) or "(unnamed place)"
            url = getattr(annotation, "url", None)
            place_lines.append(f"- [{name}]({url})" if url else f"- {name}")

    sections: list[str] = []
    if url_lines:
        sections.append("### Web sources\n" + "\n".join(url_lines))
    if file_lines:
        sections.append("### Documents\n" + "\n".join(file_lines))
    if place_lines:
        sections.append("### Places\n" + "\n".join(place_lines))

    if not sections:
        return ""
    return "\n\n---\n\n## Sources\n\n" + "\n\n".join(sections) + "\n"


def _extract_grounding_tool_counts(interaction: Any) -> dict[str, int]:
    """Extract grounding-tool usage counts (google_search/google_maps/retrieval).

    The Interactions API exposes `usage.grounding_tool_count: list[GroundingToolCount]`
    with `count` and `type` fields. We collapse to a `{type: count}` map. Unknown or
    missing counts are skipped.
    """

    usage_obj = getattr(interaction, "usage", None)
    breakdown = getattr(usage_obj, "grounding_tool_count", None) or []
    result: dict[str, int] = {}
    for entry in breakdown:
        kind = getattr(entry, "type", None)
        count = getattr(entry, "count", None)
        if kind and count is not None:
            result[str(kind)] = result.get(str(kind), 0) + int(count)
    return result


async def _run_deep_research(
    cog: "GeminiCog",
    research_params: ResearchParameters,
) -> _ResearchResult:
    """Run a deep research task using the Interactions API."""

    kwargs: dict[str, Any] = {
        "input": research_params.prompt,
        "agent": research_params.agent,
        "background": True,
    }
    agent_config = _build_deep_research_agent_config(research_params)
    if agent_config is not None:
        kwargs["agent_config"] = agent_config
    tools: list[dict[str, Any]] = []

    if research_params.file_search:
        if not GEMINI_FILE_SEARCH_STORE_IDS:
            raise ValidationError(
                "File Search requires GEMINI_FILE_SEARCH_STORE_IDS to be set in your .env file."
            )
        tools.append(
            {
                "type": "file_search",
                "file_search_store_names": GEMINI_FILE_SEARCH_STORE_IDS.copy(),
            }
        )
    if research_params.google_maps:
        tools.append({"google_maps": {}})
    if tools:
        kwargs["tools"] = tools

    # google-genai 2.9.0 widened interactions.create/get to return
    # Interaction | AsyncStream[...] (no stream-vs-non-stream overloads). This path
    # never streams, so narrow to Any to keep .id/.status access well-typed.
    interaction: Any = await cog.client.aio.interactions.create(**kwargs)
    cog.logger.info("Started deep research: %s", interaction.id)

    max_wait_time = 1200
    start_time = time.time()
    poll_interval = 15
    while interaction.status not in (
        "completed",
        "failed",
        "cancelled",
        "requires_action",
        "budget_exceeded",
    ):
        if time.time() - start_time > max_wait_time:
            raise TimeoutError("Deep research timed out after 20 minutes")
        await asyncio.sleep(poll_interval)
        interaction = await cog.client.aio.interactions.get(interaction.id)
        cog.logger.debug("Research %s status: %s", interaction.id, interaction.status)

    if interaction.status == "failed":
        raise APICallError(f"Research failed: {interaction.status}")
    if interaction.status == "cancelled":
        raise APICallError("Research was cancelled")
    if interaction.status == "budget_exceeded":
        raise APICallError("Research stopped: token/cost budget exceeded.")
    if interaction.status == "requires_action":
        plan_preview = truncate_text(_extract_interaction_text(interaction), 600)
        message = (
            "Deep research returned a plan that requires confirmation before it can continue. "
            "This bot does not yet support confirming research plans in Discord."
        )
        if plan_preview:
            message += f"\n\nPlan preview:\n{plan_preview}"
        raise ValidationError(message)

    usage_counts = usage.extract_usage_counts(interaction)
    return _ResearchResult(
        report_text=_extract_interaction_text(interaction),
        input_tokens=usage_counts.input_tokens,
        output_tokens=usage_counts.output_tokens,
        thinking_tokens=usage_counts.thinking_tokens,
        annotations=_extract_interaction_annotations(interaction),
        grounding_tool_counts=_extract_grounding_tool_counts(interaction),
    )


def _create_research_response_embeds(research_params: ResearchParameters) -> list[Embed]:
    """Create the header embeds for a deep research report."""

    description = f"**Prompt:** {truncate_text(research_params.prompt, 2000)}\n"
    description += f"**Agent:** {research_params.agent}\n"
    if research_params.file_search:
        description += "**File Search:** Enabled\n"
    if research_params.google_maps:
        description += "**Google Maps:** Enabled\n"
    if research_params.thinking_summaries is not None:
        description += f"**Thinking Summaries:** {research_params.thinking_summaries}\n"
    if research_params.collaborative_planning:
        description += "**Collaborative Planning:** Enabled\n"
    if research_params.visualization is not None:
        description += f"**Visualization:** {research_params.visualization}\n"
    return [
        Embed(
            title="Deep Research",
            description=description,
            color=embeds.GEMINI_BLUE,
        )
    ]


def _build_citations_embed(annotations: list[Any], report_text: str | None = None) -> Embed | None:
    """Build a compact 'Sources' embed mirroring the chat grounding-citations format.

    Groups by annotation kind (URL, file, place) and caps each group so the embed
    description stays under Discord's 4096-char limit. Returns None when there are
    no surfaceable citations. `report_text` is used to recover URL labels from the
    model's own `**Sources:**` footer when the API leaves `URLCitation.title` empty.
    """

    if not annotations:
        return None

    footer_titles = _model_footer_url_titles(report_text)
    url_lines: list[str] = []
    file_lines: list[str] = []
    place_lines: list[str] = []

    for annotation in annotations:
        kind = getattr(annotation, "type", None)
        if kind == "url_citation":
            title = truncate_text(_url_citation_label(annotation, footer_titles), 120) or ""
            url = getattr(annotation, "url", None)
            url_lines.append(f"[{title}]({url})" if url else title)
        elif kind == "file_citation":
            file_name = getattr(annotation, "file_name", None) or "(unnamed file)"
            page = getattr(annotation, "page_number", None)
            uri = getattr(annotation, "document_uri", None)
            label = f"{file_name}" if page is None else f"{file_name} (p. {page})"
            label = truncate_text(label, 160) or label
            file_lines.append(f"[{label}]({uri})" if uri else label)
        elif kind == "place_citation":
            name = truncate_text(getattr(annotation, "name", None) or "(unnamed place)", 120) or ""
            url = getattr(annotation, "url", None)
            place_lines.append(f"[{name}]({url})" if url else name)

    sections: list[str] = []
    if url_lines:
        numbered = [f"{i}. {entry}" for i, entry in enumerate(url_lines[:8], start=1)]
        if len(url_lines) > 8:
            numbered.append(f"_…and {len(url_lines) - 8} more_")
        sections.append("**Web sources**\n" + "\n".join(numbered))
    if file_lines:
        numbered = [f"{i}. {entry}" for i, entry in enumerate(file_lines[:8], start=1)]
        if len(file_lines) > 8:
            numbered.append(f"_…and {len(file_lines) - 8} more_")
        sections.append("**Documents**\n" + "\n".join(numbered))
    if place_lines:
        numbered = [f"{i}. {entry}" for i, entry in enumerate(place_lines[:8], start=1)]
        if len(place_lines) > 8:
            numbered.append(f"_…and {len(place_lines) - 8} more_")
        sections.append("**Places**\n" + "\n".join(numbered))

    if not sections:
        return None
    return Embed(
        title="Sources",
        description=truncate_text("\n\n".join(sections), 4000),
        color=embeds.GEMINI_BLUE,
    )


def _format_grounding_breakdown(grounding_tool_counts: dict[str, int]) -> str:
    """Render grounding tool counts as a compact ' · '-joined fragment for the cost embed."""

    if not grounding_tool_counts:
        return ""
    pretty_names = {
        "google_search": "search",
        "google_maps": "maps",
        "retrieval": "file search",
    }
    parts = [
        f"{pretty_names.get(kind, kind)}: {count}"
        for kind, count in sorted(grounding_tool_counts.items())
    ]
    return " · ".join(parts)


async def research_command(
    cog: "GeminiCog",
    ctx: ApplicationContext,
    prompt: str,
    agent: str = "deep-research-preview-04-2026",
    file_search: bool = False,
    google_maps: bool = False,
    thinking_summaries: Literal["auto", "none"] | None = None,
) -> None:
    """Run the `/gemini-tools research` command."""

    await ctx.defer()
    try:
        research_params = ResearchParameters(
            prompt=prompt,
            agent=agent,
            file_search=file_search,
            google_maps=google_maps,
            thinking_summaries=thinking_summaries,
        )
        result = await _run_deep_research(cog, research_params)

        research_model = "gemini-3.1-pro-preview"
        cost = calculate_cost(
            research_model, result.input_tokens, result.output_tokens, result.thinking_tokens
        )
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "research",
            ctx.author.id,
            research_params.agent,
            cost,
            daily_cost,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thinking_tokens=result.thinking_tokens,
            file_search=file_search,
            google_maps=google_maps,
            grounding_tool_counts=result.grounding_tool_counts or None,
        )

        if result.report_text:
            response_embeds = _create_research_response_embeds(research_params)
            citations_embed = _build_citations_embed(result.annotations, result.report_text)
            if citations_embed is not None:
                response_embeds.append(citations_embed)
            if SHOW_COST_EMBEDS:
                pricing_parts = [f"${cost:.2f}"]
                if result.thinking_tokens > 0:
                    pricing_parts.append(
                        f"{result.input_tokens:,} in / {result.output_tokens:,} out / "
                        f"{result.thinking_tokens:,} thinking"
                    )
                else:
                    pricing_parts.append(
                        f"{result.input_tokens:,} in / {result.output_tokens:,} out"
                    )
                grounding_fragment = _format_grounding_breakdown(result.grounding_tool_counts)
                if grounding_fragment:
                    pricing_parts.append(grounding_fragment)
                pricing_parts.append(f"daily ${daily_cost:.2f}")
                response_embeds.append(
                    Embed(description=" · ".join(pricing_parts), color=embeds.GEMINI_BLUE)
                )

            report_body = result.report_text
            if not _has_model_sources_footer(result.report_text):
                report_body += _format_citations_section(result.annotations)
            report_file = File(BytesIO(report_body.encode("utf-8")), filename="research_report.md")
            await send_embed_batches(
                ctx.send_followup,
                embeds=response_embeds,
                file=report_file,
                logger=cog.logger,
            )
            return

        await send_embed_batches(
            ctx.send_followup,
            embed=Embed(
                title="No Research Results",
                description=(
                    "The research agent did not produce any output. Please try again with a "
                    "different prompt."
                ),
                color=Colour.orange(),
            ),
            logger=cog.logger,
        )
    except Exception as error:
        await cog._send_error_followup(ctx, error, "research")


__all__ = [
    "_ResearchResult",
    "_build_citations_embed",
    "_create_research_response_embeds",
    "_extract_grounding_tool_counts",
    "_extract_interaction_annotations",
    "_format_citations_section",
    "_format_grounding_breakdown",
    "_has_model_sources_footer",
    "_hostname_label",
    "_model_footer_url_titles",
    "_run_deep_research",
    "_url_citation_label",
    "research_command",
]
