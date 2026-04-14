"""Deep research helpers for the Gemini cog."""

import asyncio
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any

from discord import Colour, Embed, File
from discord.commands import ApplicationContext

from ...config.auth import GEMINI_FILE_SEARCH_STORE_IDS, SHOW_COST_EMBEDS
from ...util import ResearchParameters, calculate_cost, truncate_text
from . import embeds, state, usage
from .responses import APICallError, ValidationError

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
    """Return the newest text output emitted by an interaction, if any."""

    outputs = getattr(interaction, "outputs", None) or []
    for output in reversed(outputs):
        text = getattr(output, "text", None)
        if text:
            return str(text)
    return None


async def _run_deep_research(
    cog: "GeminiCog",
    research_params: ResearchParameters,
) -> tuple[str | None, int, int, int]:
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

    interaction = await cog.client.aio.interactions.create(**kwargs)
    cog.logger.info("Started deep research: %s", interaction.id)

    max_wait_time = 1200
    start_time = time.time()
    poll_interval = 15
    while interaction.status not in ("completed", "failed", "cancelled", "requires_action"):
        if time.time() - start_time > max_wait_time:
            raise TimeoutError("Deep research timed out after 20 minutes")
        await asyncio.sleep(poll_interval)
        interaction = await cog.client.aio.interactions.get(interaction.id)
        cog.logger.debug("Research %s status: %s", interaction.id, interaction.status)

    if interaction.status == "failed":
        raise APICallError(f"Research failed: {interaction.status}")
    if interaction.status == "cancelled":
        raise APICallError("Research was cancelled")
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
    input_tokens = usage_counts.input_tokens
    output_tokens = usage_counts.output_tokens
    thinking_tokens = usage_counts.thinking_tokens

    report_text = _extract_interaction_text(interaction)
    if report_text:
        return report_text, input_tokens, output_tokens, thinking_tokens

    return None, input_tokens, output_tokens, thinking_tokens


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


async def research_command(
    cog: "GeminiCog",
    ctx: ApplicationContext,
    prompt: str,
    file_search: bool = False,
    google_maps: bool = False,
    thinking_summaries: str | None = None,
) -> None:
    """Run the `/gemini research` command."""

    await ctx.defer()
    try:
        research_params = ResearchParameters(
            prompt=prompt,
            file_search=file_search,
            google_maps=google_maps,
            thinking_summaries=thinking_summaries,
        )
        report_text, input_tokens, output_tokens, thinking_tokens = await _run_deep_research(
            cog,
            research_params,
        )

        research_model = "gemini-3.1-pro-preview"
        cost = calculate_cost(research_model, input_tokens, output_tokens, thinking_tokens)
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "research",
            ctx.author.id,
            research_params.agent,
            cost,
            daily_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            file_search=file_search,
            google_maps=google_maps,
        )

        if report_text:
            response_embeds = _create_research_response_embeds(research_params)
            if SHOW_COST_EMBEDS:
                if thinking_tokens > 0:
                    pricing_desc = (
                        f"${cost:.2f} · {input_tokens:,} in / {output_tokens:,} out / "
                        f"{thinking_tokens:,} thinking · daily ${daily_cost:.2f}"
                    )
                else:
                    pricing_desc = (
                        f"${cost:.2f} · {input_tokens:,} in / {output_tokens:,} out · "
                        f"daily ${daily_cost:.2f}"
                    )
                response_embeds.append(Embed(description=pricing_desc, color=embeds.GEMINI_BLUE))

            report_file = File(BytesIO(report_text.encode("utf-8")), filename="research_report.md")
            await ctx.send_followup(embeds=response_embeds, file=report_file)
            return

        await ctx.send_followup(
            embed=Embed(
                title="No Research Results",
                description=(
                    "The research agent did not produce any output. Please try again with a "
                    "different prompt."
                ),
                color=Colour.orange(),
            )
        )
    except Exception as error:
        await cog._send_error_followup(ctx, error, "research")


__all__ = [
    "_create_research_response_embeds",
    "_run_deep_research",
    "research_command",
]
