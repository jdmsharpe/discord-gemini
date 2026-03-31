"""Helpers for normalizing usage metadata across Gemini API surfaces."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UsageCounts:
    """Normalized token counts extracted from Gemini SDK response objects."""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    tool_use_prompt_tokens: int = 0


def _coalesce_int(source: Any, *field_names: str) -> int:
    """Return the first non-None integer-ish field from an object."""

    for field_name in field_names:
        value = getattr(source, field_name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def extract_usage_counts(source: Any) -> UsageCounts:
    """Normalize usage fields from generate_content or interactions responses."""

    if source is None:
        return UsageCounts()

    usage = getattr(source, "usage_metadata", None)
    if usage is None:
        usage = getattr(source, "usage", None)
    if usage is None:
        usage = source

    return UsageCounts(
        input_tokens=_coalesce_int(usage, "prompt_token_count", "total_input_tokens"),
        output_tokens=_coalesce_int(
            usage,
            "response_token_count",
            "candidates_token_count",
            "total_output_tokens",
        ),
        thinking_tokens=_coalesce_int(usage, "thoughts_token_count", "total_thought_tokens"),
        cached_tokens=_coalesce_int(usage, "cached_content_token_count"),
        tool_use_prompt_tokens=_coalesce_int(usage, "tool_use_prompt_token_count"),
    )


__all__ = ["UsageCounts", "extract_usage_counts"]
