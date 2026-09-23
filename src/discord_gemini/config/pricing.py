"""Load Gemini model pricing from pricing.yaml.

The YAML file ships with the package so pricing is always available. Set the
``GEMINI_PRICING_PATH`` environment variable to point at a different YAML file
for runtime overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _resolve_pricing_path() -> Path:
    override = os.getenv("GEMINI_PRICING_PATH")
    if override:
        return Path(override)
    return Path(__file__).with_name("pricing.yaml")


def _load_raw() -> dict[str, Any]:
    path = _resolve_pricing_path()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a YAML mapping at the top level.")
    return data


_RAW: dict[str, Any] = _load_raw()
_MODELS: dict[str, dict[str, Any]] = _RAW.get("models") or {}
_IMAGE: dict[str, dict[str, Any]] = _RAW.get("image_generation") or {}
_VIDEO: dict[str, dict[str, Any]] = _RAW.get("video_generation") or {}
_VIDEO_TOKENIZED: dict[str, dict[str, Any]] = _RAW.get("video_tokenized") or {}
_TTS: dict[str, dict[str, Any]] = _RAW.get("text_to_speech") or {}
_MUSIC: dict[str, dict[str, Any]] = _RAW.get("music_generation") or {}
_TOOLS: dict[str, dict[str, Any]] = _RAW.get("tools") or {}
_FALLBACKS: dict[str, dict[str, Any]] = _RAW.get("fallbacks") or {}


MODEL_PRICING: dict[str, tuple[float, float]] = {
    model_id: (float(cfg["input_per_million"]), float(cfg["output_per_million"]))
    for model_id, cfg in _MODELS.items()
}


# Rate for input tokens served from a context cache. Rows with no
# cached_input_per_million are left out so `calculate_cost` bills their cache
# hits at the full input rate instead of an invented discount.
CACHED_INPUT_PRICING: dict[str, float] = {
    model_id: float(cfg["cached_input_per_million"])
    for model_id, cfg in _MODELS.items()
    if cfg.get("cached_input_per_million") is not None
}


def _build_image_pricing() -> dict[str, tuple[float, dict[str | None, float]]]:
    """Build IMAGE_PRICING. YAML 'default' key maps to Python None for legacy lookup."""
    result: dict[str, tuple[float, dict[str | None, float]]] = {}
    for model_id, cfg in _IMAGE.items():
        input_rate = float(cfg.get("input_per_million", 0.0))
        by_size_raw: dict[str, Any] = cfg.get("per_image_by_size") or {}
        by_size: dict[str | None, float] = {}
        for size_key, price in by_size_raw.items():
            normalized_key: str | None = None if size_key == "default" else str(size_key)
            by_size[normalized_key] = float(price)
            if size_key == "default":
                # also store the literal string so a future caller passing "default" works
                by_size["default"] = float(price)
        result[model_id] = (input_rate, by_size)
    return result


IMAGE_PRICING: dict[str, tuple[float, dict[str | None, float]]] = _build_image_pricing()


VIDEO_PRICING: dict[str, dict[str, float]] = {
    model_id: {
        size: float(price) for size, price in (cfg.get("per_second_by_resolution") or {}).items()
    }
    for model_id, cfg in _VIDEO.items()
}


# Per-video-output-token pricing for Interactions-API video models (Gemini Omni).
VIDEO_TOKEN_PRICING: dict[str, float] = {
    model_id: float(cfg["video_output_per_million"]) for model_id, cfg in _VIDEO_TOKENIZED.items()
}


TTS_PRICING: dict[str, tuple[float, float]] = {
    model_id: (float(cfg["input_per_million"]), float(cfg["output_per_million"]))
    for model_id, cfg in _TTS.items()
}


# Per-song pricing for music models. A ``None`` value marks a model with no
# published per-song price (lyria-realtime-exp streams audio), so callers must
# report it as unpriced instead of falling back to a made-up rate.
MUSIC_PRICING: dict[str, float | None] = {
    model_id: (None if cfg.get("per_song") is None else float(cfg["per_song"]))
    for model_id, cfg in _MUSIC.items()
}


_MAPS_GROUNDING: dict[str, Any] = _TOOLS.get("google_maps_grounding") or {}

# Maps grounding surcharge per grounded prompt, keyed by model-id prefix
# ("gemini-3" -> $14/1K, "gemini-2.5" -> $25/1K, both after untracked free tiers).
MAPS_GROUNDING_COST_BY_MODEL_PREFIX: dict[str, float] = {
    str(prefix): float(rate)
    for prefix, rate in (_MAPS_GROUNDING.get("per_request_by_model_prefix") or {}).items()
}
# Fallback for model ids that match no prefix above.
MAPS_GROUNDING_COST_PER_REQUEST: float = float(_MAPS_GROUNDING.get("per_request", 0.025))


def maps_grounding_cost_for_model(model: str) -> float:
    """Per-request Maps grounding surcharge for ``model``; longest matching prefix wins."""
    matches = [prefix for prefix in MAPS_GROUNDING_COST_BY_MODEL_PREFIX if model.startswith(prefix)]
    if not matches:
        return MAPS_GROUNDING_COST_PER_REQUEST
    return MAPS_GROUNDING_COST_BY_MODEL_PREFIX[max(matches, key=len)]


_SEARCH_GROUNDING: dict[str, Any] = _TOOLS.get("google_search_grounding") or {}

# Google Search grounding charge, keyed by model-id prefix. Per-query prefixes
# ("gemini-3" -> $14/1K search queries) bill each search query the model ran;
# per-prompt prefixes ("gemini-2.5" -> $35/1K) bill each grounded prompt. Both
# ignore the free allowances, so the charge is an upper bound.
SEARCH_GROUNDING_COST_PER_QUERY_BY_MODEL_PREFIX: dict[str, float] = {
    str(prefix): float(rate)
    for prefix, rate in (_SEARCH_GROUNDING.get("per_query_by_model_prefix") or {}).items()
}
SEARCH_GROUNDING_COST_PER_PROMPT_BY_MODEL_PREFIX: dict[str, float] = {
    str(prefix): float(rate)
    for prefix, rate in (_SEARCH_GROUNDING.get("per_prompt_by_model_prefix") or {}).items()
}
# Per-prompt fallback for model ids that match no prefix above.
SEARCH_GROUNDING_COST_PER_PROMPT: float = float(_SEARCH_GROUNDING.get("per_prompt", 0.035))


def search_grounding_cost_for_model(
    model: str, search_queries: int, grounded_prompts: int
) -> float:
    """Google Search grounding charge for ``model``.

    The longest matching prefix across both prefix maps picks the unit: a per-query
    prefix bills ``search_queries``, and a per-prompt prefix (or no matching prefix,
    at ``SEARCH_GROUNDING_COST_PER_PROMPT``) bills ``grounded_prompts``.
    """
    per_query = SEARCH_GROUNDING_COST_PER_QUERY_BY_MODEL_PREFIX
    per_prompt = SEARCH_GROUNDING_COST_PER_PROMPT_BY_MODEL_PREFIX
    query_prefix = max((p for p in per_query if model.startswith(p)), key=len, default="")
    prompt_prefix = max((p for p in per_prompt if model.startswith(p)), key=len, default="")
    if query_prefix and len(query_prefix) >= len(prompt_prefix):
        return max(search_queries, 0) * per_query[query_prefix]
    rate = per_prompt[prompt_prefix] if prompt_prefix else SEARCH_GROUNDING_COST_PER_PROMPT
    return max(grounded_prompts, 0) * rate


def _fallback(key: str, field: str, default: float) -> float:
    value = (_FALLBACKS.get(key) or {}).get(field)
    return float(value) if value is not None else default


UNKNOWN_CHAT_MODEL_PRICING: tuple[float, float] = (
    _fallback("unknown_chat_model", "input_per_million", 2.0),
    _fallback("unknown_chat_model", "output_per_million", 12.0),
)
UNKNOWN_IMAGE_MODEL_INPUT_RATE: float = _fallback("unknown_image_model", "input_per_million", 0.50)
UNKNOWN_IMAGE_PER_IMAGE: float = _fallback("unknown_image_model", "per_image", 0.067)
UNKNOWN_VIDEO_PER_SECOND: float = _fallback("unknown_video_model", "per_second", 0.35)
UNKNOWN_VIDEO_TOKEN_PER_MILLION: float = _fallback(
    "unknown_video_tokenized_model", "video_output_per_million", 17.50
)
UNKNOWN_TTS_MODEL_PRICING: tuple[float, float] = (
    _fallback("unknown_tts_model", "input_per_million", 0.50),
    _fallback("unknown_tts_model", "output_per_million", 10.00),
)


__all__ = [
    "CACHED_INPUT_PRICING",
    "IMAGE_PRICING",
    "MAPS_GROUNDING_COST_BY_MODEL_PREFIX",
    "MAPS_GROUNDING_COST_PER_REQUEST",
    "MODEL_PRICING",
    "MUSIC_PRICING",
    "SEARCH_GROUNDING_COST_PER_PROMPT",
    "SEARCH_GROUNDING_COST_PER_PROMPT_BY_MODEL_PREFIX",
    "SEARCH_GROUNDING_COST_PER_QUERY_BY_MODEL_PREFIX",
    "TTS_PRICING",
    "UNKNOWN_CHAT_MODEL_PRICING",
    "UNKNOWN_IMAGE_MODEL_INPUT_RATE",
    "UNKNOWN_IMAGE_PER_IMAGE",
    "UNKNOWN_TTS_MODEL_PRICING",
    "UNKNOWN_VIDEO_PER_SECOND",
    "UNKNOWN_VIDEO_TOKEN_PER_MILLION",
    "VIDEO_PRICING",
    "VIDEO_TOKEN_PRICING",
    "maps_grounding_cost_for_model",
    "search_grounding_cost_for_model",
]
