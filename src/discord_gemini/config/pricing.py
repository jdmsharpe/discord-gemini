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
_TTS: dict[str, dict[str, Any]] = _RAW.get("text_to_speech") or {}
_TOOLS: dict[str, dict[str, Any]] = _RAW.get("tools") or {}
_FALLBACKS: dict[str, dict[str, Any]] = _RAW.get("fallbacks") or {}


MODEL_PRICING: dict[str, tuple[float, float]] = {
    model_id: (float(cfg["input_per_million"]), float(cfg["output_per_million"]))
    for model_id, cfg in _MODELS.items()
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


TTS_PRICING: dict[str, tuple[float, float]] = {
    model_id: (float(cfg["input_per_million"]), float(cfg["output_per_million"]))
    for model_id, cfg in _TTS.items()
}


MAPS_GROUNDING_COST_PER_REQUEST: float = float(
    (_TOOLS.get("google_maps_grounding") or {}).get("per_request", 0.025)
)


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
UNKNOWN_TTS_MODEL_PRICING: tuple[float, float] = (
    _fallback("unknown_tts_model", "input_per_million", 0.50),
    _fallback("unknown_tts_model", "output_per_million", 10.00),
)


__all__ = [
    "IMAGE_PRICING",
    "MAPS_GROUNDING_COST_PER_REQUEST",
    "MODEL_PRICING",
    "TTS_PRICING",
    "UNKNOWN_CHAT_MODEL_PRICING",
    "UNKNOWN_IMAGE_MODEL_INPUT_RATE",
    "UNKNOWN_IMAGE_PER_IMAGE",
    "UNKNOWN_TTS_MODEL_PRICING",
    "UNKNOWN_VIDEO_PER_SECOND",
    "VIDEO_PRICING",
]
