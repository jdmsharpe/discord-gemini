"""Tests for the YAML-backed pricing loader."""

import importlib
import sys
import textwrap
from pathlib import Path


def _reload_pricing():
    for mod_name in ("discord_gemini.config.pricing",):
        sys.modules.pop(mod_name, None)
    return importlib.import_module("discord_gemini.config.pricing")


class TestPricingLoader:
    def test_bundled_yaml_loads_model_pricing(self):
        pricing = _reload_pricing()
        assert pricing.MODEL_PRICING["gemini-2.5-pro"] == (1.25, 10.0)
        assert pricing.MODEL_PRICING["gemini-2.0-flash"] == (0.10, 0.40)

    def test_image_pricing_preserves_none_key_for_default(self):
        pricing = _reload_pricing()
        input_rate, size_prices = pricing.IMAGE_PRICING["gemini-3.1-flash-image-preview"]
        assert input_rate == 0.50
        # Both None (legacy) and "default" should resolve to the same price.
        assert size_prices[None] == 0.067
        assert size_prices["1k"] == 0.067
        assert size_prices["2k"] == 0.101

    def test_imagen_models_have_zero_input_rate(self):
        pricing = _reload_pricing()
        input_rate, size_prices = pricing.IMAGE_PRICING["imagen-4.0-generate-001"]
        assert input_rate == 0.0
        assert size_prices[None] == 0.04

    def test_video_pricing_keyed_by_resolution(self):
        pricing = _reload_pricing()
        assert pricing.VIDEO_PRICING["veo-3.1-generate-preview"]["default"] == 0.40
        assert pricing.VIDEO_PRICING["veo-3.1-generate-preview"]["4k"] == 0.60

    def test_tts_and_maps_grounding(self):
        pricing = _reload_pricing()
        assert pricing.TTS_PRICING["gemini-2.5-flash-preview-tts"] == (0.50, 10.00)
        assert pricing.MAPS_GROUNDING_COST_PER_REQUEST == 0.025

    def test_fallback_constants(self):
        pricing = _reload_pricing()
        assert pricing.UNKNOWN_CHAT_MODEL_PRICING == (2.0, 12.0)
        assert pricing.UNKNOWN_IMAGE_MODEL_INPUT_RATE == 0.50
        assert pricing.UNKNOWN_IMAGE_PER_IMAGE == 0.067
        assert pricing.UNKNOWN_VIDEO_PER_SECOND == 0.35
        assert pricing.UNKNOWN_TTS_MODEL_PRICING == (0.50, 10.00)

    def test_env_var_override_path(self, monkeypatch, tmp_path: Path):
        custom_yaml = tmp_path / "custom-pricing.yaml"
        custom_yaml.write_text(
            textwrap.dedent(
                """
                models:
                  custom-gemini: { input_per_million: 1.0, output_per_million: 5.0 }
                image_generation:
                  custom-imagen:
                    input_per_million: 0.0
                    per_image_by_size: { default: 0.10 }
                video_generation:
                  custom-veo:
                    per_second_by_resolution: { default: 0.5 }
                text_to_speech:
                  custom-tts: { input_per_million: 0.2, output_per_million: 2.0 }
                tools:
                  google_maps_grounding: { per_request: 0.05 }
                fallbacks:
                  unknown_chat_model: { input_per_million: 9.9, output_per_million: 99.0 }
                """
            ).strip()
        )
        monkeypatch.setenv("GEMINI_PRICING_PATH", str(custom_yaml))

        pricing = _reload_pricing()

        assert pricing.MODEL_PRICING == {"custom-gemini": (1.0, 5.0)}
        input_rate, size_prices = pricing.IMAGE_PRICING["custom-imagen"]
        assert input_rate == 0.0
        assert size_prices[None] == 0.10
        assert pricing.VIDEO_PRICING == {"custom-veo": {"default": 0.5}}
        assert pricing.MAPS_GROUNDING_COST_PER_REQUEST == 0.05
        assert pricing.UNKNOWN_CHAT_MODEL_PRICING == (9.9, 99.0)
