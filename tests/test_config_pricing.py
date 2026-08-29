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
        assert pricing.MODEL_PRICING["gemini-3.7-flash"] == (0.75, 3.75)
        assert pricing.MODEL_PRICING["gemini-3.6-flash"] == (0.75, 3.75)
        assert pricing.MODEL_PRICING["gemini-3.5-flash-lite"] == (0.30, 2.50)
        assert pricing.MODEL_PRICING["gemini-2.5-pro"] == (1.25, 10.0)
        assert pricing.MODEL_PRICING["gemini-2.5-flash-lite"] == (0.10, 0.40)

    def test_image_pricing_preserves_none_key_for_default(self):
        pricing = _reload_pricing()
        input_rate, size_prices = pricing.IMAGE_PRICING["gemini-3.1-flash-image"]
        assert input_rate == 0.50
        # Both None (legacy) and "default" should resolve to the same price.
        assert size_prices[None] == 0.067
        assert size_prices["1k"] == 0.067
        assert size_prices["2k"] == 0.101

    def test_lite_image_pricing_has_no_2k_tier(self):
        """Lite generates 1K only, so pricing it for 2K (or 512/4K, which 400 the same
        way — probed 2026-08-28) would imply a size the API rejects."""
        pricing = _reload_pricing()
        input_rate, size_prices = pricing.IMAGE_PRICING["gemini-3.1-flash-lite-image"]
        assert input_rate == 0.25
        assert size_prices[None] == 0.0336
        assert size_prices["1k"] == 0.0336
        assert "2k" not in size_prices
        assert set(size_prices) == {None, "default", "1k"}

    def test_gemini_2_5_pricing_only_has_its_fixed_1k_tier(self):
        """Gemini 2.5 silently renders 1K for larger requests, so only 1K is priced."""
        pricing = _reload_pricing()
        input_rate, size_prices = pricing.IMAGE_PRICING["gemini-2.5-flash-image"]
        assert input_rate == 0.30
        assert size_prices[None] == 0.039
        assert size_prices["1k"] == 0.039
        assert set(size_prices) == {None, "default", "1k"}

    def test_flash_and_pro_image_pricing_carry_the_probed_512_and_4k_rows(self):
        """Flash Image 512 = $0.045 and 4K = $0.151; Pro 4K = $0.24 (probed 2026-08-28).
        The 512 key is quoted in the YAML so it loads as the string the lookup uses."""
        pricing = _reload_pricing()
        _, flash = pricing.IMAGE_PRICING["gemini-3.1-flash-image"]
        assert flash["512"] == 0.045
        assert flash["4k"] == 0.151
        assert flash["1k"] == 0.067 and flash["2k"] == 0.101
        _, pro = pricing.IMAGE_PRICING["gemini-3-pro-image"]
        assert pro["4k"] == 0.24
        assert "512" not in pro

    def test_video_pricing_keyed_by_resolution(self):
        pricing = _reload_pricing()
        assert pricing.VIDEO_PRICING["veo-3.1-generate-preview"]["default"] == 0.40
        assert pricing.VIDEO_PRICING["veo-3.1-generate-preview"]["4k"] == 0.60

    def test_music_pricing_is_per_song(self):
        pricing = _reload_pricing()
        assert pricing.MUSIC_PRICING["lyria-3-clip-preview"] == 0.04
        assert pricing.MUSIC_PRICING["lyria-3-pro-preview"] == 0.08

    def test_music_pricing_keeps_realtime_explicitly_unpriced(self):
        """lyria-realtime-exp streams audio and has no published per-song price."""
        pricing = _reload_pricing()
        assert "lyria-realtime-exp" in pricing.MUSIC_PRICING
        assert pricing.MUSIC_PRICING["lyria-realtime-exp"] is None

    def test_every_chat_row_declares_a_cached_input_rate_below_its_input_rate(self):
        """Cache hits were billed at the full input rate; every row now carries a cheaper one."""
        pricing = _reload_pricing()
        assert set(pricing.CACHED_INPUT_PRICING) == set(pricing.MODEL_PRICING)
        for model, (input_rate, _output_rate) in pricing.MODEL_PRICING.items():
            cached_rate = pricing.CACHED_INPUT_PRICING[model]
            assert 0 < cached_rate < input_rate, model

    def test_cached_input_rates_match_the_live_page(self):
        pricing = _reload_pricing()
        assert pricing.CACHED_INPUT_PRICING["gemini-3.7-flash"] == 0.075
        assert pricing.CACHED_INPUT_PRICING["gemini-3.1-pro-preview"] == 0.20
        assert pricing.CACHED_INPUT_PRICING["gemini-2.5-pro"] == 0.125
        assert pricing.CACHED_INPUT_PRICING["gemini-2.5-flash-lite"] == 0.01

    def test_tts_and_maps_grounding(self):
        pricing = _reload_pricing()
        assert pricing.TTS_PRICING["gemini-2.5-flash-preview-tts"] == (0.50, 10.00)
        assert pricing.MAPS_GROUNDING_COST_BY_MODEL_PREFIX == {
            "gemini-3": 0.014,
            "gemini-2.5": 0.025,
        }
        assert pricing.MAPS_GROUNDING_COST_PER_REQUEST == 0.025

    def test_maps_grounding_rate_is_picked_by_model_generation(self):
        """Gemini 3.x: $14/1K (upper bound, free tier untracked); 2.5: $25/1K; else fallback."""
        pricing = _reload_pricing()
        assert pricing.maps_grounding_cost_for_model("gemini-3.7-flash") == 0.014
        assert pricing.maps_grounding_cost_for_model("gemini-3-flash-preview") == 0.014
        assert pricing.maps_grounding_cost_for_model("gemini-2.5-flash") == 0.025
        assert pricing.maps_grounding_cost_for_model("gemini-2.5-pro") == 0.025
        assert (
            pricing.maps_grounding_cost_for_model("some-unknown-model")
            == pricing.MAPS_GROUNDING_COST_PER_REQUEST
        )

    def test_fallback_constants(self):
        pricing = _reload_pricing()
        assert pricing.UNKNOWN_CHAT_MODEL_PRICING == (2.0, 12.0)
        assert pricing.UNKNOWN_IMAGE_MODEL_INPUT_RATE == 0.50
        assert pricing.UNKNOWN_IMAGE_PER_IMAGE == 0.067
        assert pricing.UNKNOWN_VIDEO_PER_SECOND == 0.35
        assert pricing.UNKNOWN_VIDEO_TOKEN_PER_MILLION == 17.50
        assert pricing.UNKNOWN_TTS_MODEL_PRICING == (0.50, 10.00)

    def test_video_tokenized_fallback_is_declared_in_yaml(self):
        """pricing.py read a fallbacks key the yaml never declared; the code default hid it."""
        pricing = _reload_pricing()
        declared = pricing._FALLBACKS.get("unknown_video_tokenized_model") or {}
        assert declared.get("video_output_per_million") == 17.50

    def test_env_var_override_path(self, monkeypatch, tmp_path: Path):
        custom_yaml = tmp_path / "custom-pricing.yaml"
        custom_yaml.write_text(
            textwrap.dedent(
                """
                models:
                  custom-gemini: { input_per_million: 1.0, output_per_million: 5.0, cached_input_per_million: 0.1 }
                  custom-gemini-uncached: { input_per_million: 2.0, output_per_million: 6.0 }
                image_generation:
                  custom-imagen:
                    input_per_million: 0.0
                    per_image_by_size: { default: 0.10 }
                video_generation:
                  custom-veo:
                    per_second_by_resolution: { default: 0.5 }
                text_to_speech:
                  custom-tts: { input_per_million: 0.2, output_per_million: 2.0 }
                music_generation:
                  custom-lyria: { per_song: 0.02 }
                  custom-lyria-stream: { per_song: null }
                tools:
                  google_maps_grounding:
                    per_request_by_model_prefix: { custom-gemini: 0.01 }
                    per_request: 0.05
                fallbacks:
                  unknown_chat_model: { input_per_million: 9.9, output_per_million: 99.0 }
                """
            ).strip()
        )
        monkeypatch.setenv("GEMINI_PRICING_PATH", str(custom_yaml))

        pricing = _reload_pricing()

        assert pricing.MODEL_PRICING == {
            "custom-gemini": (1.0, 5.0),
            "custom-gemini-uncached": (2.0, 6.0),
        }
        # A row without cached_input_per_million is left out, not defaulted.
        assert pricing.CACHED_INPUT_PRICING == {"custom-gemini": 0.1}
        input_rate, size_prices = pricing.IMAGE_PRICING["custom-imagen"]
        assert input_rate == 0.0
        assert size_prices[None] == 0.10
        assert pricing.VIDEO_PRICING == {"custom-veo": {"default": 0.5}}
        assert pricing.MUSIC_PRICING == {"custom-lyria": 0.02, "custom-lyria-stream": None}
        assert pricing.MAPS_GROUNDING_COST_BY_MODEL_PREFIX == {"custom-gemini": 0.01}
        assert pricing.MAPS_GROUNDING_COST_PER_REQUEST == 0.05
        assert pricing.maps_grounding_cost_for_model("custom-gemini-uncached") == 0.01
        assert pricing.maps_grounding_cost_for_model("other") == 0.05
        assert pricing.UNKNOWN_CHAT_MODEL_PRICING == (9.9, 99.0)
