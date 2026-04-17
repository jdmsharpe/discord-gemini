# Changelog

## v1.1.0

### feat
- Add conversation TTL pruning: `Conversation.touch()` + `updated_at` field, with `_prune_runtime_state()` evicting conversations older than 12h via `@tasks.loop(minutes=15)`, cascade-cleaning orphaned entries in `message_to_conversation_id`. Caps active conversations at 100 and retains `daily_costs` for 30 days.
- Configure `genai.Client` with `HttpRetryOptions(attempts=5, initial_delay=0.5, max_delay=60.0, http_status_codes=[429, 500, 502, 503, 504])` for both `build_gemini_client()` and the `v1alpha` Lyria realtime client; the google-genai SDK handles backoff internally.
- Extract pricing tables to `src/discord_gemini/config/pricing.yaml` (`MODEL_PRICING`, `IMAGE_PRICING` with size tiers, nested `VIDEO_PRICING` per-resolution, `TTS_PRICING`, `MAPS_GROUNDING_COST_PER_REQUEST`) loaded by `config/pricing.py`; override at runtime via `GEMINI_PRICING_PATH`. Preserves Python `None` key for "default size" in `IMAGE_PRICING` for backward compatibility.
- Add structured logging with per-request IDs: `src/discord_gemini/logging_setup.py` exposes `REQUEST_ID` (ContextVar), `bind_request_id()`, `configure_logging()`. `cog_before_invoke` and `on_message` bind fresh 8-char hex IDs. Set `LOG_FORMAT=json` for JSON-lines output.

### fix
- Plug memory leak in long-running bots: prior runtime-state dicts accumulated conversations and follow-up-message lookups indefinitely; the new TTL prune + cascade cleanup bounds growth.

### chore
- Bump project version to `1.1.0`.
- Add `PyYAML~=6.0` runtime dependency for the pricing loader.
- Canonical pre-commit hook: `.githooks/pre-commit` now does `ruff format` (auto-applied + re-staged), `ruff check` (blocking), `pyright` (warning-only), and `pytest --collect-only` (warning-only smoke). Byte-identical across all 6 discord-* repos.

### test
- Add 7 pricing-loader tests (`tests/test_config_pricing.py`) covering YAML load, `GEMINI_PRICING_PATH` override, and `None`/`"default"` key preservation.
- Add 8 structured-logging tests (`tests/test_logging_setup.py`) covering `bind_request_id`, ContextVar propagation, and JSON formatter output.
- Add 4 new state-prune tests (`tests/test_gemini_state.py`) covering TTL eviction, cascade cleanup of `message_to_conversation_id`, and daily-cost retention.
- Total test count: 381 to 396.

### docs
- Refresh `README.md` with new env vars (`GEMINI_PRICING_PATH`, `LOG_FORMAT`).
- Update `.claude/CLAUDE.md` with runtime conventions for pricing, retry, conversation TTL, and request IDs.
- Refresh `.env.example` with the new opt-in env vars.

### compare
- [`v1.0.4...v1.1.0`](https://github.com/jdmsharpe/discord-gemini/compare/v1.0.4...v1.1.0)
