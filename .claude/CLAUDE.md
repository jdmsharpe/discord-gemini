# Discord Gemini Bot - Developer Reference

## Environment Setup

Copy `.env.example` to `.env` and fill in the values:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `BOT_TOKEN` | **Yes** | — | Discord bot token |
| `GUILD_IDS` | **Yes** | — | Comma-separated Discord server IDs |
| `GEMINI_API_KEY` | **Yes** | — | Google Gemini API key |
| `GEMINI_API_VERSION` | No | SDK default (`v1beta`) | Override API version (`v1` for stable, `v1alpha` for preview) |
| `GEMINI_FILE_SEARCH_STORE_IDS` | No | `""` | Comma-separated file search store IDs |
| `ENABLE_CUSTOM_TOOLS` | No | `true` | Enable custom function tool calling in `/gemini chat` |
| `SHOW_COST_EMBEDS` | No | `true` | Show per-request cost embeds in supported responses |
| `GEMINI_PRICING_PATH` | No | bundled | Override the bundled `src/discord_gemini/config/pricing.yaml` |
| `LOG_FORMAT` | No | `text` | Set to `json` for structured JSON-lines output |

## Supported Entry Points

- Launcher: `python src/bot.py` remains supported and delegates to `discord_gemini.bot.main`.
- Cog composition contract:

  ```python
  from discord_gemini import GeminiCog

  bot.add_cog(GeminiCog(bot=bot))
  ```

- `discord_gemini.bot.main()` now calls `validate_required_config()` before connecting, so missing or blank `BOT_TOKEN` and `GEMINI_API_KEY` values fail fast at startup.
- Compatibility export: `Conversation` remains re-exported from `discord_gemini`. Both the top-level package and `cogs/gemini/__init__.py` use lazy `__getattr__` exports so `GeminiCog` and `Conversation` can be imported without eagerly pulling in the full Discord/runtime graph. Type-only imports keep `pyright src/` aware of those public exports.

## Package Layout

```text
src/
├── bot.py                           # Thin repo-local launcher
└── discord_gemini/
    ├── __init__.py
    ├── bot.py
    ├── logging_setup.py             # Structured logging + request-id ContextVar
    ├── util.py
    ├── config/
    │   ├── __init__.py
    │   ├── auth.py
    │   ├── pricing.py                # YAML loader exposing MODEL_PRICING, IMAGE_PRICING, VIDEO_PRICING, etc.
    │   └── pricing.yaml              # Canonical pricing data (override via GEMINI_PRICING_PATH)
    └── cogs/
        ├── __init__.py
        └── gemini/
            ├── __init__.py
            ├── attachments.py
            ├── cache.py
            ├── chat.py
            ├── client.py
            ├── cog.py
            ├── command_options.py
            ├── embed_delivery.py
            ├── embeds.py
            ├── image.py
            ├── music.py
            ├── models.py
            ├── research.py
            ├── responses.py
            ├── speech.py
            ├── state.py
            ├── tool_registry.py
            ├── tooling.py
            ├── usage.py
            ├── video.py
            └── views.py
```

`discord_gemini.cogs.gemini.cog` is now the thin registration/orchestration layer. Helper modules own the extracted state, parsing, attachments, cache lifecycle, and feature flows.
Only `src/bot.py` remains at the repo root; code imports should target `discord_gemini...`.

## Testing And Patch Targets

- `pytest` runs with `pythonpath = ["src"]`.
- The test suite uses module-aligned files (`test_gemini_<module>.py`); `tests/test_package_import.py` is the import smoke test and `tests/support.py` holds shared helpers.
- `pytest` runs with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` decorator needed on async test functions.
- New tests and patches should target real owners under `discord_gemini...`.
- Examples:
  - `discord_gemini.cogs.gemini.tooling.GEMINI_FILE_SEARCH_STORE_IDS`
  - `discord_gemini.cogs.gemini.research.GEMINI_FILE_SEARCH_STORE_IDS`
  - `discord_gemini.cogs.gemini.responses.MusicGenerationError`
  - `discord_gemini.cogs.gemini.views.ButtonView`
- Import `GeminiCog` from `discord_gemini`; do not reintroduce legacy `gemini_api` shim paths.

## Validation Commands

```bash
ruff check src/ tests/
ruff format src/ tests/
pyright src/
pytest -q

# Run the bot via Docker (uses .env for config):
docker compose up
```

- The repo pre-commit hook (`.githooks/pre-commit`) runs `ruff format` (auto-applied + re-staged), then `ruff check` (blocking), then `pyright` and `pytest --collect-only` as warning-only smoke tests. Resolves tools from `.venv/bin` or `.venv/Scripts` first, then `PATH`. Committed code may differ from what you wrote if formatting was needed.

## Provider Notes

- Dependency baseline is `google-genai~=1.74`.
- Preserve the current cache/file-search/maps/tool compatibility behavior when refactoring further.
- Custom tool dispatch uses a `ToolProvider` protocol in `discord_gemini.cogs.gemini.tooling`. `LocalFunctionProvider` wraps `@tool` callables, `BuiltinGeminiToolProvider` surfaces model-supported server-side tools, and `McpToolProvider` is a stub for future MCP transport. `execute_tool_call` routes namespaced names (`provider_id.tool_name`) to the correct provider and falls back to local lookup for un-namespaced names.
- `GEMINI_FILE_SEARCH_STORE_IDS` is the runtime gate for file-search-enabled flows.
- Gemini chat now supports built-in + custom tool combinations only on Gemini 3 chat models.
- When a request combines Gemini server-side tools with custom functions, `discord_gemini.cogs.gemini.chat` must enable `tool_config.include_server_side_tool_invocations = True`.
- Combined built-in + custom tool requests should also set `tool_config.function_calling_config.mode = VALIDATED`.
- Manual function tool execution must preserve the Gemini-provided function-call `id` when building the `functionResponse` part for the next turn.
- Unsupported built-in + custom tool combinations should fail fast with a user-visible validation error rather than silently falling back.
- `discord_gemini.cogs.gemini.chat`, `image`, `music`, `research`, `speech`, and `video` now own their respective orchestration flows.
- Default music model is `lyria-3-clip-preview`; keep `discord_gemini.util.DEFAULT_MUSIC_MODEL`, the `/gemini music` slash-command metadata, and user-facing docs aligned when changing it.
- Default video model is `veo-3.1-lite-generate-preview`; keep `discord_gemini.util.VIDEO_PRICING`, the `/gemini video` slash-command metadata, and user-facing docs aligned when changing it.
- `/gemini video` now exposes `resolution`; keep the resolution choices, model-specific validation rules, and resolution-aware pricing aligned with current Veo docs when changing video support.
- `/gemini music` response embeds should show raw model IDs, not friendly-name rewrites.
- Slash-command `duration` applies only to `lyria-realtime-exp`; Lyria 3 Clip stays fixed at 30 seconds and Lyria 3 Pro should not echo a target duration from the slash option.
- When Lyria 3 returns long lyrics or structure notes, keep a short embed preview and attach the full text as `music_notes.txt`.
- Attachment MIME handling explicitly normalizes `.opus`, `.alaw`, and `.mulaw` inputs to `audio/opus`, `audio/alaw`, and `audio/mulaw` for both Discord attachments and URL-based file inputs.
- Discord caps message embeds at 10 per message and 6000 total characters; `discord_gemini.cogs.gemini.embed_delivery.pack_embeds` enforces both, and `send_embed_batches` falls back to plain-text chunks if Discord rejects an embed batch.

## Runtime Conventions (Cross-Project)

- **Pricing** is loaded from `src/discord_gemini/config/pricing.yaml` by `config/pricing.py` at import time. Supports nested image/video size-tier pricing. Override via `GEMINI_PRICING_PATH`. Cross-referenced against [genai-prices/google.yml](https://github.com/pydantic/genai-prices/blob/main/prices/providers/google.yml).
- **Retry**: the `genai.Client` is built with `HttpRetryOptions(attempts=5, initial_delay=0.5, max_delay=60.0, http_status_codes=[429, 500, 502, 503, 504])` in `client.py` — the SDK handles backoff internally.
- **Conversation TTL**: `_prune_runtime_state` in `cogs/gemini/state.py` evicts conversations older than `CONVERSATION_TTL` (12h) every 15 minutes via `@tasks.loop`. Also cascade-cleans orphaned entries in `message_to_conversation_id`. Daily costs retained for `DAILY_COST_RETENTION_DAYS` (30).
- **Request IDs**: `cog_before_invoke` (and `on_message`) bind a fresh 8-char hex id via `discord_gemini.logging_setup.bind_request_id`. All downstream `logger.info`/`warning`/`error` calls automatically include the id. Set `LOG_FORMAT=json` for JSON-lines output.
