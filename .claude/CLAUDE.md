# Discord Gemini Bot - Developer Reference

## Quick Start

```bash
uv sync --extra dev                   # creates .venv from uv.lock (no pip inside — use `uv pip` if needed)
cp .env.example .env                  # then fill in required values
git config core.hooksPath .githooks   # enable repo pre-commit hook
uv run python src/bot.py   # or: docker compose up
```

## Gotchas

- Uses **`py-cord`** (not `discord.py`). The slash-command API differs; don't mix docs between the two.
- `GUILD_IDS` must list at least one guild ID. Empty or unset parses to `[]`, and py-cord only registers a command globally when `guild_ids is None`, so the commands land **nowhere** — not globally, not per-guild. `validate_required_config()` does not check it, so the bot starts clean and silently serves no commands.

## Environment Setup

Copy `.env.example` to `.env` and fill in the values:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `BOT_TOKEN` | **Yes** | — | Discord bot token |
| `GUILD_IDS` | **Yes** | — | Comma-separated Discord server IDs; empty or unset registers **no** commands anywhere |
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

- Preserve the current cache/file-search/maps/tool compatibility behavior when refactoring further.
- Explicit cache creation is gated on the **cacheable payload**, not `prompt_token_count`. The prompt count includes tool declarations a cache never stores, so a 1,100-token turn can carry only 782 cacheable tokens and Gemini rejects the create under `CACHE_MIN_TOKEN_COUNT` (`util.py`). `cache.py::_create_cache` pre-counts the exact payload with `count_tokens` behind the cheap `prompt_tokens` pre-filter, so it costs roughly one extra call per conversation and both the first create and `_recache` share the guard. The Developer API **rejects** `system_instruction` and `tools` in `CountTokensConfig` — the SDK raises client-side before any HTTP call — so the system instruction is folded into `contents` as a leading turn to keep the count comparable. A failed count falls through to the create attempt, so a `count_tokens` outage cannot silently disable caching.
- Every `GenerateContentConfig` must carry `automatic_function_calling=disable_afc()` (`cogs/gemini/client.py`). Tool execution belongs to `chat._run_agentic_loop`, never the SDK, and leaving AFC on routes the call through the SDK's AFC wrapper and its one-shot warning. `tests/test_gemini_afc.py` guards all five request paths (chat command, conversation continuation, image, music, speech).
- Custom tool dispatch uses a `ToolProvider` protocol in `discord_gemini.cogs.gemini.tooling`. `LocalFunctionProvider` wraps `@tool` callables, `BuiltinGeminiToolProvider` surfaces model-supported server-side tools, and `McpToolProvider` is a stub for future MCP transport. `execute_tool_call` routes namespaced names (`provider_id.tool_name`) to the correct provider and falls back to local lookup for un-namespaced names.
- `GEMINI_FILE_SEARCH_STORE_IDS` is the runtime gate for file-search-enabled flows.
- Gemini chat now supports built-in + custom tool combinations only on Gemini 3 chat models.
- When a request combines Gemini server-side tools with custom functions, `discord_gemini.cogs.gemini.chat` must enable `tool_config.include_server_side_tool_invocations = True`.
- Combined built-in + custom tool requests should also set `tool_config.function_calling_config.mode = VALIDATED`.
- Manual function tool execution must preserve the Gemini-provided function-call `id` when building the `functionResponse` part for the next turn.
- Unsupported built-in + custom tool combinations should fail fast with a user-visible validation error rather than silently falling back.
- `discord_gemini.cogs.gemini.chat`, `image`, `music`, `research`, `speech`, and `video` now own their respective orchestration flows.
- Default chat model is `gemini-3.7-flash`; keep the `/gemini chat` `model` param default (`cogs/gemini/cog.py`), `CHAT_MODEL_CHOICES` ordering (`cogs/gemini/command_options.py`), the tool `model_allowlist`s in `cogs/gemini/tool_registry.py`, the `models` block in `config/pricing.yaml`, and `CACHE_MIN_TOKEN_COUNT` (`util.py`) aligned when changing it. `MODEL_PRICING.get(model, UNKNOWN_CHAT_MODEL_PRICING)` falls back silently, so an unpriced model bills at the fallback rate with no error. Also check the two thinking guards in `util.py`: `MINIMAL_THINKING_UNSUPPORTED_MODELS` (`gemini-3.7-flash` and `gemini-3.1-pro-preview` 400 on `thinking_level="minimal"`) and `THINKING_LEVEL_UNSUPPORTED_MODELS` (every Gemini 2.5 chat model 400s on ANY `thinking_level` with "Thinking level is not supported for this model" — they take `thinking_budget` instead; live-reproduced 2026-08-28). `_validate_thinking_request` (`cogs/gemini/responses.py`) is what keeps both — and the level/budget mutual exclusion — off the wire; `tests/test_gemini_api.py::TestThinkingValidation` pins all three (`test_rejects_minimal_on_models_that_do_not_support_it`, `test_rejects_every_level_on_gemini_2_5`, `test_thinking_level_unsupported_set_is_exactly_the_2_5_family`).
- Default music model is `lyria-3-clip-preview`; keep `discord_gemini.util.DEFAULT_MUSIC_MODEL`, the `/gemini-tools music` slash-command metadata, and user-facing docs aligned when changing it.
- Music is billed **per generated song** (`music_generation` in `pricing.yaml` → `MUSIC_PRICING` / `calculate_music_cost`), not per second, and a run that returned no audio is not charged. `lyria-realtime-exp` streams audio and has no published per-song price: it is stored as `null`, `calculate_music_cost` returns `None` for it, and `music.py` logs `unpriced=True` at $0. Never substitute an invented rate — add a real one only when Google publishes it.
- Default video model is `gemini-omni-1.1-flash` (`DEFAULT_OMNI_VIDEO_MODEL` in `video.py`, GA 2026-08-27); the legacy `gemini-omni-flash-preview` stays selectable until its 2026-09-30 shutdown. Both ids live in `OMNI_VIDEO_MODELS`, which is what `video_command`'s routing, `_validate_omni_video_request`, and the `video_tokenized` rows in `pricing.yaml` key off — add a new Omni id to the set, `VIDEO_MODEL_CHOICES`, and `pricing.yaml` together (`tests/test_gemini_video.py::TestOmniVideoModels` pins routing, Veo-option rejection, and the 17.50/MTok rate for every id in the set). Omni is generated via the **Interactions API** (`_generate_video_with_omni` in `video.py`), NOT the Veo `generate_videos` path, and it MUST run in background mode: a synchronous `interactions.create` on the GA id was closed server-side after 60.26 s twice on 2026-08-28 (`httpx.RemoteProtocolError`, "Server disconnected without sending a response", no interaction id), while `background=True` returned an id in 0.79 s and completed after 55.7 s of `interactions.get` polling (`in_progress` → `completed`); the preview still completed synchronously on 2026-08-20, so this is GA-specific. The poller mirrors `research.py` — every `OMNI_VIDEO_POLL_INTERVAL` (5 s), bounded by `VIDEO_GENERATION_TIMEOUT` with the same `TimeoutError` as the Veo path; a non-`completed` terminal status raises `APICallError` with the interaction's `errors[].message` — and the completed interaction's `output_video.uri` downloads first try via `files.download` (no ACTIVE polling). `resolution` is accepted on `gemini-omni-1.1-flash` only, through its `VIDEO_SUPPORTED_RESOLUTIONS` entry (`720p`, `1080p`): `1080p` returned a real 1920x1080 avc1 MP4 on 2026-08-28 (tkhd and stsd agree); the preview ignored resolution on 2026-08-20 and stays rejected with the aspect-ratio-only message; `4k`/`360p` are documented but unprobed with no price row, so `_validate_omni_video_request` refuses them as "not yet supported for Gemini Omni" while `VIDEO_RESOLUTION_CHOICES` stays the Veo list. `duration` is accepted by the GA id (`"<n>s"`, minimum 3 s) but deliberately NOT exposed: a 3 s 1080p clip billed 57,920 video tokens, identical to a default 10 s 720p clip, so the count is flat per clip — never derive a duration from it (the old `~Ns 720p` label was wrong); the cost log/embed shows the REQUESTED resolution (`OMNI_DEFAULT_VIDEO_RESOLUTION` = `720p` when unset) plus the exact token count (`tests/test_gemini_video.py::TestOmniVideoGeneration`, `TestOmniVideoValidation`, `TestOmniVideoModels::test_omni_cost_embed_shows_the_requested_resolution`). Veo 3.1 models remain for image-to-video, last-frame interpolation, and 4k/duration control. Omni is billed per video-modality OUTPUT token (`VIDEO_TOKEN_PRICING` / `calculate_omni_video_cost`, exact from the response's usage), while Veo is per-second-by-resolution (`VIDEO_PRICING`, estimated). Veo-only options are rejected for Omni via `_validate_omni_video_request`. Keep the default model, `discord_gemini.util.VIDEO_PRICING`/`VIDEO_TOKEN_PRICING`, the `/gemini-media video` slash-command metadata, and user-facing docs aligned when changing it.
- `/gemini-media video` exposes `resolution` for Veo and Omni 1.1; keep `VIDEO_RESOLUTION_CHOICES`, `VIDEO_SUPPORTED_RESOLUTIONS`, both validators, the option description in `cog.py`, and resolution-aware pricing aligned with current docs when changing video support.
- **Image sizes** are per model (`IMAGE_SUPPORTED_SIZES` / `_validate_image_size_request` in `image.py`, lower-cased compare; unlisted models such as `gemini-2.5-flash-image` pass through), from live probes with the bot's exact request shape on 2026-08-28: `gemini-3.1-flash-image` takes `512` (704x384 at the model's own 11:6 when no aspect ratio is sent, 512x512 with `1:1`; 747 IMAGE tokens = $0.045), `1K`, `2K` and `4K` (5632x3072, 2,520 tokens = $0.151); `gemini-3-pro-image` takes `1K`/`2K`/`4K` (2,000 tokens = $0.24) and 400s on `512`; `gemini-3.1-flash-lite-image` is `1K` only (`512`, `2K`, `4K` all 400 with "Image size X is not supported for this model"); `0.5K` 400s everywhere ("Supported values are: 1K, 2K, 4K, 512, 512P, 512PX."). The validator refuses with that verbatim API text (`tests/test_gemini_api.py::TestImageSizeValidation`), and `pricing.yaml` carries the `512`/`4k` rows (`tests/test_config_pricing.py::test_flash_and_pro_image_pricing_carry_the_probed_512_and_4k_rows`, `tests/test_util.py::test_calculate_image_cost_512_and_4k_tiers`). **Case:** `IMAGE_SIZE_CHOICES` values are the documented uppercase `1K`/`2K`/`4K` (plus `512`); lowercase `1k`/`2k` are accepted and return the requested pixels but are METERED at the 1K tier (`2k` billed 1,120 IMAGE tokens vs 1,680 for `2K`), a Google-side inconsistency that can be corrected at any time — `calculate_image_cost` lower-cases the key so the uppercase values still hit the yaml tiers. **Delivery:** `_generate_image_with_gemini` returns `GeneratedImage(data, mime_type)` (the API's own bytes, not decoded PIL images) and `_create_image_response_embed` attaches `image/png`/`image/jpeg` unchanged with the matching extension (`DISCORD_NATIVE_IMAGE_EXTENSIONS`), re-encoding only other formats to PNG: re-encoding the probe's 4K JPEGs to PNG produced 10.06-12.95 MB files, over Discord's 10 MB bot upload cap, while the originals were 5.5-7.1 MB (`tests/test_gemini_api.py::TestGeminiImageDelivery`). **Aspect ratio** is always sent, `1:1` included — omitted, the model picks its own ratio (the 11:6 above), so the advertised `(default: 1:1)` only holds on the wire (`test_generate_image_with_gemini_default_config`).
- `/gemini-tools music` response embeds should show raw model IDs, not friendly-name rewrites.
- Slash-command `duration` applies only to `lyria-realtime-exp`; Lyria 3 Clip stays fixed at 30 seconds and Lyria 3 Pro should not echo a target duration from the slash option.
- When Lyria 3 returns long lyrics or structure notes, keep a short embed preview and attach the full text as `music_notes.txt`.
- Attachment MIME handling explicitly normalizes `.opus`, `.alaw`, and `.mulaw` inputs to `audio/opus`, `audio/alaw`, and `audio/mulaw` for both Discord attachments and URL-based file inputs.
- Discord caps message embeds at 10 per message and 6000 total characters; `discord_gemini.cogs.gemini.embed_delivery.pack_embeds` enforces both, and `send_embed_batches` falls back to plain-text chunks if Discord rejects an embed batch.

## Runtime Conventions (Cross-Project)

- **Pricing** is loaded from `src/discord_gemini/config/pricing.yaml` by `config/pricing.py` at import time. Supports nested image/video size-tier pricing. Override via `GEMINI_PRICING_PATH`. Cross-referenced against [genai-prices/google.yml](https://github.com/pydantic/genai-prices/blob/main/prices/providers/google.yml).
- **Cached-input billing**: `calculate_cost(..., cached_tokens=)` bills the response's `cached_content_token_count` at the row's `cached_input_per_million` (`CACHED_INPUT_PRICING`) and the remainder at the input rate, clamped so neither side goes negative. A row without a cached rate silently bills cache hits at the full input rate, so every chat row must carry one (`tests/test_config_pricing.py::test_every_chat_row_declares_a_cached_input_rate_below_its_input_rate`); explicit-cache STORAGE (per token-hour) is not modeled. All `calculate_cost` call sites pass it through — `chat.py` (both sites, from `AgenticResult.total_cached_tokens`, summed per iteration in `_run_agentic_loop`), `research.py` (`_ResearchResult.cached_tokens`), and `embeds.append_pricing_embed`, which also shows an `N cached` label (`tests/test_util.py::test_calculate_cost_bills_cached_tokens_at_the_cached_rate`, `tests/test_gemini_chat.py::test_chat_passes_cached_tokens_to_calculate_cost`, `tests/test_gemini_api.py::test_research_bills_cached_tokens_through_calculate_cost`).
- **Maps grounding surcharge** is keyed by model-id prefix, not flat: `tools.google_maps_grounding.per_request_by_model_prefix` → `MAPS_GROUNDING_COST_BY_MODEL_PREFIX` / `maps_grounding_cost_for_model` (longest prefix wins, `per_request` is the fallback for unmatched ids). `gemini-3` bills 0.014 ($14 / 1K after a 5,000-prompt/month free tier shared across Gemini 3 that cannot be tracked here, so it is an UPPER BOUND) and `gemini-2.5` bills 0.025 ($25 / 1K, no free tier). Pinned by `tests/test_util.py::test_maps_surcharge_is_split_by_generation` and `tests/test_config_pricing.py::test_maps_grounding_rate_is_picked_by_model_generation`. Every `calculate_cost` call site passes `google_maps_grounded` from ACTUAL grounding use, never from the tool merely being enabled: `chat.py` derives it from `tool_info["tools_used"]` (response grounding metadata), `research.py` from the Interactions API's `usage.grounding_tool_count` (`_ResearchResult.grounding_tool_counts["google_maps"]`, the same source as the cost embed's `maps: N` fragment), and `embeds.append_pricing_embed` takes it as a parameter. Google bills EVERY grounded prompt, so `calculate_cost(google_maps_grounded: bool | int)` adds `int(value)` surcharges: chat and the embed pass `True` (exactly one, unchanged), research passes the count so N Maps calls bill N surcharges (`tests/test_util.py::test_calculate_cost_bills_each_maps_grounded_prompt`; `tests/test_gemini_api.py::test_research_bills_maps_grounding_surcharge_for_research_model` pins count=2 → two surcharges). `research.py` also logs `google_maps_grounded=` (bool) beside the requested `google_maps=` flag and the raw counts.
- **SDK migration triggers**: pinned to `google-genai ~=2.20`, **locked/running 2.20.0**. Veo generation calls `generate_videos(model=..., source=types.GenerateVideosSource(...), config=...)` — the `prompt`/`image`/`video` arguments were deprecated in 2.14.0 and slated for removal no earlier than 2026-07-31; `last_frame` stays a `GenerateVideosConfig` field, not a source field. `_build_veo_image` must keep wrapping raw bytes in `types.Image`: a `PIL.Image` is silently coerced to an all-`None` `types.Image` rather than raising. **All rechecked at 2.20.0: (1) `generate_videos` STILL accepts `prompt`/`image`/`video` (the removal date passed without action, and the bot passes `source=` regardless); (2) the PIL coercion STILL happens — both `GenerateVideosSource(image=<PIL>)` and `GenerateVideosConfig(last_frame=<PIL>)` yield a bare `Image()` with no bytes, so `_build_veo_image` remains load-bearing; (3) the `Turn`/`TurnContent` interactions types that 2.18.1 removed (along with `List[Turn]` on `InteractionsInput`) are STILL absent — inert here only because the Omni video and research flows pass kwargs to `interactions.create` and never name `Turn`, so re-check if either flow starts building typed turns; (4) the SDK `Client` has no `disable_afc` method — the bot's own `disable_afc()` helper in `cogs/gemini/client.py` returns `types.AutomaticFunctionCallingConfig(disable=True)`, which the SDK's `should_disable_afc` honors (unchanged at 2.20.0), so every request config skips the SDK's AFC wrapper and its one-shot `AsyncModels._logged_afc_warning` (through v1.10.0 the flag was set only when custom functions were enabled); (5) NEW in 2.20.0: `models.generate_images` raises `ValueError` in Developer-API mode — inert here since Imagen was removed and image generation goes through `generate_content`.** Re-check all five on the next bump rather than blind-bumping the pin.
- **Retry**: the `genai.Client` is built with `HttpRetryOptions(attempts=5, initial_delay=0.5, max_delay=60.0, http_status_codes=[429, 500, 502, 503, 504])` in `client.py` — the SDK handles backoff internally.
- **Conversation TTL**: `_prune_runtime_state` in `cogs/gemini/state.py` evicts conversations older than `CONVERSATION_TTL` (12h) every 15 minutes via `@tasks.loop`. Also cascade-cleans orphaned entries in `message_to_conversation_id`. Daily costs retained for `DAILY_COST_RETENTION_DAYS` (30).
- **Request IDs**: `cog_before_invoke` (and `on_message`) bind a fresh 8-char hex id via `discord_gemini.logging_setup.bind_request_id`. All downstream `logger.info`/`warning`/`error` calls automatically include the id. Set `LOG_FORMAT=json` for JSON-lines output.
- **Async file I/O**: blocking `open()` and `pathlib` methods (`read_bytes`, `write_bytes`, `unlink`, etc.) inside `async def` freeze the Discord event loop and stall every concurrent slash command. Wrap them with `asyncio.to_thread(...)` so the I/O runs on a worker thread. Enforced by `ruff` (`ASYNC230`/`ASYNC240`).
