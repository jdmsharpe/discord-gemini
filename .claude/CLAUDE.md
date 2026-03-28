# Discord Gemini Bot - Developer Reference

A Discord bot integrating Google's Gemini AI API for text generation, image generation, video generation, TTS, music generation, and deep research via Discord slash commands.

## Project Structure

```text
discord-gemini/
├── src/
│   ├── bot.py                 # Main bot entry point
│   ├── gemini_api.py          # Core API integration & slash commands (~2700 lines)
│   ├── button_view.py         # Discord UI button controls (pause/resume/regenerate/stop + tool select)
│   ├── exceptions.py          # Custom exception hierarchy (GeminiBotError base)
│   ├── tools.py               # Custom function tool registry, @tool decorator, starter tools
│   ├── util.py                # Dataclasses, constants, pricing, and utility functions
│   └── config/
│       └── auth.py            # API keys, guild IDs, env var config (fail-fast via _require_env)
├── tests/
│   ├── test_util.py           # Dataclass and utility function tests
│   ├── test_button_view.py    # ButtonView UI tests
│   ├── test_gemini_api.py     # GeminiAPI cog tests
│   └── test_tools.py          # Tool registry and execution tests
├── .githooks/
│   └── pre-commit             # ruff format + lint on staged Python files
├── .github/workflows/
│   └── main.yml               # CI: tests + Docker build
├── pyproject.toml             # ruff + pyright config (single source of truth)
└── requirements.txt
```

## Core Dependencies

- **google-genai** ~1.68 — Gemini API client (native async via `client.aio`)
- **py-cord** ~2.7 — Discord bot framework (fork of discord.py)
- **Pillow** ~12.1 — Image processing
- **aiohttp** — Async HTTP for attachment downloads
- **ruff** ~0.15 — Linting and formatting

## Architecture

### Key Components

- **GeminiAPI Cog** (`gemini_api.py`): All slash commands, conversation state, HTTP sessions, API calls, agentic tool-calling loop. Implements the `ConversationHost` protocol for ButtonView decoupling.
- **ButtonView** (`button_view.py`): Interactive buttons + tool select dropdown (6 options incl. Custom Functions) for mid-conversation toggling. Communicates with the cog via the `ConversationHost` protocol (not direct attribute access).
- **exceptions.py**: Custom exception hierarchy rooted at `GeminiBotError` — `APICallError`, `CacheError`, `FileUploadError`, `ValidationError`, `MusicGenerationError`. Replaces bare `Exception` raises for precise catch blocks.
- **tools.py**: Custom function tool registry with `@tool` decorator, `execute_tool_call()`, starter tools (`get_current_time`, `roll_dice`)
- **util.py**: Parameter dataclasses (`ChatCompletionParameters`, `ImageGenerationParameters`, `VideoGenerationParameters`, `SpeechGenerationParameters`, `MusicGenerationParameters`, `ResearchParameters`, `AgenticResult`), pricing dicts/functions, constants, text utilities

### Slash Commands

All commands use `SlashCommandGroup` under `/gemini` — `guild_ids` is set on the group, not individual commands.

| Command | API Method | Notes |
|---------|-----------|-------|
| `/gemini chat` | `client.aio.models.generate_content()` | Multi-turn, 19 params, `on_message` for follow-ups, agentic tool-calling loop |
| `/gemini image` | `generate_content` (Gemini) or `generate_images` (Imagen) | Different APIs per model family |
| `/gemini video` | `client.aio.models.generate_videos()` | Long-running op, polls every 20s, 10min timeout |
| `/gemini tts` | `generate_content` with `response_modalities` | WAV output (24kHz, 16-bit, Mono) |
| `/gemini music` | WebSocket streaming (`api_version='v1alpha'`) | Stereo WAV (48kHz), Lyria model |
| `/gemini research` | `client.aio.interactions.create/get()` | Interactions API, polls 15s, 20min timeout |

### Key Design Patterns

1. **Native async**: All API calls use `client.aio.*` — no thread offloading
2. **Conversation state**: `self.conversations: Dict[int, Conversation]`, `self.message_to_conversation_id: Dict[int, int]`, `self.views: Dict[Member|User, ButtonView]`, `self.last_view_messages: Dict[Member|User, Message]`
3. **One conversation per user per channel** — enforced, persists until ended or bot restarts
4. **Explicit context caching**: Gemini 3.x models auto-cache when token count exceeds `CACHE_MIN_TOKEN_COUNT`; Gemini 2.5 and below use implicit caching (automatic). Cache includes system instruction + `display_name` for observability, TTL refreshed each turn, re-cached when uncached tail grows large, cleaned up on conversation end/cog unload
5. **Typing indicators**: `asyncio.create_task(self.keep_typing(ctx.channel))` for long ops
6. **Shared HTTP session**: Lazy-initialized with lock via `_get_http_session()`, configured with `ClientTimeout(total=300, connect=15)` and `TCPConnector(limit=20, limit_per_host=10, ttl_dns_cache=300)`
7. **Agentic tool-calling loop**: `_run_agentic_loop()` wraps `generate_content()` calls — when the model returns `function_calls`, registered Python tools are executed and results fed back, up to `MAX_AGENTIC_ITERATIONS` (10) rounds. Token usage accumulated across iterations for accurate pricing.
8. **Error handling**: Centralized via `_send_error_followup()` for slash commands; custom exceptions from `exceptions.py` for domain-specific errors. Error followup is wrapped in try/except to ensure cleanup runs even if Discord API fails.

### Tool System

**Server-side tools** (executed by Google): `google_search`, `code_execution`, `google_maps`, `url_context`, `file_search`

**Custom function tools** (executed locally): Registered via `@tool` decorator in `tools.py`. Model calls them via `function_calls` in the response; `_run_agentic_loop()` executes them and feeds `FunctionResponse` back. Toggled via `custom_functions` param on `/gemini chat` or ButtonView dropdown. Controlled by `ENABLE_CUSTOM_TOOLS` env var (default: true).

**Constraints:**

- `file_search` is incompatible with `google_search`, `google_maps`, `url_context` (enforced via `FILE_SEARCH_INCOMPATIBLE_TOOLS`)
- `google_search` and `google_maps` are mutually exclusive (enforced via `MUTUALLY_EXCLUSIVE_TOOLS`)
- File Search requires `GEMINI_FILE_SEARCH_STORE_IDS` env var; store IDs injected at runtime via `enrich_file_search_tools()`
- Custom function callables pass through model compatibility and file_search filters unchanged

### Attachment Handling

- Small files (≤20 MB) → inline_data; large files (>20 MB) → Gemini File API upload
- Max 2 GB via File API; PDFs max 50 MB inline
- Media parts placed before text in API requests (Google recommendation)
- YouTube URLs detected and set to `video/mp4` MIME type
- Uploaded files tracked in `ChatCompletionParameters.uploaded_file_names`, cleaned up on conversation end

### Discord Embed Constraints

- Embed description max: 4096 chars; total embed: 6000 chars
- User prompts truncated to 2000 chars; model responses chunked at 3500 chars via `append_response_embeds()`
- Gemini image text responses truncated to 3800 chars
- `chunk_text()` and `truncate_text()` in `util.py`

### Pricing

- Per-request cost + daily accumulation tracked for all commands
- Pricing dicts in `util.py`: `MODEL_PRICING`, `IMAGE_PRICING`, `VIDEO_PRICING`, `TTS_PRICING`
- Cost functions: `calculate_cost()` (with `thinking_tokens`), `calculate_image_cost()`, `calculate_video_cost()`, `calculate_tts_cost()`
- Embeds toggled via `SHOW_COST_EMBEDS` env var (default: true)
- Structured cost logging via `_log_cost()` on every API call

### Resource Cleanup (`cog_unload`)

- HTTP session, async client (`client.aio.aclose()`), sync client (`client.close()`)
- All active caches deleted, uploaded files cleaned up, temp files removed

## Linting & Formatting

- **ruff** handles both linting and formatting, configured in `pyproject.toml`
- Rules: `E`, `W`, `F`, `I`, `UP`, `B`, `SIM` (E501 ignored — formatter handles line length)
- Line length: 100, target: Python 3.13
- Pre-commit hook in `.githooks/pre-commit`: auto-formats staged files, blocks commit on lint failure
- After cloning, run `git config core.hooksPath .githooks` to enable the hook

## Testing

- `pytest` from project root — pytest-native with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- `pythonpath = ["src"]` configured in `pyproject.toml` — use direct imports (`from util import ...`)
- Mocked Discord/Gemini clients, no real API calls

```bash
.venv/Scripts/python.exe -m pytest -q    # Windows
.venv/bin/python -m pytest -q            # Unix
```

- `test_util.py` — dataclasses, constants, pricing, utility functions, AgenticResult, callable tool handling
- `test_button_view.py` — button callbacks, tool select behavior, custom functions toggle
- `test_gemini_api.py` — cog methods, attachments, caching, pricing, embeds
- `test_tools.py` — tool registry, @tool decorator, execute_tool_call, starter tools
- CI runs on every push/PR; Docker build only proceeds if tests pass

## Adding a New Slash Command

1. Define with `@gemini.command` (not `@slash_command`) + `@option` decorators
2. `await ctx.defer()` immediately, wrap in try/except, errors as red embeds
3. Use parameter dataclass from `util.py` for config
4. Add tests in `test_gemini_api.py`

## Adding a Custom Tool

1. Define an async or sync function in `src/tools.py` with type-hinted params and a docstring
2. Decorate with `@tool` — the SDK auto-generates JSON schema from type hints
3. The tool is automatically available when "Custom Functions" is enabled in a chat
4. Add tests in `tests/test_tools.py`

## External Resources

- [Gemini API Docs](https://ai.google.dev/gemini-api/docs) · [py-cord Docs](https://docs.pycord.dev/) · [AI Studio](https://aistudio.google.com/)
