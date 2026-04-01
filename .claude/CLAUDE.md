# Discord Gemini Bot - Developer Reference

## Supported Entry Points

- Launcher: `python src/bot.py` remains supported and delegates to `discord_gemini.bot.main`.
- Cog composition contract:

  ```python
  from discord_gemini import GeminiCog

  bot.add_cog(GeminiCog(bot=bot))
  ```

- Compatibility export: `Conversation` remains re-exported from `discord_gemini` during this refactor pass.

## Package Layout

```text
src/
├── bot.py                           # Thin repo-local launcher
└── discord_gemini/
    ├── __init__.py
    ├── bot.py
    ├── util.py
    ├── config/
    │   ├── __init__.py
    │   └── auth.py
    └── cogs/
        ├── __init__.py
        └── gemini/
            ├── __init__.py
            ├── attachments.py
            ├── cache.py
            ├── chat.py
            ├── client.py
            ├── cog.py
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
            ├── video.py
            └── views.py
```

`discord_gemini.cogs.gemini.cog` is now the thin registration/orchestration layer. Helper modules own the extracted state, parsing, attachments, cache lifecycle, and feature flows.
Only `src/bot.py` remains at the repo root; code imports should target `discord_gemini...`.

## Testing And Patch Targets

- `pytest` runs with `pythonpath = ["src"]`.
- The test suite is organized into module-aligned files such as `tests/test_gemini_models.py`, `tests/test_gemini_responses.py`, `tests/test_gemini_attachments.py`, `tests/test_gemini_music.py`, and `tests/test_gemini_video.py`.
- `tests/test_package_import.py` is the package import smoke test, and `tests/support.py` holds shared Gemini test helpers.
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
```

## Provider Notes

- Preserve the current cache/file-search/maps/tool compatibility behavior when refactoring further.
- `GEMINI_FILE_SEARCH_STORE_IDS` is the runtime gate for file-search-enabled flows.
- Gemini chat now supports built-in + custom tool combinations only on Gemini 3 chat models.
- When a request combines Gemini server-side tools with custom functions, `discord_gemini.cogs.gemini.chat` must enable `tool_config.include_server_side_tool_invocations = True`.
- Combined built-in + custom tool requests should also set `tool_config.function_calling_config.mode = VALIDATED`.
- Manual function tool execution must preserve the Gemini-provided function-call `id` when building the `functionResponse` part for the next turn.
- Unsupported built-in + custom tool combinations should fail fast with a user-visible validation error rather than silently falling back.
- `discord_gemini.cogs.gemini.chat`, `image`, `music`, `research`, `speech`, and `video` now own their respective orchestration flows.
- Default music model is `lyria-3-clip-preview`; keep `discord_gemini.util.DEFAULT_MUSIC_MODEL`, the `/gemini music` slash-command metadata, and user-facing docs aligned when changing it.
- `/gemini music` response embeds should show raw model IDs, not friendly-name rewrites.
- Slash-command `duration` applies only to `lyria-realtime-exp`; Lyria 3 Clip stays fixed at 30 seconds and Lyria 3 Pro should not echo a target duration from the slash option.
- When Lyria 3 returns long lyrics or structure notes, keep a short embed preview and attach the full text as `music_notes.txt`.
