# Discord Gemini Bot - Developer Reference

## Supported Entry Points

- Launcher: `python src/bot.py` remains supported and delegates to `discord_gemini.bot.main`.
- Cog composition contract:

  ```python
  from discord_gemini import GeminiCog

  bot.add_cog(GeminiCog(bot=bot))
  ```

- Compatibility export: `Conversation` remains re-exported from `discord_gemini` during this refactor pass.
- Legacy shim: `src/gemini_api.py` exists only for module-path compatibility, emits a `DeprecationWarning`, and re-exports `GeminiCog` plus `Conversation` only.

## Package Layout

```text
src/
├── bot.py                           # Thin repo-local launcher
├── gemini_api.py                    # Temporary compatibility shim
├── button_view.py                   # Top-level compatibility shim
├── config/                          # Top-level compatibility shim
├── exceptions.py                    # Top-level compatibility shim
├── tools.py                         # Top-level compatibility shim
├── util.py                          # Top-level compatibility shim
    └── discord_gemini/
    ├── __init__.py
    ├── bot.py
    ├── util.py
    ├── config/
    │   ├── __init__.py
    │   └── auth.py
    └── cogs/gemini/
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
        ├── tooling.py
        ├── video.py
        └── views.py
```

`discord_gemini.cogs.gemini.cog` is now the thin registration/orchestration layer. Helper modules own the extracted state, parsing, attachments, cache lifecycle, and feature flows.

## Testing And Patch Targets

- `pytest` runs with `pythonpath = ["src"]`.
- New tests and patches should target real owners under `discord_gemini...`, not `gemini_api`.
- Examples:
  - `discord_gemini.cogs.gemini.tooling.GEMINI_FILE_SEARCH_STORE_IDS`
  - `discord_gemini.cogs.gemini.research.GEMINI_FILE_SEARCH_STORE_IDS`
  - `discord_gemini.cogs.gemini.responses.MusicGenerationError`
  - `discord_gemini.cogs.gemini.views.ButtonView`
- `tests/test_gemini_api_shim.py` is the shim-only compatibility test.

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
- `discord_gemini.cogs.gemini.chat`, `image`, `music`, `research`, `speech`, and `video` now own their respective orchestration flows.
- Default music model is `lyria-3-clip-preview`; keep `discord_gemini.util.DEFAULT_MUSIC_MODEL`, the `/gemini music` slash-command metadata, and user-facing docs aligned when changing it.
- `/gemini music` response embeds should show raw model IDs, not friendly-name rewrites.
- Slash-command `duration` applies only to `lyria-realtime-exp`; Lyria 3 Clip stays fixed at 30 seconds and Lyria 3 Pro should not echo a target duration from the slash option.
- When Lyria 3 returns long lyrics or structure notes, keep a short embed preview and attach the full text as `music_notes.txt`.
