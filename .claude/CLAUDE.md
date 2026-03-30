# Discord Gemini Bot - Developer Reference

## Supported Entry Points

- Launcher: `python src/bot.py` remains supported and delegates to `discord_gemini.bot.main`.
- Cog composition contract:

  ```python
  from discord_gemini import GeminiAPI

  bot.add_cog(GeminiAPI(bot=bot))
  ```

- Compatibility export: `Conversation` remains re-exported from `discord_gemini` during this refactor pass.
- Legacy shim: `src/gemini_api.py` exists only for import compatibility and emits a `DeprecationWarning`.

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
        ├── client.py
        ├── cog.py
        ├── embeds.py
        ├── models.py
        ├── responses.py
        ├── tooling.py
        └── views.py
```

Most command flow still lives in `discord_gemini.cogs.gemini.cog`. Helper modules currently own the explicitly extracted support code.

## Testing And Patch Targets

- `pytest` runs with `pythonpath = ["src"]`.
- New tests and patches should target real owners under `discord_gemini...`, not `gemini_api`.
- Examples:
  - `discord_gemini.cogs.gemini.cog.GEMINI_FILE_SEARCH_STORE_IDS`
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
- `discord_gemini.cogs.gemini.cog` remains the canonical owner for most chat, music, video, speech, and research orchestration in this pass.
