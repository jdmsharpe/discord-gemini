# Discord Gemini Bot

![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-gemini&label=discord-gemini&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![Workflow](https://github.com/jdmsharpe/discord-gemini/actions/workflows/main.yml/badge.svg)](https://hub.docker.com/r/jsgreen152/discord-gemini)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

## Features

All commands are grouped under `/gemini` for clean namespacing.

### Text Generation

- **`/gemini chat`**: Have conversations with Gemini AI models
- Support for multiple Gemini models:
  - Gemini 3.1 Pro (Default), 3.0 Flash
  - Gemini 2.5 Pro, 2.5 Flash, 2.5 Flash Lite
  - Gemini 2.0 Flash, 2.0 Flash Lite
- Persistent conversation history with button controls
- Built-in tool calling support:
  - `google_search`
  - `code_execution`
  - `google_maps` (model-dependent)
  - `url_context` (model-dependent)
  - `file_search` (model-dependent, RAG over uploaded document stores)
- Tools can be enabled:
  - In the initial slash command options
  - Mid-conversation via the tool select dropdown
- Tool/model compatibility is enforced automatically (unsupported tools are skipped with a user-visible message)
- File Search cannot be combined with Google Search, Google Maps, or URL Context (incompatible tools are automatically disabled)
- Responses can include a Sources embed with:
  - Web and Maps citations
  - Search queries
  - URL Context retrieval metadata
  - File Search document citations
- Thinking configuration:
  - `thinking_level` for Gemini 3 models (Minimal, Low, Medium, High)
  - `thinking_budget` for Gemini 2.5 models (token budget, -1 for dynamic, 0 to disable)
  - Thought summaries displayed as spoilered embeds when thinking params are set
- Thought signatures preserved in conversation history for multi-turn reasoning context
- Automatic explicit context caching for long conversations on Gemini 3.x models with TTL refresh and periodic re-caching; 2.5 and below use implicit caching
- Multimodal support (text + images, PDFs, audio, video, documents)
- External URL file input (pass a URL to a PDF, image, YouTube video, etc. directly to the model)
- Automatic File API routing for large attachments (>20 MB) with transparent fallback
- Attachment size validation (max 2 GB via File API)

### Image Generation

- **`/gemini image`**: Generate images from text prompts
- Multiple model options:
  - Gemini 3.1 Flash Image (Default)
  - Gemini 3.0 Pro Image
  - Gemini 2.5 Flash Image
  - Imagen 4, 4 Ultra, and 4 Fast
- Multiple aspect ratios and image count options (1-4 images)
- Output resolution control (1K, 2K) for Gemini models
- Google Image Search grounding for real-world visual context (Gemini 3.1 Flash Image only)
- Image editing support with reference images (Gemini model exclusive)

### Video Generation

- **`/gemini video`**: Generate videos from text prompts or image inputs
- Model options:
  - Veo 3.1 - Latest with native audio, video extension, and reference images (Default)
  - Veo 3.1 Fast - Faster Veo 3.1 variant
  - Veo 3 - Enhanced realism and detail
  - Veo 3 Fast - Faster Veo 3 variant
  - Veo 2 - High-quality video generation
- Support for both text-to-video and image-to-video generation
- Customizable duration (5-8 seconds), aspect ratio, resolution (720p/1080p)

### Text-to-Speech

- **`/gemini tts`**: Convert text to lifelike speech
- 25+ voice options with different personalities
- Natural language style control
- High-quality WAV output (24kHz, 16-bit)

### Music Generation

- **`/gemini music`**: Create music using Lyria 3 or Lyria RealTime
- Default music model: Lyria 3 Clip Preview (`lyria-3-clip-preview`)
- Multiple music model options:
  - Lyria 3 Pro Preview for full-length songs with structural coherence
  - Lyria 3 Clip Preview for 30-second clips, loops, and previews
  - Lyria RealTime Experimental for streaming instrumental generation
- Customizable parameters:
  - Duration for Lyria RealTime only
  - BPM (60-200)
  - Musical scale/key selection
  - Density and brightness controls
  - Prompt guidance strength
- Lyria 3 Clip is always 30 seconds; Lyria 3 Pro ignores the slash-command `duration` value
- Optional reference image input for Lyria 3 Pro/Clip
- Lyria 3 can return lyrics or structure notes alongside audio
- Long Lyria lyrics/structure notes are previewed in the embed and attached as `music_notes.txt`
- High-quality stereo audio output (MP3/WAV for Lyria 3, WAV for RealTime)
- Support for various genres, instruments, moods, songs, and clips

### Deep Research

- **`/gemini research`**: Run autonomous deep research tasks
- Powered by the Deep Research agent (`deep-research-pro-preview-12-2025`)
- Agent autonomously plans, searches the web, reads sources, and synthesizes detailed, cited reports
- Research tasks typically take 2-10 minutes
- Optional `file_search` to also search your uploaded document stores
- Great for market analysis, literature reviews, due diligence, and competitive landscaping

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your values:
   - `BOT_TOKEN`: Your Discord bot token
   - `GUILD_IDS`: Comma-separated list of Discord server IDs
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `GEMINI_FILE_SEARCH_STORE_IDS` (optional): Comma-separated File Search store IDs for RAG
   - `SHOW_COST_EMBEDS` (optional): Show per-request cost embeds in chat responses (default: `true`)
   - `ENABLE_CUSTOM_TOOLS` (optional): Enable custom function tool calling in chat (default: `true`)
3. Set up Discord bot permissions in your server
4. Run the bot: `python src/bot.py`

`src/bot.py` remains a thin repo-local launcher that delegates to `discord_gemini.bot.main`.

### Using as a Cog

To compose this repo into a larger bot, import the namespaced package:

```python
from discord_gemini import GeminiCog

bot.add_cog(GeminiCog(bot=bot))
```

`Conversation` remains re-exported from `discord_gemini` for compatibility during this refactor pass.
Top-level `bot.py`, `button_view.py`, `config`, `exceptions.py`, `tools.py`, and `util.py` are repo-local helpers or compatibility surfaces, not the installed public API surface.

## Requirements

- Python 3.10+
- google-genai ~1.69
- py-cord ~2.7
- Pillow ~12.1
- aiohttp ~3.13
- python-dotenv ~1.2

## Music Generation Examples

Try these prompts with `/gemini music`:

- **Genres**: "minimal techno", "jazz fusion", "acoustic folk", "orchestral score"
- **Instruments**: "piano and strings", "guitar and drums", "electronic synthesizers"
- **Moods**: "upbeat and energetic", "calm and meditative", "dark and atmospheric"
- **Combinations**: "upbeat electronic dance music with heavy bass", "peaceful acoustic guitar with nature sounds"

## Development

### Testing

Tests use pytest with pytest-asyncio (`asyncio_mode = "auto"`). All tests are mocked — no real API calls.
The suite is organized around the refactored package layout, with focused files such as `tests/test_gemini_models.py`, `tests/test_gemini_responses.py`, `tests/test_gemini_attachments.py`, `tests/test_gemini_music.py`, and `tests/test_gemini_video.py`.
Import from `discord_gemini` directly; legacy top-level shim modules are no longer part of the supported workflow.
GitHub Actions runs the test suite against Python 3.10, 3.11, 3.12, and 3.13.

```bash
# Run tests
.venv/Scripts/python.exe -m pytest -q    # Windows
.venv/bin/python -m pytest -q            # Unix

# Run tests in Docker
docker build --build-arg PYTHON_VERSION=3.10 -f Dockerfile.test -t discord-gemini-test:3.10 .
docker run --rm discord-gemini-test:3.10
```

### Linting & Type Checking

```bash
ruff check src/ tests/
ruff format src/ tests/
pyright src/
```

After cloning, run `git config core.hooksPath .githooks` to enable the pre-commit hook.

## Troubleshooting

### General Issues

- Ensure your API key has the necessary permissions
- Check that your bot has proper Discord permissions in the channel
- Verify your internet connection for streaming features
