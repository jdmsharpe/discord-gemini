# Discord Gemini Bot

![Hits](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-gemini&label=discord-gemini&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![Version](https://img.shields.io/github/v/tag/jdmsharpe/discord-gemini?sort=semver&label=version)](https://github.com/jdmsharpe/discord-gemini/tags)
[![License](https://img.shields.io/github/license/jdmsharpe/discord-gemini?label=license)](./LICENSE)
[![CI](https://github.com/jdmsharpe/discord-gemini/actions/workflows/main.yml/badge.svg)](https://hub.docker.com/r/jsgreen152/discord-gemini)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

## Overview

A Discord bot built on Pycord 2.0 that integrates Google's Gemini API, providing a unified interface for text and multi-turn conversations, image creation, video generation, text-to-speech, music generation, and deep research—all grouped cleanly under the `/gemini` namespace.

## Features

- **Multi-turn Conversations:** Persistent conversation history with interactive button controls and explicit context caching for long conversations.
- **Multiple Gemini Models:** Supports Gemini 3.1 Pro, 3.0 Flash, 2.5 Pro/Flash/Flash Lite, and 2.0 Flash/Flash Lite.
- **Multimodal Input:** Supports text, images, PDFs, audio, video, and documents. Includes external URL file input, `opus` / `alaw` / `mulaw` audio MIME handling, and automatic File API routing for large attachments (up to 2 GB).
- **Advanced Tool Calling:** Features built-in tools (`google_search`, `code_execution`, `google_maps`, `url_context`, `file_search`). Gemini 3 chat models seamlessly combine built-in tools with custom functions.
- **Thinking Configuration:** Customizable thinking levels for Gemini 3 models (Minimal, Low, Medium, High) and token budgets for Gemini 2.5 models, with thought summaries displayed in spoilered embeds.
- **Rich Embeds:** Responses include a Sources embed displaying web and map citations, search queries, URL context retrieval, and file search document citations.
- **Media Generation:**
  - **Images:** High-quality image generation and editing using Imagen 4 and Gemini Flash/Pro Image models.
  - **Video:** Text-to-video, image-to-video, and Veo 3.1 last-frame-constrained generation powered by Google's Veo models.
  - **Music:** Music generation using Lyria 3 (Pro/Clip Preview) and Lyria RealTime Experimental.
  - **Text-to-Speech:** Lifelike speech conversion with 25+ voice options.
- **Deep Research Agent:** Run autonomous deep research tasks that search, read, and synthesize cited reports.

## Commands

### `/gemini chat`

Start a conversation with Gemini AI models.

- Features tool enablement mid-conversation via a dropdown.
- Incompatible tools are automatically disabled depending on the model (e.g., File Search cannot be combined with Google Search).

### `/gemini image`

Generate images from text prompts or edit using reference images.

- **Models:** Gemini 3.1 Flash Image, 3.0 Pro Image, 2.5 Flash Image, Imagen 4 (Ultra/Fast).
- **Options:** Multiple aspect ratios, image count (1-4), resolution control (1K, 2K), and Google Image Search grounding.

### `/gemini video`

Generate videos from text prompts or image inputs.

- **Models:** Veo 3.1 Lite Preview (default), Veo 3.1 Preview, Veo 3.1 Fast Preview, Veo 3, Veo 3 Fast, Veo 2.
- **Options:** Customizable aspect ratio, resolution (`720p`, `1080p`, `4k` where supported), first-frame image input, optional `last_frame` interpolation on Veo 3.1 models, negative prompts, and prompt enhancement control.
- **Validation:** Veo 3.x models support 4/6/8-second outputs, Veo 2 supports 5/6/8-second outputs, `1080p` and `4k` require 8 seconds, `4k` is unavailable on Veo 3.1 Lite, Veo 2 does not expose explicit resolution control, and requesting 2 videos is only supported on Veo 2.

### `/gemini music`

Create music using Google Lyria models.

- **Models:** Lyria 3 Pro Preview, Lyria 3 Clip Preview, Lyria RealTime Experimental.
- **Options:** Customizable BPM, scale/key, density, brightness, duration (RealTime only), and reference image inputs. Long lyrics or structure notes are previewed and attached as text files.

### `/gemini tts`

Convert text to high-quality WAV speech audio (24kHz, 16-bit). Features 25+ voice options with natural language style controls.

### `/gemini research`

Run autonomous deep research tasks (takes 2-10 minutes) using a Gemini Deep Research agent. The default `deep-research-preview-04-2026` (Apr 2026) is tuned for low latency and cost; choose `deep-research-max-preview-04-2026` for higher-quality, more comprehensive reports at the cost of longer runtimes. The legacy `deep-research-pro-preview-12-2025` agent remains selectable for parity with prior runs. Great for market analysis and literature reviews. Optionally search uploaded document stores via `file_search`, enable Google Maps grounding, and control whether the response includes deep-research thinking summaries.

### `/gemini check_permissions`

Check if the bot has the necessary permissions in the current channel.

## Setup & Installation

### Prerequisites

- Python 3.10+
- `google-genai` ~1.73
- `py-cord` ~2.7
- `Pillow` ~12.1
- `aiohttp` ~3.13
- `python-dotenv` ~1.2

### Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and its runtime dependencies:

   ```bash
   python -m pip install .
   ```

4. Copy the environment example file:

   ```bash
   cp .env.example .env
   ```

### Contributor Setup

Install development tooling for tests, linting, and type checking:

```bash
python -m pip install -e ".[dev]"
```

### Configuration (`.env`)

| Variable | Required | Description |
| --- | --- | --- |
| `BOT_TOKEN` | **Yes** | Your Discord bot token |
| `GUILD_IDS` | **Yes** | Comma-separated Discord server IDs |
| `GEMINI_API_KEY` | **Yes** | Your Google Gemini API key |
| `GEMINI_API_VERSION` | No | Override version (`v1` for stable, `v1alpha` for preview features) |
| `GEMINI_FILE_SEARCH_STORE_IDS` | No | Comma-separated File Search store IDs for RAG |
| `SHOW_COST_EMBEDS` | No | Show per-request cost embeds in supported responses (Default: `true`) |
| `ENABLE_CUSTOM_TOOLS` | No | Enable custom function tool calling in chat (Default: `true`) |
| `GEMINI_PRICING_PATH` | No | Path to a pricing YAML that overrides the bundled `src/discord_gemini/config/pricing.yaml` |
| `LOG_FORMAT` | No | `text` (default) for human-readable logs, or `json` for structured JSON-lines output with per-request IDs |

### Running the Bot

**Locally:**

```bash
python src/bot.py
```

*(Note: `src/bot.py` is a thin launcher that delegates to `discord_gemini.bot.main`)*

**With Docker:**

```bash
docker compose up -d --build
```

**Using as a Cog:**
To compose this repo into a larger bot, import the namespaced package:

```python
from discord_gemini import GeminiCog

bot.add_cog(GeminiCog(bot=bot))
```

## Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Create a new application and add a bot in the "Bot" section.
3. Enable **Server Members Intent** and **Message Content Intent** under Privileged Gateway Intents.
4. Copy the bot token and add it to your `.env` file.
5. Go to OAuth2 > URL Generator.
6. Select scopes: `bot`, `applications.commands`.
7. Select permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`, `Embed Links`, `Attach Files`.
8. Use the generated URL to invite the bot to your server.

## Usage

### Music Generation Prompt Examples

Try these prompts with `/gemini music`:

- **Genres:** "minimal techno", "jazz fusion", "acoustic folk", "orchestral score"
- **Instruments:** "piano and strings", "guitar and drums", "electronic synthesizers"
- **Moods:** "upbeat and energetic", "calm and meditative", "dark and atmospheric"
- **Combinations:** "upbeat electronic dance music with heavy bass", "peaceful acoustic guitar with nature sounds"

### Troubleshooting

- Ensure your API key has the necessary permissions.
- Check that your bot has proper Discord permissions in the channel.
- Verify your internet connection for streaming features.

## Development

### Testing

Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). All tests are mocked (no real API calls).

```bash
# Install developer tooling if you have not already
python -m pip install -e ".[dev]"

# Run tests locally
python -m pytest -q

# Run tests in Docker
docker build --build-arg PYTHON_VERSION=3.10 -f Dockerfile.test -t discord-gemini-test:3.10 .
docker run --rm discord-gemini-test:3.10 python -m pytest -q

# Run linting and type checks in Docker
docker run --rm discord-gemini-test:3.10 sh -lc 'ruff check src tests && ruff format --check src tests && pyright'
```

### Linting & Type Checking

```bash
ruff check src tests
ruff format --check src tests
pyright
```

*Run `git config core.hooksPath .githooks` after cloning to enable the pre-commit hook.*

## License

MIT License - see [LICENSE](LICENSE) for details.
