<h1 align="center">Discord Gemini Bot</h1>

<div align="center">

![Hits](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-gemini&label=discord-gemini&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![Version](https://img.shields.io/github/v/tag/jdmsharpe/discord-gemini?sort=semver&label=version)](https://github.com/jdmsharpe/discord-gemini/tags)
[![License](https://img.shields.io/github/license/jdmsharpe/discord-gemini?label=license)](./LICENSE)
[![CI](https://github.com/jdmsharpe/discord-gemini/actions/workflows/main.yml/badge.svg)](https://github.com/jdmsharpe/discord-gemini/actions/workflows/main.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/jsgreen152/discord-gemini?logo=docker&logoColor=white)](https://hub.docker.com/r/jsgreen152/discord-gemini)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Codecov](https://codecov.io/gh/jdmsharpe/discord-gemini/branch/main/graph/badge.svg)](https://app.codecov.io/gh/jdmsharpe/discord-gemini)

</div>

## Overview

A Discord bot for Google's Gemini API, built on Pycord 2.0. It provides chat, image and video generation, music, text-to-speech and deep research through slash commands.

## Features

- **Chat:** Multi-turn conversations with Gemini 4 Argon (default; needs a paid API key), 3.8 Flash, 3.6 Flash, 3.5 Flash Lite, 3.1 Pro, 3.1 Flash Lite, 3.0 Flash and 2.5 Pro / Flash / Flash Lite. Messages you send in the channel continue the conversation. Google limits the 2.5 models to API accounts that used them before.
- **Attachments:** Text, images, PDFs, audio, video, documents and file URLs, up to 2 GB. On Gemini 4 Argon, 3.8 / 3.6 Flash and 3.5 Flash Lite, videos and YouTube URLs use agentic video understanding: the model requests only the segments it needs.
- **Reasoning:** Thinking levels on Gemini 3 and 4, token budgets on Gemini 2.5, with thought summaries in a spoiler. Unsupported combinations are refused before the request is sent.
- **Built-in tools:** `google_search`, `code_execution`, `google_maps`, `url_context` and `file_search`. Gemini 3 and 4 can combine them with custom functions.
- **Citations:** Web and map citations, search queries, URL context and file search citations appear in a Sources embed.
- **Context caching:** Conversations above the model's cache minimum are cached (not on Gemini 4 Argon), and cache hits are billed at the cached-input rate.
- **Request storage:** Chat, image, speech and music requests are sent with `store=False`.
- **Media generation:**
  - **Images:** Generation and editing with Nano Banana 2.1 (default) and the Gemini Flash / Pro Image models, up to 4K.
  - **Video:** Gemini Omni 1.2 Flash (default) for text-to-video, image-to-video and first-to-last-frame clips, 360p to 4K, 3 to 15 seconds.
  - **Music:** Lyria 3.5 (default), Lyria 3 and Lyria RealTime.
  - **Text-to-speech:** Gemini 3.8 Flash TTS (default) with 25+ voices.
- **Deep research:** Research runs that search the web and return a report with citations.
- **Cost line:** Each reply ends with the request cost, input tokens (cached in parentheses), output tokens (thinking in parentheses), tool calls and your total for the day; grounding charges are included. Set `SHOW_COST_EMBEDS=false` to hide it.

## Commands

### `/gemini chat`

Start a conversation with a Gemini model.

- **`prompt` (required):** Your first message.
- **`system_instruction`:** System prompt.
- **`model`:** Gemini model (default: Gemini 4 Argon).
- **`attachment` / `url`:** A file attachment, or a file URL (Gemini 2.5 and newer).
- **`temperature` / `top_p` / `frequency_penalty` / `presence_penalty` / `seed`:** Sampling options; Gemini 4 ignores `temperature` and `top_p`.
- **`media_resolution`:** Resolution for image, video and PDF inputs.
- **`thinking_level`:** `minimal`, `low`, `medium` or `high` on Gemini 3 and 4.
- **`thinking_budget`:** Thinking token budget on Gemini 2.5 (`-1` dynamic, `0` off).
- **`google_search` / `code_execution` / `google_maps` / `url_context` / `file_search` / `custom_functions`:** Enable tools (default: off). `file_search` needs `GEMINI_FILE_SEARCH_STORE_IDS`. Tools that can't be combined on the chosen model (for example File Search with Google Search) are turned off.

### `/gemini-media image`

Generate an image from a prompt, or edit an attached image.

- **`model`:** Gemini Nano Banana 2.1 (default), 3.1 Flash Image, 3.1 Flash Lite Image, 3.0 Pro Image, 2.5 Flash Image.
- **`image_size`:** `512`, `1K`, `2K` or `4K`. 3.1 Flash Image takes all four, Nano Banana 2.1 and 3.0 Pro Image take `1K`–`4K`, and 3.1 Flash Lite Image and 2.5 Flash Image take `1K` only.
- **`aspect_ratio`:** `1:1` (default), `3:4`, `4:3`, `9:16` or `16:9`. Nano Banana 2.1 and 3.1 Flash / Flash Lite Image also take `1:4`, `4:1`, `1:8` and `8:1`.
- **`google_image_search`:** Ground the image with Google Search (Nano Banana 2.1 and 3.1 Flash Image only).
- One image per request. For variations, run the command again with a different `seed`.
- Images are attached as the API returned them. An image over the server's upload limit is re-encoded as JPEG to fit, which drops transparency.

### `/gemini-media video`

Generate a video from a prompt or an image.

- **`model`:** Gemini Omni 1.2 Flash (default) or Gemini Omni 1.1 Flash.
- **`resolution`:** `360p`, `720p` (default), `1080p` or `4K`; `360p` and `4K` need Omni 1.2.
- **`duration`:** 3 to 15 seconds (default 10; at most 10 on Omni 1.1 or with `last_frame`).
- **`attachment` / `last_frame`:** First-frame and last-frame images; `last_frame` requires `attachment`.

### `/gemini-tools music`

Create music with Google Lyria.

- **`model`:** Lyria 3.5 (default), Lyria 3 Pro Preview, Lyria 3 Clip Preview or Lyria RealTime Experimental.
- **`bpm` / `scale` / `density` / `brightness` / `guidance`:** Tempo, key and style controls.
- **`duration`:** Target length in seconds (Lyria RealTime only).
- **`attachment`:** A reference image (Lyria 3 / 3.5).
- Long lyrics or structure notes are attached as a text file. Lyria RealTime has no published price, so its cost line shows no request cost.

### `/gemini-tools tts`

Convert text to WAV speech (24 kHz, 16-bit, mono).

- **`model`:** Gemini 3.8 Flash TTS (default), 3.8 Flash Lite TTS, 3.1 Flash Preview TTS, 2.5 Flash Preview TTS or 2.5 Pro Preview TTS.
- **`voice_name`:** 25+ voices (default: Kore).
- **`style_prompt`:** Instructions for voice style, tone, accent and pace.

### `/gemini-tools research`

Run a deep research task (2–10 minutes) that searches the web and returns a report with citations.

- **`agent`:** `deep-research-preview-04-2026` (default; faster and cheaper) or `deep-research-max-preview-04-2026` (more thorough, longer runs).
- **`file_search` / `google_maps`:** Also search your File Search stores, or ground on Google Maps (default: off).
- **`thinking_summaries`:** Include or suppress the agent's thought summaries.
- The agent always grounds on Google Search. Each search and Maps call is billed and counted in the cost line.

### `/gemini check_permissions`

Check if the bot has the necessary permissions in the current channel.

## Setup & Installation

### Prerequisites

- Python 3.11+
- Discord Bot Token
- Gemini API Key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jdmsharpe/discord-gemini.git
   cd discord-gemini
   ```

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
| `GEMINI_API_VERSION` | No | Override version (default `v1beta`; `v1` does not support text-to-speech or context caching) |
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

1. Run `/gemini chat` to start a conversation.
2. Send messages in the same channel to continue it.
3. Use the controls under each reply:
   - 🔄 Regenerate the last response
   - ⏯️ Pause or resume the conversation
   - ⏹️ End the conversation
   - **Tools** menu: turn tools on or off mid-conversation

### Music Prompt Examples

Try these prompts with `/gemini-tools music`:

- **Genres:** "minimal techno", "jazz fusion", "acoustic folk", "orchestral score"
- **Instruments:** "piano and strings", "guitar and drums", "electronic synthesizers"
- **Moods:** "upbeat and energetic", "calm and meditative", "dark and atmospheric"
- **Combinations:** "upbeat electronic dance music with heavy bass", "peaceful acoustic guitar with nature sounds"

## Development

### Testing

Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). All tests are mocked (no real API calls).

```bash
# Install developer tooling if you have not already
python -m pip install -e ".[dev]"

# Run tests locally
python -m pytest -q

# Run tests in Docker
docker build --build-arg PYTHON_VERSION=3.11 -f Dockerfile.test -t discord-gemini-test:3.11 .
docker run --rm discord-gemini-test:3.11 python -m pytest -q

# Run linting and type checks in Docker
docker run --rm discord-gemini-test:3.11 sh -lc 'ruff check src tests && ruff format --check src tests && pyright'
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
