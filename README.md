# Discord Gemini Bot

![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-gemini&label=discord-gemini&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![Workflow](https://github.com/jdmsharpe/discord-gemini/actions/workflows/main.yml/badge.svg)](https://hub.docker.com/r/jsgreen152/discord-gemini)

## Features

All commands are grouped under `/gemini` for clean namespacing.

### Text Generation

- **`/gemini converse`**: Have conversations with Gemini AI models
- Support for multiple Gemini models:
  - Gemini 3.0 Flash (Default), 3.0 Pro
  - Gemini 2.5 Pro, 2.5 Flash, 2.5 Flash Lite
  - Gemini 2.0 Flash, 2.0 Flash Lite
  - Legacy support for Gemini 1.5 models
- Persistent conversation history with button controls
- Multimodal support (text + images)

### Image Generation

- **`/gemini image`**: Generate images from text prompts
- Multiple model options:
  - Gemini 3.0 Pro Image (Default)
  - Gemini 2.5 Flash Image
  - Imagen 4, 4 Ultra, and 4 Fast
- Multiple aspect ratios and image count options (1-4 images)
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

- **`/gemini music`**: Create instrumental music using Lyria RealTime
- Real-time streaming music generation
- Customizable parameters:
  - BPM (60-200)
  - Musical scale/key selection
  - Density and brightness controls
  - Prompt guidance strength
- High-quality stereo audio output (48kHz, 16-bit)
- Support for various genres, instruments, and moods

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your values:
   - `BOT_TOKEN`: Your Discord bot token
   - `GUILD_IDS`: Comma-separated list of Discord server IDs
   - `GEMINI_API_KEY`: Your Google Gemini API key
3. Set up Discord bot permissions in your server
4. Run the bot: `python src/bot.py`

## Requirements

- Python 3.10+
- google-genai ~1.52
- py-cord ~2.6
- Pillow ~12.0
- python-dotenv ~1.2

## Music Generation Examples

Try these prompts with `/gemini music`:

- **Genres**: "minimal techno", "jazz fusion", "acoustic folk", "orchestral score"
- **Instruments**: "piano and strings", "guitar and drums", "electronic synthesizers"
- **Moods**: "upbeat and energetic", "calm and meditative", "dark and atmospheric"
- **Combinations**: "upbeat electronic dance music with heavy bass", "peaceful acoustic guitar with nature sounds"

## Troubleshooting

### General Issues

- Ensure your API key has the necessary permissions
- Check that your bot has proper Discord permissions in the channel
- Verify your internet connection for streaming features
