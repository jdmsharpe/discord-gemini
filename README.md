# Discord Gemini Bot

<div align="center">

![Hits](https://hitscounter.dev/count/tag.svg?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-gemini&title=repo%20views)
<a href="https://hub.docker.com/r/jsgreen152/discord-gemini" target="_blank" rel="noopener noreferrer">![Workflow](https://github.com/jdmsharpe/discord-gemini/actions/workflows/main.yml/badge.svg)</a>
  
</div>

## Features

### Text Generation
- **`/converse`**: Have conversations with Gemini AI models
- Support for multiple Gemini models (2.5 Flash, 2.5 Pro, etc.)
- Persistent conversation history with button controls

### Image Generation
- **`/generate_image`**: Generate images from text prompts
- Multiple aspect ratios and image count options
- Support for Gemini 2.5 Flash, along with Imagen 3, 4, and 4 Ultra
- Additional support for image editing with reference images

### Video Generation  
- **`/generate_video`**: Generate videos from text prompts or image inputs
- Uses Veo 2 to generate realistic video clips

### Text-to-Speech
- **`/text_to_speech`**: Convert text to lifelike speech
- 25+ voice options with different personalities
- Natural language style control
- High-quality WAV output (24kHz, 16-bit)

### Music Generation
- **`/generate_music`**: Create instrumental music using Lyria RealTime
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
2. Configure your Gemini API key in `src/config/auth.py`
3. Set up Discord bot permissions and guild IDs
4. Run the bot: `python src/bot.py`

## Requirements

- Python 3.8+
- google-genai ~1.21
- py-cord ~2.6  
- Pillow ~10.0

## Music Generation Examples

Try these prompts with `/generate_music`:

- **Genres**: "minimal techno", "jazz fusion", "acoustic folk", "orchestral score"
- **Instruments**: "piano and strings", "guitar and drums", "electronic synthesizers"
- **Moods**: "upbeat and energetic", "calm and meditative", "dark and atmospheric"
- **Combinations**: "upbeat electronic dance music with heavy bass", "peaceful acoustic guitar with nature sounds"

## Troubleshooting

### General Issues
- Ensure your API key has the necessary permissions
- Check that your bot has proper Discord permissions in the channel
- Verify your internet connection for streaming features
