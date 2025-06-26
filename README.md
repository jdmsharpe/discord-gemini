# Discord Gemini Bot

A Discord bot that enables Google Gemini API interactions, including text generation, image creation, video generation, text-to-speech, and music generation.

## Features

### Text Generation
- **`/chat`**: Have conversations with Gemini AI models
- Support for multiple Gemini models (2.5 Flash, 2.5 Pro, etc.)
- Persistent conversation history with button controls

### Image Generation
- **`/image`**: Generate images from text prompts
- Multiple aspect ratios and image count options
- Support for Gemini 2.5 Flash and Imagen
- Additional support for image editing with reference images

### Video Generation  
- **`/video`**: Generate videos from text prompts or image inputs
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
