# Discord Gemini Bot - Developer Reference

This document serves as a comprehensive reference for future development work on the Discord Gemini Bot.

## Project Overview

A Discord bot that integrates Google's Gemini AI API to provide text generation, image generation, video generation, text-to-speech, and music generation capabilities through Discord slash commands.

## Project Structure

```text
discord-gemini/
├── src/
│   ├── bot.py                 # Main bot entry point
│   ├── gemini_api.py          # Core API integration & slash commands (1960+ lines)
│   ├── button_view.py         # Discord UI button controls
│   ├── util.py                # Data classes and utility functions
│   └── config/
│       ├── __init__.py
│       └── auth.py            # API keys and guild IDs configuration
├── tests/
│   ├── __init__.py
│   ├── test_util.py           # Tests for dataclasses and utility functions
│   ├── test_button_view.py    # Tests for ButtonView UI controls
│   └── test_gemini_api.py     # Tests for GeminiAPI cog
├── .github/workflows/
│   └── main.yml               # CI pipeline (tests + Docker build)
├── .venv/                     # Virtual environment
├── requirements.txt           # Python dependencies
├── README.md                  # User-facing documentation
└── CLAUDE.md                  # This file - developer reference
```

## Core Dependencies

- **google-genai** ~1.48: Google's Gemini API client library
- **py-cord** ~2.6: Discord bot framework (fork of discord.py)
- **Pillow** ~12.0: Image processing library
- **aiohttp**: Async HTTP client for downloading attachments

## Available Models (As of December 2025)

### Text Generation Models (`/gemini converse`)

- `gemini-3-flash-preview` - Gemini 3.0 Flash (default)
- `gemini-3-pro-preview` - Gemini 3.0 Pro
- `gemini-2.5-pro` - State-of-the-art reasoning model
- `gemini-2.5-flash` - Best price-performance
- `gemini-2.5-flash-lite` - Ultra-fast, cost-efficient
- `gemini-2.0-flash` - Second generation
- `gemini-2.0-flash-lite` - Small workhorse
- `gemini-1.5-flash` - Legacy support
- `gemini-1.5-flash-8b` - Legacy support
- `gemini-1.5-pro` - Legacy support

### Image Generation Models (`/gemini image`)

- `gemini-3-pro-image-preview` - Gemini 3.0 Pro Image (default, supports editing)
- `gemini-2.5-flash-image` - Gemini 2.5 Flash Image (supports editing)
- `imagen-4.0-generate-001` - Imagen 4 standard
- `imagen-4.0-ultra-generate-001` - Imagen 4 ultra quality
- `imagen-4.0-fast-generate-001` - Imagen 4 fast generation

### Video Generation Models (`/gemini video`)

- `veo-3.1-generate-preview` - Veo 3.1 with native audio and video extension (default)
- `veo-3.1-fast-generate-preview` - Veo 3.1 fast variant
- `veo-3.0-generate-001` - Veo 3 with improved realism
- `veo-3.0-fast-generate-001` - Veo 3 fast variant
- `veo-2.0-generate-001` - Veo 2

### Text-to-Speech Models (`/gemini tts`)

- `gemini-2.5-flash-preview-tts` - Flash TTS (default)
- `gemini-2.5-pro-preview-tts` - Pro TTS

### Music Generation Models (`/gemini music`)

- `lyria-realtime-exp` - Lyria RealTime experimental model

## Architecture Overview

### Main Components

1. **GeminiAPI Cog** (`src/gemini_api.py`)
   - Discord Cog that handles all slash commands
   - Manages conversation state, HTTP sessions, and API calls
   - Uses async/await for non-blocking operations

2. **ButtonView** (`src/button_view.py`)
   - Handles interactive UI buttons for conversations
   - Provides pause, resume, regenerate, and end conversation controls

3. **Utility Classes** (`src/util.py`)
   - `ChatCompletionParameters`: Conversation state
   - `ImageGenerationParameters`: Image generation config
   - `VideoGenerationParameters`: Video generation config
   - `SpeechGenerationParameters`: TTS config
   - `MusicGenerationParameters`: Music generation config
   - `chunk_text()`: Splits long text for Discord embeds

### Key Design Patterns

#### 1. Native Async Client

The google-genai SDK provides a native async client via `client.aio`:

```python
# Direct async calls - no thread offloading needed
response = await self.client.aio.models.generate_content(
    model=model,
    contents=contents,
    config=config,
)

# Available async methods:
# - client.aio.models.generate_content()
# - client.aio.models.generate_images()
# - client.aio.models.generate_videos()
# - client.aio.operations.get()
# - client.aio.files.download()  # Note: only accepts str|File, not Video
```

#### 2. HTTP Session Management

A shared aiohttp session is used for downloading attachments:

```python
async def _get_http_session(self) -> aiohttp.ClientSession:
    if self._http_session and not self._http_session.closed:
        return self._http_session
    async with self._session_lock:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session
```

#### 3. Conversation State Management

Conversations are tracked using dictionaries:

- `self.conversations`: Maps conversation ID → Conversation object
- `self.message_to_conversation_id`: Maps message ID → conversation ID
- `self.views`: Maps user → ButtonView UI

#### 4. Typing Indicators

Long-running operations show typing indicators to improve UX:

```python
typing_task = asyncio.create_task(self.keep_typing(ctx.channel))
# ... do work ...
typing_task.cancel()
```

## Slash Commands Reference

All commands are grouped under `/gemini` using `SlashCommandGroup` for clean namespacing. This allows multiple AI provider bots to coexist (e.g., `/gemini converse` vs `/openai converse`).

### `/gemini converse`

**Purpose**: Multi-turn conversations with context preservation
**Parameters**:

- `prompt` (required): Initial message
- `model`: Gemini model selection (default: gemini-3-flash-preview)
- `system_instruction`: Behavioral guidelines
- `attachment`: Optional image for multimodal input
- Advanced: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`

**Implementation Notes**:

- Stores conversation in `self.conversations` dict
- Uses `on_message` listener to handle follow-ups
- Button controls pause/resume conversation

### `/gemini image`

**Purpose**: Generate images from text prompts
**Parameters**:

- `prompt` (required): Image description
- `model`: Gemini or Imagen model (default: gemini-3-pro-image-preview)
- `number_of_images`: 1-4 images (default: 1)
- `aspect_ratio`: 1:1, 3:4, 4:3, 9:16, 16:9
- `attachment`: Reference image for editing (Gemini only)
- Advanced: `negative_prompt`, `seed`, `guidance_scale`, `person_generation`

**Implementation Notes**:

- Uses different APIs for Gemini vs Imagen models
- Gemini: `generate_content` with `response_modalities=['TEXT', 'IMAGE']`
  - Prompts automatically prefixed with "Create image: " (or "Create image(s): " for multiple images) to reduce text-only responses
  - Image editing (with attachment) preserves original prompt
- Imagen: `generate_images` with full config support
- Text responses truncated to 3800 chars to prevent Discord embed errors

### `/gemini video`

**Purpose**: Generate videos from text or image prompts
**Parameters**:

- `prompt` (required): Video description
- `model`: Veo 2 or Veo 3 (default: veo-2.0-generate-001)
- `aspect_ratio`: 16:9 or 9:16
- `attachment`: Starting image for image-to-video
- `number_of_videos`: 1-2 videos
- `duration_seconds`: 5-8 seconds
- Advanced: `negative_prompt`, `enhance_prompt`, `person_generation`

**Implementation Notes**:

- Returns a long-running operation that must be polled
- Polls every 20 seconds, max 10 minutes timeout
- Downloads video files when complete

### `/gemini tts`

**Purpose**: Convert text to lifelike speech
**Parameters**:

- `input_text` (required): Text to speak (max 32k tokens)
- `model`: Flash TTS or Pro TTS
- `voice_name`: 25+ voice options (default: Kore)
- `style_prompt`: Natural language style control

**Implementation Notes**:

- Outputs WAV format (24kHz, 16-bit, Mono)
- Uses `response_modalities` to request audio

### `/gemini music`

**Purpose**: Generate instrumental music
**Parameters**:

- `prompt` (required): Musical description
- `duration`: 5-120 seconds (default: 30)
- `bpm`: 60-200 beats per minute
- `scale`: Musical key selection
- Advanced: `density`, `brightness`, `guidance`

**Implementation Notes**:

- Uses WebSocket streaming API
- Collects audio chunks in real-time
- Outputs stereo WAV (48kHz, 16-bit)
- Requires special API client with `api_version='v1alpha'`

## Common Implementation Patterns

### Adding a New Slash Command

Commands are added to the `gemini` SlashCommandGroup for consistent namespacing:

1. **Define the command** (uses `@gemini.command` instead of `@slash_command`):

```python
@gemini.command(
    name="command_name",
    description="Command description",
)
@option("param_name", description="Param desc", required=True, type=str)
async def command_name(self, ctx: ApplicationContext, param_name: str):
    await ctx.defer()  # Acknowledge immediately
    try:
        # Implementation
        await ctx.send_followup(embed=embed)
    except Exception as e:
        await ctx.send_followup(
            embed=Embed(title="Error", description=str(e), color=Colour.red())
        )
```

Note: `guild_ids` is set on the `SlashCommandGroup` definition, not individual commands.

1. **Create helper methods** for complex operations:

```python
async def _helper_method(self, params):
    """Private helper method for command logic."""
    # Implementation
    return result
```

1. **Use parameter dataclasses** from `util.py` for clean parameter handling

### Working with Attachments

```python
# Download attachment
attachment_bytes = await self._fetch_attachment_bytes(attachment)

# Convert to PIL Image
image = Image.open(BytesIO(attachment_bytes))

# Use in API call
contents = [prompt, image]
```

### Creating Discord Embeds

```python
embed = Embed(
    title="Title",
    description="Description with **markdown** support",
    color=Colour.dark_blue(),  # or .red(), .green(), .orange()
)

# Add image
embed.set_image(url="attachment://filename.png")

# Send with files
await ctx.send_followup(embed=embed, files=[File(path)])
```

### Discord Embed Limits

Discord enforces strict limits on embed content. The bot handles these automatically:

| Limit | Value |
|-------|-------|
| Embed description | 4096 chars |
| Total embed content | 6000 chars |

**Truncation strategy by command:**

| Command | Field | Limit | Reason |
|---------|-------|-------|--------|
| converse | user prompt | 2000 chars | Leave room for metadata |
| converse | model response | 3500 char chunks | Via `append_response_embeds()` |
| image | user prompt | 2000 chars | Leave room for metadata |
| image | model text response | 3800 chars | When no images generated |
| video | user prompt | 2000 chars | Leave room for metadata |
| tts | input text | 500 chars | Displayed in embed summary |
| music | user prompt | 2000 chars | Leave room for metadata |

**Key functions:**

- `append_response_embeds()` in `gemini_api.py` - Chunks model responses and enforces 20000 char total limit with 3500 char chunks
- `chunk_text()` in `util.py` - Splits text into configurable segments (default 4096 chars)
- `truncate_text()` in `util.py` - Truncates text to max length with customizable suffix (default "...")

### Handling Long-Running Operations

```python
# Start typing indicator
typing_task = asyncio.create_task(self.keep_typing(ctx.channel))

try:
    # Long operation
    result = await long_operation()
finally:
    # Always cancel typing
    if typing_task:
        typing_task.cancel()
```

## Important Implementation Details

### Conversation Management

- Only ONE conversation per user per channel allowed
- Conversations persist until explicitly ended or bot restarts
- Follow-up messages automatically handled by `on_message` listener
- Button state managed through `ButtonView` class

### Multimodal Input

- Attachments converted to inline_data format for API
- Format: `{"inline_data": {"mime_type": "...", "data": bytes}}`
- Supported by most Gemini models

### Response Handling

- Long responses split into 3500-char chunks to avoid Discord limits
- Multiple embeds sent as separate messages
- All messages in a conversation tracked via `message_to_conversation_id`

### Error Handling

- All slash commands wrapped in try/except
- Errors shown to users via red-colored embeds
- Detailed logging with `self.logger`

### Resource Cleanup

- HTTP session closed on cog unload (`cog_unload()`)
- Gemini async client closed via `client.aio.aclose()`
- Gemini sync client closed via `client.close()`
- Proper async task cancellation
- Temporary files deleted after sending

## Testing

### Running Tests

Run tests from the project root using `PYTHONPATH` to ensure proper imports:

```bash
# Windows PowerShell (with venv)
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/ -v

# Unix/macOS (with venv)
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

### Test Structure

- **`test_util.py`** (43 tests): Tests for all dataclasses (`ChatCompletionParameters`, `ImageGenerationParameters`, `VideoGenerationParameters`, `SpeechGenerationParameters`, `MusicGenerationParameters`, `EmbeddingParameters`) and utility functions (`chunk_text()`, `truncate_text()`).

- **`test_button_view.py`** (9 tests): Tests for ButtonView button callbacks (regenerate, play/pause, stop) including user permission checks and conversation state management.

- **`test_gemini_api.py`** (21 tests): Tests for GeminiAPI cog initialization, HTTP session management, message handling, attachment fetching, response embed generation, and image generation text/prompt truncation.

### CI Pipeline

Tests run automatically on every push and PR via GitHub Actions (`.github/workflows/main.yml`). The Docker build only proceeds if tests pass.

### Writing New Tests

When adding new functionality:

1. Add unit tests for new dataclasses/utility functions in `test_util.py`
2. Add tests for new button interactions in `test_button_view.py`
3. Add tests for new cog methods in `test_gemini_api.py`
4. Use `unittest.IsolatedAsyncioTestCase` for async tests
5. Use `MagicMock`/`AsyncMock` to mock Discord and API dependencies

### Manual Testing Checklist

When making changes, also manually test:

1. **Basic functionality**: Does the command work with minimal parameters?
2. **Error cases**: Invalid inputs, API errors, network issues
3. **Edge cases**: Very long prompts, many attachments, rapid requests
4. **UI interactions**: Buttons work correctly, no state conflicts
5. **Concurrent usage**: Multiple users, multiple channels
6. **Resource cleanup**: No memory leaks, sessions closed properly

## Common Issues & Solutions

### Issue: Command not showing up in Discord

**Solution**: Check `GUILD_IDS` in `auth.py`, restart bot, wait for sync

### Issue: "Missing key inputs argument" error

**Solution**: Check `GEMINI_API_KEY` is set in `auth.py`

### Issue: Conversation not responding to follow-ups

**Solution**: Check `on_message` listener is working, verify conversation stored in dict

### Issue: Images/videos not downloading

**Solution**: Check HTTP session not closed, verify file download logic

### Issue: Music generation hangs

**Solution**: Check WebSocket connection, verify API version, check receiver task

### Issue: Image generation returns text instead of images

**Solution**: For Gemini models, prompts are automatically prefixed with "Create image: " (or "Create image(s): " for multiple) to guide the model. If still getting text-only responses, try being more explicit in your prompt (e.g., "a red sports car in a sunny parking lot" provides more detail than just "car")

### Issue: "Invalid Form Body - embed description must be 4096 or fewer" error

**Solution**: All commands now automatically truncate content to fit Discord's 4096 character embed limit:
- **User prompts**: Truncated to 2000 characters (with "..." indicator)
- **Model text responses**: Truncated to 3800 characters (with "..." indicator)
- This applies to all `/gemini` commands: converse, image, video, tts, music

If you see truncated content, either shorten your input or the model returned an unusually long response.

## Future Enhancement Ideas

- [ ] Support for Gemini Live API (real-time voice conversations)
- [ ] Multi-speaker TTS support
- [ ] Video editing capabilities
- [ ] Persistent conversation storage (database)
- [ ] Custom voice cloning
- [ ] Batch generation commands
- [ ] User preference storage
- [ ] Rate limiting per user
- [ ] Admin commands for bot management

## Version History

### December 2025 - Image Generation Fixes

- Fixed Discord embed error when image generation returns long text responses
- Added automatic truncation to prevent "must be 4096 or fewer" errors:
  - User prompts truncated to 2000 chars across all commands
  - Model text responses truncated to 3800 chars
- Added `truncate_text()` utility function in `util.py` for consistent truncation across codebase
- Improved Gemini image prompts with "Create image:" / "Create image(s):" prefix to reduce text-only responses
- Image editing workflows preserve original prompts without prefix

### December 2025 - Native Async & Model Updates

- Migrated from `asyncio.to_thread()` wrappers to native `client.aio.*` async calls
- Added `veo-3.0-fast-generate-001` video model
- Removed deprecated Imagen 3 (no longer available in API)
- Improved resource cleanup in `cog_unload()` with proper client shutdown

### November 2025 - Gemini 3.0 Pro

- Added `gemini-3-pro-preview` for text generation (new default)
- Added `gemini-3-pro-image-preview` for image generation (new default)

### November 2025 - Model Updates

- Updated to google-genai 1.48.0
- Replaced deprecated models:
  - `gemini-2.0-flash-preview-image-generation` → `gemini-2.5-flash-image`
  - Imagen 4 preview models → GA versions
  - Fixed `gemini-2.5-flash-lite-preview-06-17` → `gemini-2.5-flash-lite`
- Added Veo 3, 3.1, and 3.1 Fast support
- Added Imagen 4 Fast variant
- Veo 3.1 features: native audio generation, video extension, reference images

## Useful Commands

```bash
# Create virtual environment
python -m venv .venv

# Activate venv (Windows)
.venv\Scripts\activate

# Install/update requirements
pip install -r requirements.txt

# Update all packages
pip list --outdated
pip install --upgrade package-name

# Run bot
python src/bot.py
```

## Additional Resources

- [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [py-cord Documentation](https://docs.pycord.dev/)
- [Discord Developer Portal](https://discord.com/developers/applications)
- [Google AI Studio](https://aistudio.google.com/) - Test models & get API key

---

**Last Updated**: December 2025
**Maintained by**: AI Assistant (Claude)
