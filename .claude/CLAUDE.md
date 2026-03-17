# Discord Gemini Bot - Developer Reference

This document serves as a comprehensive reference for future development work on the Discord Gemini Bot.

## Project Overview

A Discord bot that integrates Google's Gemini AI API to provide text generation, image generation, video generation, text-to-speech, music generation, and deep research capabilities through Discord slash commands.

## Project Structure

```text
discord-gemini/
├── src/
│   ├── bot.py                 # Main bot entry point
│   ├── gemini_api.py          # Core API integration & slash commands (2590+ lines)
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

- **google-genai** ~1.65: Google's Gemini API client library
- **py-cord** ~2.6: Discord bot framework (fork of discord.py)
- **Pillow** ~12.0: Image processing library
- **aiohttp**: Async HTTP client for downloading attachments

## Available Models (As of March 2026)

### Text Generation Models (`/gemini chat`)

- `gemini-3.1-pro-preview` - Gemini 3.1 Pro (default)
- `gemini-3.1-flash-lite-preview` - Gemini 3.1 Flash Lite
- `gemini-3-flash-preview` - Gemini 3.0 Flash
- `gemini-2.5-pro` - State-of-the-art reasoning model
- `gemini-2.5-flash` - Best price-performance
- `gemini-2.5-flash-lite` - Ultra-fast, cost-efficient
- `gemini-2.0-flash` - Second generation
- `gemini-2.0-flash-lite` - Small workhorse

### Image Generation Models (`/gemini image`)

- `gemini-3.1-flash-image-preview` - Gemini 3.1 Flash Image (default, supports editing)
- `gemini-3-pro-image-preview` - Gemini 3.0 Pro Image (supports editing)
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

### Deep Research Agent (`/gemini research`)

- `deep-research-pro-preview-12-2025` - Gemini Deep Research (powered by Gemini 3.1 Pro)

## Architecture Overview

### Main Components

1. **GeminiAPI Cog** (`src/gemini_api.py`)
   - Discord Cog that handles all slash commands
   - Manages conversation state, HTTP sessions, and API calls
   - Uses async/await for non-blocking operations

2. **ButtonView** (`src/button_view.py`)
   - Handles interactive UI buttons for conversations
   - Provides pause, resume, regenerate, and end conversation controls
   - Provides a tool select menu for Google Search and Code Execution toggling

3. **Utility Classes** (`src/util.py`)
   - `ChatCompletionParameters`: Conversation state (includes `tools` list, `cache_name`, `cached_history_length`, `uploaded_file_names`, `thinking_level`, `thinking_budget`)
   - Cache constants: `CACHE_MIN_TOKEN_COUNT` (Gemini 3.x models → min tokens), `CACHE_TTL` (default 1 hour)
   - Attachment size constants: `ATTACHMENT_MAX_INLINE_SIZE` (100 MB), `ATTACHMENT_PDF_MAX_INLINE_SIZE` (50 MB), `ATTACHMENT_FILE_API_THRESHOLD` (20 MB), `ATTACHMENT_FILE_API_MAX_SIZE` (2 GB)
   - Tool constants: `TOOL_GOOGLE_SEARCH`, `TOOL_CODE_EXECUTION`, `TOOL_FILE_SEARCH`, `AVAILABLE_TOOLS`
   - `FILE_SEARCH_INCOMPATIBLE_TOOLS`: Tools that cannot be combined with file_search
   - `ImageGenerationParameters`: Image generation config
   - `VideoGenerationParameters`: Video generation config
   - `SpeechGenerationParameters`: TTS config
   - `MusicGenerationParameters`: Music generation config
   - `ResearchParameters`: Deep research config (prompt, agent, file_search flag)
   - Pricing constants: `MODEL_PRICING` (model → (input, output) per million tokens), `calculate_cost()` (accepts optional `thinking_tokens` billed at output rate)
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
# - client.aio.interactions.create()  # Deep Research (Interactions API)
# - client.aio.interactions.get()     # Poll research status
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
- `self.last_view_messages`: Maps user → last Discord message with buttons (for stripping previous view)
- `_strip_previous_view(user)`: Removes buttons from the previous turn's message; called in both chat command and on_message

#### 4. Explicit Context Caching

Long conversations on Gemini 3.x models automatically use explicit caching to reduce cost. Gemini 2.5 and below rely on implicit caching (automatic, no dev work). After each response, `_maybe_create_cache()` checks `usage_metadata.prompt_token_count` against the model's minimum threshold (`CACHE_MIN_TOKEN_COUNT`). When crossed, a cache is created containing the system instruction and conversation history so far. Subsequent turns send only the uncached portion of history with `cached_content` in the config, falling back to full history if the cache expires.

- Cache is created when the token threshold is first exceeded on a supported model
- System instruction is included in the cache, so it's omitted from per-request config when a cache is active
- TTL is 1 hour (`CACHE_TTL`); refreshed on each turn via `_refresh_cache_ttl()` using `caches.update()`
- If the cache expires despite TTL refresh, the next request retries transparently with full history
- When the uncached tail grows large enough (≥ model threshold), `_recache()` replaces the old cache with a new one covering the full history
- Caches are deleted when the conversation ends (stop button) or when the cog unloads
- `_delete_conversation_cache()` handles cleanup and is called from both `ButtonView.stop_button` and `cog_unload`

#### 5. Typing Indicators

Long-running operations show typing indicators to improve UX:

```python
typing_task = asyncio.create_task(self.keep_typing(ctx.channel))
# ... do work ...
typing_task.cancel()
```

## Slash Commands Reference

All commands are grouped under `/gemini` using `SlashCommandGroup` for clean namespacing. This allows multiple AI provider bots to coexist (e.g., `/gemini chat` vs `/openai chat`).

### `/gemini chat`

**Purpose**: Multi-turn conversations with context preservation
**Parameters**:

- Total parameters: 18 (1 required + 17 optional)

- `prompt` (required): Initial message
- `model`: Gemini model selection (default: gemini-3.1-pro-preview)
- `system_instruction`: Behavioral guidelines
- `attachment`: File attachment for multimodal input (image, PDF, audio, video, document). Max 2 GB.
- `url`: URL to a file (PDF, image, etc.) for the model to read. Gemini 2.5+ only.
- `thinking_level`: Thinking depth for Gemini 3 models — Minimal, Low, Medium, High (default: not set / model default)
- `thinking_budget`: Thinking token budget for Gemini 2.5 models — -1 for dynamic, 0 to disable (default: not set)
- `google_search`: Enable grounding with Google Search
- `code_execution`: Enable built-in code execution
- `google_maps`: Enable Google Maps grounding (model-dependent)
- `url_context`: Enable URL Context retrieval (model-dependent)
- `file_search`: Enable File Search over configured document stores (model-dependent)
- `media_resolution`: Resolution for media inputs — Low, Medium, or High (default: not set / API default)
- Advanced: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`

**Implementation Notes**:

- Stores conversation in `self.conversations` dict
- Uses `on_message` listener to handle follow-ups
- Button controls pause/resume conversation
- Tool select menu updates `conversation.params.tools` mid-conversation
- Grounded responses may include a "Sources" embed built from `grounding_metadata`
- File Search requires `GEMINI_FILE_SEARCH_STORE_IDS` env var with comma-separated store IDs
- File Search is incompatible with Google Search, Google Maps, and URL Context (enforced automatically)
- File Search store IDs are injected into the tool config at runtime via `enrich_file_search_tools()`
- Supported models for File Search: gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash-lite
- `media_resolution` is set globally on `GenerateContentConfig` and persists across follow-up messages in the conversation
- `thinking_level` controls reasoning depth on Gemini 3 models (minimal, low, medium, high); defaults to "high" (dynamic) if not set
- `thinking_budget` controls reasoning token count on Gemini 2.5 models; -1 for dynamic, 0 to disable
- When either thinking param is set, `include_thoughts=True` is added to `ThinkingConfig` and thought summaries are shown as spoilered Discord embeds (light grey color)
- Full response parts (including thought signatures) are stored in conversation history to preserve multi-turn reasoning context
- Thinking tokens are tracked separately in pricing via `thoughts_token_count` from `usage_metadata`

### `/gemini image`

**Purpose**: Generate images from text prompts
**Parameters**:

- `prompt` (required): Image description
- `model`: Gemini or Imagen model (default: gemini-3.1-flash-image-preview)
- `number_of_images`: 1-4 images (default: 1)
- `aspect_ratio`: 1:1, 3:4, 4:3, 9:16, 16:9
- `attachment`: Reference image for editing (Gemini only)
- `image_size`: Output image resolution — 1K or 2K (Gemini only, default: not set / model default)
- `google_image_search`: Ground generation with Google Image Search (Gemini 3.1 Flash Image only)
- Advanced: `negative_prompt`, `seed`, `guidance_scale`, `person_generation`

**Implementation Notes**:

- Uses different APIs for Gemini vs Imagen models
- Gemini: `generate_content` with `response_modalities=['TEXT', 'IMAGE']`
  - Prompts automatically prefixed with "Create image: " (or "Create image(s): " for multiple images) to reduce text-only responses
  - Image editing (with attachment) preserves original prompt
  - `aspect_ratio` and `image_size` passed via `ImageConfig` on `GenerateContentConfig`
  - `google_image_search` adds a `GoogleSearch` tool with `SearchTypes(web_search, image_search)` — only for `gemini-3.1-flash-image-preview`
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

### `/gemini research`

**Purpose**: Run autonomous deep research tasks that produce detailed, cited reports
**Parameters**:

- `prompt` (required): Research question or topic to investigate
- `file_search`: Also search uploaded document stores (default: False)

**Implementation Notes**:

- Uses the Interactions API (`client.aio.interactions.create/get`), not `generate_content`
- Runs as a background task with polling (every 15 seconds, 20-minute timeout)
- Agent is `deep-research-pro-preview-12-2025` (powered by Gemini 3.1 Pro)
- Agent autonomously plans, searches the web, reads sources, and synthesizes a report
- Typical research tasks take 2-10 minutes to complete
- No typing indicator (too long-running)
- Report sent as a downloadable `.md` file attachment; header embed shows prompt/agent info only
- Optional `file_search` support requires `GEMINI_FILE_SEARCH_STORE_IDS` env var
- Estimated cost: $2-5 per task depending on complexity

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

| Limit               | Value       |
|---------------------|-------------|
| Embed description   | 4096 chars  |
| Total embed content | 6000 chars  |

**Truncation strategy by command:**

| Command   | Field               | Limit             | Reason                          |
|-----------|---------------------|-------------------|---------------------------------|
| chat      | user prompt         | 2000 chars        | Leave room for metadata         |
| chat      | model response      | 3500 char chunks  | Via `append_response_embeds()`  |
| image     | user prompt         | 2000 chars        | Leave room for metadata         |
| image     | model text response | 3800 chars        | When no images generated        |
| video     | user prompt         | 2000 chars        | Leave room for metadata         |
| tts       | input text          | 500 chars         | Displayed in embed summary      |
| music     | user prompt         | 2000 chars        | Leave room for metadata         |

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

- Supports all file types: images, PDFs, audio, video, documents, code
- Small files (≤20 MB) use inline_data: `{"inline_data": {"mime_type": "...", "data": bytes}}`
- Large files (>20 MB) automatically use the Gemini File API for efficiency
- File API uploads are tracked in `ChatCompletionParameters.uploaded_file_names` and cleaned up on conversation end
- External URLs use `Part.from_uri()` stored as `{"file_data": {"file_uri": "...", "mime_type": "..."}}`
- Attachment size validated before download (max 2 GB via File API)
- MIME types from Discord's `attachment.content_type`; URL MIME types inferred via `mimetypes.guess_type()`

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

- **`test_util.py`** (90 tests): Tests for all dataclasses (`ChatCompletionParameters`, `ImageGenerationParameters`, `VideoGenerationParameters`, `SpeechGenerationParameters`, `MusicGenerationParameters`, `ResearchParameters`, `EmbeddingParameters`) and utility functions (`chunk_text()`, `truncate_text()`, `resolve_tool_name()`, `filter_file_search_incompatible_tools()`, `calculate_cost()`), including conversation tools state, file search model compatibility, caching constants, cache field coverage, attachment size constants, `uploaded_file_names` isolation, `MODEL_PRICING` validation, thinking field defaults/isolation, `calculate_cost()` with thinking tokens, and `ImageGenerationParameters` new fields (`image_size`, `google_image_search`) defaults and isolation.

- **`test_button_view.py`** (13 tests): Tests for ButtonView button callbacks (regenerate, play/pause, stop) plus tool select initialization and callback behavior, including file_search option.

- **`test_gemini_api.py`** (100 tests): Tests for GeminiAPI cog initialization, HTTP session management, message handling, attachment fetching, response embed generation, image generation text/prompt truncation, image generation config (`_generate_image_with_gemini()` with `image_size`, `aspect_ratio`, `google_image_search`, combined config, and model-gating), tool metadata extraction (including file_search detection), `enrich_file_search_tools()`, attachment validation (`_validate_attachment_size()`), attachment preparation (`_prepare_attachment_part()` with inline/File API routing and fallback), uploaded file cleanup (`_cleanup_uploaded_files()`), deep research (`_run_deep_research()`, `_create_research_response_embeds()`), explicit context caching (create/delete/TTL refresh/periodic re-caching/error handling), pricing (`append_pricing_embed()`, `_track_daily_cost()` accumulation and per-user isolation), URL MIME type detection (`_guess_url_mime_type()` with YouTube URL handling and fallback), and thinking features (`_build_thinking_config()`, `extract_thinking_text()`, `_get_response_content_parts()`, `append_thinking_embeds()`, pricing with thinking tokens).

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
- This applies to all `/gemini` commands: chat, image, video, tts, music

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

### March 2026 - Image Generation Enhancements

- Added `image_size` parameter to `/gemini image` for controlling output resolution on Gemini models (1K, 2K)
- Added `google_image_search` parameter to `/gemini image` for grounding image generation with Google Image Search (Gemini 3.1 Flash Image only)
- `image_size` and `aspect_ratio` are now passed to Gemini models via `ImageConfig` on `GenerateContentConfig`
- `google_image_search` adds a `GoogleSearch` tool with `SearchTypes(web_search, image_search)` — model-gated to `gemini-3.1-flash-image-preview`
- `aspect_ratio` is now supported for Gemini models (previously listed as unsupported)
- Added `image_size` and `google_image_search` fields to `ImageGenerationParameters` in `util.py`
- Refactored `_generate_image_with_gemini()` to accept `ImageGenerationParameters` instead of individual args

### March 2026 - Thinking Configuration & Thought Signatures

- Added `thinking_level` parameter to `/gemini chat` for Gemini 3 models (Minimal, Low, Medium, High)
- Added `thinking_budget` parameter to `/gemini chat` for Gemini 2.5 models (-1 dynamic, 0 disable, or token count)
- When either thinking param is set, `ThinkingConfig` with `include_thoughts=True` is sent to the API
- Thought summaries displayed as spoilered embeds with `Colour.light_grey()` (matching discord-claude pattern)
- Added `append_thinking_embeds()`, `extract_thinking_text()`, `_get_response_content_parts()`, `_build_thinking_config()` helper functions
- **Fixed conversation history** to store full response content parts instead of only `response.text`, preserving thought signatures for multi-turn reasoning context
- **Fixed pricing** to include `thoughts_token_count` from `usage_metadata` — thinking tokens are billed at the output token rate
- Pricing embed now shows thinking token count when non-zero (e.g., `$0.0042 · 1,000 in / 500 out / 2,000 thinking · daily $0.10`)
- Added `thinking_level`, `thinking_budget` fields to `ChatCompletionParameters` in `util.py`
- Updated `calculate_cost()` to accept optional `thinking_tokens` parameter
- Updated `_track_daily_cost()` and `append_pricing_embed()` to propagate thinking tokens

### March 2026 - Multimodal Input Improvements

- **Media-first part ordering**: Attachments and URL parts are now placed before the text prompt in API requests, following Google's recommendation for better multimodal results. Applies to both initial `/gemini chat` commands and follow-up messages.
- **YouTube URL support**: The `url` parameter on `/gemini chat` now correctly detects YouTube URLs (youtube.com, youtu.be) and sets `mime_type` to `video/mp4` instead of falling back to `application/octet-stream`
- Added `_guess_url_mime_type()` helper function with YouTube regex detection and `mimetypes.guess_type()` fallback
- Added `re` import to `gemini_api.py`

### March 2026 - File Input Methods & Attachment Improvements

- **File size validation**: All attachment commands now validate size before downloading (max 2 GB)
- **Multi-format support**: `/gemini chat` attachment param now accepts any file type (image, PDF, audio, video, document) — no longer image-only
- **URL parameter**: New `url` option on `/gemini chat` for passing external file URLs directly to Gemini via `Part.from_uri()` (Gemini 2.5+ only)
- **File API for large files**: Attachments over 20 MB automatically use the Gemini File API instead of inline data for efficiency
  - `_prepare_attachment_part()` routes by size: ≤20 MB → inline, >20 MB → File API upload
  - `_upload_attachment_to_file_api()` saves to temp file, uploads with explicit MIME type, cleans up
  - Uploaded file names tracked in `ChatCompletionParameters.uploaded_file_names` for cleanup
  - Files cleaned up on conversation end (stop button) and cog unload
  - Falls back to inline data if File API upload fails
- Added `ATTACHMENT_MAX_INLINE_SIZE` (100 MB), `ATTACHMENT_PDF_MAX_INLINE_SIZE` (50 MB), `ATTACHMENT_FILE_API_THRESHOLD` (20 MB), `ATTACHMENT_FILE_API_MAX_SIZE` (2 GB) constants to `util.py`
- Added `_validate_attachment_size()`, `_prepare_attachment_part()`, `_upload_attachment_to_file_api()`, `_cleanup_uploaded_files()` methods to `GeminiAPI`
- Parts conversion logic in chat now handles `file_data` dicts via `types.Part.from_uri()`

### March 2026 - Pricing Feedback

- Added per-request cost and daily cost tracking to `/gemini chat` responses
- Each chat response (initial and follow-up) shows a pricing embed: `$cost · N in / N out · daily $total`
- Pricing and sources embeds are included in the same message as the response, before the buttons/tool select view
- Pricing embed color matches the response embed color (dark blue)
- Pricing embeds are configurable via `SHOW_COST_EMBEDS` env var (default: `true`); daily cost is always tracked regardless of this setting
- Added `MODEL_PRICING` dict to `util.py` with per-million-token rates for all 8 chat models
- Added `calculate_cost()` utility function in `util.py`
- Added `append_pricing_embed()` standalone function in `gemini_api.py`
- Added `daily_costs` dict and `_track_daily_cost()` method to `GeminiAPI` for per-user daily accumulation
- Token counts extracted from `response.usage_metadata` (`prompt_token_count`, `candidates_token_count`)
- Added `SHOW_COST_EMBEDS` to `config/auth.py` (default: `true`)

### March 2026 - Media Resolution

- Added `media_resolution` parameter to `/gemini chat` for controlling media tokenization quality
- Choices: Low, Medium, High (default: not set / API default)
- Set globally on `GenerateContentConfig` and persists across follow-up messages
- Added `media_resolution` field to `ChatCompletionParameters` in `util.py`

### March 2026 - Deep Research

- Added `/gemini research` command for autonomous multi-step research tasks
- Uses the Interactions API with `deep-research-pro-preview-12-2025` agent
- Agent autonomously plans, searches the web, reads sources, and synthesizes detailed reports
- Background polling with 15-second intervals and 20-minute timeout
- Optional `file_search` parameter to also search uploaded document stores
- Added `ResearchParameters` dataclass to `util.py`
- Added `_run_deep_research()` and `_create_research_response_embeds()` helper methods

### March 2026 - File Search (RAG)

- Added File Search tool to `/gemini chat` for retrieval-augmented generation over uploaded document stores
- New `file_search` boolean parameter on the chat slash command
- File Search can be toggled mid-conversation via the tool select dropdown in ButtonView
- Requires `GEMINI_FILE_SEARCH_STORE_IDS` env var (comma-separated store IDs)
- File Search is incompatible with Google Search, Google Maps, and URL Context (enforced automatically)
- Supported models: `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash-lite`
- Added `TOOL_FILE_SEARCH`, `FILE_SEARCH_INCOMPATIBLE_TOOLS`, `filter_file_search_incompatible_tools()` to `util.py`
- Added `enrich_file_search_tools()` method to `GeminiAPI` cog
- Updated `resolve_tool_name()` to handle enriched file_search configs with dynamic store IDs
- Updated `extract_tool_info()` to detect file_search usage via `retrieval_metadata`
- Added `GEMINI_FILE_SEARCH_STORE_IDS` to `config/auth.py`

### March 2026 - Explicit Context Caching

- Added automatic explicit caching for `/gemini chat` conversations on Gemini 3.x models
- When prompt token count exceeds the model's minimum threshold, conversation history is cached to reduce cost
- Supported models: `gemini-3.1-pro-preview` (4096), `gemini-3-flash-preview` (1024)
- Gemini 2.5 and below rely on implicit caching (automatic, no dev work needed)
- TTL refresh on each turn via `_refresh_cache_ttl()` using `caches.update()` prevents cache expiration between turns
- Periodic re-caching via `_recache()`: when the uncached tail exceeds the model's token threshold, the old cache is replaced with a new one covering the full conversation history
- Transparent fallback to full history if the cache expires despite TTL refresh
- Caches are cleaned up on conversation end and cog unload
- Added `CACHE_MIN_TOKEN_COUNT`, `CACHE_TTL` constants to `util.py`
- Added `cache_name`, `cached_history_length` fields to `ChatCompletionParameters`
- Added `_maybe_create_cache()`, `_recache()`, `_refresh_cache_ttl()`, `_delete_conversation_cache()` methods to `GeminiAPI`

### March 2026 - Gemini 3.1 Flash Image Release & Deprecation Cleanup

- Added `gemini-3.1-flash-image-preview` for image generation (new default)
- Removed `gemini-3-pro-preview` (shut down March 9, 2026; replaced by `gemini-3.1-pro-preview`)
- Removed `gemini-1.5-flash`, `gemini-1.5-flash-8b`, `gemini-1.5-pro` (no longer available in API)

### February 2026 - Chat Tool Calling

- Added built-in Gemini tool support to `/gemini chat`:
  - `google_search` slash option
  - `code_execution` slash option
- Added tool select dropdown in `ButtonView` for mid-conversation toggling
- Added `tools` to `ChatCompletionParameters` for persisted per-conversation tool state
- Added response tool metadata parsing and optional "Sources" embed for grounded citations

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

### February 2026 - Gemini 3.1 Pro

- Added `gemini-3.1-pro-preview` for text generation (new default)
- Updated google-genai from ~1.64 to ~1.65

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

**Last Updated**: March 2026
**Maintained by**: AI Assistant (Claude)
