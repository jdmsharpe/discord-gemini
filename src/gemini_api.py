# Standard library imports
import asyncio
import contextlib
import logging
import mimetypes
import re
import time
import wave
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

# Third-party imports
import aiohttp

# Discord imports
import discord
from discord import Attachment, Colour, Embed, File, Member, User
from discord.commands import (
    ApplicationContext,
    OptionChoice,
    SlashCommandGroup,
    option,
)
from discord.ext import commands

# Google AI imports
from google import genai
from google.genai import types
from PIL import Image

# Local imports
from button_view import ButtonView
from config.auth import (
    ENABLE_CUSTOM_TOOLS,
    GEMINI_API_KEY,
    GEMINI_FILE_SEARCH_STORE_IDS,
    GUILD_IDS,
    SHOW_COST_EMBEDS,
)
from exceptions import APICallError, MusicGenerationError, ValidationError
from tools import execute_tool_call, get_tool_callables
from util import (
    ATTACHMENT_FILE_API_MAX_SIZE,
    ATTACHMENT_FILE_API_THRESHOLD,
    AVAILABLE_TOOLS,
    CACHE_MIN_TOKEN_COUNT,
    CACHE_TTL,
    LYRIA_3_MODELS,
    LYRIA_REALTIME_MODEL,
    MAX_AGENTIC_ITERATIONS,
    TOOL_CODE_EXECUTION,
    TOOL_FILE_SEARCH,
    TOOL_GOOGLE_MAPS,
    TOOL_GOOGLE_SEARCH,
    TOOL_URL_CONTEXT,
    TYPING_INDICATOR_INTERVAL,
    VIDEO_GENERATION_TIMEOUT,
    WS_DRAIN_INTERVAL,
    AgenticResult,
    ChatCompletionParameters,
    ImageGenerationParameters,
    MusicGenerationParameters,
    ResearchParameters,
    SpeechGenerationParameters,
    VideoGenerationParameters,
    calculate_cost,
    calculate_image_cost,
    calculate_tts_cost,
    calculate_video_cost,
    check_mutually_exclusive_tools,
    chunk_text,
    filter_file_search_incompatible_tools,
    filter_supported_tools_for_model,
    resolve_tool_name,
    truncate_text,
)

# Google Gemini brand blue (#4285F4) used for all response embeds
GEMINI_BLUE = Colour(0x4285F4)


@dataclass
class Conversation:
    """A dataclass to store conversation state."""

    params: ChatCompletionParameters
    history: list[dict[str, Any]]


class CitationInfo(TypedDict):
    title: str
    uri: str


class UrlContextInfo(TypedDict):
    retrieved_url: str
    status: str


class ToolInfo(TypedDict):
    tools_used: list[str]
    citations: list[CitationInfo]
    search_queries: list[str]
    url_context_sources: list[UrlContextInfo]
    maps_widget_token: str | None


class PermissionAwareChannel(Protocol):
    def permissions_for(self, member: Any) -> Any: ...


_YOUTUBE_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:youtube\.com/|youtu\.be/)", re.IGNORECASE)


def _guess_url_mime_type(url: str) -> str:
    """Guess MIME type for a URL, with special handling for YouTube URLs."""
    if _YOUTUBE_URL_RE.match(url):
        return "video/mp4"
    mime_type, _ = mimetypes.guess_type(url)
    return mime_type if mime_type is not None else "application/octet-stream"


def _build_lyria3_prompt(music_params: MusicGenerationParameters) -> str:
    """Translate structured music controls into natural-language guidance for Lyria 3."""
    weighted_prompts = music_params.to_weighted_prompts()
    prompt_blocks = []
    for weighted_prompt in weighted_prompts:
        text = weighted_prompt["text"]
        weight = weighted_prompt["weight"]
        if weight == 1.0 or len(weighted_prompts) == 1:
            prompt_blocks.append(text)
        else:
            prompt_blocks.append(f"{text} (importance {weight})")

    production_notes: list[str] = []
    if music_params.model == "lyria-3-clip-preview":
        production_notes.append("Generate a 30-second music clip.")
    else:
        production_notes.append(f"Target duration: about {music_params.duration} seconds.")

    if music_params.bpm is not None:
        production_notes.append(f"Tempo: {music_params.bpm} BPM.")
    if music_params.scale is not None:
        production_notes.append(f"Musical key or scale: {music_params.scale.replace('_', ' ')}.")
    if music_params.density is not None:
        production_notes.append(f"Density: {music_params.density} on a 0 to 1 scale.")
    if music_params.brightness is not None:
        production_notes.append(f"Brightness: {music_params.brightness} on a 0 to 1 scale.")
    if music_params.guidance != 4.0:
        production_notes.append(
            f"Prompt adherence target: {music_params.guidance} on a 0 to 6 scale."
        )

    if not production_notes:
        return "\n".join(prompt_blocks)

    return "\n".join(prompt_blocks) + "\n\nProduction notes:\n" + "\n".join(
        f"- {note}" for note in production_notes
    )


def _music_file_suffix_for_mime_type(mime_type: str | None) -> str:
    """Map an audio MIME type to a Discord-friendly file extension."""
    if mime_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return "wav"
    return "mp3"


def _music_model_display_name(model: str) -> str:
    """Return a human-friendly name for supported music models."""
    if model == "lyria-3-pro-preview":
        return "Lyria 3 Pro Preview"
    if model == "lyria-3-clip-preview":
        return "Lyria 3 Clip Preview"
    return "Lyria RealTime Experimental"


def _guess_attachment_mime_type(attachment: Attachment) -> str:
    """Guess MIME type for a Discord attachment when content_type is absent."""
    if attachment.content_type:
        return attachment.content_type
    mime_type, _ = mimetypes.guess_type(attachment.filename)
    return mime_type if mime_type is not None else "application/octet-stream"


def append_response_embeds(embeds, response_text):
    # Compute how many characters are already used by existing embeds (e.g. thinking embed).
    # Discord enforces a 6000-char total per message; leave 500 chars headroom for pricing/sources.
    used = sum(len(e.title or "") + len(e.description or "") for e in embeds)
    available = max(500, 5500 - used)
    if len(response_text) > available:
        response_text = response_text[: available - 40] + "\n\n... [Response truncated]"

    for index, chunk in enumerate(chunk_text(response_text, 3500), start=1):
        embeds.append(
            Embed(
                title="Response" + f" (Part {index})" if index > 1 else "Response",
                description=chunk,
                color=GEMINI_BLUE,
            )
        )


def append_thinking_embeds(embeds: list[Embed], thinking_text: str) -> None:
    """Append thinking summary as a spoilered Discord embed."""
    if not thinking_text:
        return

    if len(thinking_text) > 3500:
        thinking_text = thinking_text[:3450] + "\n\n... [thinking truncated]"

    embeds.append(
        Embed(
            title="Thinking",
            description=f"||{thinking_text}||",
            color=Colour.light_grey(),
        )
    )


def extract_thinking_text(response) -> str:
    """Extract thought summary text from a Gemini response."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return ""
    thinking_parts = []
    for part in parts:
        if getattr(part, "thought", False) and getattr(part, "text", None):
            thinking_parts.append(part.text)
    return "\n\n".join(thinking_parts)


def _get_response_content_parts(response) -> list | None:
    """Get the raw content parts from a Gemini response for history storage.

    Preserves all parts including thought signatures for multi-turn context.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    if content is None:
        return None
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    return list(parts)


def _build_thinking_config(thinking_level: str | None, thinking_budget: int | None):
    """Build a ThinkingConfig from user parameters, or return None."""
    if thinking_level is None and thinking_budget is None:
        return None
    kwargs: dict[str, Any] = {"include_thoughts": True}
    if thinking_level is not None:
        kwargs["thinking_level"] = thinking_level
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    return types.ThinkingConfig(**kwargs)


def extract_tool_info(response) -> ToolInfo:
    """
    Extract tool usage and citation data from a Gemini response object.
    """
    tool_info: ToolInfo = {
        "tools_used": [],
        "citations": [],
        "search_queries": [],
        "url_context_sources": [],
        "maps_widget_token": None,
    }
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return tool_info

    candidate = candidates[0]
    grounding_metadata = getattr(candidate, "grounding_metadata", None)

    search_used = False
    maps_used = False
    if grounding_metadata is not None:
        web_search_queries = getattr(grounding_metadata, "web_search_queries", None)
        if web_search_queries:
            tool_info["search_queries"] = [
                query for query in web_search_queries if isinstance(query, str) and query
            ]

        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
        seen_uris = set()
        citations: list[CitationInfo] = []
        for chunk in grounding_chunks:
            web_chunk = getattr(chunk, "web", None)
            maps_chunk = getattr(chunk, "maps", None)

            if web_chunk is not None:
                uri = getattr(web_chunk, "uri", None)
                if uri and uri not in seen_uris:
                    title = getattr(web_chunk, "title", None) or uri
                    citations.append({"title": title, "uri": uri})
                    seen_uris.add(uri)
                    search_used = True

            if maps_chunk is not None:
                uri = getattr(maps_chunk, "uri", None)
                if uri and uri not in seen_uris:
                    title = getattr(maps_chunk, "title", None) or uri
                    citations.append({"title": title, "uri": uri})
                    seen_uris.add(uri)
                maps_used = True

        if citations:
            tool_info["citations"] = citations

        if getattr(grounding_metadata, "search_entry_point", None) is not None:
            search_used = True

        maps_widget_token = getattr(grounding_metadata, "google_maps_widget_context_token", None)
        if maps_widget_token:
            tool_info["maps_widget_token"] = str(maps_widget_token)
            maps_used = True

        if tool_info["search_queries"]:
            search_used = True

    if search_used:
        tool_info["tools_used"].append("google_search")
    if maps_used:
        tool_info["tools_used"].append("google_maps")

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if parts:
        code_execution_used = any(
            getattr(part, "executable_code", None) or getattr(part, "code_execution_result", None)
            for part in parts
        )
        if code_execution_used:
            tool_info["tools_used"].append("code_execution")

    url_context_metadata = getattr(candidate, "url_context_metadata", None)
    if url_context_metadata is not None:
        url_metadata_entries = getattr(url_context_metadata, "url_metadata", None) or []
        parsed_sources: list[UrlContextInfo] = []
        for entry in url_metadata_entries:
            retrieved_url = getattr(entry, "retrieved_url", None)
            if not retrieved_url:
                continue
            status = getattr(entry, "url_retrieval_status", None)
            parsed_sources.append(
                {
                    "retrieved_url": str(retrieved_url),
                    "status": str(status) if status is not None else "UNKNOWN",
                }
            )

        if parsed_sources:
            tool_info["url_context_sources"] = parsed_sources
            tool_info["tools_used"].append("url_context")

    # Detect file_search usage via retrieval_metadata on the candidate
    retrieval_metadata = getattr(candidate, "retrieval_metadata", None)
    if retrieval_metadata is not None:
        tool_info["tools_used"].append("file_search")
    elif grounding_metadata is not None and not search_used and not maps_used:
        # Grounding metadata present without search/maps indicates file_search
        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
        if grounding_chunks:
            tool_info["tools_used"].append("file_search")

    return tool_info


def append_sources_embed(embeds: list[Embed], tool_info: ToolInfo) -> None:
    """
    Add a compact sources embed for grounded web responses.
    """
    citations = tool_info["citations"]
    url_context_sources = tool_info["url_context_sources"]
    if (not citations and not url_context_sources) or len(embeds) >= 10:
        return

    source_lines: list[str] = []
    for index, citation in enumerate(citations[:8], start=1):
        safe_title = truncate_text(citation["title"], 120)
        source_lines.append(f"{index}. [{safe_title}]({citation['uri']})")

    if url_context_sources:
        if source_lines:
            source_lines.append("")
        source_lines.append("**URL Context**")
        for source in url_context_sources[:6]:
            safe_url = truncate_text(source["retrieved_url"], 200)
            source_lines.append(f"- {safe_url} ({source['status']})")

    description = "\n".join(source_lines)
    queries = tool_info["search_queries"]
    if queries:
        query_preview = truncate_text(", ".join(queries[:3]), 500)
        description += f"\n\n**Queries:** {query_preview}"
    if tool_info["maps_widget_token"]:
        description += "\n\n**Maps Widget:** `google_maps_widget_context_token` returned."

    embeds.append(Embed(title="Sources", description=description, color=GEMINI_BLUE))


def append_pricing_embed(
    embeds: list[Embed],
    model: str,
    input_tokens: int,
    output_tokens: int,
    daily_cost: float,
    thinking_tokens: int = 0,
    google_maps_grounded: bool = False,
) -> None:
    """Append a compact pricing embed showing cost and token usage."""
    cost = calculate_cost(model, input_tokens, output_tokens, thinking_tokens, google_maps_grounded)
    parts = []
    parts.append(f"${cost:.4f}")
    if thinking_tokens > 0:
        parts.append(f"{input_tokens:,} in / {output_tokens:,} out / {thinking_tokens:,} thinking")
    else:
        parts.append(f"{input_tokens:,} tokens in / {output_tokens:,} tokens out")
    if google_maps_grounded:
        parts.append("Maps grounded")
    parts.append(f"daily ${daily_cost:.2f}")
    embeds.append(Embed(description=" · ".join(parts), color=GEMINI_BLUE))


class GeminiAPI(commands.Cog):
    # Slash command group for all Gemini commands: /gemini <subcommand>
    gemini = SlashCommandGroup("gemini", "Gemini AI commands", guild_ids=GUILD_IDS)

    def __init__(self, bot):
        """
        Initialize the GeminiAPI class.

        Args:
            bot: The bot instance.
        """
        self.bot = bot
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        logging.basicConfig(
            level=logging.DEBUG,  # Capture all levels of logs
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Dictionary to store conversation state for each chat interaction
        self.conversations: dict[int, Conversation] = {}
        # Dictionary to map any message ID to the main conversation ID for tracking
        self.message_to_conversation_id: dict[int, int] = {}
        # Dictionary to store UI views keyed by the conversation-starting user
        self.views: dict[Member | User, ButtonView] = {}
        # Track the last message carrying the view per user for cleanup
        self.last_view_messages: dict[Member | User, discord.Message] = {}
        # Daily cost tracking: (user_id, date_iso) -> cumulative cost
        self.daily_costs: dict[tuple[int, str], float] = {}
        self._http_session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    # -- ConversationHost protocol methods (used by ButtonView) --

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        """Return the Conversation for *conversation_id*, or ``None``."""
        return self.conversations.get(conversation_id)

    async def end_conversation(self, conversation_id: int, user: Member | User) -> None:
        """Fully tear down a conversation: delete cache, files, and state."""
        conversation = self.conversations.get(conversation_id)
        if conversation is not None:
            await self._delete_conversation_cache(conversation.params)
            await self._cleanup_uploaded_files(conversation.params)
            del self.conversations[conversation_id]
        # Remove all message-to-conversation mappings for this conversation
        stale_keys = [
            msg_id
            for msg_id, conv_id in self.message_to_conversation_id.items()
            if conv_id == conversation_id
        ]
        for key in stale_keys:
            del self.message_to_conversation_id[key]
        await self._cleanup_conversation(user)

    def _resolve_tools_for_view(
        self,
        selected_values: list[str],
        custom_functions_selected: bool,
        conversation,
    ) -> tuple[set[str], str | None]:
        """Resolve tool selection from ButtonView. Returns (active_names, message)."""
        from copy import deepcopy

        exclusive_error = check_mutually_exclusive_tools(set(selected_values))
        if exclusive_error:
            return set(), exclusive_error

        requested_tools = [deepcopy(AVAILABLE_TOOLS[name]) for name in selected_values]
        supported_tools, unsupported_tools = filter_supported_tools_for_model(
            conversation.params.model, requested_tools
        )
        supported_tools, incompatible_tools = filter_file_search_incompatible_tools(supported_tools)

        enrich_error = self.enrich_file_search_tools(supported_tools)
        if enrich_error:
            return set(), enrich_error

        conversation.params.tools = supported_tools
        conversation.params.custom_functions_enabled = custom_functions_selected

        active_names: set[str] = {
            name for tool in supported_tools if (name := resolve_tool_name(tool)) is not None
        }
        if custom_functions_selected:
            active_names.add("custom_functions")

        parts: list[str] = []
        if active_names:
            parts.append(f"Tools updated: {', '.join(sorted(active_names))}.")
        else:
            parts.append("Tools disabled for this conversation.")
        if unsupported_tools:
            unsupported_text = ", ".join(sorted(set(unsupported_tools)))
            parts.append(f"Skipped for model `{conversation.params.model}`: {unsupported_text}.")
        if incompatible_tools:
            incompatible_text = ", ".join(sorted(set(incompatible_tools)))
            parts.append(f"Disabled (incompatible with file_search): {incompatible_text}.")

        return active_names, " ".join(parts)

    def _track_daily_cost(self, user_id: int, cost: float) -> float:
        """Add a pre-calculated cost to the user's daily total and return the new daily total."""
        key = (user_id, date.today().isoformat())
        self.daily_costs[key] = self.daily_costs.get(key, 0.0) + cost
        return self.daily_costs[key]

    def _log_cost(
        self,
        command: str,
        user_id: int,
        model: str,
        cost: float,
        daily_total: float,
        **details: Any,
    ) -> None:
        """Log cost data persistently for auditing and analytics."""
        detail_parts = [f"{k}={v}" for k, v in details.items()]
        detail_str = " | ".join(detail_parts) if detail_parts else ""
        self.logger.info(
            "COST | command=%s | user=%s | model=%s | cost=$%.4f%s | daily=$%.4f",
            command,
            user_id,
            model,
            cost,
            f" | {detail_str}" if detail_str else "",
            daily_total,
        )

    async def _send_error_followup(
        self,
        ctx: ApplicationContext,
        error: Exception,
        command_name: str,
    ) -> None:
        """Log an error and send a red error embed as a followup message."""
        description = str(error)
        self.logger.error("Error in %s: %s", command_name, description, exc_info=True)
        if len(description) > 4000:
            description = description[:4000] + "\n\n... (error message truncated)"
        try:
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )
        except Exception as followup_err:
            self.logger.error(
                "Failed to send error followup for %s: %s",
                command_name,
                followup_err,
            )

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session and not self._http_session.closed:
            return self._http_session
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                timeout = aiohttp.ClientTimeout(total=300, connect=15)
                connector = aiohttp.TCPConnector(limit=20, limit_per_host=10, ttl_dns_cache=300)
                self._http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            return self._http_session

    async def _fetch_attachment_bytes(self, attachment: Attachment) -> bytes | None:
        session = await self._get_http_session()
        try:
            async with session.get(attachment.url) as response:
                if response.status == 200:
                    return await response.read()
                self.logger.warning(
                    "Failed to fetch attachment %s: HTTP %s",
                    attachment.url,
                    response.status,
                )
        except aiohttp.ClientError as error:
            self.logger.warning("Error fetching attachment %s: %s", attachment.url, error)
        return None

    def _validate_attachment_size(self, attachment: Attachment) -> str | None:
        """Validate an attachment's size against Gemini API limits.

        Returns an error message string if the attachment is too large,
        otherwise None.
        """
        if attachment.size > ATTACHMENT_FILE_API_MAX_SIZE:
            size_mb = attachment.size / (1024 * 1024)
            return f"Attachment is too large ({size_mb:.1f} MB). Maximum file size is 2 GB."
        return None

    async def _prepare_attachment_part(
        self,
        attachment: Attachment,
        uploaded_file_names: list[str] | None = None,
    ) -> dict | None:
        """Prepare a Discord attachment as a Gemini API content part.

        Uses inline data for small files and the File API for files
        exceeding ATTACHMENT_FILE_API_THRESHOLD.  Appends uploaded file
        names to uploaded_file_names for cleanup when provided.

        Returns a dict part (inline_data or file_data), or None on failure.
        """
        content_type = attachment.content_type or "application/octet-stream"
        use_file_api = attachment.size > ATTACHMENT_FILE_API_THRESHOLD

        attachment_data = await self._fetch_attachment_bytes(attachment)
        if attachment_data is None:
            return None

        if use_file_api:
            uploaded_file = await self._upload_attachment_to_file_api(
                attachment_data, attachment.filename, content_type
            )
            if uploaded_file:
                if uploaded_file_names is not None:
                    uploaded_file_names.append(uploaded_file.name)
                return {
                    "file_data": {
                        "file_uri": uploaded_file.uri,
                        "mime_type": uploaded_file.mime_type,
                    }
                }
            # Fall back to inline data if upload fails
            self.logger.warning("File API upload failed, falling back to inline data")

        return {
            "inline_data": {
                "mime_type": content_type,
                "data": attachment_data,
            }
        }

    async def _upload_attachment_to_file_api(
        self, data: bytes, filename: str, mime_type: str
    ) -> Any | None:
        """Upload attachment bytes to the Gemini File API.

        Saves to a temporary file, uploads, then cleans up the temp file.
        Returns the uploaded file object, or None on failure.
        """
        temp_path = Path(f"temp_upload_{filename}")
        try:
            temp_path.write_bytes(data)
            uploaded_file = await self.client.aio.files.upload(
                file=str(temp_path),
                config={"mime_type": mime_type},
            )
            self.logger.info(
                "Uploaded %s to File API: %s (%d bytes)",
                filename,
                uploaded_file.name,
                len(data),
            )
            return uploaded_file
        except Exception as e:
            self.logger.warning("Failed to upload %s to File API: %s", filename, e)
            return None
        finally:
            temp_path.unlink(missing_ok=True)

    async def _cleanup_uploaded_files(self, params: ChatCompletionParameters) -> None:
        """Delete files uploaded to the File API for a conversation."""
        for file_name in params.uploaded_file_names:
            try:
                await self.client.aio.files.delete(name=file_name)
                self.logger.info("Deleted uploaded file %s", file_name)
            except Exception as e:
                self.logger.warning("Failed to delete uploaded file %s: %s", file_name, e)
        params.uploaded_file_names.clear()

    def enrich_file_search_tools(self, tools: list[dict[str, Any]]) -> str | None:
        """Inject file search store IDs into file_search tool configs.

        Mutates the list in-place.  Returns an error message string if
        GEMINI_FILE_SEARCH_STORE_IDS is not configured, otherwise None.
        """
        for i, tool in enumerate(tools):
            if "file_search" in tool:
                if not GEMINI_FILE_SEARCH_STORE_IDS:
                    return (
                        "File Search requires GEMINI_FILE_SEARCH_STORE_IDS "
                        "to be set in your .env file."
                    )
                tools[i] = {
                    "file_search": {"file_search_store_names": GEMINI_FILE_SEARCH_STORE_IDS.copy()}
                }
        return None

    async def _maybe_create_cache(
        self, params: ChatCompletionParameters, history: list[dict[str, Any]], response
    ) -> None:
        """Create, refresh, or re-create an explicit cache based on conversation size."""
        threshold = CACHE_MIN_TOKEN_COUNT.get(params.model)
        if threshold is None:
            return  # Model relies on implicit caching

        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0

        # If a cache already exists, decide whether to refresh TTL or re-cache
        if params.cache_name:
            cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
            uncached_tokens = prompt_tokens - cached_tokens
            if uncached_tokens >= threshold:
                # Uncached tail has grown large enough — re-cache the full history
                await self._recache(params, history, prompt_tokens, uncached_tokens)
            else:
                # Extend TTL so the cache doesn't expire between turns
                await self._refresh_cache_ttl(params)
            return

        # No cache yet — create one if the conversation is large enough
        if prompt_tokens < threshold:
            return

        try:
            contents = [{"role": entry["role"], "parts": entry["parts"]} for entry in history]
            cache = await self.client.aio.caches.create(
                model=params.model,
                config=types.CreateCachedContentConfig(
                    display_name=f"conv-{params.conversation_id}",
                    system_instruction=params.system_instruction,
                    contents=cast(Any, contents),
                    ttl=CACHE_TTL,
                ),
            )
            params.cache_name = cache.name
            params.cached_history_length = len(history)
            self.logger.info(
                "Created cache %s for conversation %s (%d prompt tokens)",
                cache.name,
                params.conversation_id,
                prompt_tokens,
            )
        except Exception as e:
            self.logger.warning("Failed to create cache: %s", e)

    async def _recache(
        self,
        params: ChatCompletionParameters,
        history: list[dict[str, Any]],
        prompt_tokens: int,
        uncached_tokens: int,
    ) -> None:
        """Delete the old cache and create a new one covering the full history."""
        old_cache_name = params.cache_name
        try:
            contents = [{"role": entry["role"], "parts": entry["parts"]} for entry in history]
            cache = await self.client.aio.caches.create(
                model=params.model,
                config=types.CreateCachedContentConfig(
                    display_name=f"conv-{params.conversation_id}",
                    system_instruction=params.system_instruction,
                    contents=cast(Any, contents),
                    ttl=CACHE_TTL,
                ),
            )
            params.cache_name = cache.name
            params.cached_history_length = len(history)
            self.logger.info(
                "Re-cached conversation %s as %s (%d prompt tokens, %d were uncached)",
                params.conversation_id,
                cache.name,
                prompt_tokens,
                uncached_tokens,
            )
        except Exception as e:
            self.logger.warning("Failed to re-cache: %s", e)
            return  # Keep the old cache

        # Clean up the old cache after the new one is safely created
        if old_cache_name is not None:
            try:
                await self.client.aio.caches.delete(name=old_cache_name)
            except Exception as e:
                self.logger.warning("Failed to delete old cache %s: %s", old_cache_name, e)

    async def _refresh_cache_ttl(self, params: ChatCompletionParameters) -> None:
        """Extend the TTL of an existing cache so it doesn't expire between turns."""
        cache_name = params.cache_name
        if cache_name is None:
            return
        try:
            await self.client.aio.caches.update(
                name=cache_name,
                config=types.UpdateCachedContentConfig(ttl=CACHE_TTL),
            )
        except Exception as e:
            self.logger.warning("Failed to refresh cache TTL for %s: %s", params.cache_name, e)

    async def _delete_conversation_cache(self, params: ChatCompletionParameters) -> None:
        """Delete the explicit cache for a conversation, if one exists."""
        if not params.cache_name:
            return
        try:
            await self.client.aio.caches.delete(name=params.cache_name)
            self.logger.info("Deleted cache %s", params.cache_name)
        except Exception as e:
            self.logger.warning("Failed to delete cache %s: %s", params.cache_name, e)
        params.cache_name = None
        params.cached_history_length = 0

    async def _run_agentic_loop(
        self,
        model: str,
        contents: list,
        config: types.GenerateContentConfig | None,
    ) -> AgenticResult:
        """Run generate_content in a loop, executing custom function calls.

        When the model returns function_calls, the registered tool functions
        are executed and their results fed back. The loop continues until the
        model returns a text-only response or MAX_AGENTIC_ITERATIONS is hit.

        The final model response is NOT appended to ``contents`` — the caller
        is responsible for adding it to the conversation history.
        """
        result = AgenticResult(response=None, contents=contents)

        for iteration in range(MAX_AGENTIC_ITERATIONS):
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            result.response = response
            result.iterations = iteration + 1

            # Accumulate tokens (each API call is billed separately)
            usage = getattr(response, "usage_metadata", None)
            result.total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
            result.total_output_tokens += getattr(usage, "candidates_token_count", 0) or 0
            result.total_thinking_tokens += getattr(usage, "thoughts_token_count", 0) or 0

            function_calls = response.function_calls
            if not function_calls:
                break  # Model returned text — done

            # Append model's function-call turn to contents
            model_parts = _get_response_content_parts(response)
            if model_parts:
                contents.append({"role": "model", "parts": model_parts})

            # Execute each function and build response parts
            func_response_parts = []
            for fc in function_calls:
                name = fc.name or ""
                if not name:
                    continue
                self.logger.info(
                    "Agentic loop iter %d: calling %s(%s)",
                    iteration + 1,
                    name,
                    fc.args,
                )
                result.tool_calls_made.append(name)
                call_result = await execute_tool_call(name, fc.args)
                func_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response=call_result,
                    )
                )

            contents.append({"role": "user", "parts": func_response_parts})
        else:
            self.logger.warning(
                "Agentic loop hit max iterations (%d)",
                MAX_AGENTIC_ITERATIONS,
            )

        return result

    def cog_unload(self):
        loop = getattr(self.bot, "loop", None)

        # Close HTTP session
        session = self._http_session
        if session and not session.closed:
            if loop and loop.is_running():
                loop.create_task(session.close())
            else:
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(session.close())
                finally:
                    new_loop.close()
        self._http_session = None

        # Delete any active caches and uploaded files
        for conversation in self.conversations.values():
            cache_name = conversation.params.cache_name
            if cache_name and loop and loop.is_running():
                loop.create_task(self.client.aio.caches.delete(name=cache_name))
            for file_name in conversation.params.uploaded_file_names:
                if loop and loop.is_running():
                    loop.create_task(self.client.aio.files.delete(name=file_name))

        self.last_view_messages.clear()

        # Close Gemini clients
        if loop and loop.is_running():
            loop.create_task(self.client.aio.aclose())
        self.client.close()

    async def _strip_previous_view(self, user) -> None:
        """Remove the button view from the previous turn's message for a user."""
        prev_view_msg = self.last_view_messages.pop(user, None)
        if prev_view_msg is not None:
            try:
                await prev_view_msg.edit(view=None)
            except discord.NotFound:
                pass  # Message was deleted before we could edit it
            except discord.HTTPException as exc:
                self.logger.debug(
                    "Failed to strip view from message %s: %s",
                    prev_view_msg.id,
                    exc,
                )

    async def _cleanup_conversation(self, user) -> None:
        """Remove button view from the last message and clean up view state."""
        await self._strip_previous_view(user)
        self.views.pop(user, None)

    async def handle_new_message_in_conversation(self, message, conversation_wrapper: Conversation):
        """
        Handles a new message in an ongoing conversation.

        Args:
            message: The incoming Discord Message object.
            conversation_wrapper: The conversation object wrapper.
        """
        params = conversation_wrapper.params
        history = conversation_wrapper.history

        self.logger.info(f"Handling new message in conversation {params.conversation_id}.")
        typing_task = None
        embeds = []

        try:
            # Only respond to the user who started the conversation and if not paused.
            if message.author != params.conversation_starter or params.paused:
                return

            self.logger.debug(
                f"Starting typing indicator for followup message from {message.author}"
            )
            typing_task = asyncio.create_task(self.keep_typing(message.channel))

            # Build parts with media first, text last (recommended by Gemini docs)
            user_parts: list[str | dict] = []
            if message.attachments:
                for attachment in message.attachments:
                    validation_error = self._validate_attachment_size(attachment)
                    if validation_error:
                        await message.reply(
                            embed=Embed(
                                title="Error",
                                description=validation_error,
                                color=Colour.red(),
                            )
                        )
                        return

                    attachment_part = await self._prepare_attachment_part(
                        attachment, params.uploaded_file_names
                    )
                    if attachment_part is None:
                        continue
                    user_parts.append(attachment_part)

            if message.content:
                user_parts.append({"text": message.content})
            history.append({"role": "user", "parts": user_parts})

            self.logger.debug(f"Sending history to Gemini: {history}")

            config_args = {}
            if params.cache_name:
                config_args["cached_content"] = params.cache_name
            elif params.system_instruction:
                config_args["system_instruction"] = params.system_instruction
            if params.temperature is not None:
                config_args["temperature"] = params.temperature
            if params.top_p is not None:
                config_args["top_p"] = params.top_p
            if params.media_resolution is not None:
                config_args["media_resolution"] = params.media_resolution
            thinking_config = _build_thinking_config(params.thinking_level, params.thinking_budget)
            if thinking_config is not None:
                config_args["thinking_config"] = thinking_config
            if params.tools:
                supported_tools, unsupported_tools = filter_supported_tools_for_model(
                    params.model, params.tools
                )
                if unsupported_tools:
                    self.logger.warning(
                        "Skipping unsupported tools for model %s: %s",
                        params.model,
                        ", ".join(sorted(set(unsupported_tools))),
                    )
                params.tools = supported_tools
                if supported_tools:
                    config_args["tools"] = supported_tools

            # Inject custom function callables when enabled
            if params.custom_functions_enabled and ENABLE_CUSTOM_TOOLS:
                callables = get_tool_callables()
                if callables:
                    tool_list = config_args.get("tools", [])
                    tool_list.extend(callables)
                    config_args["tools"] = tool_list

            # Convert history to the format expected by Gemini API.
            # If a cache exists, send only the uncached portion of history.
            history_start = params.cached_history_length if params.cache_name else 0
            contents = []
            for entry in history[history_start:]:
                contents.append({"role": entry["role"], "parts": entry["parts"]})

            self.logger.debug(f"Sending contents to Gemini: {contents}")

            generation_config = types.GenerateContentConfig(**config_args) if config_args else None
            pre_loop_len = len(contents)
            try:
                result = await self._run_agentic_loop(
                    params.model,
                    contents,
                    generation_config,
                )
            except Exception as cache_err:
                if not params.cache_name:
                    raise
                # Cache may have expired; retry with full history
                self.logger.warning("Cached request failed, retrying without cache: %s", cache_err)
                params.cache_name = None
                params.cached_history_length = 0
                config_args.pop("cached_content", None)
                if params.system_instruction:
                    config_args["system_instruction"] = params.system_instruction
                contents = []
                for entry in history:
                    contents.append({"role": entry["role"], "parts": entry["parts"]})
                pre_loop_len = len(contents)
                generation_config = (
                    types.GenerateContentConfig(**config_args) if config_args else None
                )
                result = await self._run_agentic_loop(
                    params.model,
                    contents,
                    generation_config,
                )
            response = result.response

            response_text = response.text
            tool_info = extract_tool_info(response)
            # Merge function calls made during agentic loop into tool_info
            for fc_name in result.tool_calls_made:
                if fc_name not in tool_info["tools_used"]:
                    tool_info["tools_used"].append(fc_name)
            self.logger.debug(f"Received response from Gemini: {response_text}")

            # Stop typing indicator as soon as we have the response
            if typing_task:
                self.logger.debug(
                    f"Stopping typing indicator for conversation {params.conversation_id}"
                )
                typing_task.cancel()
                typing_task = None

            # Handle case where response text might be None
            if response_text is None:
                response_text = "No response generated by the model."
                self.logger.warning("Model returned None as response text")

            # Sync intermediate agentic turns (function call/response pairs) to history
            for entry in contents[pre_loop_len:]:
                history.append(entry)

            # Store full response parts to preserve thought signatures for multi-turn context
            response_parts = _get_response_content_parts(response)
            history.append({"role": "model", "parts": response_parts or [{"text": response_text}]})

            # Create an explicit cache if the conversation is large enough
            await self._maybe_create_cache(params, history, response)

            thinking_text = extract_thinking_text(response)
            append_thinking_embeds(embeds, thinking_text)
            append_response_embeds(embeds, response_text)

            # Append sources and pricing embeds to main embeds (before buttons)
            append_sources_embed(embeds, tool_info)

            input_tokens = result.total_input_tokens
            output_tokens = result.total_output_tokens
            thinking_tokens = result.total_thinking_tokens
            maps_grounded = "google_maps" in tool_info.get("tools_used", [])
            cost = calculate_cost(
                params.model, input_tokens, output_tokens, thinking_tokens, maps_grounded
            )
            daily_cost = self._track_daily_cost(message.author.id, cost)
            self._log_cost(
                "chat",
                message.author.id,
                params.model,
                cost,
                daily_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                google_maps_grounded=maps_grounded,
            )
            if SHOW_COST_EMBEDS:
                append_pricing_embed(
                    embeds,
                    params.model,
                    input_tokens,
                    output_tokens,
                    daily_cost,
                    thinking_tokens,
                    maps_grounded,
                )

            view = self.views.get(message.author)
            main_conversation_id = conversation_wrapper.params.conversation_id

            # Ensure conversation_id is not None
            if main_conversation_id is None:
                self.logger.error("Conversation ID is None, cannot track message")
                return

            # Remove the view from the previous turn's message to reduce clutter
            await self._strip_previous_view(message.author)

            if embeds:
                # Send response embeds with view
                try:
                    reply_message = await message.reply(embeds=embeds, view=view)
                    self.message_to_conversation_id[reply_message.id] = main_conversation_id
                    self.last_view_messages[message.author] = reply_message
                except Exception as embed_error:
                    # If embed fails due to size, try sending as plain text
                    self.logger.warning(f"Embed failed, sending as text: {embed_error}")
                    safe_response_text = response_text or "No response text available"
                    reply_message = await message.reply(
                        content=f"**Response:**\n{safe_response_text[:1900]}{'...' if len(safe_response_text) > 1900 else ''}",
                        view=view,
                    )
                    self.message_to_conversation_id[reply_message.id] = main_conversation_id
                    self.last_view_messages[message.author] = reply_message

                self.logger.debug("Replied with generated response.")
            else:
                self.logger.warning("No embeds to send in the reply.")
                await message.reply(
                    content="An error occurred: No content to send.",
                    view=view,
                )

        except Exception as e:
            description = str(e)
            self.logger.error(
                "Error in handle_new_message_in_conversation: %s",
                description,
                exc_info=True,
            )
            if len(description) > 4000:
                description = description[:4000] + "\n\n... (error message truncated)"
            await message.reply(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )
            # Clean up the conversation on error
            await self._delete_conversation_cache(conversation_wrapper.params)
            await self._cleanup_uploaded_files(conversation_wrapper.params)
            conv_id = conversation_wrapper.params.conversation_id
            if conv_id is not None:
                self.conversations.pop(conv_id, None)
            await self._cleanup_conversation(message.author)

        finally:
            # Cancel typing task if it exists
            if typing_task:
                typing_task.cancel()

    async def keep_typing(self, channel):
        """
        Coroutine to keep the typing indicator alive in a channel.

        Args:
            channel: The Discord channel object.
        """
        try:
            self.logger.debug(f"Starting typing indicator loop in channel {channel.id}")
            while True:
                async with channel.typing():
                    self.logger.debug(f"Sent typing indicator to channel {channel.id}")
                    await asyncio.sleep(TYPING_INDICATOR_INTERVAL)
        except asyncio.CancelledError:
            # Task was cancelled, stop typing
            self.logger.debug(f"Typing indicator cancelled for channel {channel.id}")
            raise

    # Added for debugging purposes
    @commands.Cog.listener()
    async def on_ready(self):
        """
        Event listener that runs when the bot is ready.
        Logs bot details and attempts to synchronize commands.
        """
        self.logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.owner_id})")
        self.logger.info(f"Attempting to sync commands for guilds: {GUILD_IDS}")
        try:
            await self.bot.sync_commands()
            self.logger.info("Commands synchronized successfully.")
        except Exception as e:
            self.logger.error(f"Error during command synchronization: {e}", exc_info=True)

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Event listener that runs when a message is sent.
        Generates a response using chat completion API when a new message from the conversation author is detected.

        Args:
            message: The incoming Discord Message object.
        """
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        self.logger.debug(
            f"Received message from {message.author} in channel {message.channel.id}: '{message.content}'"
        )

        # Check for active conversations in this channel
        for conversation_wrapper in self.conversations.values():
            # Skip conversations that are not in the same channel
            if message.channel.id != conversation_wrapper.params.channel_id:
                continue

            # Skip if the message is not from the conversation starter
            if message.author != conversation_wrapper.params.conversation_starter:
                continue

            self.logger.info(
                f"Processing followup message for conversation {conversation_wrapper.params.conversation_id}"
            )
            # Process the message for the matching conversation
            await self.handle_new_message_in_conversation(message, conversation_wrapper)
            break  # Stop looping once we've handled the message

        self.logger.debug("No matching conversations found for this message")

    @commands.Cog.listener()
    async def on_error(self, event, *args, **kwargs):
        """
        Event listener that runs when an error occurs.

        Args:
            event: The name of the event that raised the error.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.logger.error(f"Error in event {event}: {args} {kwargs}", exc_info=True)

    @gemini.command(
        name="check_permissions",
        description="Check if bot has necessary permissions in this channel",
    )
    async def check_permissions(self, ctx: ApplicationContext):
        """
        Checks and reports the bot's permissions in the current channel.
        """
        guild = ctx.guild
        channel = ctx.channel
        if guild is None or channel is None:
            await ctx.respond("This command must be used in a server channel.")
            return

        me = guild.me
        if me is None:
            await ctx.respond("Unable to resolve bot permissions in this server.")
            return

        if not hasattr(channel, "permissions_for"):
            await ctx.respond("Cannot inspect permissions for this channel type.")
            return

        permission_channel = cast(PermissionAwareChannel, channel)
        permissions = permission_channel.permissions_for(me)
        can_read = bool(getattr(permissions, "read_messages", False))
        can_read_history = bool(getattr(permissions, "read_message_history", False))

        if can_read and can_read_history:
            await ctx.respond("Bot has permission to read messages and message history.")
        else:
            await ctx.respond("Bot is missing necessary permissions in this channel.")

    @gemini.command(
        name="chat",
        description="Starts a conversation with a model.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "system_instruction",
        description="Additional instructions for the model. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "model",
        description="Choose from the following Gemini models. (default: Gemini 3.1 Pro)",
        required=False,
        choices=[
            OptionChoice(name="Gemini 3.1 Pro", value="gemini-3.1-pro-preview"),
            OptionChoice(name="Gemini 3.1 Flash Lite", value="gemini-3.1-flash-lite-preview"),
            OptionChoice(name="Gemini 3.0 Flash", value="gemini-3-flash-preview"),
            OptionChoice(name="Gemini 2.5 Pro", value="gemini-2.5-pro"),
            OptionChoice(name="Gemini 2.5 Flash", value="gemini-2.5-flash"),
            OptionChoice(name="Gemini 2.5 Flash Lite", value="gemini-2.5-flash-lite"),
            OptionChoice(name="Gemini 2.0 Flash", value="gemini-2.0-flash"),
            OptionChoice(name="Gemini 2.0 Flash Lite", value="gemini-2.0-flash-lite"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="Attach an image, PDF, audio, video, document, or code file. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "url",
        description="URL to a file (PDF, image, etc.) for the model to read. Gemini 2.5+ only. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "frequency_penalty",
        description="(Advanced) Controls how much the model should repeat itself. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "presence_penalty",
        description="(Advanced) Controls how much the model should talk about the prompt. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "seed",
        description="(Advanced) Seed for deterministic outputs. (default: not set)",
        required=False,
        type=int,
    )
    @option(
        "temperature",
        description="(Advanced) Controls randomness. Gemini 3 recommends 1.0. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "top_p",
        description="(Advanced) Nucleus sampling. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "media_resolution",
        description="Resolution for media inputs (images, video, PDFs). (default: not set)",
        required=False,
        choices=[
            OptionChoice(name="Low", value="MEDIA_RESOLUTION_LOW"),
            OptionChoice(name="Medium", value="MEDIA_RESOLUTION_MEDIUM"),
            OptionChoice(name="High", value="MEDIA_RESOLUTION_HIGH"),
        ],
        type=str,
    )
    @option(
        "thinking_level",
        description="Thinking depth for Gemini 3 models: Minimal, Low, Medium, High. (default: not set / model default)",
        required=False,
        choices=[
            OptionChoice(name="Minimal", value="minimal"),
            OptionChoice(name="Low", value="low"),
            OptionChoice(name="Medium", value="medium"),
            OptionChoice(name="High", value="high"),
        ],
        type=str,
    )
    @option(
        "thinking_budget",
        description="(Advanced) Thinking token budget for Gemini 2.5 models. -1=dynamic, 0=off. (default: not set)",
        required=False,
        type=int,
        min_value=-1,
        max_value=32768,
    )
    @option(
        "google_search",
        description="Enable Google Search grounding for current web information. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "code_execution",
        description="Enable code execution for calculations and data processing. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "google_maps",
        description="Enable Google Maps grounding (model-dependent). (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "url_context",
        description="Enable URL Context retrieval from links in your prompt. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "file_search",
        description="Enable File Search over configured document stores. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "custom_functions",
        description="Enable custom function tool calling (time, dice, etc.). (default: false)",
        required=False,
        type=bool,
    )
    async def chat(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "gemini-3.1-pro-preview",
        system_instruction: str | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        attachment: Attachment | None = None,
        url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        media_resolution: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        google_search: bool = False,
        code_execution: bool = False,
        google_maps: bool = False,
        url_context: bool = False,
        file_search: bool = False,
        custom_functions: bool = False,
    ):
        """
        Creates a persistent conversation session with a Gemini model.

        Initiates an interactive conversation with context preservation across multiple exchanges.
        Supports multimodal inputs (text + files) and provides interactive UI controls for
        conversation management.

        Args:
            ctx: Discord application context
            prompt: Initial conversation prompt or question
            model: Gemini model variant (default: gemini-3.1-pro-preview)
            system_instruction: Optional behavioral guidelines for the AI
            frequency_penalty: Controls repetition reduction (experimental)
            presence_penalty: Controls topic focus (experimental)
            seed: Random seed for deterministic responses
            attachment: Attach an image, PDF, audio, video, document, or code file.
            url: URL to a file for the model to process (Gemini 2.5+ only)
            temperature: Response creativity (0.0 conservative → 2.0 creative)
            top_p: Nucleus sampling threshold (0.0 focused → 1.0 diverse)
            google_search: Enable Google Search grounding
            code_execution: Enable code execution
            google_maps: Enable Google Maps grounding (model-dependent)
            url_context: Enable URL Context retrieval (model-dependent)
            file_search: Enable File Search over configured document stores (model-dependent)
            media_resolution: Media resolution for tokenization (Low/Medium/High)

        Returns:
            Discord response with initial AI message and interactive conversation controls

        Note:
            Only one conversation per user per channel allowed. Conversations persist until
            explicitly ended or bot restarts. Follow-up messages automatically handled.
        """
        # Acknowledge the interaction immediately - reply can take some time
        await ctx.defer()
        typing_task = None
        channel = ctx.channel
        channel_id = getattr(channel, "id", None)
        if channel is None or channel_id is None:
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description="Unable to determine the channel for this conversation.",
                    color=Colour.red(),
                )
            )
            return

        for conv_wrapper in self.conversations.values():
            if (
                conv_wrapper.params.conversation_starter == ctx.author
                and conv_wrapper.params.channel_id == channel_id
            ):
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description="You already have an active conversation in this channel. Please finish it before starting a new one.",
                        color=Colour.red(),
                    )
                )
                return

        try:
            # Start typing and keep it alive until the response is ready
            typing_task = asyncio.create_task(self.keep_typing(channel))

            # Build parts with media first, text last (recommended by Gemini docs)
            parts: list[dict] = []
            uploaded_file_names: list[str] = []
            if attachment:
                validation_error = self._validate_attachment_size(attachment)
                if validation_error:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description=validation_error,
                            color=Colour.red(),
                        )
                    )
                    if typing_task:
                        typing_task.cancel()
                    return

                attachment_part = await self._prepare_attachment_part(
                    attachment, uploaded_file_names
                )
                if attachment_part is not None:
                    parts.append(attachment_part)

            if url:
                mime_type = _guess_url_mime_type(url)
                parts.append(
                    {
                        "file_data": {
                            "file_uri": url,
                            "mime_type": mime_type,
                        }
                    }
                )

            parts.append({"text": prompt})

            selected_tool_names = {
                "google_search": (google_search, TOOL_GOOGLE_SEARCH),
                "code_execution": (code_execution, TOOL_CODE_EXECUTION),
                "google_maps": (google_maps, TOOL_GOOGLE_MAPS),
                "url_context": (url_context, TOOL_URL_CONTEXT),
                "file_search": (file_search, TOOL_FILE_SEARCH),
            }

            enabled_names = {name for name, (enabled, _) in selected_tool_names.items() if enabled}
            exclusive_error = check_mutually_exclusive_tools(enabled_names)
            if exclusive_error:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=exclusive_error,
                        color=Colour.red(),
                    )
                )
                if typing_task:
                    typing_task.cancel()
                return

            requested_tools = [
                deepcopy(tool_config)
                for enabled, tool_config in selected_tool_names.values()
                if enabled
            ]
            tools, unsupported_tools = filter_supported_tools_for_model(model, requested_tools)
            tools, incompatible_tools = filter_file_search_incompatible_tools(tools)

            # Enrich file_search tools with configured store IDs
            enrich_error = self.enrich_file_search_tools(tools)
            if enrich_error:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=enrich_error,
                        color=Colour.red(),
                    )
                )
                if typing_task:
                    typing_task.cancel()
                return

            config_args = {}
            if system_instruction is not None:
                config_args["system_instruction"] = system_instruction
            if frequency_penalty is not None:
                config_args["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                config_args["presence_penalty"] = presence_penalty
            if seed is not None:
                config_args["seed"] = seed
            if temperature is not None:
                config_args["temperature"] = temperature
            if top_p is not None:
                config_args["top_p"] = top_p
            if media_resolution is not None:
                config_args["media_resolution"] = media_resolution
            thinking_config = _build_thinking_config(thinking_level, thinking_budget)
            if thinking_config is not None:
                config_args["thinking_config"] = thinking_config
            if tools:
                config_args["tools"] = tools

            # Inject custom function callables when enabled
            custom_functions_enabled = custom_functions and ENABLE_CUSTOM_TOOLS
            if custom_functions_enabled:
                callables = get_tool_callables()
                if callables:
                    tool_list = config_args.get("tools", [])
                    tool_list.extend(callables)
                    config_args["tools"] = tool_list

            generation_config = types.GenerateContentConfig(**config_args) if config_args else None

            # Convert parts to the format expected by Gemini API
            formatted_parts = []
            for part in parts:
                if isinstance(part, dict):
                    if "text" in part:
                        formatted_parts.append(types.Part(text=part["text"]))
                    elif "inline_data" in part:
                        formatted_parts.append(types.Part(inline_data=part["inline_data"]))
                    elif "file_data" in part:
                        formatted_parts.append(
                            types.Part.from_uri(
                                file_uri=part["file_data"]["file_uri"],
                                mime_type=part["file_data"]["mime_type"],
                            )
                        )
                else:
                    # Assume it's a string or other supported type
                    formatted_parts.append(part)

            initial_contents = [{"role": "user", "parts": formatted_parts}]
            result = await self._run_agentic_loop(
                model,
                initial_contents,
                generation_config,
            )
            response = result.response
            response_text = response.text
            tool_info = extract_tool_info(response)
            # Merge function calls made during agentic loop into tool_info
            for fc_name in result.tool_calls_made:
                if fc_name not in tool_info["tools_used"]:
                    tool_info["tools_used"].append(fc_name)

            self.logger.debug(f"Received response from Gemini: {response_text}")

            # Update initial response description based on input parameters
            # Truncate prompt to avoid exceeding Discord's 4096 char embed limit
            truncated_prompt = truncate_text(prompt, 2000)
            description = f"**Prompt:** {truncated_prompt}\n"
            description += f"**Model:** {model}\n"
            description += (
                f"**System Instruction:** {system_instruction}\n" if system_instruction else ""
            )
            description += (
                f"**Frequency Penalty:** {frequency_penalty}\n" if frequency_penalty else ""
            )
            description += f"**Presence Penalty:** {presence_penalty}\n" if presence_penalty else ""
            description += f"**Seed:** {seed}\n" if seed else ""
            if temperature is not None:
                description += f"**Temperature:** {temperature}"
                if model.startswith("gemini-3") and temperature != 1.0:
                    description += (
                        " (warning: Gemini 3 recommends 1.0; lower values may cause looping)"
                    )
                description += "\n"
            description += f"**Nucleus Sampling:** {top_p}\n" if top_p else ""
            description += f"**Media Resolution:** {media_resolution}\n" if media_resolution else ""
            description += (
                f"**Thinking Level:** {thinking_level.capitalize()}\n" if thinking_level else ""
            )
            description += (
                f"**Thinking Budget:** {thinking_budget}\n" if thinking_budget is not None else ""
            )
            if tools:
                active_tool_labels = [
                    resolve_tool_name(tool_config) or "unknown" for tool_config in tools
                ]
                description += f"**Tools:** {', '.join(active_tool_labels)}\n"
                if unsupported_tools:
                    description += f"**Tools Skipped (model unsupported):** {', '.join(sorted(set(unsupported_tools)))}\n"
                if incompatible_tools:
                    description += f"**Tools Skipped (incompatible with file_search):** {', '.join(sorted(set(incompatible_tools)))}\n"

            # Assemble all embeds for a single message
            embeds = [
                Embed(
                    title="Conversation Started",
                    description=description,
                    color=Colour.green(),
                )
            ]
            thinking_text = extract_thinking_text(response)
            append_thinking_embeds(embeds, thinking_text)
            embed_count_before_response = len(embeds)
            append_response_embeds(embeds, response_text)
            has_response = len(embeds) > embed_count_before_response

            # Append sources and pricing embeds to main embeds (before buttons)
            append_sources_embed(embeds, tool_info)

            input_tokens = result.total_input_tokens
            output_tokens = result.total_output_tokens
            thinking_tokens = result.total_thinking_tokens
            maps_grounded = "google_maps" in tool_info.get("tools_used", [])
            cost = calculate_cost(
                model, input_tokens, output_tokens, thinking_tokens, maps_grounded
            )
            daily_cost = self._track_daily_cost(ctx.author.id, cost)
            self._log_cost(
                "chat",
                ctx.author.id,
                model,
                cost,
                daily_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                google_maps_grounded=maps_grounded,
            )
            if SHOW_COST_EMBEDS:
                append_pricing_embed(
                    embeds,
                    model,
                    input_tokens,
                    output_tokens,
                    daily_cost,
                    thinking_tokens,
                    maps_grounded,
                )

            if not has_response:
                await ctx.send_followup("No response generated.")
                return

            # Create the view with buttons
            interaction = ctx.interaction
            if interaction is None:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description="Unable to determine interaction context for this conversation.",
                        color=Colour.red(),
                    )
                )
                return

            main_conversation_id = interaction.id
            view = ButtonView(
                conversation_starter=ctx.author,
                conversation_id=main_conversation_id,
                initial_tools=tools,
                custom_functions_enabled=custom_functions_enabled,
                get_conversation=lambda cid: self.conversations.get(cid),
                on_regenerate=self.handle_new_message_in_conversation,
                on_stop=self.end_conversation,
                on_tools_changed=self._resolve_tools_for_view,
            )
            self.views[ctx.author] = view

            # Strip buttons from previous conversation's last message
            await self._strip_previous_view(ctx.author)

            # Send all embeds with view in a single message
            message = await ctx.send_followup(embeds=embeds, view=view)
            self.message_to_conversation_id[message.id] = main_conversation_id
            self.last_view_messages[ctx.author] = message

            # Store the conversation details
            params = ChatCompletionParameters(
                model=model,
                system_instruction=system_instruction,
                conversation_starter=ctx.author,
                channel_id=channel_id,
                conversation_id=main_conversation_id,
                temperature=temperature,
                top_p=top_p,
                media_resolution=media_resolution,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                tools=tools,
                uploaded_file_names=uploaded_file_names,
                custom_functions_enabled=custom_functions_enabled,
            )
            # Build history: original user parts + agentic turns + final model response
            response_parts = _get_response_content_parts(response)
            history = [{"role": "user", "parts": parts}]
            # Add intermediate agentic turns (function call/response pairs)
            for entry in initial_contents[1:]:
                history.append(entry)
            history.append({"role": "model", "parts": response_parts or [{"text": response_text}]})
            conversation_wrapper = Conversation(params=params, history=history)
            self.conversations[main_conversation_id] = conversation_wrapper

        except Exception as e:
            await self._send_error_followup(ctx, e, "chat")
            await self._cleanup_conversation(ctx.author)

        finally:
            if typing_task:
                typing_task.cancel()

    @gemini.command(
        name="image",
        description="Generates an image based on a prompt.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "model",
        description="Choose between Gemini or Imagen models. (default: Gemini 3.1 Flash Image)",
        required=False,
        choices=[
            OptionChoice(name="Gemini 3.1 Flash Image", value="gemini-3.1-flash-image-preview"),
            OptionChoice(name="Gemini 3.0 Pro Image", value="gemini-3-pro-image-preview"),
            OptionChoice(name="Gemini 2.5 Flash Image", value="gemini-2.5-flash-image"),
            OptionChoice(name="Imagen 4", value="imagen-4.0-generate-001"),
            OptionChoice(name="Imagen 4 Ultra", value="imagen-4.0-ultra-generate-001"),
            OptionChoice(name="Imagen 4 Fast", value="imagen-4.0-fast-generate-001"),
        ],
        type=str,
    )
    @option(
        "number_of_images",
        description="Number of images to generate (1-4). (default: 1)",
        required=False,
        type=int,
        min_value=1,
        max_value=4,
    )
    @option(
        "aspect_ratio",
        description="Aspect ratio of the generated image. (default: 1:1)",
        required=False,
        choices=[
            OptionChoice(name="Square (1:1)", value="1:1"),
            OptionChoice(name="Portrait (3:4)", value="3:4"),
            OptionChoice(name="Landscape (4:3)", value="4:3"),
            OptionChoice(name="Portrait (9:16)", value="9:16"),
            OptionChoice(name="Landscape (16:9)", value="16:9"),
        ],
        type=str,
    )
    @option(
        "person_generation",
        description="(Imagen only) Control generation of people in images. (default: allow_adult)",
        required=False,
        choices=[
            OptionChoice(name="Don't Allow", value="dont_allow"),
            OptionChoice(name="Allow Adults", value="allow_adult"),
            OptionChoice(name="Allow All", value="allow_all"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="(Gemini only) Image to edit. Upload an image for image editing tasks. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "negative_prompt",
        description="(Advanced) Description of what to discourage in the generated images. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "seed",
        description="(Advanced) Random seed for image generation. (default: not set)",
        required=False,
        type=int,
    )
    @option(
        "guidance_scale",
        description="(Advanced) Controls adherence to prompt. Ranges from 0.0 to 20.0. (default: not set)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=20.0,
    )
    @option(
        "image_size",
        description="(Gemini only) Output image resolution. (default: not set / model default)",
        required=False,
        choices=[
            OptionChoice(name="1K", value="1k"),
            OptionChoice(name="2K", value="2k"),
        ],
        type=str,
    )
    @option(
        "google_image_search",
        description="(Gemini 3.1 Flash Image only) Ground image generation with Google Image Search. (default: False)",
        required=False,
        type=bool,
    )
    async def image(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "gemini-3.1-flash-image-preview",
        number_of_images: int = 1,
        aspect_ratio: str = "1:1",
        person_generation: str = "allow_adult",
        attachment: Attachment | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        guidance_scale: float | None = None,
        image_size: str | None = None,
        google_image_search: bool | None = None,
    ):
        """
        Generates images from a prompt using either Gemini or Imagen models.

        This function supports both Google's Gemini and Imagen image generation models,
        automatically selecting the appropriate API based on the model chosen:

        Gemini Models (via generate_content API):
        - Text-to-image generation with response_modalities=['TEXT', 'IMAGE']
        - Image editing capabilities when attachments are provided
        - Multiple image generation via candidate_count
        - Seed control for reproducible outputs
        - Returns both text and image responses

        Imagen Models (via generate_images API):
        - Advanced image generation with full parameter support
        - number_of_images, aspect_ratio, negative_prompt, seed, guidance_scale
        - person_generation controls for content safety
        - No image editing support (attachment not allowed)
        - Image-only responses

        The function uses a clean modular architecture with separate helper methods
        for each model type, ensuring maintainable and extensible code.

        Args:
            ctx: Discord application context
            prompt: Text description of the image to generate
            model: Model to use (Gemini or Imagen variants)
            number_of_images: Number of images to generate (1-4)
            aspect_ratio: Image dimensions (1:1, 3:4, 4:3, 9:16, 16:9)
            person_generation: Control people in images (Imagen only)
            attachment: Image to edit (Gemini only)
            negative_prompt: What to avoid in generation (Imagen only)
            seed: Random seed for reproducible results
            guidance_scale: Text prompt adherence control (Imagen only)

        Returns:
            Discord response with generated images and parameter information
        """
        await ctx.defer()
        try:
            if attachment:
                validation_error = self._validate_attachment_size(attachment)
                if validation_error:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description=validation_error,
                            color=Colour.red(),
                        )
                    )
                    return

            # Create ImageGenerationParameters for clean parameter handling
            image_params = ImageGenerationParameters(
                prompt=prompt,
                model=model,
                number_of_images=number_of_images,
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                negative_prompt=negative_prompt,
                seed=seed,
                guidance_scale=guidance_scale,
                image_size=image_size,
                google_image_search=bool(google_image_search),
            )

            # Check if this is a Gemini or Imagen model and use appropriate API
            is_gemini_model = model.startswith("gemini-")

            # Process response - handle both Gemini and Imagen response formats
            text_response = None
            generated_images = []
            input_tokens = 0

            if is_gemini_model:
                # Handle Gemini models using generate_content
                (
                    text_response,
                    generated_images,
                    input_tokens,
                ) = await self._generate_image_with_gemini(image_params, attachment)
            else:
                # Handle Imagen models using generate_images
                if attachment:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Not Supported",
                            description="Image editing is not supported with Imagen models. Please use a Gemini model for image editing.",
                            color=Colour.orange(),
                        )
                    )
                    return

                generated_images = await self._generate_image_with_imagen(image_params)

            # Track and log cost
            num_images = len(generated_images)
            cost = calculate_image_cost(model, num_images, input_tokens, image_size)
            daily_cost = self._track_daily_cost(ctx.author.id, cost)
            self._log_cost(
                "image",
                ctx.author.id,
                model,
                cost,
                daily_cost,
                images=num_images,
                input_tokens=input_tokens,
            )

            # Send response
            if generated_images:
                embed, files = await self._create_image_response_embed(
                    image_params=image_params,
                    generated_images=generated_images,
                    attachment=attachment,
                    text_response=text_response,
                )

                embeds = [embed]
                if SHOW_COST_EMBEDS:
                    pricing_desc = (
                        f"${cost:.4f} · {num_images} image{'s' if num_images != 1 else ''}"
                    )
                    if input_tokens:
                        pricing_desc += f" · {input_tokens:,} input tokens"
                    pricing_desc += f" · daily ${daily_cost:.2f}"
                    embeds.append(Embed(description=pricing_desc, color=GEMINI_BLUE))

                await ctx.send_followup(embeds=embeds, files=files)
            else:
                # No images generated, but maybe there's text (only for Gemini)
                embed_description = "The model did not generate any images.\n"
                if text_response:
                    # Truncate text response to avoid exceeding Discord's 4096 char limit
                    # Reserve ~200 chars for the rest of the message
                    truncated_text = truncate_text(text_response, 3800)
                    embed_description += f"Text response: {truncated_text}\n"
                elif is_gemini_model:
                    embed_description += (
                        "Try asking explicitly for image generation (e.g., 'a red car').\n"
                    )
                else:
                    embed_description += "Imagen models should generate images. Check your prompt or try different parameters.\n"

                # Show unsupported parameters info based on model type
                if is_gemini_model:
                    unsupported_params = []
                    if image_params.negative_prompt:
                        unsupported_params.append("negative_prompt")
                    if image_params.guidance_scale:
                        unsupported_params.append("guidance_scale")
                    if image_params.person_generation != "allow_adult":
                        unsupported_params.append("person_generation")

                    if unsupported_params:
                        embed_description += f"\n*Note: These parameters are not yet implemented for Gemini: {', '.join(unsupported_params)}*"

                embeds_no_img = [
                    Embed(
                        title="No Images Generated",
                        description=embed_description,
                        color=Colour.orange(),
                    )
                ]
                if SHOW_COST_EMBEDS and cost > 0:
                    pricing_desc = f"${cost:.4f} · 0 images"
                    if input_tokens:
                        pricing_desc += f" · {input_tokens:,} input tokens"
                    pricing_desc += f" · daily ${daily_cost:.2f}"
                    embeds_no_img.append(Embed(description=pricing_desc, color=GEMINI_BLUE))
                await ctx.send_followup(embeds=embeds_no_img)

        except Exception as e:
            await self._send_error_followup(ctx, e, "image")

    @gemini.command(
        name="video",
        description="Generates a video based on a prompt using Veo.",
    )
    @option("prompt", description="Prompt for video generation", required=True, type=str)
    @option(
        "model",
        description="Choose Veo model for video generation. (default: Veo 3.1 Preview)",
        required=False,
        choices=[
            OptionChoice(name="Veo 3.1 Preview", value="veo-3.1-generate-preview"),
            OptionChoice(name="Veo 3.1 Fast Preview", value="veo-3.1-fast-generate-preview"),
            OptionChoice(name="Veo 3", value="veo-3.0-generate-001"),
            OptionChoice(name="Veo 3 Fast", value="veo-3.0-fast-generate-001"),
            OptionChoice(name="Veo 2", value="veo-2.0-generate-001"),
        ],
        type=str,
    )
    @option(
        "aspect_ratio",
        description="Aspect ratio of the generated video. (default: 16:9)",
        required=False,
        choices=[
            OptionChoice(name="Landscape (16:9)", value="16:9"),
            OptionChoice(name="Portrait (9:16)", value="9:16"),
        ],
        type=str,
    )
    @option(
        "person_generation",
        description="Control generation of people in videos. (default: allow_adult)",
        required=False,
        choices=[
            OptionChoice(name="Don't Allow", value="dont_allow"),
            OptionChoice(name="Allow Adults", value="allow_adult"),
            OptionChoice(name="Allow All", value="allow_all"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="Image to use as the first frame for the video. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "last_frame",
        description="(Veo 3.1 only) Image to use as the last frame for interpolation. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "number_of_videos",
        description="Number of videos to generate, from 1 to 2. (default: 1)",
        required=False,
        type=int,
        min_value=1,
        max_value=2,
    )
    @option(
        "duration_seconds",
        description="Length of each output video in seconds, from 5 to 8 seconds. (default: not set)",
        required=False,
        type=int,
        min_value=5,
        max_value=8,
    )
    @option(
        "negative_prompt",
        description="(Advanced) Description of what to discourage. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "enhance_prompt",
        description="(Advanced) Enable or disable the prompt rewriter. (default: True)",
        required=False,
        type=bool,
    )
    async def video(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "veo-3.1-generate-preview",
        aspect_ratio: str = "16:9",
        person_generation: str = "allow_adult",
        attachment: Attachment | None = None,
        last_frame: Attachment | None = None,
        number_of_videos: int = 1,
        duration_seconds: int | None = None,
        negative_prompt: str | None = None,
        enhance_prompt: bool | None = None,
    ):
        """
        Generates videos from a prompt using Veo models (2.0, 3.0, or 3.1).

        This function uses Google's Veo video generation models to create videos based on text prompts
        and optionally starting from an image. The generation process is asynchronous and can take
        2-6 minutes to complete.

        Veo Features:
        - Text-to-video generation with detailed prompts
        - Image-to-video generation when attachments are provided
        - 5-8 second video duration at 720p or 1080p resolution and 24fps
        - Support for landscape (16:9) and portrait (9:16) aspect ratios
        - Person generation controls for content safety
        - Advanced parameters like negative prompts and prompt enhancement
        - Veo 3.1: Native audio generation, video extension, reference image support
        - Veo 3.1: First/last frame interpolation for controlled video generation

        The function handles the long-running operation by polling the API until completion,
        then downloads and sends the generated videos to Discord.

        Args:
            ctx: Discord application context
            prompt: Text description of the video to generate
            model: Veo model variant (2.0, 3.0, 3.1, or 3.1-fast)
            aspect_ratio: Video dimensions (16:9 or 9:16)
            person_generation: Control people in videos (dont_allow, allow_adult, allow_all)
            attachment: Optional image to use as first frame (image-to-video)
            last_frame: Optional image to use as last frame for interpolation (Veo 3.1 only)
            number_of_videos: Number of videos to generate (1-2)
            duration_seconds: Video length in seconds (5-8)
            negative_prompt: What to avoid in generation
            enhance_prompt: Enable/disable prompt rewriter

        Returns:
            Discord response with generated videos and parameter information
        """
        await ctx.defer()
        try:
            for att in (attachment, last_frame):
                if att:
                    validation_error = self._validate_attachment_size(att)
                    if validation_error:
                        await ctx.send_followup(
                            embed=Embed(
                                title="Error",
                                description=validation_error,
                                color=Colour.red(),
                            )
                        )
                        return

            # Veo 3.1-only: last_frame requires a Veo 3.1 model
            if last_frame and "veo-3.1" not in model:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description="The `last_frame` parameter is only supported on Veo 3.1 models.",
                        color=Colour.red(),
                    )
                )
                return

            # Create VideoGenerationParameters for clean parameter handling
            video_params = VideoGenerationParameters(
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                negative_prompt=negative_prompt,
                number_of_videos=number_of_videos,
                duration_seconds=duration_seconds,
                enhance_prompt=enhance_prompt,
                has_last_frame=last_frame is not None,
            )

            # Generate videos using Veo
            generated_videos = await self._generate_video_with_veo(
                video_params, attachment, last_frame
            )

            # Send response
            if generated_videos:
                # Track and log cost only when videos are actually generated
                # (Veo only charges for successful generations)
                num_videos = len(generated_videos)
                est_duration = video_params.duration_seconds or 8
                cost = calculate_video_cost(model, est_duration, num_videos)
                daily_cost = self._track_daily_cost(ctx.author.id, cost)
                self._log_cost(
                    "video",
                    ctx.author.id,
                    model,
                    cost,
                    daily_cost,
                    videos=num_videos,
                    duration_seconds=est_duration,
                )

                embed, files = await self._create_video_response_embed(
                    video_params=video_params,
                    generated_videos=generated_videos,
                    attachment=attachment,
                )

                embeds = [embed]
                if SHOW_COST_EMBEDS:
                    pricing_desc = (
                        f"${cost:.2f} · {num_videos} video{'s' if num_videos != 1 else ''} "
                        f"× {est_duration}s · daily ${daily_cost:.2f}"
                    )
                    embeds.append(Embed(description=pricing_desc, color=GEMINI_BLUE))

                await ctx.send_followup(embeds=embeds, files=files)
            else:
                await ctx.send_followup(
                    embed=Embed(
                        title="No Videos Generated",
                        description="The model did not generate any videos. This may be due to resource constraints or safety filters. Please try again with a different prompt or parameters.",
                        color=Colour.orange(),
                    )
                )

        except Exception as e:
            await self._send_error_followup(ctx, e, "video")

    @gemini.command(
        name="tts",
        description="Generates lifelike audio from input text using Gemini text-to-speech.",
    )
    @option(
        "input_text",
        description="Text to convert to speech. Max 32k tokens.",
        required=True,
        type=str,
    )
    @option(
        "model",
        description="Choose Gemini text-to-speech model. (default: Gemini 2.5 Flash Preview TTS)",
        required=False,
        choices=[
            OptionChoice(
                name="Gemini 2.5 Flash Preview TTS",
                value="gemini-2.5-flash-preview-tts",
            ),
            OptionChoice(name="Gemini 2.5 Pro Preview TTS", value="gemini-2.5-pro-preview-tts"),
        ],
        type=str,
    )
    @option(
        "voice_name",
        description="Voice to use for single-speaker text-to-speech. (default: Kore)",
        required=False,
        choices=[
            OptionChoice(name="Kore (Firm)", value="Kore"),
            OptionChoice(name="Puck (Upbeat)", value="Puck"),
            OptionChoice(name="Charon (Informative)", value="Charon"),
            OptionChoice(name="Zephyr (Bright)", value="Zephyr"),
            OptionChoice(name="Fenrir (Excitable)", value="Fenrir"),
            OptionChoice(name="Leda (Youthful)", value="Leda"),
            OptionChoice(name="Orus (Firm)", value="Orus"),
            OptionChoice(name="Aoede (Breezy)", value="Aoede"),
            OptionChoice(name="Callirrhoe (Easy-going)", value="Callirrhoe"),
            OptionChoice(name="Autonoe (Bright)", value="Autonoe"),
            OptionChoice(name="Enceladus (Breathy)", value="Enceladus"),
            OptionChoice(name="Iapetus (Clear)", value="Iapetus"),
            OptionChoice(name="Umbriel (Easy-going)", value="Umbriel"),
            OptionChoice(name="Algieba (Smooth)", value="Algieba"),
            OptionChoice(name="Despina (Smooth)", value="Despina"),
            OptionChoice(name="Erinome (Clear)", value="Erinome"),
            OptionChoice(name="Algenib (Gravelly)", value="Algenib"),
            OptionChoice(name="Rasalgethi (Informative)", value="Rasalgethi"),
            OptionChoice(name="Laomedeia (Upbeat)", value="Laomedeia"),
            OptionChoice(name="Achernar (Soft)", value="Achernar"),
            OptionChoice(name="Alnilam (Firm)", value="Alnilam"),
            OptionChoice(name="Schedar (Even)", value="Schedar"),
            OptionChoice(name="Gacrux (Mature)", value="Gacrux"),
            OptionChoice(name="Achird (Friendly)", value="Achird"),
            OptionChoice(name="Sulafat (Warm)", value="Sulafat"),
        ],
        type=str,
    )
    @option(
        "style_prompt",
        description="Natural language instructions to control voice style, tone, accent, pace. (default: not set)",
        required=False,
        type=str,
    )
    async def tts(
        self,
        ctx: ApplicationContext,
        input_text: str,
        model: str = "gemini-2.5-flash-preview-tts",
        voice_name: str = "Kore",
        style_prompt: str | None = None,
    ):
        """
        Generates audio from input text using Gemini's native text-to-speech capabilities.

        This function uses Google's Gemini TTS models to convert text into lifelike audio.
        The TTS capability supports natural language control over style, accent, pace, and tone.

        Features:
        - Single-speaker TTS with 30 prebuilt voice options
        - Natural language style control through prompts
        - Support for 24 languages with automatic detection
        - High-quality audio output in WAV format
        - Up to 32k token context window

        Args:
            ctx: Discord application context
            input_text: Text to convert to speech (max 32k tokens)
            model: Gemini TTS model to use
            voice_name: Voice option for speech generation
            style_prompt: Natural language instructions for voice control

        Returns:
            Discord response with generated audio file and parameter information
        """
        await ctx.defer()

        try:
            # Prepare the text input with optional style guidance
            content_text = input_text
            if style_prompt:
                content_text = f"{style_prompt}: {input_text}"

            # Create TTS parameters
            tts_params = SpeechGenerationParameters(
                input_text=content_text,
                model=model,
                voice_name=voice_name,
                multi_speaker=False,
                style_prompt=style_prompt,
            )

            # Generate audio using Gemini TTS
            audio_data, input_tokens, output_tokens = await self._generate_speech_with_gemini(
                tts_params
            )

            # Track and log cost
            cost = calculate_tts_cost(model, input_tokens, output_tokens)
            daily_cost = self._track_daily_cost(ctx.author.id, cost)
            self._log_cost(
                "tts",
                ctx.author.id,
                model,
                cost,
                daily_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            if audio_data:
                # Create temporary audio file
                audio_file_path = Path(f"tts_output_{voice_name}.wav")

                # Write audio data to WAV file
                with wave.open(str(audio_file_path), "wb") as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(24000)  # 24kHz sample rate
                    wf.writeframes(audio_data)

                # Create response embed
                description = (
                    f"**Text:** {input_text[:500]}{'...' if len(input_text) > 500 else ''}\n"
                )
                description += f"**Model:** {model}\n"
                description += f"**Voice:** {voice_name}\n"
                if style_prompt:
                    description += f"**Style Instructions:** {style_prompt}\n"
                description += "**Format:** WAV (24kHz, 16-bit, Mono)\n"

                embed = Embed(
                    title="Text-to-Speech Generation",
                    description=description,
                    color=GEMINI_BLUE,
                )

                embeds = [embed]
                if SHOW_COST_EMBEDS:
                    pricing_desc = f"${cost:.4f} · {input_tokens:,} in / {output_tokens:,} out (audio) · daily ${daily_cost:.2f}"
                    embeds.append(Embed(description=pricing_desc, color=GEMINI_BLUE))

                # Send the audio file
                await ctx.send_followup(embeds=embeds, file=File(audio_file_path))

                # Clean up the temporary file
                audio_file_path.unlink(missing_ok=True)
            else:
                await ctx.send_followup(
                    embed=Embed(
                        title="No Audio Generated",
                        description="The model did not generate any audio. Please try again with different text or parameters.",
                        color=Colour.orange(),
                    )
                )

        except Exception as e:
            await self._send_error_followup(ctx, e, "tts")

    @gemini.command(
        name="music",
        description="Generate music using Lyria 3 or Lyria RealTime.",
    )
    @option(
        "prompt",
        description="Musical prompt describing genre, mood, instruments, or style.",
        required=True,
        type=str,
    )
    @option(
        "attachment",
        description="(Lyria 3 only) Reference image for multimodal music generation.",
        required=False,
        type=Attachment,
    )
    @option(
        "model",
        description="Choose music generation model. (default: Lyria RealTime Experimental)",
        required=False,
        choices=[
            OptionChoice(name="Lyria 3 Pro Preview", value="lyria-3-pro-preview"),
            OptionChoice(name="Lyria 3 Clip Preview", value="lyria-3-clip-preview"),
            OptionChoice(name="Lyria RealTime Experimental", value=LYRIA_REALTIME_MODEL),
        ],
        type=str,
    )
    @option(
        "duration",
        description="Target duration in seconds. RealTime uses it directly; Lyria 3 uses it as guidance.",
        required=False,
        type=int,
        min_value=5,
        max_value=120,
    )
    @option(
        "bpm",
        description="Beats per minute from 60 to 200. Leave empty for model to decide.",
        required=False,
        type=int,
        min_value=60,
        max_value=200,
    )
    @option(
        "scale",
        description="Scale/key for the music.",
        required=False,
        choices=[
            OptionChoice(name="C Major / A Minor", value="C_MAJOR_A_MINOR"),
            OptionChoice(name="D♭ Major / B♭ Minor", value="D_FLAT_MAJOR_B_FLAT_MINOR"),
            OptionChoice(name="D Major / B Minor", value="D_MAJOR_B_MINOR"),
            OptionChoice(name="E♭ Major / C Minor", value="E_FLAT_MAJOR_C_MINOR"),
            OptionChoice(name="E Major / C# Minor", value="E_MAJOR_D_FLAT_MINOR"),
            OptionChoice(name="F Major / D Minor", value="F_MAJOR_D_MINOR"),
            OptionChoice(name="G♭ Major / E♭ Minor", value="G_FLAT_MAJOR_E_FLAT_MINOR"),
            OptionChoice(name="G Major / E Minor", value="G_MAJOR_E_MINOR"),
            OptionChoice(name="A♭ Major / F Minor", value="A_FLAT_MAJOR_F_MINOR"),
            OptionChoice(name="A Major / F# Minor", value="A_MAJOR_G_FLAT_MINOR"),
            OptionChoice(name="B♭ Major / G Minor", value="B_FLAT_MAJOR_G_MINOR"),
            OptionChoice(name="B Major / G# Minor", value="B_MAJOR_A_FLAT_MINOR"),
        ],
        type=str,
    )
    @option(
        "density",
        description="(Advanced) Musical density from 0.0 top 1.0. Lower is sparser, Higher is busier. (default: not set)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=1.0,
    )
    @option(
        "brightness",
        description="(Advanced) Tonal brightness from 0.0 to 1.0. Higher is brighter sound. (default: not set)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=1.0,
    )
    @option(
        "guidance",
        description="(Advanced) Prompt adherence from 0.0 to 6.0. Higher is more faithful to prompt. (default: 4.0)",
        required=False,
        type=float,
        min_value=0.0,
        max_value=6.0,
    )
    async def music(
        self,
        ctx: ApplicationContext,
        prompt: str,
        attachment: Attachment | None = None,
        model: str = LYRIA_REALTIME_MODEL,
        duration: int = 30,
        bpm: int | None = None,
        scale: str | None = None,
        density: float | None = None,
        brightness: float | None = None,
        guidance: float = 4.0,
    ):
        """
        Generate music using Lyria 3 or Lyria RealTime.

        This command supports two music-generation paths:

        - Lyria 3 Pro / Clip: standard generate_content API returning encoded audio
        - Lyria RealTime: live WebSocket streaming returning raw stereo PCM

        Features:
        - Lyria 3 song and clip generation
        - Real-time streaming music generation
        - Support for various genres, instruments, and moods
        - Optional BPM, scale, density, brightness, and guidance controls
        - High-quality stereo audio output
        - Lyria 3 can return lyrics / structure notes alongside audio

        Args:
            ctx: Discord application context
            prompt: Musical prompt describing desired style/characteristics
            attachment: Optional reference image for Lyria 3 multimodal generation
            model: Music generation model to use
            duration: Length or target length of music to generate in seconds
            bpm: Beats per minute (optional, model decides if not set)
            scale: Musical scale/key for generation
            density: Musical density (sparseness vs business)
            brightness: Tonal brightness (frequency emphasis)
            guidance: How strictly to follow the prompt

        Returns:
            Discord response with generated music file and parameter information
        """
        await ctx.defer()

        try:
            if attachment:
                validation_error = self._validate_music_attachment(model, attachment)
                if validation_error:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description=validation_error,
                            color=Colour.red(),
                        )
                    )
                    return

            # Create music generation parameters
            music_params = MusicGenerationParameters(
                prompts=[prompt],
                model=model,
                duration=duration,
                bpm=bpm,
                scale=scale,
                guidance=guidance,
                density=density,
                brightness=brightness,
            )

            text_response = None
            audio_mime_type = "audio/wav"

            if model in LYRIA_3_MODELS:
                audio_data, text_response, audio_mime_type = await self._generate_music_with_lyria3(
                    music_params,
                    attachment,
                )
            else:
                audio_data = await self._generate_music_with_lyria_realtime(music_params)

            # Log music generation (no music pricing is wired yet for these models)
            daily_cost = self._track_daily_cost(ctx.author.id, 0.0)
            self._log_cost(
                "music",
                ctx.author.id,
                model,
                0.0,
                daily_cost,
                duration_seconds=duration,
            )

            if audio_data:
                suffix = (
                    "wav"
                    if model == LYRIA_REALTIME_MODEL
                    else _music_file_suffix_for_mime_type(audio_mime_type)
                )
                audio_file_path = Path(f"music_output_{int(time.time())}.{suffix}")

                if model == LYRIA_REALTIME_MODEL:
                    with wave.open(str(audio_file_path), "wb") as wf:
                        wf.setnchannels(2)  # Stereo
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(48000)  # 48kHz sample rate
                        wf.writeframes(audio_data)
                else:
                    audio_file_path.write_bytes(audio_data)

                # Create response embed
                # Truncate prompt to avoid exceeding Discord's 4096 char embed limit
                truncated_prompt = truncate_text(prompt, 2000)
                description = f"**Prompt:** {truncated_prompt}\n"
                description += f"**Model:** {_music_model_display_name(model)} (`{model}`)\n"
                if attachment:
                    description += "**Reference Image:** Attached\n"
                if model == "lyria-3-clip-preview":
                    description += "**Mode:** 30-second clip generation\n"
                    description += "**Duration:** 30 seconds (fixed by model)\n"
                elif model == "lyria-3-pro-preview":
                    description += "**Mode:** Full song generation\n"
                    description += f"**Target Duration:** {duration} seconds\n"
                else:
                    description += "**Mode:** Real-time instrumental generation\n"
                    description += f"**Duration:** {duration} seconds\n"
                if bpm is not None:
                    description += f"**BPM:** {bpm}\n"
                if scale is not None:
                    description += f"**Scale:** {scale.replace('_', ' ')}\n"
                if density is not None:
                    description += f"**Density:** {density}\n"
                if brightness is not None:
                    description += f"**Brightness:** {brightness}\n"
                description += f"**Guidance:** {guidance}\n"
                if model == LYRIA_REALTIME_MODEL:
                    description += "**Format:** WAV (48kHz, 16-bit, Stereo)\n"
                else:
                    description += f"**Format:** {suffix.upper()}\n"
                    if text_response:
                        description += (
                            f"**Lyrics / Notes:** {truncate_text(text_response, 500)}\n"
                        )
                    if any(
                        value is not None
                        for value in (bpm, scale, density, brightness)
                    ) or guidance != 4.0:
                        description += (
                            "*Advanced controls were translated into prompt guidance for "
                            "Lyria 3.*\n"
                        )

                embed = Embed(
                    title="Music Generation",
                    description=description,
                    color=GEMINI_BLUE,
                )

                # Send the audio file
                await ctx.send_followup(embed=embed, file=File(audio_file_path))

                # Clean up the temporary file
                audio_file_path.unlink(missing_ok=True)
            else:
                await ctx.send_followup(
                    embed=Embed(
                        title="No Music Generated",
                        description=(
                            "The model did not generate any music. Please try again with a "
                            "different prompt or parameters."
                        ),
                        color=Colour.orange(),
                    )
                )

        except MusicGenerationError as e:
            self.logger.error("Music generation error: %s", e, exc_info=True)
            description = str(e)
            if len(description) > 4000:
                description = description[:4000] + "\n\n... (error message truncated)"
            await ctx.send_followup(
                embed=Embed(
                    title="Music Generation Error",
                    description=description,
                    color=Colour.orange(),
                )
            )
        except Exception as e:
            await self._send_error_followup(ctx, e, "music")

    @gemini.command(
        name="research",
        description="Run a deep research task that autonomously searches, reads, and synthesizes a detailed report.",
    )
    @option(
        "prompt",
        description="Research question or topic to investigate.",
        required=True,
        type=str,
    )
    @option(
        "file_search",
        description="Also search your uploaded document stores (File Search / RAG). (default: False)",
        required=False,
        type=bool,
    )
    @option(
        "google_maps",
        description="Enable Google Maps grounding for location-aware research. (default: False)",
        required=False,
        type=bool,
    )
    async def research(
        self,
        ctx: ApplicationContext,
        prompt: str,
        file_search: bool = False,
        google_maps: bool = False,
    ):
        """
        Run a deep research task using the Gemini Deep Research agent.

        The agent autonomously plans, searches the web, reads sources, and
        synthesizes a detailed, cited report. Research tasks typically take
        2-10 minutes to complete.

        Args:
            ctx: Discord application context
            prompt: Research question or topic to investigate
            file_search: Whether to additionally search uploaded document stores
            google_maps: Whether to enable Google Maps grounding
        """
        await ctx.defer()

        try:
            research_params = ResearchParameters(
                prompt=prompt,
                file_search=file_search,
                google_maps=google_maps,
            )

            (
                report_text,
                input_tokens,
                output_tokens,
                thinking_tokens,
            ) = await self._run_deep_research(research_params)

            # Deep Research is billed at Gemini 3.1 Pro rates
            research_model = "gemini-3.1-pro-preview"
            cost = calculate_cost(research_model, input_tokens, output_tokens, thinking_tokens)
            daily_cost = self._track_daily_cost(ctx.author.id, cost)
            self._log_cost(
                "research",
                ctx.author.id,
                research_params.agent,
                cost,
                daily_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                file_search=file_search,
                google_maps=google_maps,
            )

            if report_text:
                embeds = self._create_research_response_embeds(research_params)

                if SHOW_COST_EMBEDS:
                    if thinking_tokens > 0:
                        pricing_desc = (
                            f"${cost:.2f} · {input_tokens:,} in / {output_tokens:,} out / "
                            f"{thinking_tokens:,} thinking · daily ${daily_cost:.2f}"
                        )
                    else:
                        pricing_desc = f"${cost:.2f} · {input_tokens:,} in / {output_tokens:,} out · daily ${daily_cost:.2f}"
                    embeds.append(Embed(description=pricing_desc, color=GEMINI_BLUE))

                # Always attach the full report as a downloadable file
                report_bytes = report_text.encode("utf-8")
                report_file = File(
                    BytesIO(report_bytes),
                    filename="research_report.md",
                )
                await ctx.send_followup(embeds=embeds, file=report_file)
            else:
                await ctx.send_followup(
                    embed=Embed(
                        title="No Research Results",
                        description="The research agent did not produce any output. Please try again with a different prompt.",
                        color=Colour.orange(),
                    )
                )

        except Exception as e:
            await self._send_error_followup(ctx, e, "research")

    async def _generate_image_with_gemini(
        self,
        image_params: ImageGenerationParameters,
        attachment: Attachment | None,
    ) -> tuple[str | None, list[Image.Image], int]:
        """
        Generate images using Gemini models with generate_content API.

        Returns:
            tuple: (text_response, generated_images, input_tokens)
        """
        prompt = image_params.prompt
        number_of_images = image_params.number_of_images

        # Explicitly request image generation to reduce text-only responses
        if attachment:
            # For image editing, keep the prompt as-is for more natural edits
            contents = prompt
        else:
            # For generation, explicitly request an image
            image_word = "image(s)" if number_of_images > 1 else "image"
            contents = f"Create {image_word}: {prompt}"

        # Add attachment for image editing if provided
        if attachment:
            image_data = await self._fetch_attachment_bytes(attachment)
            if image_data:
                try:
                    image = Image.open(BytesIO(image_data))
                except Exception as error:
                    self.logger.warning("Failed to open attachment for image generation: %s", error)
                else:
                    contents = [prompt, image]

        # Build configuration for image generation
        config_kwargs: dict[str, Any] = {
            "response_modalities": ["TEXT", "IMAGE"],
        }

        if number_of_images and number_of_images > 1:
            config_kwargs["candidate_count"] = number_of_images
        elif image_params.seed is not None:
            config_kwargs["seed"] = image_params.seed

        # Image config for aspect ratio and resolution (Gemini models)
        if image_params.aspect_ratio != "1:1" or image_params.image_size:
            image_config_kwargs = {}
            if image_params.aspect_ratio != "1:1":
                image_config_kwargs["aspect_ratio"] = image_params.aspect_ratio
            if image_params.image_size:
                image_config_kwargs["image_size"] = image_params.image_size
            config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

        # Google Image Search grounding (3.1 Flash Image only)
        if (
            image_params.google_image_search
            and image_params.model == "gemini-3.1-flash-image-preview"
        ):
            config_kwargs["tools"] = [
                types.Tool(
                    google_search=types.GoogleSearch(
                        search_types=types.SearchTypes(
                            web_search=types.WebSearch(),
                            image_search=types.ImageSearch(),
                        )
                    )
                )
            ]

        generate_config = types.GenerateContentConfig(**config_kwargs)

        gemini_response = await self.client.aio.models.generate_content(
            model=image_params.model, contents=contents, config=generate_config
        )

        # Extract input token count from usage metadata
        usage = getattr(gemini_response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0

        # Process Gemini response from generate_content
        text_response = None
        generated_images = []

        if gemini_response.candidates and len(gemini_response.candidates) > 0:
            candidate = gemini_response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text is not None:
                        text_response = part.text
                    elif (
                        hasattr(part, "inline_data")
                        and part.inline_data is not None
                        and part.inline_data.data
                    ):
                        image = Image.open(BytesIO(part.inline_data.data))
                        generated_images.append(image)

        return text_response, generated_images, input_tokens

    async def _generate_image_with_imagen(
        self, image_params: ImageGenerationParameters
    ) -> list[Image.Image]:
        """
        Generate images using Imagen models with generate_images API.

        Returns:
            List of generated PIL Images
        """
        imagen_response = await self.client.aio.models.generate_images(
            model=image_params.model,
            prompt=image_params.prompt,
            config=types.GenerateImagesConfig(**image_params.to_dict()),
        )

        # Process Imagen response from generate_images
        generated_images = []
        if hasattr(imagen_response, "generated_images") and imagen_response.generated_images:
            for generated_image in imagen_response.generated_images:
                if hasattr(generated_image, "image") and generated_image.image:
                    img = generated_image.image

                    # Extract image data using the canonical image_bytes attribute
                    if hasattr(img, "image_bytes") and img.image_bytes:
                        try:
                            pil_image = Image.open(BytesIO(img.image_bytes))
                            generated_images.append(pil_image)
                        except Exception as e:
                            self.logger.error(f"Failed to convert image_bytes to PIL Image: {e}")
                            continue
                    else:
                        self.logger.error(
                            f"Image object missing image_bytes attribute. Type: {type(img)}"
                        )
                        continue

        return generated_images

    async def _create_image_response_embed(
        self,
        image_params: ImageGenerationParameters,
        generated_images: list[Image.Image],
        attachment: Attachment | None,
        text_response: str | None = None,
    ) -> tuple[Embed, list[File]]:
        """
        Create Discord embed and file attachments for image generation response.

        Returns:
            tuple: (embed, files)
        """
        is_gemini_model = image_params.model.startswith("gemini-")

        # Create files for Discord
        files = []
        for i, image in enumerate(generated_images):
            try:
                image_bytes = BytesIO()
                # All generated_images should be PIL Image objects at this point
                image.save(image_bytes, format="PNG")
                image_bytes.seek(0)
                files.append(File(image_bytes, filename=f"generated_image_{i + 1}.png"))
            except Exception as e:
                self.logger.error(f"Failed to save image {i + 1}: {e}")
                continue

        # Truncate prompt to avoid exceeding Discord's 4096 char embed limit
        truncated_prompt = truncate_text(image_params.prompt, 2000)
        description = f"**Prompt:** {truncated_prompt}\n"
        description += f"**Model:** {image_params.model}\n"
        if attachment:
            description += "**Mode:** Image Editing\n"
        else:
            description += "**Mode:** Image Generation\n"
        description += f"**Number of Images:** {len(generated_images)}"

        # Show which parameters are currently supported
        if is_gemini_model:
            # For Gemini models, show what's supported
            if image_params.number_of_images > 1:
                description += f" (requested: {image_params.number_of_images})"
            if image_params.seed is not None:
                description += f"\n**Seed:** {image_params.seed}"
            if image_params.aspect_ratio != "1:1":
                description += f"\n**Aspect Ratio:** {image_params.aspect_ratio}"
            if image_params.image_size:
                description += f"\n**Image Size:** {image_params.image_size}"
            if image_params.google_image_search:
                description += "\n**Google Image Search:** Enabled"

            # Note about unsupported parameters for Gemini
            unsupported_params = []
            if image_params.negative_prompt:
                unsupported_params.append(f"negative_prompt: {image_params.negative_prompt}")
            if image_params.guidance_scale:
                unsupported_params.append(f"guidance_scale: {image_params.guidance_scale}")
            if image_params.person_generation != "allow_adult":
                unsupported_params.append(f"person_generation: {image_params.person_generation}")

            if unsupported_params:
                description += f"\n\n*Note: Advanced parameters not yet implemented for Gemini: {', '.join(unsupported_params)}*"
        else:
            # For Imagen models, show all supported parameters
            if image_params.seed is not None:
                description += f"\n**Seed:** {image_params.seed}"
            if image_params.negative_prompt:
                description += f"\n**Negative Prompt:** {image_params.negative_prompt}"
            if image_params.guidance_scale:
                description += f"\n**Guidance Scale:** {image_params.guidance_scale}"
            if image_params.aspect_ratio != "1:1":
                description += f"\n**Aspect Ratio:** {image_params.aspect_ratio}"
            if image_params.person_generation != "allow_adult":
                description += f"\n**Person Generation:** {image_params.person_generation}"

        if text_response:
            # Truncate long text responses for the embed
            truncated_text = truncate_text(text_response, 500)
            description += f"\n\n**AI Response:** {truncated_text}"

        embed = Embed(
            title=f"{'Gemini' if is_gemini_model else 'Imagen'} Image Generation",
            description=description,
            color=GEMINI_BLUE,
        )

        if files:
            embed.set_image(url=f"attachment://{files[0].filename}")

        return embed, files

    async def _generate_video_with_veo(
        self,
        video_params: VideoGenerationParameters,
        attachment: Attachment | None = None,
        last_frame_attachment: Attachment | None = None,
    ) -> list[str]:
        """
        Generate videos using Veo models with generate_videos API.

        Returns:
            List of generated video file paths
        """

        # Build config dict; last_frame is added here (it's part of GenerateVideosConfig)
        config_dict = video_params.to_dict()

        # Load last_frame image and add to config (Veo 3.1 interpolation)
        if last_frame_attachment:
            last_frame_data = await self._fetch_attachment_bytes(last_frame_attachment)
            if last_frame_data:
                try:
                    config_dict["last_frame"] = Image.open(BytesIO(last_frame_data))
                except Exception as error:
                    self.logger.warning("Failed to open last_frame attachment: %s", error)

        # Prepare the generation call
        kwargs = {
            "model": video_params.model,
            "prompt": video_params.prompt,
            "config": types.GenerateVideosConfig(**config_dict),
        }

        # Add image if provided for image-to-video generation (first frame)
        if attachment:
            image_data = await self._fetch_attachment_bytes(attachment)
            if image_data:
                try:
                    image = Image.open(BytesIO(image_data))
                except Exception as error:
                    self.logger.warning("Failed to open attachment for video generation: %s", error)
                else:
                    kwargs["image"] = image

        # Start the video generation operation
        operation = await self.client.aio.models.generate_videos(**kwargs)

        self.logger.info(f"Started video generation operation: {operation.name}")

        # Poll for completion (this can take 2-6 minutes)
        start_time = time.time()
        poll_interval = 20  # Poll every 20 seconds

        while not operation.done:
            if time.time() - start_time > VIDEO_GENERATION_TIMEOUT:
                raise TimeoutError("Video generation timed out after 10 minutes")

            await asyncio.sleep(poll_interval)
            operation = await self.client.aio.operations.get(operation)
            self.logger.debug(f"Operation status: {operation.done}")

        # Process completed operation
        generated_videos = []
        if (
            hasattr(operation, "response")
            and operation.response
            and hasattr(operation.response, "generated_videos")
            and operation.response.generated_videos
        ):
            for i, generated_video in enumerate(operation.response.generated_videos):
                if hasattr(generated_video, "video") and generated_video.video:
                    try:
                        # Download the video file
                        # Note: async files.download doesn't support Video objects,
                        # so we use the sync version via to_thread
                        await asyncio.to_thread(
                            self.client.files.download,
                            file=generated_video.video,
                        )

                        # Save to a temporary file path
                        video_path = f"temp_video_{i}.mp4"
                        generated_video.video.save(video_path)
                        generated_videos.append(video_path)

                        self.logger.info(f"Downloaded video {i + 1}: {video_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to download video {i + 1}: {e}")
                        continue

        return generated_videos

    async def _create_video_response_embed(
        self,
        video_params: VideoGenerationParameters,
        generated_videos: list[str],
        attachment: Attachment | None,
    ) -> tuple[Embed, list[File]]:
        """
        Create Discord embed and file attachments for video generation response.

        Returns:
            tuple: (embed, files)
        """
        # Create files for Discord
        files = []
        for i, video_path in enumerate(generated_videos):
            try:
                files.append(File(video_path, filename=f"generated_video_{i + 1}.mp4"))
            except Exception as e:
                self.logger.error(f"Failed to create file for video {i + 1}: {e}")
                continue

        # Truncate prompt to avoid exceeding Discord's 4096 char embed limit
        truncated_prompt = truncate_text(video_params.prompt, 2000)
        description = f"**Prompt:** {truncated_prompt}\n"
        description += f"**Model:** {video_params.model}\n"
        if attachment and video_params.has_last_frame:
            description += "**Mode:** Interpolation (First + Last Frame)\n"
        elif attachment:
            description += "**Mode:** Image-to-Video\n"
        elif video_params.has_last_frame:
            description += "**Mode:** Last Frame Constrained\n"
        else:
            description += "**Mode:** Text-to-Video\n"
        description += f"**Number of Videos:** {len(generated_videos)}"
        if video_params.number_of_videos and video_params.number_of_videos > len(generated_videos):
            description += f" (requested: {video_params.number_of_videos})"

        # Show generation parameters
        if video_params.aspect_ratio:
            description += f"\n**Aspect Ratio:** {video_params.aspect_ratio}"
        if video_params.person_generation and video_params.person_generation != "allow_adult":
            description += f"\n**Person Generation:** {video_params.person_generation}"
        if video_params.duration_seconds:
            description += f"\n**Duration:** {video_params.duration_seconds} seconds"
        if video_params.negative_prompt:
            description += f"\n**Negative Prompt:** {video_params.negative_prompt}"
        if video_params.enhance_prompt is not None:
            description += f"\n**Prompt Enhancement:** {'Enabled' if video_params.enhance_prompt else 'Disabled'}"

        embed = Embed(
            title="Video Generation",
            description=description,
            color=GEMINI_BLUE,
        )

        return embed, files

    async def _generate_music_with_lyria3(
        self,
        music_params: MusicGenerationParameters,
        attachment: Attachment | None = None,
    ) -> tuple[bytes | None, str | None, str | None]:
        """
        Generate music using a Lyria 3 model via the standard generate_content API.

        Returns:
            tuple: (audio_data, text_response, mime_type)
        """
        try:
            response = await self.client.aio.models.generate_content(
                model=music_params.model,
                contents=await self._build_lyria3_music_contents(music_params, attachment),
                config=types.GenerateContentConfig(response_modalities=["AUDIO", "TEXT"]),
            )

            text_parts: list[str] = []
            audio_data: bytes | None = None
            mime_type: str | None = None
            response_parts = _get_response_content_parts(response) or []

            for part in response_parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)

                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    audio_data = inline_data.data
                    mime_type = getattr(inline_data, "mime_type", None)

            text_response = "\n\n".join(text_parts) if text_parts else None
            return audio_data, text_response, mime_type

        except Exception as e:
            self.logger.error(f"Error generating music with Lyria 3: {e}")
            raise MusicGenerationError(f"Music generation failed: {e}") from e

    def _validate_music_attachment(self, model: str, attachment: Attachment) -> str | None:
        """Validate an attachment for use with music generation."""
        size_error = self._validate_attachment_size(attachment)
        if size_error:
            return size_error

        if model == LYRIA_REALTIME_MODEL:
            return (
                "Reference images are only supported for Lyria 3 Pro Preview and "
                "Lyria 3 Clip Preview."
            )

        mime_type = _guess_attachment_mime_type(attachment)
        if not mime_type.startswith("image/"):
            return "Music reference attachments must be image files."

        return None

    async def _build_lyria3_music_contents(
        self,
        music_params: MusicGenerationParameters,
        attachment: Attachment | None = None,
    ) -> str | list[Any]:
        """Build generate_content contents for Lyria 3 text-only or multimodal requests."""
        prompt = _build_lyria3_prompt(music_params)
        if attachment is None:
            return prompt

        attachment_data = await self._fetch_attachment_bytes(attachment)
        if attachment_data is None:
            raise MusicGenerationError(
                "Failed to read the uploaded image for music generation. Please try again."
            )

        try:
            image = Image.open(BytesIO(attachment_data))
            image.load()
        except Exception as error:
            raise MusicGenerationError(
                "The uploaded music reference attachment could not be processed as an image."
            ) from error

        return [prompt, image]

    async def _generate_music_with_lyria_realtime(
        self, music_params: MusicGenerationParameters
    ) -> bytes | None:
        """
        Generate music using Gemini's Lyria RealTime model.

        Args:
            music_params: Music generation parameters

        Returns:
            Raw audio data as bytes, or None if generation failed
        """
        try:
            # Create a client specifically for music generation with proper API version
            music_client = genai.Client(
                api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"}
            )

            # Collect audio chunks
            audio_chunks = []
            stop_receiving = False

            async def receive_audio(session):
                """Background task to process incoming audio."""
                self.logger.info("Audio receiver task started.")
                try:
                    # Iterate through the async generator to receive messages with timeout
                    async for message in session.receive():
                        if stop_receiving:
                            self.logger.info("Stop signal received, breaking from audio receiver")
                            break

                        if hasattr(message, "server_content") and hasattr(
                            message.server_content, "audio_chunks"
                        ):
                            audio_data = message.server_content.audio_chunks[0].data
                            if audio_data:
                                audio_chunks.append(audio_data)
                                self.logger.debug(
                                    f"Received audio chunk, size: {len(audio_data)} bytes"
                                )
                except asyncio.CancelledError:
                    self.logger.info("Audio receiver task cancelled")
                    raise
                except Exception as e:
                    # This might catch errors if the connection is closed unexpectedly
                    self.logger.error(f"Error in audio receiver: {e}")
                finally:
                    self.logger.info("Audio receiver task finished.")

            # Connect to Lyria RealTime using WebSocket with proper structure
            try:
                async with music_client.aio.live.music.connect(
                    model="models/lyria-realtime-exp"
                ) as session:
                    # Set up task to receive server messages
                    receiver_task = asyncio.create_task(receive_audio(session))

                    try:
                        # Send initial prompts and config following the official pattern
                        await session.set_weighted_prompts(
                            prompts=[
                                types.WeightedPrompt(
                                    text=prompt_data["text"],
                                    weight=prompt_data["weight"],
                                )
                                for prompt_data in music_params.to_weighted_prompts()
                            ]
                        )

                        # Set music generation configuration
                        config_dict = music_params.to_music_config()

                        # Convert scale string to enum if provided
                        if music_params.scale:
                            scale_enum = getattr(types.Scale, music_params.scale, None)
                            if scale_enum:
                                config_dict["scale"] = scale_enum

                        await session.set_music_generation_config(
                            config=types.LiveMusicGenerationConfig(**config_dict)
                        )

                        # Start streaming music
                        await session.play()

                        # Wait for the specified duration to collect audio
                        self.logger.info(
                            f"Waiting {music_params.duration} seconds for music generation"
                        )
                        await asyncio.sleep(music_params.duration)

                        # Stop the session
                        self.logger.info("Stopping music generation session")
                        await session.stop()

                        # Signal the receiver to stop and wait a bit for final chunks
                        self.logger.info("Signaling receiver to stop")
                        stop_receiving = True

                    finally:
                        # Cancel the receiver task if it's still running
                        if not receiver_task.done():
                            self.logger.info("Cancelling receiver task")
                            receiver_task.cancel()

                        # Give the task a moment to finish cancellation
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.sleep(WS_DRAIN_INTERVAL)

                # Combine all audio chunks
                if audio_chunks:
                    self.logger.info(f"Successfully collected {len(audio_chunks)} audio chunks")
                    return b"".join(audio_chunks)
                else:
                    self.logger.warning("No audio chunks received from Lyria RealTime")
                    return None

            except Exception as websocket_error:
                # Handle specific WebSocket connection errors
                if "404" in str(websocket_error):
                    self.logger.error("Lyria RealTime endpoint not found (404).")
                    raise MusicGenerationError(
                        "Music generation is currently unavailable. This could be due to:\n"
                        "1. The Lyria RealTime model is not available in your region\n"
                        "2. Your account doesn't have access to the music generation API\n"
                        "3. The service is temporarily unavailable\n\n"
                        "Please check Google AI Studio or contact support for more information."
                    ) from websocket_error
                elif "401" in str(websocket_error) or "403" in str(websocket_error):
                    self.logger.error("Authentication/authorization error for Lyria RealTime")
                    raise MusicGenerationError(
                        "Authentication error: Please check your API key permissions for music generation."
                    ) from websocket_error
                else:
                    self.logger.error(f"WebSocket connection error: {websocket_error}")
                    raise MusicGenerationError(
                        f"Connection error: {websocket_error}"
                    ) from websocket_error

        except MusicGenerationError:
            raise
        except Exception as e:
            self.logger.error(f"Error generating music with Lyria RealTime: {e}")
            raise MusicGenerationError(f"Music generation failed: {e}") from e

    async def _generate_speech_with_gemini(
        self, tts_params: SpeechGenerationParameters
    ) -> tuple[bytes | None, int, int]:
        """
        Generate speech using Gemini TTS models.

        Args:
            tts_params: TTS parameters containing input text, model, voice, etc.

        Returns:
            tuple: (audio_data, input_tokens, output_tokens)
        """
        try:
            # Generate speech using Gemini TTS
            response = await self.client.aio.models.generate_content(
                model=tts_params.model,
                contents=tts_params.input_text,
                config=types.GenerateContentConfig(**tts_params.to_dict()),
            )

            # Extract token counts from usage metadata
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            # Extract audio data from response
            if (
                response.candidates
                and len(response.candidates) > 0
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        return part.inline_data.data, input_tokens, output_tokens

            self.logger.warning("No audio data found in Gemini TTS response")
            return None, input_tokens, output_tokens

        except Exception as e:
            self.logger.error(f"Error generating speech with Gemini: {e}")
            raise

    async def _run_deep_research(
        self, research_params: ResearchParameters
    ) -> tuple[str | None, int, int, int]:
        """
        Run a deep research task using the Interactions API.

        Starts a background research task and polls until completion.
        Typically takes 2-10 minutes.

        Returns:
            tuple: (report_text, input_tokens, output_tokens, thinking_tokens)
        """
        kwargs: dict[str, Any] = {
            "input": research_params.prompt,
            "agent": research_params.agent,
            "background": True,
        }

        tools: list[dict[str, Any]] = []

        if research_params.file_search:
            if not GEMINI_FILE_SEARCH_STORE_IDS:
                raise ValidationError(
                    "File Search requires GEMINI_FILE_SEARCH_STORE_IDS to be set in your .env file."
                )
            tools.append(
                {
                    "type": "file_search",
                    "file_search_store_names": GEMINI_FILE_SEARCH_STORE_IDS.copy(),
                }
            )

        if research_params.google_maps:
            tools.append({"google_maps": {}})

        if tools:
            kwargs["tools"] = tools

        interaction = await self.client.aio.interactions.create(**kwargs)
        self.logger.info(f"Started deep research: {interaction.id}")

        max_wait_time = 1200  # 20 minutes timeout
        start_time = time.time()
        poll_interval = 15

        while interaction.status not in ("completed", "failed", "cancelled"):
            if time.time() - start_time > max_wait_time:
                raise TimeoutError("Deep research timed out after 20 minutes")

            await asyncio.sleep(poll_interval)
            interaction = await self.client.aio.interactions.get(interaction.id)
            self.logger.debug(f"Research {interaction.id} status: {interaction.status}")

        if interaction.status == "failed":
            raise APICallError(f"Research failed: {interaction.status}")

        if interaction.status == "cancelled":
            raise APICallError("Research was cancelled")

        # Extract token counts from interaction usage
        usage = getattr(interaction, "usage", None)
        input_tokens = getattr(usage, "total_input_tokens", 0) or 0
        output_tokens = getattr(usage, "total_output_tokens", 0) or 0
        thinking_tokens = getattr(usage, "total_thought_tokens", 0) or 0

        # Extract text from outputs
        if interaction.outputs:
            for output in reversed(interaction.outputs):
                text = getattr(output, "text", None)
                if text:
                    return text, input_tokens, output_tokens, thinking_tokens

        return None, input_tokens, output_tokens, thinking_tokens

    def _create_research_response_embeds(
        self,
        research_params: ResearchParameters,
    ) -> list[Embed]:
        """
        Create Discord embeds for a deep research report header.

        Returns:
            List of embeds containing the prompt info (report sent as file).
        """
        embeds: list[Embed] = []

        # Header embed with parameters
        truncated_prompt = truncate_text(research_params.prompt, 2000)
        description = f"**Prompt:** {truncated_prompt}\n"
        description += f"**Agent:** {research_params.agent}\n"
        if research_params.file_search:
            description += "**File Search:** Enabled\n"
        if research_params.google_maps:
            description += "**Google Maps:** Enabled\n"

        embeds.append(
            Embed(
                title="Deep Research",
                description=description,
                color=GEMINI_BLUE,
            )
        )

        return embeds
