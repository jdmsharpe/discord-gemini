"""Thin Gemini cog that delegates feature logic into provider-local modules."""

import asyncio
import logging
from typing import Any, cast

import aiohttp
import discord
from discord import Attachment, Embed, Member, User
from discord.commands import ApplicationContext, OptionChoice, SlashCommandGroup, option
from discord.ext import commands
from PIL import Image

from ...config.auth import GUILD_IDS
from ...util import (
    DEFAULT_MUSIC_MODEL,
    LYRIA_REALTIME_MODEL,
    ChatCompletionParameters,
    ImageGenerationParameters,
    MusicGenerationParameters,
    ResearchParameters,
    SpeechGenerationParameters,
    VideoGenerationParameters,
)
from . import (
    attachments as attachment_helpers,
)
from . import (
    cache as cache_helpers,
)
from . import (
    chat as chat_flow,
)
from . import (
    client as client_helpers,
)
from . import (
    image as image_flow,
)
from . import (
    music as music_flow,
)
from . import (
    research as research_flow,
)
from . import (
    speech as speech_flow,
)
from . import (
    state as state_helpers,
)
from . import (
    tooling as tooling_helpers,
)
from . import (
    video as video_flow,
)
from .embeds import build_error_embed
from .models import Conversation, PermissionAwareChannel


class GeminiCog(commands.Cog):
    """Discord cog exposing Gemini chat, image, video, speech, music, and research commands."""

    gemini = SlashCommandGroup("gemini", "Gemini AI commands", guild_ids=GUILD_IDS)

    def __init__(self, bot: Any):
        self.bot = bot
        self._client = None
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.conversations: dict[int, Conversation] = {}
        self.message_to_conversation_id: dict[int, int] = {}
        self.views: dict[Member | User, Any] = {}
        self.last_view_messages: dict[Member | User, discord.Message] = {}
        self.daily_costs: dict[tuple[int, str], float] = {}
        self._http_session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    @property
    def client(self):
        """Create the Gemini client lazily so imports work without auth."""

        if self._client is None:
            self._client = client_helpers.build_gemini_client()
        return self._client

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        return state_helpers.get_conversation(self, conversation_id)

    async def end_conversation(self, conversation_id: int, user: Member | User) -> None:
        await state_helpers.end_conversation(self, conversation_id, user)

    def _resolve_tools_for_view(
        self,
        selected_values: list[str],
        custom_functions_selected: bool,
        conversation: Conversation,
    ) -> tuple[set[str], str | None]:
        return tooling_helpers._resolve_tools_for_view(
            self,
            selected_values,
            custom_functions_selected,
            conversation,
        )

    def _track_daily_cost(self, user_id: int, cost: float) -> float:
        return state_helpers._track_daily_cost(self, user_id, cost)

    def _log_cost(
        self,
        command: str,
        user_id: int,
        model: str,
        cost: float,
        daily_total: float,
        **details: Any,
    ) -> None:
        detail_parts = [f"{key}={value}" for key, value in details.items()]
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
        description = str(error)
        self.logger.error("Error in %s: %s", command_name, description, exc_info=True)
        if len(description) > 4000:
            description = description[:4000] + "\n\n... (error message truncated)"
        try:
            await ctx.send_followup(embed=build_error_embed(description))
        except Exception as followup_error:
            self.logger.error(
                "Failed to send error followup for %s: %s",
                command_name,
                followup_error,
            )

    async def _get_http_session(self) -> aiohttp.ClientSession:
        return await attachment_helpers._get_http_session(self)

    async def _fetch_attachment_bytes(self, attachment: Attachment) -> bytes | None:
        return await attachment_helpers._fetch_attachment_bytes(self, attachment)

    def _validate_attachment_size(self, attachment: Attachment) -> str | None:
        return attachment_helpers._validate_attachment_size(attachment)

    async def _prepare_attachment_part(
        self,
        attachment: Attachment,
        uploaded_file_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        return await attachment_helpers._prepare_attachment_part(
            self,
            attachment,
            uploaded_file_names,
        )

    async def _upload_attachment_to_file_api(
        self,
        data: bytes,
        filename: str,
        mime_type: str,
    ) -> Any | None:
        return await attachment_helpers._upload_attachment_to_file_api(
            self,
            data,
            filename,
            mime_type,
        )

    async def _cleanup_uploaded_files(self, params: ChatCompletionParameters) -> None:
        await attachment_helpers._cleanup_uploaded_files(self, params)

    def enrich_file_search_tools(self, tools: list[dict[str, Any]]) -> str | None:
        return tooling_helpers.enrich_file_search_tools(tools)

    async def _maybe_create_cache(
        self,
        params: ChatCompletionParameters,
        history: list[dict[str, Any]],
        response: Any,
    ) -> None:
        await cache_helpers._maybe_create_cache(self, params, history, response)

    async def _recache(
        self,
        params: ChatCompletionParameters,
        history: list[dict[str, Any]],
        prompt_tokens: int,
        uncached_tokens: int,
    ) -> None:
        await cache_helpers._recache(self, params, history, prompt_tokens, uncached_tokens)

    async def _refresh_cache_ttl(self, params: ChatCompletionParameters) -> None:
        await cache_helpers._refresh_cache_ttl(self, params)

    async def _delete_conversation_cache(self, params: ChatCompletionParameters) -> None:
        await cache_helpers._delete_conversation_cache(self, params)

    async def _run_agentic_loop(self, model: str, contents: list[Any], config: Any):
        return await chat_flow._run_agentic_loop(self, model, contents, config)

    def cog_unload(self) -> None:
        loop = getattr(self.bot, "loop", None)
        client = self._client

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

        for conversation in self.conversations.values():
            cache_name = conversation.params.cache_name
            if cache_name and client and loop and loop.is_running():
                loop.create_task(client.aio.caches.delete(name=cache_name))
            for file_name in conversation.params.uploaded_file_names:
                if client and loop and loop.is_running():
                    loop.create_task(client.aio.files.delete(name=file_name))

        self.last_view_messages.clear()
        if client:
            if loop and loop.is_running():
                loop.create_task(client.aio.aclose())
            client.close()

    async def _strip_previous_view(self, user: Member | User) -> None:
        await state_helpers._strip_previous_view(self, user)

    async def _cleanup_conversation(self, user: Member | User) -> None:
        await state_helpers._cleanup_conversation(self, user)

    async def handle_new_message_in_conversation(
        self,
        message: Any,
        conversation_wrapper: Conversation,
    ) -> None:
        await chat_flow.handle_new_message_in_conversation(self, message, conversation_wrapper)

    async def keep_typing(self, channel: Any) -> None:
        await chat_flow.keep_typing(self, channel)

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        self.logger.info("Logged in as %s (ID: %s)", self.bot.user, self.bot.owner_id)
        self.logger.info("Attempting to sync commands for guilds: %s", GUILD_IDS)
        try:
            await self.bot.sync_commands()
            self.logger.info("Commands synchronized successfully.")
        except Exception as error:
            self.logger.error("Error during command synchronization: %s", error, exc_info=True)

    @commands.Cog.listener()
    async def on_message(self, message: Any) -> None:
        await chat_flow.handle_on_message(self, message)

    @commands.Cog.listener()
    async def on_error(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error("Error in event %s: %s %s", event, args, kwargs, exc_info=True)

    @gemini.command(
        name="check_permissions",
        description="Check if bot has necessary permissions in this channel",
    )
    async def check_permissions(self, ctx: ApplicationContext) -> None:
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

    @gemini.command(name="chat", description="Starts a conversation with a model.")
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
    ) -> None:
        await chat_flow.chat_command(
            self,
            ctx,
            prompt,
            model,
            system_instruction,
            frequency_penalty,
            presence_penalty,
            seed,
            attachment,
            url,
            temperature,
            top_p,
            media_resolution,
            thinking_level,
            thinking_budget,
            google_search,
            code_execution,
            google_maps,
            url_context,
            file_search,
            custom_functions,
        )

    @gemini.command(name="image", description="Generates an image based on a prompt.")
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
    ) -> None:
        await image_flow.image_command(
            self,
            ctx,
            prompt,
            model,
            number_of_images,
            aspect_ratio,
            person_generation,
            attachment,
            negative_prompt,
            seed,
            guidance_scale,
            image_size,
            google_image_search,
        )

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
    ) -> None:
        await video_flow.video_command(
            self,
            ctx,
            prompt,
            model,
            aspect_ratio,
            person_generation,
            attachment,
            last_frame,
            number_of_videos,
            duration_seconds,
            negative_prompt,
            enhance_prompt,
        )

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
    ) -> None:
        await speech_flow.tts_command(self, ctx, input_text, model, voice_name, style_prompt)

    @gemini.command(name="music", description="Generate music using Lyria 3 or Lyria RealTime.")
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
        description="Choose music generation model. (default: Lyria 3 Clip Preview)",
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
        description="Target duration in seconds. Applies to Lyria RealTime only; ignored for Lyria 3.",
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
        description="(Advanced) Musical density from 0.0 to 1.0. Lower is sparser, Higher is busier. (default: not set)",
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
        model: str = DEFAULT_MUSIC_MODEL,
        duration: int = 30,
        bpm: int | None = None,
        scale: str | None = None,
        density: float | None = None,
        brightness: float | None = None,
        guidance: float = 4.0,
    ) -> None:
        await music_flow.music_command(
            self,
            ctx,
            prompt,
            attachment,
            model,
            duration,
            bpm,
            scale,
            density,
            brightness,
            guidance,
        )

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
    ) -> None:
        await research_flow.research_command(self, ctx, prompt, file_search, google_maps)

    async def _generate_image_with_gemini(
        self,
        image_params: ImageGenerationParameters,
        attachment: Attachment | None,
    ) -> tuple[str | None, list[Image.Image], int]:
        return await image_flow._generate_image_with_gemini(self, image_params, attachment)

    async def _generate_image_with_imagen(
        self,
        image_params: ImageGenerationParameters,
    ) -> list[Image.Image]:
        return await image_flow._generate_image_with_imagen(self, image_params)

    async def _create_image_response_embed(
        self,
        image_params: ImageGenerationParameters,
        generated_images: list[Image.Image],
        attachment: Attachment | None,
        text_response: str | None = None,
    ) -> tuple[Embed, list[discord.File]]:
        return await image_flow._create_image_response_embed(
            self,
            image_params,
            generated_images,
            attachment,
            text_response,
        )

    async def _generate_video_with_veo(
        self,
        video_params: VideoGenerationParameters,
        attachment: Attachment | None = None,
        last_frame_attachment: Attachment | None = None,
    ) -> list[str]:
        return await video_flow._generate_video_with_veo(
            self,
            video_params,
            attachment,
            last_frame_attachment,
        )

    async def _create_video_response_embed(
        self,
        video_params: VideoGenerationParameters,
        generated_videos: list[str],
        attachment: Attachment | None,
    ) -> tuple[Embed, list[discord.File]]:
        return await video_flow._create_video_response_embed(
            self,
            video_params,
            generated_videos,
            attachment,
        )

    async def _generate_music_with_lyria3(
        self,
        music_params: MusicGenerationParameters,
        attachment: Attachment | None = None,
    ) -> tuple[bytes | None, str | None, str | None]:
        return await music_flow._generate_music_with_lyria3(self, music_params, attachment)

    def _validate_music_attachment(self, model: str, attachment: Attachment) -> str | None:
        return music_flow._validate_music_attachment(self, model, attachment)

    async def _build_lyria3_music_contents(
        self,
        music_params: MusicGenerationParameters,
        attachment: Attachment | None = None,
    ) -> str | list[Any]:
        return await music_flow._build_lyria3_music_contents(self, music_params, attachment)

    async def _generate_music_with_lyria_realtime(
        self,
        music_params: MusicGenerationParameters,
    ) -> bytes | None:
        return await music_flow._generate_music_with_lyria_realtime(self, music_params)

    async def _generate_speech_with_gemini(
        self,
        tts_params: SpeechGenerationParameters,
    ) -> tuple[bytes | None, int, int]:
        return await speech_flow._generate_speech_with_gemini(self, tts_params)

    async def _run_deep_research(
        self,
        research_params: ResearchParameters,
    ) -> tuple[str | None, int, int, int]:
        return await research_flow._run_deep_research(self, research_params)

    def _create_research_response_embeds(
        self,
        research_params: ResearchParameters,
    ) -> list[Embed]:
        return research_flow._create_research_response_embeds(research_params)


__all__ = ["Conversation", "GeminiCog"]
