from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from discord import Member, User

from .cogs.gemini.tool_registry import (
    build_runtime_tool_config,
    get_tool_registry,
)
from .config.pricing import (  # noqa: F401 — re-exported for callers
    IMAGE_PRICING,
    MAPS_GROUNDING_COST_PER_REQUEST,
    MODEL_PRICING,
    TTS_PRICING,
    UNKNOWN_CHAT_MODEL_PRICING,
    UNKNOWN_IMAGE_MODEL_INPUT_RATE,
    UNKNOWN_IMAGE_PER_IMAGE,
    UNKNOWN_TTS_MODEL_PRICING,
    UNKNOWN_VIDEO_PER_SECOND,
    VIDEO_PRICING,
)

TOOL_GOOGLE_SEARCH = build_runtime_tool_config("google_search") or {"google_search": {}}
TOOL_CODE_EXECUTION = build_runtime_tool_config("code_execution") or {"code_execution": {}}
TOOL_GOOGLE_MAPS = build_runtime_tool_config("google_maps") or {"google_maps": {}}
TOOL_URL_CONTEXT = build_runtime_tool_config("url_context") or {"url_context": {}}
TOOL_FILE_SEARCH = build_runtime_tool_config("file_search") or {"file_search": {}}
TOOL_CUSTOM_FUNCTIONS = {"_custom_functions": True}  # Sentinel for ButtonView toggle

DEFAULT_MUSIC_MODEL = "lyria-3-clip-preview"
LYRIA_REALTIME_MODEL = "lyria-realtime-exp"
LYRIA_3_MODELS = frozenset(
    {
        "lyria-3-pro-preview",
        "lyria-3-clip-preview",
    }
)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    google_maps_grounded: bool = False,
) -> float:
    """Calculate the cost in dollars for a given model and token usage.

    Thinking tokens are billed at the output token rate.
    When google_maps_grounded is True, adds the per-request Maps surcharge ($0.025).
    """
    input_price, output_price = MODEL_PRICING.get(model, UNKNOWN_CHAT_MODEL_PRICING)
    cost = (input_tokens / 1_000_000) * input_price + (
        (output_tokens + thinking_tokens) / 1_000_000
    ) * output_price
    if google_maps_grounded:
        cost += MAPS_GROUNDING_COST_PER_REQUEST
    return cost


def calculate_image_cost(
    model: str,
    num_images: int,
    input_tokens: int = 0,
    image_size: str | None = None,
) -> float:
    """Calculate the cost for image generation.

    For Gemini image models, includes input token cost plus per-image output cost
    at the requested resolution (image_size). Falls back to default (1K) rate.
    For Imagen models, uses flat per-image pricing only.
    """
    default_sizes: dict[str | None, float] = {None: UNKNOWN_IMAGE_PER_IMAGE}
    input_rate, size_prices = IMAGE_PRICING.get(
        model, (UNKNOWN_IMAGE_MODEL_INPUT_RATE, default_sizes)
    )
    # Normalize image_size to lowercase for lookup
    key = image_size.lower() if image_size else None
    per_image_cost = size_prices.get(key, size_prices.get(None, UNKNOWN_IMAGE_PER_IMAGE))
    return (input_tokens / 1_000_000) * input_rate + num_images * per_image_cost


def calculate_video_cost(
    model: str,
    duration_seconds: int,
    num_videos: int = 1,
    resolution: str | None = None,
) -> float:
    """Calculate the cost for video generation using model and optional resolution."""
    resolution_prices = VIDEO_PRICING.get(model, {"default": UNKNOWN_VIDEO_PER_SECOND})
    price_key = resolution.lower() if resolution else "default"
    price_per_second = resolution_prices.get(
        price_key, resolution_prices.get("default", UNKNOWN_VIDEO_PER_SECOND)
    )
    return duration_seconds * num_videos * price_per_second


def calculate_tts_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost for text-to-speech generation."""
    input_price, output_price = TTS_PRICING.get(model, UNKNOWN_TTS_MODEL_PRICING)
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


# Minimum input token counts required for this bot's explicit context caching policy.
# Gemini 2.5 and newer models also support implicit caching automatically.
# Models not listed here fall back to implicit caching only.
CACHE_MIN_TOKEN_COUNT: dict[str, int] = {
    "gemini-3.1-pro-preview": 4096,
    "gemini-3-flash-preview": 1024,
    "gemini-2.5-pro": 4096,
    "gemini-2.5-flash": 1024,
}

CACHE_TTL = "3600s"  # 60-minute TTL for explicit caches

MAX_AGENTIC_ITERATIONS = 10  # Max tool-calling round-trips per user message
TYPING_INDICATOR_INTERVAL = 5  # Seconds between typing indicator resends
VIDEO_GENERATION_TIMEOUT = 600  # Max seconds to wait for video generation
WS_DRAIN_INTERVAL = 0.1  # Seconds to wait for WebSocket cleanup drain

# Attachment size limits for Gemini API file input (bytes)
ATTACHMENT_MAX_INLINE_SIZE = 100 * 1024 * 1024  # 100 MB general inline limit
ATTACHMENT_PDF_MAX_INLINE_SIZE = 50 * 1024 * 1024  # 50 MB PDF inline limit
ATTACHMENT_FILE_API_THRESHOLD = 20 * 1024 * 1024  # 20 MB — prefer File API above this
ATTACHMENT_FILE_API_MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB File API limit
_TOOL_REGISTRY = get_tool_registry()

AVAILABLE_TOOLS = {
    tool_id: config
    for tool_id in _TOOL_REGISTRY
    if tool_id != "custom_functions"
    if (config := build_runtime_tool_config(tool_id)) is not None
}
SERVER_SIDE_TOOLS = frozenset(AVAILABLE_TOOLS)

# Tools that cannot be combined with file_search per API limitations.
FILE_SEARCH_INCOMPATIBLE_TOOLS = frozenset(
    tool_id for tool_id, metadata in _TOOL_REGISTRY.items() if metadata.file_search_incompatible
)

# Sets of tools that are mutually exclusive — only one from each set can be active at a time.
MUTUALLY_EXCLUSIVE_TOOLS: list[tuple[str, str]] = []
for _tool_id, _metadata in _TOOL_REGISTRY.items():
    for _other in _metadata.mutually_exclusive_with:
        _pair = (_tool_id, _other)
        if (
            _pair not in MUTUALLY_EXCLUSIVE_TOOLS
            and (_other, _tool_id) not in MUTUALLY_EXCLUSIVE_TOOLS
        ):
            MUTUALLY_EXCLUSIVE_TOOLS.append(_pair)

# Model-specific compatibility constraints for tools that are not universally supported.
TOOL_MODEL_COMPATIBILITY: dict[str, set[str]] = {
    tool_id: set(metadata.model_allowlist)
    for tool_id, metadata in _TOOL_REGISTRY.items()
    if metadata.model_allowlist is not None
}


@dataclass
class ChatCompletionParameters:
    """A dataclass to store the parameters for a chat completion."""

    model: str
    system_instruction: str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    media_resolution: str | None = None
    thinking_level: str | None = None
    thinking_budget: int | None = None
    conversation_starter: Member | User | None = None
    conversation_id: int | None = None
    channel_id: int | None = None
    paused: bool | None = False
    history: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    cache_name: str | None = None
    cached_history_length: int = 0
    uploaded_file_names: list[str] = field(default_factory=list)
    custom_functions_enabled: bool = False


@dataclass
class ImageGenerationParameters:
    """A dataclass to store the parameters for an image generation."""

    prompt: str
    model: str
    number_of_images: int = 1
    aspect_ratio: str | None = None
    person_generation: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    guidance_scale: float | None = None
    image_size: str | None = None
    google_image_search: bool = False

    def to_dict(self):
        """Convert to dictionary for API calls, filtering out None values and handling special cases."""
        config_dict = {}

        if self.number_of_images is not None:
            config_dict["number_of_images"] = self.number_of_images
        if self.aspect_ratio is not None:
            config_dict["aspect_ratio"] = self.aspect_ratio
        if self.negative_prompt is not None:
            config_dict["negative_prompt"] = self.negative_prompt
        if self.seed is not None:
            config_dict["seed"] = self.seed
        if self.guidance_scale is not None:
            config_dict["guidance_scale"] = self.guidance_scale

        # Handle person_generation mapping for Imagen models
        if self.person_generation and self.person_generation != "allow_adult":
            person_gen_map = {
                "dont_allow": "DONT_ALLOW",
                "allow_adult": "ALLOW_ADULT",
                "allow_all": "ALLOW_ALL",
            }
            if self.person_generation in person_gen_map:
                config_dict["person_generation"] = person_gen_map[self.person_generation]

        return config_dict


@dataclass
class VideoGenerationParameters:
    """A dataclass to store the parameters for a video generation."""

    prompt: str
    model: str
    aspect_ratio: str | None = None
    resolution: str | None = None
    person_generation: str | None = None
    negative_prompt: str | None = None
    number_of_videos: int | None = None
    duration_seconds: int | None = None
    enhance_prompt: bool | None = None
    has_last_frame: bool = False

    def to_dict(self):
        """Convert to dictionary for API calls, filtering out None values and handling special cases."""
        config_dict = {}

        if self.aspect_ratio is not None:
            config_dict["aspect_ratio"] = self.aspect_ratio
        if self.resolution is not None:
            config_dict["resolution"] = self.resolution
        if self.negative_prompt is not None:
            config_dict["negative_prompt"] = self.negative_prompt
        if self.number_of_videos is not None:
            config_dict["number_of_videos"] = self.number_of_videos
        if self.duration_seconds is not None:
            config_dict["duration_seconds"] = self.duration_seconds
        if self.enhance_prompt is not None:
            config_dict["enhance_prompt"] = self.enhance_prompt

        # Handle person_generation mapping for Veo models
        if self.person_generation and self.person_generation != "allow_adult":
            person_gen_map = {
                "dont_allow": "dont_allow",
                "allow_adult": "allow_adult",
                "allow_all": "allow_all",
            }
            if self.person_generation in person_gen_map:
                config_dict["person_generation"] = person_gen_map[self.person_generation]

        return config_dict


@dataclass
class SpeechGenerationParameters:
    """A dataclass to store the parameters for a speech generation."""

    input_text: str
    model: str = "gemini-2.5-flash-preview-tts"
    voice_name: str | None = "Kore"
    multi_speaker: bool = False
    speaker_configs: list[dict[str, str]] | None = None
    style_prompt: str | None = None

    def to_dict(self):
        """Convert to dictionary for API calls, filtering out None values."""
        config_dict: dict[str, Any] = {"response_modalities": ["AUDIO"]}

        # Add speech config
        speech_config: dict[str, Any] = {}

        if self.multi_speaker and self.speaker_configs:
            # Multi-speaker configuration
            speaker_voice_configs = []
            for speaker_config in self.speaker_configs:
                speaker_voice_configs.append(
                    {
                        "speaker": speaker_config["speaker"],
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": speaker_config["voice_name"]}
                        },
                    }
                )
            speech_config["multi_speaker_voice_config"] = {
                "speaker_voice_configs": speaker_voice_configs
            }
        else:
            # Single speaker configuration
            speech_config["voice_config"] = {
                "prebuilt_voice_config": {"voice_name": self.voice_name}
            }

        config_dict["speech_config"] = speech_config
        return config_dict


@dataclass
class MusicGenerationParameters:
    """A dataclass to store parameters for music generation using Lyria models."""

    prompts: list[str]
    model: str = DEFAULT_MUSIC_MODEL
    prompt_weights: list[float] | None = None
    duration: int = 30  # seconds
    bpm: int | None = None  # 60-200
    scale: str | None = None  # e.g., "C_MAJOR_A_MINOR"
    guidance: float = 4.0  # 0.0-6.0
    density: float | None = None  # 0.0-1.0
    brightness: float | None = None  # 0.0-1.0
    temperature: float = 1.1  # 0.0-3.0
    top_k: int = 40  # 1-1000
    seed: int | None = None
    mute_bass: bool = False
    mute_drums: bool = False
    only_bass_and_drums: bool = False

    def to_weighted_prompts(self) -> list[dict[str, Any]]:
        """Convert prompts to WeightedPrompt format for Gemini API."""
        if self.prompt_weights and len(self.prompt_weights) == len(self.prompts):
            return [
                {"text": prompt, "weight": weight}
                for prompt, weight in zip(self.prompts, self.prompt_weights, strict=True)
            ]
        else:
            # Default weight of 1.0 for all prompts
            return [{"text": prompt, "weight": 1.0} for prompt in self.prompts]

    def to_music_config(self) -> dict[str, Any]:
        """Convert to MusicGenerationConfig format for Gemini API."""
        config = {
            "guidance": self.guidance,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "mute_bass": self.mute_bass,
            "mute_drums": self.mute_drums,
            "only_bass_and_drums": self.only_bass_and_drums,
        }

        # Add optional parameters if provided
        if self.bpm is not None:
            config["bpm"] = self.bpm
        if self.scale is not None:
            config["scale"] = self.scale
        if self.density is not None:
            config["density"] = self.density
        if self.brightness is not None:
            config["brightness"] = self.brightness
        if self.seed is not None:
            config["seed"] = self.seed

        return config


@dataclass
class ResearchParameters:
    """A dataclass to store the parameters for a deep research task."""

    prompt: str
    agent: str = "deep-research-pro-preview-12-2025"
    file_search: bool = False
    google_maps: bool = False
    collaborative_planning: bool = False
    thinking_summaries: Literal["auto", "none"] | None = None
    visualization: Literal["auto", "off"] | None = None


@dataclass
class AgenticResult:
    """Aggregated result from an agentic tool-calling loop."""

    response: Any  # Final GenerateContentResponse
    contents: list[dict[str, Any]]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0
    iterations: int = 0
    tool_calls_made: list[str] = field(default_factory=list)


def resolve_tool_name(tool_config: Any) -> str | None:
    """
    Resolve a tool config back to its canonical tool name.

    Handles dict-based server-side tools, tools with dynamic configuration
    (e.g., file_search with injected store IDs), function_declarations dicts,
    and Python callables (custom function tools).
    """
    # Python callable — custom function tool
    if callable(tool_config):
        return "custom_functions"
    if not isinstance(tool_config, dict):
        return None
    # function_declarations dict — custom function tools
    if "function_declarations" in tool_config:
        return "custom_functions"
    for tool_name, available_config in AVAILABLE_TOOLS.items():
        if tool_config == available_config:
            return tool_name
    # Fallback: match by top-level key for tools with dynamic config
    for key in tool_config:
        if key in AVAILABLE_TOOLS:
            return key
    return None


def filter_supported_tools_for_model(model: str, tools: list[Any]) -> tuple[list[Any], list[str]]:
    """
    Return model-compatible tools and a list of unsupported tool names.

    Callables (custom function tools) are passed through unchanged since
    they have no model compatibility constraints.
    """
    supported_tools: list[Any] = []
    unsupported_tools: list[str] = []

    for tool_config in tools:
        tool_name = resolve_tool_name(tool_config)

        # Callables and unknown tools pass through
        if tool_name is None or tool_name == "custom_functions":
            supported_tools.append(tool_config if callable(tool_config) else deepcopy(tool_config))
            continue

        supported_models = TOOL_MODEL_COMPATIBILITY.get(tool_name)
        if supported_models is not None and model not in supported_models:
            unsupported_tools.append(tool_name)
            continue

        supported_tools.append(deepcopy(tool_config))

    return supported_tools, unsupported_tools


def has_server_side_tools(tools: list[Any]) -> bool:
    """Return True when the provided tool list includes Gemini server-side tools."""

    return any(resolve_tool_name(tool_config) in SERVER_SIDE_TOOLS for tool_config in tools)


def model_supports_tool_combinations(model: str) -> bool:
    """Return True when the model supports built-in/custom tool combinations."""

    return model.startswith("gemini-3")


def validate_builtin_custom_tool_combination(
    model: str,
    tools: list[Any],
    custom_functions_enabled: bool,
) -> str | None:
    """Validate whether built-in and custom tools can be combined for the model."""

    if not custom_functions_enabled or not has_server_side_tools(tools):
        return None
    if model_supports_tool_combinations(model):
        return None
    return (
        f"Built-in tools and custom functions cannot be combined on `{model}`. "
        "Tool combination preview requires a Gemini 3 model."
    )


def filter_file_search_incompatible_tools(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Remove tools incompatible with file_search when file_search is present.

    Returns the filtered tools list and a list of removed tool names.
    """
    has_file_search = any("file_search" in tool for tool in tools)
    if not has_file_search:
        return tools, []

    filtered: list[dict[str, Any]] = []
    removed: list[str] = []
    for tool in tools:
        name = resolve_tool_name(tool)
        if name in FILE_SEARCH_INCOMPATIBLE_TOOLS:
            removed.append(name)
        else:
            filtered.append(tool)
    return filtered, removed


def check_mutually_exclusive_tools(tool_names: set[str]) -> str | None:
    """
    Check whether any mutually exclusive tools have been selected together.

    Returns an error message string if a conflict is found, or None if OK.
    """
    for tool_a, tool_b in MUTUALLY_EXCLUSIVE_TOOLS:
        if tool_a in tool_names and tool_b in tool_names:
            return (
                f"`{tool_a}` and `{tool_b}` cannot be combined in the same "
                f"request. Please choose one to continue."
            )
    return None


def chunk_text(text: str, chunk_size: int = 4096) -> list[str]:
    """
    Splits a string into chunks of a specified size.

    Args:
        text: The string to split.
        chunk_size: The maximum size of each chunk.

    Returns:
        A list of strings, where each string is a chunk of the original text.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def truncate_text(text: str | None, max_length: int, suffix: str = "...") -> str | None:
    """Truncate text to max_length, adding suffix if truncated.

    Args:
        text: The text to truncate (``None`` passes through unchanged).
        max_length: Maximum length before truncation.
        suffix: String to append when truncated (default ``"..."``).

    Returns:
        Original text if under max_length, truncated with suffix otherwise,
        or ``None`` if *text* was ``None``.
    """
    if text is None:
        return None
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if len(suffix) >= max_length:
        return suffix[:max_length]
    return text[: max_length - len(suffix)] + suffix
