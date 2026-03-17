from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from discord import Member, User

TOOL_GOOGLE_SEARCH = {"google_search": {}}
TOOL_CODE_EXECUTION = {"code_execution": {}}
TOOL_GOOGLE_MAPS = {"google_maps": {}}
TOOL_URL_CONTEXT = {"url_context": {}}
TOOL_FILE_SEARCH = {"file_search": {}}

# Per-million-token pricing: (input_cost, output_cost)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
}


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int, thinking_tokens: int = 0
) -> float:
    """Calculate the cost in dollars for a given model and token usage.

    Thinking tokens are billed at the output token rate.
    """
    input_price, output_price = MODEL_PRICING.get(model, (2.0, 12.0))
    return (input_tokens / 1_000_000) * input_price + (
        (output_tokens + thinking_tokens) / 1_000_000
    ) * output_price

# Minimum input token counts required for explicit context caching per model.
# Models not listed here rely on implicit caching (automatic, no dev work).
# Only Gemini 3.x models use explicit caching; 2.5 and below use implicit.
CACHE_MIN_TOKEN_COUNT: Dict[str, int] = {
    "gemini-3.1-pro-preview": 4096,
    "gemini-3-flash-preview": 1024,
}

CACHE_TTL = "3600s"  # 60-minute TTL for explicit caches

# Attachment size limits for Gemini API file input (bytes)
ATTACHMENT_MAX_INLINE_SIZE = 100 * 1024 * 1024  # 100 MB general inline limit
ATTACHMENT_PDF_MAX_INLINE_SIZE = 50 * 1024 * 1024  # 50 MB PDF inline limit
ATTACHMENT_FILE_API_THRESHOLD = 20 * 1024 * 1024  # 20 MB — prefer File API above this
ATTACHMENT_FILE_API_MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB File API limit
AVAILABLE_TOOLS = {
    "google_search": TOOL_GOOGLE_SEARCH,
    "code_execution": TOOL_CODE_EXECUTION,
    "google_maps": TOOL_GOOGLE_MAPS,
    "url_context": TOOL_URL_CONTEXT,
    "file_search": TOOL_FILE_SEARCH,
}

# Tools that cannot be combined with file_search per API limitations.
FILE_SEARCH_INCOMPATIBLE_TOOLS = frozenset({"google_search", "google_maps", "url_context"})

# Model-specific compatibility constraints for tools that are not universally supported.
TOOL_MODEL_COMPATIBILITY: Dict[str, Set[str]] = {
    "google_maps": {
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    },
    "url_context": {
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    },
    "file_search": {
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    },
}


@dataclass
class ChatCompletionParameters:
    """A dataclass to store the parameters for a chat completion."""

    model: str
    system_instruction: Optional[str] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    media_resolution: Optional[str] = None
    thinking_level: Optional[str] = None
    thinking_budget: Optional[int] = None
    conversation_starter: Optional[Union[Member, User]] = None
    conversation_id: Optional[int] = None
    channel_id: Optional[int] = None
    paused: Optional[bool] = False
    history: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    cache_name: Optional[str] = None
    cached_history_length: int = 0
    uploaded_file_names: List[str] = field(default_factory=list)


@dataclass
class ImageGenerationParameters:
    """A dataclass to store the parameters for an image generation."""

    prompt: str
    model: str
    number_of_images: int = 1
    aspect_ratio: Optional[str] = None
    person_generation: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: Optional[float] = None
    image_size: Optional[str] = None
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
                config_dict["person_generation"] = person_gen_map[
                    self.person_generation
                ]

        return config_dict


@dataclass
class VideoGenerationParameters:
    """A dataclass to store the parameters for a video generation."""

    prompt: str
    model: str
    aspect_ratio: Optional[str] = None
    person_generation: Optional[str] = None
    negative_prompt: Optional[str] = None
    number_of_videos: Optional[int] = None
    duration_seconds: Optional[int] = None
    enhance_prompt: Optional[bool] = None

    def to_dict(self):
        """Convert to dictionary for API calls, filtering out None values and handling special cases."""
        config_dict = {}

        if self.aspect_ratio is not None:
            config_dict["aspect_ratio"] = self.aspect_ratio
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
                config_dict["person_generation"] = person_gen_map[
                    self.person_generation
                ]

        return config_dict


@dataclass
class SpeechGenerationParameters:
    """A dataclass to store the parameters for a speech generation."""

    input_text: str
    model: str = "gemini-2.5-flash-preview-tts"
    voice_name: Optional[str] = "Kore"
    multi_speaker: bool = False
    speaker_configs: Optional[List[Dict[str, str]]] = None
    style_prompt: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for API calls, filtering out None values."""
        config_dict: Dict[str, Any] = {"response_modalities": ["AUDIO"]}

        # Add speech config
        speech_config: Dict[str, Any] = {}

        if self.multi_speaker and self.speaker_configs:
            # Multi-speaker configuration
            speaker_voice_configs = []
            for speaker_config in self.speaker_configs:
                speaker_voice_configs.append(
                    {
                        "speaker": speaker_config["speaker"],
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": speaker_config["voice_name"]
                            }
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
    """A dataclass to store parameters for music generation using Lyria RealTime."""

    prompts: List[str]
    prompt_weights: Optional[List[float]] = None
    duration: int = 30  # seconds
    bpm: Optional[int] = None  # 60-200
    scale: Optional[str] = None  # e.g., "C_MAJOR_A_MINOR"
    guidance: float = 4.0  # 0.0-6.0
    density: Optional[float] = None  # 0.0-1.0
    brightness: Optional[float] = None  # 0.0-1.0
    temperature: float = 1.1  # 0.0-3.0
    top_k: int = 40  # 1-1000
    seed: Optional[int] = None
    mute_bass: bool = False
    mute_drums: bool = False
    only_bass_and_drums: bool = False

    def to_weighted_prompts(self) -> List[Dict[str, Any]]:
        """Convert prompts to WeightedPrompt format for Gemini API."""
        if self.prompt_weights and len(self.prompt_weights) == len(self.prompts):
            return [
                {"text": prompt, "weight": weight}
                for prompt, weight in zip(self.prompts, self.prompt_weights)
            ]
        else:
            # Default weight of 1.0 for all prompts
            return [{"text": prompt, "weight": 1.0} for prompt in self.prompts]

    def to_music_config(self) -> Dict[str, Any]:
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


@dataclass
class EmbeddingParameters:
    """A dataclass to store the parameters for an embedding."""

    prompt: str
    model: str = "gemini-embedding-exp"
    model_options: Literal["gemini-embedding-exp"] = "gemini-embedding-exp"


def resolve_tool_name(tool_config: Dict[str, Any]) -> Optional[str]:
    """
    Resolve a tool config dictionary back to its canonical tool name.

    Handles both exact matches and tools with dynamic configuration
    (e.g., file_search with injected store IDs) by falling back to
    top-level key matching.
    """
    for tool_name, available_config in AVAILABLE_TOOLS.items():
        if tool_config == available_config:
            return tool_name
    # Fallback: match by top-level key for tools with dynamic config
    for key in tool_config:
        if key in AVAILABLE_TOOLS:
            return key
    return None


def filter_supported_tools_for_model(
    model: str, tools: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Return model-compatible tools and a list of unsupported tool names.
    """
    supported_tools: List[Dict[str, Any]] = []
    unsupported_tools: List[str] = []

    for tool_config in tools:
        tool_name = resolve_tool_name(tool_config)
        if tool_name is None:
            supported_tools.append(deepcopy(tool_config))
            continue

        supported_models = TOOL_MODEL_COMPATIBILITY.get(tool_name)
        if supported_models is not None and model not in supported_models:
            unsupported_tools.append(tool_name)
            continue

        supported_tools.append(deepcopy(tool_config))

    return supported_tools, unsupported_tools


def filter_file_search_incompatible_tools(
    tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Remove tools incompatible with file_search when file_search is present.

    Returns the filtered tools list and a list of removed tool names.
    """
    has_file_search = any("file_search" in tool for tool in tools)
    if not has_file_search:
        return tools, []

    filtered: List[Dict[str, Any]] = []
    removed: List[str] = []
    for tool in tools:
        name = resolve_tool_name(tool)
        if name in FILE_SEARCH_INCOMPATIBLE_TOOLS:
            removed.append(name)
        else:
            filtered.append(tool)
    return filtered, removed


def chunk_text(text: str, chunk_size: int = 4096) -> List[str]:
    """
    Splits a string into chunks of a specified size.

    Args:
        text: The string to split.
        chunk_size: The maximum size of each chunk.

    Returns:
        A list of strings, where each string is a chunk of the original text.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max_length, adding suffix if truncated.

    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: String to append when truncated (default "...")

    Returns:
        Original text if under max_length, otherwise truncated with suffix
    """
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix
