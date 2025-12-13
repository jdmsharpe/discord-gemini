from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from discord import Member, User


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
    conversation_starter: Optional[Union[Member, User]] = None
    conversation_id: Optional[int] = None
    channel_id: Optional[int] = None
    paused: Optional[bool] = False
    history: List[Dict[str, str]] = field(default_factory=list)


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
class EmbeddingParameters:
    """A dataclass to store the parameters for an embedding."""

    prompt: str
    model: str = "gemini-embedding-exp"
    model_options: Literal["gemini-embedding-exp"] = "gemini-embedding-exp"


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
