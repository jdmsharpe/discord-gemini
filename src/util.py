from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal


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
    conversation_starter: Optional[str] = None
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
                "allow_all": "ALLOW_ALL"
            }
            if self.person_generation in person_gen_map:
                config_dict["person_generation"] = person_gen_map[self.person_generation]
        
        return config_dict


@dataclass
class VideoGenerationParameters:
    """A dataclass to store the parameters for a video generation."""

    prompt: str
    model: str = "veo-2.0-generate-001"
    model_options: Literal["veo-2.0-generate-001"] = "veo-2.0-generate-001"


@dataclass
class SpeechGenerationParameters:
    """A dataclass to store the parameters for a speech generation."""

    prompt: str
    model: str = "gemini-2.5-flash-preview-tts"
    model_options: Literal[
        "gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"
    ] = "gemini-2.5-flash-preview-tts"


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
