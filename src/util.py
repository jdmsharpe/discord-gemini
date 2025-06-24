from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal


@dataclass
class ChatCompletionParameters:
    """A dataclass to store the parameters for a chat completion."""

    model: str = "gemini-2.5-flash"
    model_options: Literal[
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ] = "gemini-1.5-flash"
    history: List[Dict[str, str]] = field(default_factory=list)
    persona: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    conversation_starter: Optional[str] = None
    conversation_id: Optional[int] = None
    channel_id: Optional[int] = None
    paused: Optional[bool] = False


@dataclass
class ImageGenerationParameters:
    """A dataclass to store the parameters for an image generation."""

    prompt: str
    model: str = "imagen-3.0-generate-002"
    model_options: Literal[
        "imagen-3.0-generate-002", "gemini-2.0-flash-preview-image-generation"
    ] = "imagen-3.0-generate-002"


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
