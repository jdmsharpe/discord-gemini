"""Gemini client builders."""

from google import genai

from ...config.auth import GEMINI_API_KEY


def build_gemini_client() -> genai.Client:
    """Build the standard Gemini client used by the cog."""

    return genai.Client(api_key=GEMINI_API_KEY)


def build_lyria_realtime_client() -> genai.Client:
    """Build the v1alpha client required for Lyria RealTime."""

    return genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"})


__all__ = ["build_gemini_client", "build_lyria_realtime_client"]
