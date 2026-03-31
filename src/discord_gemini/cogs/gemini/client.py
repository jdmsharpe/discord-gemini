"""Gemini client builders."""

from google import genai
from google.genai import types

from ...config.auth import GEMINI_API_KEY, GEMINI_API_VERSION


def build_gemini_client() -> genai.Client:
    """Build the standard Gemini client used by the cog."""

    client_kwargs = {"api_key": GEMINI_API_KEY}
    if GEMINI_API_VERSION:
        client_kwargs["http_options"] = types.HttpOptions(api_version=GEMINI_API_VERSION)
    return genai.Client(**client_kwargs)


def build_lyria_realtime_client() -> genai.Client:
    """Build the v1alpha client required for Lyria RealTime."""

    return genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"})


__all__ = ["build_gemini_client", "build_lyria_realtime_client"]
