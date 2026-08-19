"""Gemini client builders."""

from google import genai
from google.genai import types

from ...config.auth import GEMINI_API_KEY, GEMINI_API_VERSION

MAX_API_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 60.0
RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]


def _build_retry_options() -> types.HttpRetryOptions:
    return types.HttpRetryOptions(
        attempts=MAX_API_ATTEMPTS,
        initial_delay=INITIAL_RETRY_DELAY_SECONDS,
        max_delay=MAX_RETRY_DELAY_SECONDS,
        http_status_codes=list(RETRYABLE_STATUS_CODES),
    )


def disable_afc() -> types.AutomaticFunctionCallingConfig:
    """Turn the SDK's automatic function calling off for a request.

    No flow wants the SDK executing callables: chat runs its own tool loop in
    ``chat._run_agentic_loop`` and the media flows pass no callables at all.
    Leaving AFC nominally on also routes ``generate_content`` through the SDK's
    AFC wrapper, which logs a one-shot "direct use of AFC" warning per process.
    """

    return types.AutomaticFunctionCallingConfig(disable=True)


def build_gemini_client() -> genai.Client:
    """Build the standard Gemini client used by the cog."""

    retry_options = _build_retry_options()
    if GEMINI_API_VERSION:
        return genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                api_version=GEMINI_API_VERSION,
                retry_options=retry_options,
            ),
        )
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(retry_options=retry_options),
    )


def build_lyria_realtime_client() -> genai.Client:
    """Build the v1alpha client required for Lyria RealTime."""

    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            api_version="v1alpha",
            retry_options=_build_retry_options(),
        ),
    )


__all__ = ["build_gemini_client", "build_lyria_realtime_client"]
