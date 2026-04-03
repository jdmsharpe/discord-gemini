"""Explicit cache lifecycle helpers for Gemini conversations."""

import aiohttp
from typing import TYPE_CHECKING, Any, cast

from google.genai import types
from google.genai.errors import APIError

from ...util import CACHE_MIN_TOKEN_COUNT, CACHE_TTL
from . import usage

if TYPE_CHECKING:
    from .cog import GeminiCog


CACHE_API_EXCEPTIONS = (APIError, aiohttp.ClientError, TimeoutError)


async def _maybe_create_cache(
    cog: "GeminiCog",
    params: Any,
    history: list[dict[str, Any]],
    response: Any,
) -> None:
    """Create, refresh, or re-create an explicit cache based on conversation size."""

    threshold = CACHE_MIN_TOKEN_COUNT.get(params.model)
    if threshold is None:
        return

    usage_counts = usage.extract_usage_counts(response)
    prompt_tokens = usage_counts.input_tokens
    if prompt_tokens <= 0:
        return

    if params.cache_name:
        cached_tokens = usage_counts.cached_tokens
        uncached_tokens = prompt_tokens - cached_tokens
        if uncached_tokens >= threshold:
            await _recache(cog, params, history, prompt_tokens, uncached_tokens)
        else:
            await _refresh_cache_ttl(cog, params)
        return

    if prompt_tokens < threshold:
        return

    try:
        contents = [{"role": entry["role"], "parts": entry["parts"]} for entry in history]
        cache = await cog.client.aio.caches.create(
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
        cog.logger.info(
            "Created cache %s for conversation %s (%d prompt tokens)",
            cache.name,
            params.conversation_id,
            prompt_tokens,
        )
    except CACHE_API_EXCEPTIONS as error:
        cog.logger.warning("Failed to create cache: %s", error)


async def _recache(
    cog: "GeminiCog",
    params: Any,
    history: list[dict[str, Any]],
    prompt_tokens: int,
    uncached_tokens: int,
) -> None:
    """Delete the old cache and create a new one covering the full history."""

    old_cache_name = params.cache_name
    try:
        contents = [{"role": entry["role"], "parts": entry["parts"]} for entry in history]
        cache = await cog.client.aio.caches.create(
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
        cog.logger.info(
            "Re-cached conversation %s as %s (%d prompt tokens, %d were uncached)",
            params.conversation_id,
            cache.name,
            prompt_tokens,
            uncached_tokens,
        )
    except CACHE_API_EXCEPTIONS as error:
        cog.logger.warning("Failed to re-cache: %s", error)
        return

    if old_cache_name is not None:
        try:
            await cog.client.aio.caches.delete(name=old_cache_name)
        except CACHE_API_EXCEPTIONS as error:
            cog.logger.warning("Failed to delete old cache %s: %s", old_cache_name, error)


async def _refresh_cache_ttl(cog: "GeminiCog", params: Any) -> None:
    """Extend the TTL of an existing cache so it does not expire between turns."""

    cache_name = params.cache_name
    if cache_name is None:
        return

    try:
        await cog.client.aio.caches.update(
            name=cache_name,
            config=types.UpdateCachedContentConfig(ttl=CACHE_TTL),
        )
    except CACHE_API_EXCEPTIONS as error:
        cog.logger.warning("Failed to refresh cache TTL for %s: %s", params.cache_name, error)


async def _delete_conversation_cache(cog: "GeminiCog", params: Any) -> None:
    """Delete the explicit cache for a conversation, if one exists."""

    if not params.cache_name:
        return

    try:
        await cog.client.aio.caches.delete(name=params.cache_name)
        cog.logger.info("Deleted cache %s", params.cache_name)
    except CACHE_API_EXCEPTIONS as error:
        cog.logger.warning("Failed to delete cache %s: %s", params.cache_name, error)
    params.cache_name = None
    params.cached_history_length = 0


__all__ = [
    "_delete_conversation_cache",
    "_maybe_create_cache",
    "_recache",
    "_refresh_cache_ttl",
]
