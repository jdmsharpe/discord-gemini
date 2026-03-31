from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gemini_test_support import AsyncGeminiCogTestCase


class TestGeminiCaching(AsyncGeminiCogTestCase):
    """Tests for explicit context caching logic."""

    @pytest.fixture(autouse=True)
    async def setup_caching(self, setup):
        self.mock_client_instance.aio.caches.create = AsyncMock()
        self.mock_client_instance.aio.caches.delete = AsyncMock()
        self.mock_client_instance.aio.caches.update = AsyncMock()

    async def test_maybe_create_cache_below_threshold(self):
        """Test that _maybe_create_cache does nothing when below token threshold."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        history = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=500))

        await self.cog._maybe_create_cache(params, history, response)

        assert params.cache_name is None
        assert params.cached_history_length == 0
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_above_threshold(self):
        """Test that _maybe_create_cache creates a cache when above threshold."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview", conversation_id=100)
        history = [
            {"role": "user", "parts": [{"text": "long prompt " * 200}]},
            {"role": "model", "parts": [{"text": "long response " * 200}]},
        ]
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=2000))

        mock_cache = SimpleNamespace(name="cachedContents/test-cache-id")
        self.cog.client.aio.caches.create.return_value = mock_cache

        await self.cog._maybe_create_cache(params, history, response)

        assert params.cache_name == "cachedContents/test-cache-id"
        assert params.cached_history_length == 2
        self.cog.client.aio.caches.create.assert_called_once()

    async def test_maybe_create_cache_refreshes_ttl(self):
        """Test that _maybe_create_cache refreshes TTL when uncached tail is small."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/existing",
            cached_history_length=4,
        )
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=3000,
                cached_content_token_count=2500,
            )
        )

        await self.cog._maybe_create_cache(params, [], response)

        self.cog.client.aio.caches.create.assert_not_called()
        self.cog.client.aio.caches.update.assert_called_once()
        assert params.cache_name == "cachedContents/existing"
        assert params.cached_history_length == 4

    async def test_maybe_create_cache_refreshes_ttl_error_handled(self):
        """Test that TTL refresh errors are handled gracefully."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/existing",
            cached_history_length=4,
        )
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1500,
                cached_content_token_count=1200,
            )
        )
        self.cog.client.aio.caches.update.side_effect = Exception("TTL error")

        await self.cog._maybe_create_cache(params, [], response)

        assert params.cache_name == "cachedContents/existing"

    async def test_maybe_create_cache_recaches_when_uncached_tail_large(self):
        """Test that _maybe_create_cache re-caches when uncached portion exceeds threshold."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            conversation_id=100,
            cache_name="cachedContents/old-cache",
            cached_history_length=4,
        )
        history = [{"role": "user", "parts": [{"text": f"msg {i}"}]} for i in range(10)]
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=5000,
                cached_content_token_count=2000,
            )
        )

        mock_cache = SimpleNamespace(name="cachedContents/new-cache")
        self.cog.client.aio.caches.create.return_value = mock_cache

        await self.cog._maybe_create_cache(params, history, response)

        assert params.cache_name == "cachedContents/new-cache"
        assert params.cached_history_length == 10
        self.cog.client.aio.caches.create.assert_called_once()
        self.cog.client.aio.caches.delete.assert_called_once_with(name="cachedContents/old-cache")
        self.cog.client.aio.caches.update.assert_not_called()

    async def test_maybe_create_cache_recache_keeps_old_on_create_error(self):
        """Test that re-cache failure preserves the old cache."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            conversation_id=100,
            cache_name="cachedContents/old-cache",
            cached_history_length=4,
        )
        history = [{"role": "user", "parts": [{"text": "msg"}]}] * 10
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=5000,
                cached_content_token_count=2000,
            )
        )
        self.cog.client.aio.caches.create.side_effect = Exception("Create failed")

        await self.cog._maybe_create_cache(params, history, response)

        assert params.cache_name == "cachedContents/old-cache"
        assert params.cached_history_length == 4
        self.cog.client.aio.caches.delete.assert_not_called()

    async def test_maybe_create_cache_unsupported_model(self):
        """Test that _maybe_create_cache skips for models without explicit caching."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-2.0-flash")
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=5000))

        await self.cog._maybe_create_cache(params, [], response)

        assert params.cache_name is None
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_implicit_only_models_skipped(self):
        """Test that 2.5 models rely on implicit caching and are skipped."""
        from discord_gemini.util import ChatCompletionParameters

        for model in ("gemini-2.5-pro", "gemini-2.5-flash"):
            params = ChatCompletionParameters(model=model)
            response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10000))

            await self.cog._maybe_create_cache(params, [], response)

            assert params.cache_name is None
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_no_usage_metadata(self):
        """Test that _maybe_create_cache handles missing usage_metadata."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        response = SimpleNamespace()

        await self.cog._maybe_create_cache(params, [], response)

        assert params.cache_name is None
        self.cog.client.aio.caches.create.assert_not_called()

    async def test_maybe_create_cache_handles_api_error(self):
        """Test that _maybe_create_cache logs warning on API error."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview", conversation_id=100)
        history = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=2000))

        self.cog.client.aio.caches.create.side_effect = Exception("API error")

        await self.cog._maybe_create_cache(params, history, response)

        assert params.cache_name is None
        assert params.cached_history_length == 0

    async def test_delete_conversation_cache(self):
        """Test that _delete_conversation_cache deletes and clears fields."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/to-delete",
            cached_history_length=4,
        )

        await self.cog._delete_conversation_cache(params)

        self.cog.client.aio.caches.delete.assert_called_once_with(name="cachedContents/to-delete")
        assert params.cache_name is None
        assert params.cached_history_length == 0

    async def test_delete_conversation_cache_noop_when_none(self):
        """Test that _delete_conversation_cache is a no-op when no cache exists."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(model="gemini-3-flash-preview")

        await self.cog._delete_conversation_cache(params)

        self.cog.client.aio.caches.delete.assert_not_called()

    async def test_delete_conversation_cache_handles_api_error(self):
        """Test that _delete_conversation_cache handles API errors gracefully."""
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-3-flash-preview",
            cache_name="cachedContents/broken",
            cached_history_length=2,
        )

        self.cog.client.aio.caches.delete.side_effect = Exception("Not found")

        await self.cog._delete_conversation_cache(params)

        assert params.cache_name is None
        assert params.cached_history_length == 0
