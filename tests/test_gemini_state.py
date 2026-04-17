from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.support import AsyncGeminiCogTestCase, GeminiCogTestCase


class TestGeminiStateHelpers(AsyncGeminiCogTestCase):
    async def test_cleanup_conversation_strips_view_and_clears_state(self):
        """Test that _cleanup_conversation edits the message to remove the view and cleans up views dict."""
        user = MagicMock()
        mock_message = AsyncMock()
        self.cog.last_view_messages[user] = mock_message
        self.cog.views[user] = MagicMock()

        await self.cog._cleanup_conversation(user)

        mock_message.edit.assert_awaited_once_with(view=None)
        assert user not in self.cog.last_view_messages
        assert user not in self.cog.views

    async def test_cleanup_conversation_no_previous_message(self):
        """Test that _cleanup_conversation handles missing last_view_messages gracefully."""
        user = MagicMock()
        self.cog.views[user] = MagicMock()

        await self.cog._cleanup_conversation(user)

        assert user not in self.cog.views

    async def test_cleanup_conversation_edit_fails(self):
        """Test that _cleanup_conversation handles deleted messages without raising."""
        import discord

        user = MagicMock()
        mock_message = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.reason = "Not Found"
        mock_message.edit.side_effect = discord.NotFound(mock_response, "Unknown Message")
        self.cog.last_view_messages[user] = mock_message
        self.cog.views[user] = MagicMock()

        await self.cog._cleanup_conversation(user)

        mock_message.edit.assert_awaited_once_with(view=None)
        assert user not in self.cog.last_view_messages
        assert user not in self.cog.views

    async def test_cleanup_conversation_unknown_user(self):
        """Test that _cleanup_conversation is a no-op for users with no state."""
        user = MagicMock()

        await self.cog._cleanup_conversation(user)

        assert user not in self.cog.last_view_messages
        assert user not in self.cog.views


class TestGeminiCostTracking(GeminiCogTestCase):
    """Tests for daily cost tracking and structured cost logging."""

    def test_cog_init_daily_costs(self):
        """Test that GeminiCog initializes daily_costs dict."""
        assert self.cog.daily_costs == {}

    def test_track_daily_cost_accumulates(self):
        """Test that _track_daily_cost accumulates costs for the same user and day."""
        daily1 = self.cog._track_daily_cost(user_id=12345, cost=0.50)
        assert daily1 == pytest.approx(0.50)

        daily2 = self.cog._track_daily_cost(user_id=12345, cost=0.50)
        assert daily2 == pytest.approx(1.00)

    def test_track_daily_cost_separate_users(self):
        """Test that _track_daily_cost tracks users independently."""
        self.cog._track_daily_cost(user_id=111, cost=0.10)
        daily_user2 = self.cog._track_daily_cost(user_id=222, cost=0.10)
        assert daily_user2 == pytest.approx(0.10)

    def test_track_daily_cost_with_thinking_tokens(self):
        """Test that _track_daily_cost accumulates pre-calculated cost including thinking."""
        from discord_gemini.util import calculate_cost

        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 500_000, thinking_tokens=500_000)
        daily = self.cog._track_daily_cost(user_id=99, cost=cost)
        assert daily == pytest.approx(0.50)

    def test_log_cost_basic(self):
        """Test that _log_cost calls logger.info with structured cost data."""
        self.cog.logger = MagicMock()
        self.cog._log_cost(
            "chat",
            12345,
            "gemini-2.0-flash",
            0.50,
            1.00,
            input_tokens=1000,
            output_tokens=500,
        )
        self.cog.logger.info.assert_called_once()
        call_args = self.cog.logger.info.call_args
        fmt = call_args[0][0]
        assert "COST" in fmt
        assert "command=%s" in fmt
        assert "daily=$%.4f" in fmt

    def test_log_cost_no_details(self):
        """Test _log_cost with no extra detail kwargs."""
        self.cog.logger = MagicMock()
        self.cog._log_cost("video", 999, "veo-3.1-generate-preview", 3.20, 5.00)
        self.cog.logger.info.assert_called_once()

    def test_log_cost_image_details(self):
        """Test _log_cost with image-specific details."""
        self.cog.logger = MagicMock()
        self.cog._log_cost(
            "image",
            42,
            "gemini-3.1-flash-image-preview",
            0.134,
            0.50,
            images=2,
            input_tokens=500,
        )
        self.cog.logger.info.assert_called_once()
        call_args = self.cog.logger.info.call_args
        assert "images=2" in str(call_args)


class TestGeminiPruneRuntimeState(AsyncGeminiCogTestCase):
    """Tests for _prune_runtime_state — TTL, overflow cap, cascade cleanup."""

    def _make_conversation(self, *, starter=None, age: timedelta = timedelta(0)):
        from discord_gemini.cogs.gemini.models import Conversation
        from discord_gemini.util import ChatCompletionParameters

        params = ChatCompletionParameters(
            model="gemini-2.0-flash",
            conversation_starter=starter,
        )
        conversation = Conversation(params=params, history=[])
        conversation.updated_at = datetime.now(timezone.utc) - age
        return conversation

    async def test_drops_conversations_older_than_ttl(self):
        from discord_gemini.cogs.gemini.state import (
            CONVERSATION_TTL,
            _prune_runtime_state,
        )

        user = MagicMock(spec=["id"])
        user.id = 42
        self.cog._delete_conversation_cache = AsyncMock()
        self.cog._cleanup_uploaded_files = AsyncMock()
        self.cog.conversations[1] = self._make_conversation(starter=user, age=CONVERSATION_TTL * 2)
        self.cog.conversations[2] = self._make_conversation(starter=user, age=timedelta(minutes=5))

        await _prune_runtime_state(self.cog)

        assert 1 not in self.cog.conversations
        assert 2 in self.cog.conversations

    async def test_overflow_cap_drops_oldest(self, monkeypatch):
        from discord_gemini.cogs.gemini import state as state_mod
        from discord_gemini.cogs.gemini.state import _prune_runtime_state

        monkeypatch.setattr(state_mod, "MAX_ACTIVE_CONVERSATIONS", 2)
        self.cog._delete_conversation_cache = AsyncMock()
        self.cog._cleanup_uploaded_files = AsyncMock()
        for i in range(4):
            self.cog.conversations[i] = self._make_conversation(age=timedelta(minutes=i))

        await _prune_runtime_state(self.cog)

        assert len(self.cog.conversations) == 2
        assert {0, 1} == set(self.cog.conversations)

    async def test_cascades_orphaned_message_map_entries(self):
        from discord_gemini.cogs.gemini.state import (
            CONVERSATION_TTL,
            _prune_runtime_state,
        )

        user = MagicMock(spec=["id"])
        user.id = 99
        self.cog._delete_conversation_cache = AsyncMock()
        self.cog._cleanup_uploaded_files = AsyncMock()
        self.cog.conversations[7] = self._make_conversation(starter=user, age=CONVERSATION_TTL * 2)
        self.cog.message_to_conversation_id[500] = 7
        self.cog.message_to_conversation_id[501] = 7

        await _prune_runtime_state(self.cog)

        assert 500 not in self.cog.message_to_conversation_id
        assert 501 not in self.cog.message_to_conversation_id

    async def test_prunes_old_daily_costs(self):
        from discord_gemini.cogs.gemini.state import (
            DAILY_COST_RETENTION_DAYS,
            _prune_runtime_state,
        )

        self.cog._delete_conversation_cache = AsyncMock()
        self.cog._cleanup_uploaded_files = AsyncMock()
        old_date = (
            datetime.now(timezone.utc) - timedelta(days=DAILY_COST_RETENTION_DAYS + 2)
        ).date()
        fresh_date = datetime.now(timezone.utc).date()
        self.cog.daily_costs[(1, old_date.isoformat())] = (10.0, datetime.now(timezone.utc))
        self.cog.daily_costs[(1, fresh_date.isoformat())] = (5.0, datetime.now(timezone.utc))

        await _prune_runtime_state(self.cog)

        assert (1, old_date.isoformat()) not in self.cog.daily_costs
        assert (1, fresh_date.isoformat()) in self.cog.daily_costs


class TestConversationTouch:
    def test_touch_advances_updated_at(self):
        from discord_gemini.cogs.gemini.models import Conversation
        from discord_gemini.util import ChatCompletionParameters

        conv = Conversation(
            params=ChatCompletionParameters(model="gemini-2.0-flash"),
            history=[],
        )
        original = conv.updated_at
        conv.updated_at = original - timedelta(hours=1)
        conv.touch()
        assert conv.updated_at > original - timedelta(seconds=1)
