from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gemini_test_support import AsyncGeminiCogTestCase, GeminiCogTestCase


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
