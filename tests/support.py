import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discord_gemini import GeminiCog
from discord_gemini.cogs.gemini import research as gemini_research
from discord_gemini.cogs.gemini import tooling as gemini_tooling


def build_mock_bot() -> MagicMock:
    bot = MagicMock()
    bot.user = MagicMock(name="bot-user")
    bot.owner_id = 1234567890
    bot.sync_commands = AsyncMock()
    bot.add_cog = MagicMock()
    bot.loop = None
    return bot


class GeminiCogTestCase:
    file_search_store_ids = ["store-1", "store-2"]

    @pytest.fixture(autouse=True)
    def setup(self):
        with (
            patch.object(
                gemini_tooling,
                "GEMINI_FILE_SEARCH_STORE_IDS",
                self.file_search_store_ids.copy(),
            ),
            patch.object(
                gemini_research,
                "GEMINI_FILE_SEARCH_STORE_IDS",
                self.file_search_store_ids.copy(),
            ),
            patch("discord_gemini.cogs.gemini.client.build_gemini_client") as mock_genai_client,
        ):
            self.mock_genai_client = mock_genai_client
            self.mock_client_instance = mock_genai_client.return_value
            self.mock_client_instance.aio.aclose = AsyncMock()
            self.mock_client_instance.close = MagicMock()

            self.bot = build_mock_bot()
            self.cog = GeminiCog(bot=self.bot)
            yield


class AsyncGeminiCogTestCase:
    file_search_store_ids = ["store-1", "store-2"]

    @pytest.fixture(autouse=True)
    async def setup(self):
        with (
            patch.object(
                gemini_tooling,
                "GEMINI_FILE_SEARCH_STORE_IDS",
                self.file_search_store_ids.copy(),
            ),
            patch.object(
                gemini_research,
                "GEMINI_FILE_SEARCH_STORE_IDS",
                self.file_search_store_ids.copy(),
            ),
            patch("discord_gemini.cogs.gemini.client.build_gemini_client") as mock_genai_client,
        ):
            self.mock_genai_client = mock_genai_client
            self.mock_client_instance = mock_genai_client.return_value
            self.mock_client_instance.aio.aclose = AsyncMock()
            self.mock_client_instance.close = MagicMock()

            self.bot = build_mock_bot()
            self.bot.loop = asyncio.get_running_loop()
            self.cog = GeminiCog(bot=self.bot)
            yield
