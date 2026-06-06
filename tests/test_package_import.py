from discord import Bot, Intents

from discord_gemini import GeminiCog


async def test_package_import_registers_cog():
    # async so pytest-asyncio (auto mode) provides a running event loop —
    # py-cord's Bot() calls asyncio.get_event_loop(), which raises when no loop
    # is current (e.g. after pytest-asyncio tears one down on Python 3.11-3.13).
    bot = Bot(intents=Intents.default())
    bot.add_cog(GeminiCog(bot=bot))
    assert bot.get_cog("GeminiCog") is not None
