from discord import Bot, Intents

from discord_gemini import GeminiCog


def test_package_import_registers_cog():
    bot = Bot(intents=Intents.default())
    bot.add_cog(GeminiCog(bot=bot))
    assert bot.get_cog("GeminiCog") is not None
