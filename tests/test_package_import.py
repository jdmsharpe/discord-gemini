from discord import Bot, Intents

from discord_gemini import GeminiCog


def test_package_import_regains_cog():
    intents = Intents.default()
    bot = Bot(intents=intents)
    bot.add_cog(GeminiCog(bot=bot))
    assert bot.get_cog("GeminiCog") is not None
