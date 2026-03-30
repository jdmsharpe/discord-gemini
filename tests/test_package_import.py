from discord import Bot, Intents

from discord_gemini import GeminiAPI


def test_package_import_regains_cog():
    intents = Intents.default()
    bot = Bot(intents=intents)
    bot.add_cog(GeminiAPI(bot=bot))
    assert bot.get_cog("GeminiAPI") is not None
