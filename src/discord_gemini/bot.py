"""Thin bot launcher inside the namespaced package."""

from discord import Bot, Intents

from .cogs.gemini.cog import GeminiCog
from .config.auth import BOT_TOKEN, validate_required_config
from .logging_setup import configure_logging


def build_bot() -> Bot:
    intents = Intents.default()
    intents.presences = False
    intents.members = True
    intents.message_content = True
    intents.guilds = True
    bot = Bot(intents=intents)
    bot.add_cog(GeminiCog(bot=bot))
    return bot


def main() -> None:
    validate_required_config()
    configure_logging()
    bot = build_bot()
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
