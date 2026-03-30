"""Thin bot launcher inside the namespaced package."""

import logging

from discord import Bot, Intents

from . import GeminiCog
from .config.auth import BOT_TOKEN


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
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    bot = build_bot()
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
