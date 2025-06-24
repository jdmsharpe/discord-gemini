import os

BOT_TOKEN = str(os.getenv("BOT_TOKEN"))
GUILD_IDS = os.getenv("GUILD_IDS", "").split(",")
GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY"))
