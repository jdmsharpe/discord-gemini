import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = str(os.getenv("BOT_TOKEN"))
GUILD_IDS = [int(id) for id in os.getenv("GUILD_IDS", "").split(",") if id]
GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY"))
