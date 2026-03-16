import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = str(os.getenv("BOT_TOKEN"))
GUILD_IDS = [int(id) for id in os.getenv("GUILD_IDS", "").split(",") if id]
GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY"))
GEMINI_FILE_SEARCH_STORE_IDS = [
    store_id
    for store_id in os.getenv("GEMINI_FILE_SEARCH_STORE_IDS", "").split(",")
    if store_id
]
SHOW_COST_EMBEDS = os.getenv("SHOW_COST_EMBEDS", "true").lower() in ("true", "1", "yes")
