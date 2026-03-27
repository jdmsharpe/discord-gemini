import os

from dotenv import load_dotenv

load_dotenv()


def require_env(key: str) -> str:
    """Return an environment variable's value or raise if unset/empty."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable {key} is not set or is empty")
    return value


BOT_TOKEN = os.getenv("BOT_TOKEN")
GUILD_IDS = [int(id) for id in os.getenv("GUILD_IDS", "").split(",") if id]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FILE_SEARCH_STORE_IDS = [
    store_id for store_id in os.getenv("GEMINI_FILE_SEARCH_STORE_IDS", "").split(",") if store_id
]
SHOW_COST_EMBEDS = os.getenv("SHOW_COST_EMBEDS", "true").lower() in ("true", "1", "yes")
ENABLE_CUSTOM_TOOLS = os.getenv("ENABLE_CUSTOM_TOOLS", "true").lower() in ("true", "1", "yes")
