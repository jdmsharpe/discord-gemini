import os

from dotenv import load_dotenv

load_dotenv()

TRUE_ENV_VALUES = frozenset({"true", "1", "yes"})
REQUIRED_ENV_VARS = ("BOT_TOKEN", "GEMINI_API_KEY")


def _get_env_or_none(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped_value = value.strip()
    return stripped_value or None


def _parse_csv_values(raw_values: str) -> list[str]:
    return [token.strip() for token in raw_values.split(",") if token.strip()]


def _parse_guild_ids(raw_guild_ids: str) -> list[int]:
    guild_ids: list[int] = []

    for token in raw_guild_ids.split(","):
        stripped_token = token.strip()
        if not stripped_token:
            continue
        try:
            guild_ids.append(int(stripped_token))
        except ValueError as exc:
            raise RuntimeError(
                "Invalid GUILD_IDS value. Expected a comma-separated list of integers, "
                f"but received invalid token: {stripped_token!r}."
            ) from exc

    return guild_ids


def _parse_bool_env(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in TRUE_ENV_VALUES


def validate_required_config() -> None:
    missing_vars = [name for name in REQUIRED_ENV_VARS if _get_env_or_none(name) is None]
    if missing_vars:
        missing_list = ", ".join(missing_vars)
        raise RuntimeError(
            "Missing required environment configuration: "
            f"{missing_list}. Please set these variables before starting the bot."
        )


BOT_TOKEN = _get_env_or_none("BOT_TOKEN")
GUILD_IDS = _parse_guild_ids(os.getenv("GUILD_IDS", ""))
GEMINI_API_KEY = _get_env_or_none("GEMINI_API_KEY")
GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION")
GEMINI_FILE_SEARCH_STORE_IDS = _parse_csv_values(os.getenv("GEMINI_FILE_SEARCH_STORE_IDS", ""))
SHOW_COST_EMBEDS = _parse_bool_env("SHOW_COST_EMBEDS")
ENABLE_CUSTOM_TOOLS = _parse_bool_env("ENABLE_CUSTOM_TOOLS")
