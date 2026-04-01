import importlib

import pytest

MODULE_NAME = "discord_gemini.config.auth"


def _import_fresh_auth_module(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr("dotenv.load_dotenv", lambda *_, **__: None)
    module = importlib.import_module(MODULE_NAME)
    return importlib.reload(module)


def test_validate_required_config_reports_missing_vars(monkeypatch):
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    auth = _import_fresh_auth_module(monkeypatch)

    with pytest.raises(RuntimeError, match="BOT_TOKEN, GEMINI_API_KEY"):
        auth.validate_required_config()


def test_validate_required_config_rejects_whitespace_only_values(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "   ")
    monkeypatch.setenv("GEMINI_API_KEY", "\t")

    auth = _import_fresh_auth_module(monkeypatch)

    with pytest.raises(RuntimeError, match="BOT_TOKEN, GEMINI_API_KEY"):
        auth.validate_required_config()


def test_validate_required_config_allows_present_vars(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "discord-token")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-token")

    auth = _import_fresh_auth_module(monkeypatch)

    auth.validate_required_config()


def test_invalid_guild_ids_raise_clear_error(monkeypatch):
    monkeypatch.setenv("GUILD_IDS", "123, abc, 456")

    with pytest.raises(RuntimeError, match="invalid token: 'abc'"):
        _import_fresh_auth_module(monkeypatch)


def test_guild_ids_parsing_ignores_whitespace_and_empty_tokens(monkeypatch):
    monkeypatch.setenv("GUILD_IDS", " 123 , , 456 ,   ")

    auth = _import_fresh_auth_module(monkeypatch)

    assert auth.GUILD_IDS == [123, 456]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
    ],
)
def test_show_cost_embeds_uses_standard_boolean_parser(monkeypatch, raw_value, expected):
    monkeypatch.setenv("SHOW_COST_EMBEDS", raw_value)

    auth = _import_fresh_auth_module(monkeypatch)

    assert auth.SHOW_COST_EMBEDS is expected
