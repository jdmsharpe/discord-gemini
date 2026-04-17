"""Tests for the structured logging module."""

import json
import logging
from io import StringIO

import pytest

from discord_gemini.logging_setup import (
    REQUEST_ID,
    _JsonFormatter,
    _RequestIdFilter,
    bind_request_id,
    new_request_id,
)


def test_new_request_id_returns_8_hex_chars():
    rid = new_request_id()
    assert len(rid) == 8
    int(rid, 16)  # no exception → valid hex


def test_bind_request_id_returns_bound_value():
    rid = bind_request_id("my-custom-id")
    assert rid == "my-custom-id"
    assert REQUEST_ID.get() == "my-custom-id"


def test_bind_request_id_generates_when_not_provided():
    # Reset so we don't carry state between tests.
    REQUEST_ID.set(None)
    rid = bind_request_id()
    assert rid is not None and len(rid) == 8
    assert REQUEST_ID.get() == rid


def test_request_id_filter_injects_bound_id():
    record = logging.makeLogRecord({})
    REQUEST_ID.set("abc12345")
    _RequestIdFilter().filter(record)
    assert record.request_id == "abc12345"


def test_request_id_filter_uses_dash_when_unbound():
    record = logging.makeLogRecord({})
    REQUEST_ID.set(None)
    _RequestIdFilter().filter(record)
    assert record.request_id == "-"


def test_json_formatter_emits_expected_fields():
    REQUEST_ID.set("deadbeef")
    record = logging.makeLogRecord(
        {
            "name": "discord_gemini.test",
            "levelname": "INFO",
            "msg": "hello %s",
            "args": ("world",),
        }
    )
    _RequestIdFilter().filter(record)
    output = _JsonFormatter().format(record)
    payload = json.loads(output)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "discord_gemini.test"
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "deadbeef"


def test_text_format_includes_request_id_bracket(monkeypatch, caplog):
    """Integration: configure_logging in text mode prefixes with [rid=...]."""
    import discord_gemini.logging_setup as logging_setup

    monkeypatch.delenv("LOG_FORMAT", raising=False)

    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter(logging_setup._TEXT_FORMAT))
    handler.addFilter(logging_setup._RequestIdFilter())

    logger = logging.getLogger("discord_gemini.test.text")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    REQUEST_ID.set("cafef00d")
    logger.info("hello world")

    output = buffer.getvalue()
    assert "[rid=cafef00d]" in output
    assert "hello world" in output


@pytest.mark.asyncio
async def test_request_id_is_isolated_between_tasks():
    """Each asyncio task gets its own request-id copy of the ContextVar."""
    import asyncio

    bind_request_id("outer")

    async def inner_task() -> str | None:
        bind_request_id("inner")
        return REQUEST_ID.get()

    inner_rid = await asyncio.create_task(inner_task())
    assert inner_rid == "inner"
    # Outer task's binding survives — asyncio.create_task runs in a copy so
    # inner's set() doesn't leak up to the outer scope.
    assert REQUEST_ID.get() == "outer"
