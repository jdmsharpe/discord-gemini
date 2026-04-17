"""Structured logging with contextvars-backed request IDs.

Each user-initiated action (slash command, message event, button callback)
binds a short random request ID via :func:`bind_request_id`. Every
``logger.info``/``warning``/``error`` call inside that async task then emits
the id alongside its message, so a multi-step flow can be traced end-to-end.

Set ``LOG_FORMAT=json`` for JSON-lines output (suitable for log aggregators)
or leave it unset for plain-text dev output.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from contextvars import ContextVar
from typing import Any

REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s [rid=%(request_id)s] %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def new_request_id() -> str:
    """Return a short random request id (8 hex chars)."""
    return secrets.token_hex(4)


def bind_request_id(request_id: str | None = None) -> str:
    """Bind a request id to the current async task and return it.

    Pass an explicit id to thread through from an external source (Discord
    interaction id, upstream trace); omit to generate a fresh one.
    """
    rid = request_id or new_request_id()
    REQUEST_ID.set(rid)
    return rid


class _RequestIdFilter(logging.Filter):
    """Attach the currently bound request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID.get() or "-"
        return True


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once, reading ``LOG_FORMAT`` for output style.

    Safe to call multiple times; a sentinel on the installed handler prevents
    duplicate registration.
    """
    formatter: logging.Formatter
    if os.getenv("LOG_FORMAT", "text").lower() == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT)

    root_logger = logging.getLogger()
    for existing in root_logger.handlers:
        if getattr(existing, "_configured_by_logging_setup", False):
            return  # already configured

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())
    handler._configured_by_logging_setup = True  # type: ignore[attr-defined]

    root_logger.setLevel(level)
    root_logger.addHandler(handler)


__all__ = [
    "REQUEST_ID",
    "bind_request_id",
    "configure_logging",
    "new_request_id",
]
