"""Describe the stickers in a Discord message as text.

A message that carries only a sticker has empty ``content`` and no
``attachments``; the sticker is in ``message.stickers``. Each sticker is sent to
the model as a ``[Sticker: <name>]`` line after the message text, the same way a
custom emoji reaches the model as text inside the content. The sticker image is
not sent.

The Discord bots in this fleet share this module unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class StickerLike(Protocol):
    @property
    def name(self) -> str: ...


def sticker_label(name: str) -> str:
    """Return the text that stands for one sticker."""
    return f"[Sticker: {name}]"


def text_with_stickers(text: str | None, stickers: Iterable[StickerLike] | None) -> str:
    """Return the message text followed by one label line per sticker.

    Returns ``""`` when the message has neither text nor stickers.
    """
    lines = [text] if text else []
    lines.extend(sticker_label(sticker.name) for sticker in stickers or ())
    return "\n".join(lines)
