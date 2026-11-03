from types import SimpleNamespace
from unittest.mock import MagicMock

from discord import StickerItem

from discord_gemini.stickers import sticker_label, text_with_stickers


def test_sticker_label_names_the_sticker():
    assert sticker_label("EEYUP") == "[Sticker: EEYUP]"


def test_sticker_only_message_becomes_its_label():
    assert text_with_stickers("", [SimpleNamespace(name="EEYUP")]) == "[Sticker: EEYUP]"


def test_text_comes_first_and_each_sticker_gets_its_own_line():
    stickers = [SimpleNamespace(name="EEYUP"), SimpleNamespace(name="wave")]

    assert text_with_stickers("look", stickers) == "look\n[Sticker: EEYUP]\n[Sticker: wave]"


def test_message_without_stickers_keeps_its_text():
    assert text_with_stickers("hi", []) == "hi"
    assert text_with_stickers("hi", None) == "hi"


def test_message_with_neither_text_nor_stickers_is_empty():
    assert text_with_stickers("", []) == ""
    assert text_with_stickers(None, None) == ""


def test_reads_the_name_of_a_py_cord_sticker_item():
    """Discord's built-in stickers are Lottie (format_type 3); every format is sent by name."""
    sticker = StickerItem(state=MagicMock(), data={"id": "1", "name": "EEYUP", "format_type": 3})

    assert text_with_stickers("", [sticker]) == "[Sticker: EEYUP]"
