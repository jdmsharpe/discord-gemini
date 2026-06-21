"""Guard against breaching Discord's 25-static-choices-per-option cap.

Discord rejects any slash-command option carrying more than 25 static
``choices`` with API error 50035. py-cord syncs every command in a single
all-or-nothing bulk PUT on ``on_connect``, so a SINGLE over-limit list
silently aborts slash-command registration for EVERY cog in the bot -- the
most catastrophic silent failure in this repo.

This test discovers every module-level ``*_CHOICES`` list in
``command_options`` and asserts each stays within the cap. It is intentionally
generic: any future choices list following the ``*_CHOICES`` naming convention
is protected automatically with no edits here.
"""

import pytest

from discord_gemini.cogs.gemini import command_options

#: Discord's hard limit on static ``choices`` per slash-command option.
DISCORD_MAX_STATIC_CHOICES = 25


def _discover_choice_lists() -> list[tuple[str, list]]:
    """Return every resolved module-level ``*_CHOICES`` list as (name, list).

    Counting the already-resolved module-level objects (rather than parsing the
    AST) means any list built dynamically -- e.g. via a comprehension at import
    time -- is measured at its true runtime length.
    """
    return [
        (name, value)
        for name, value in vars(command_options).items()
        if name.endswith("_CHOICES") and isinstance(value, list)
    ]


CHOICE_LISTS = _discover_choice_lists()


def test_choice_lists_discovered():
    """Sanity check: discovery actually found the choices lists."""
    names = {name for name, _ in CHOICE_LISTS}
    assert names, "no *_CHOICES lists discovered in command_options"
    # Known menus that must always exist; protects against an import/rename
    # regression silently emptying the discovered set.
    assert "CHAT_MODEL_CHOICES" in names
    assert "TTS_VOICE_CHOICES" in names


@pytest.mark.parametrize("name, choices", CHOICE_LISTS, ids=[n for n, _ in CHOICE_LISTS])
def test_choice_list_within_discord_cap(name: str, choices: list):
    """Each slash-command choices list must stay within Discord's 25 cap."""
    count = len(choices)
    assert count <= DISCORD_MAX_STATIC_CHOICES, (
        f"{name} has {count} choices, exceeding Discord's hard cap of "
        f"{DISCORD_MAX_STATIC_CHOICES} static choices per option. Discord "
        f"rejects the command with API error 50035, and because py-cord syncs "
        f"all commands in one all-or-nothing bulk PUT, this silently aborts "
        f"slash-command registration for EVERY cog. Trim {name} back to "
        f"{DISCORD_MAX_STATIC_CHOICES} or fewer."
    )


def test_tts_voice_choices_at_the_line():
    """TTS_VOICE_CHOICES sits at exactly the 25-choice limit.

    This is a deliberate at-the-line guard: the voice menu is full, so adding
    even one more voice breaches the cap and kills registration for the whole
    bot. Keep this list at 25 or fewer; do not raise the assertion to make room.
    """
    count = len(command_options.TTS_VOICE_CHOICES)
    assert count <= DISCORD_MAX_STATIC_CHOICES, (
        f"TTS_VOICE_CHOICES has {count} choices, over Discord's 25 cap "
        f"(API error 50035 -> all slash-command registration aborts)."
    )
    assert count == DISCORD_MAX_STATIC_CHOICES, (
        f"TTS_VOICE_CHOICES is expected to sit at exactly the 25-choice limit "
        f"but has {count}. If this dropped intentionally, update this guard; "
        f"if it is meant to be full, a voice was lost."
    )
