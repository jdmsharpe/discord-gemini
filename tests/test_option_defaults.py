"""Guard against a slash-command option advertising a default it no longer uses.

An option description ending in ``(default: X)`` is the ONLY place a Discord
user learns what happens when they leave the option blank. When a "promote the
new default" change updates the option's ``default=`` but misses the sentence
that describes it, nothing breaks and no test fails -- the bot simply shows
every user WRONG INFORMATION in the Discord UI, indefinitely. Four such
mismatches were found across this bot fleet at once, each one traceable to a
default that moved while its description stayed behind.

This module discovers every ``SlashCommandGroup`` on ``GeminiCog``, walks its
subcommands and their options, and asserts each stated default actually names
the choice the code falls back to. It is intentionally generic: any future
option is protected automatically with no edits here.

Scope is deliberately narrow. An option is asserted over only when it carries
``choices`` AND a non-``None`` ``default`` AND a ``(default: X)`` clause AND
that default resolves to one of its own choices. Options with no choices or no
declared default usually have their real default applied downstream in the
command body, where introspection cannot see it, so asserting over them
produces a flood of false alarms. Do not widen this rule: a guard that cries
wolf gets muted, which is worse than no guard at all.

The matching rule is under test too, because a rule that is too generous is
indistinguishable from no guard at all. It has been holed twice, in mirror-image
ways, and both holes must stay closed:

* accepting a claim the display name merely *started with* waved through a
  "Foo 1" -> "Foo 1.5" promotion whose description still said "Foo 1";
* plain substring containment waved through the mirror case -- "Claude Opus 5"
  is contained in a claim of "Claude Opus 5.1", so promoting 5 to 5.1 with a
  stale description passed as well.

Every match is therefore anchored with :data:`NOT_EXTENDED`, a lookahead that
rejects a continuation into a longer identifier while still allowing ordinary
sentence punctuation, because real descriptions write things like
``(default: Claude Opus 5. warning: Opus is expensive!)``.
``test_describes_accepts_only_honest_claims`` pins the rule to a fixed table so
it cannot quietly loosen again, and it runs whether or not discovery finds
anything.
"""

import re

import discord
import pytest

from discord_gemini.cogs.gemini.cog import GeminiCog

#: Matches the ``(default: X)`` clause an option description uses to tell users
#: what they get when they leave the option blank.
DEFAULT_CLAUSE_RE = re.compile(r"\(default:\s*([^)]+)\)", re.I)

#: Anchors every match so a claim may not *extend* the name it matched. A word
#: character or hyphen continues an identifier ("Foo 1" inside "Foo 1-mini"),
#: and a dot followed by a digit continues a version ("Claude Opus 5" inside
#: "Claude Opus 5.1") -- both are drift, not a match. A dot followed by anything
#: else is sentence punctuation and stays acceptable, which is why a real
#: description like "Claude Opus 5. warning: Opus is expensive!" still passes.
NOT_EXTENDED = r"(?![\w-])(?!\.\d)(?!\s+\w)"

#: Exact discovered population, recorded from the cog as it stands. This is an
#: equality check, not a floor: a ">= N" floor in a repo whose real count IS N
#: degrades to a mere non-emptiness check, and a partial discovery collapse --
#: py-cord moving where options hang off subcommands, a group renamed, an import
#: regression -- would slip through as a shrunken-but-non-empty set.
#:
#: NEXT CONTRIBUTOR: when you add or remove a choice-backed option that states a
#: default, UPDATE these numbers deliberately in the same change. A mismatch
#: means either a real change to the command surface or a discovery regression,
#: and both deserve a human look -- do not relax the assertion to make it pass.
EXPECTED_ASSERTABLE_OPTIONS = 10
EXPECTED_UNASSERTABLE_OPTIONS = 0


def _discover_documented_defaults() -> tuple[list[tuple[str, str, str, object]], list[str]]:
    """Return (assertable, unassertable) options that document a default.

    Assertable entries are ``(option_id, claimed, choice_name, default_value)``.
    Unassertable entries are option ids whose declared default matches none of
    their own choices: there is no true display name to compare the claim
    against, so they are counted and reported separately rather than quietly
    dropped.

    Walks the cog's command groups rather than naming commands one by one, so
    options added later are covered without touching this file.
    """
    assertable: list[tuple[str, str, str, object]] = []
    unassertable: list[str] = []
    for value in vars(GeminiCog).values():
        if not isinstance(value, discord.SlashCommandGroup):
            continue
        for subcommand in value.subcommands:
            for opt in getattr(subcommand, "options", []):
                choices = getattr(opt, "choices", None) or []
                default = getattr(opt, "default", None)
                # Out of scope by design: without choices, or without a
                # declared default, the effective default usually lives in the
                # command body where this test cannot see it.
                if not choices or default is None:
                    continue
                match = DEFAULT_CLAUSE_RE.search(getattr(opt, "description", "") or "")
                if not match:
                    continue
                option_id = f"{value.name} {subcommand.name} --{opt.name}"
                resolved = next((c for c in choices if c.value == default), None)
                if resolved is None:
                    unassertable.append(f"{option_id} (default: {default!r})")
                    continue
                assertable.append((option_id, match.group(1).strip(), resolved.name, default))
    return assertable, unassertable


ASSERTABLE_DEFAULTS, UNASSERTABLE_DEFAULTS = _discover_documented_defaults()


def _describes(display_name: str, raw_value: object, claimed: str) -> bool:
    """Whether ``claimed`` is an honest description of the resolved default choice.

    A description is prose and a choice name is a menu label, so four spellings
    are accepted and nothing else -- each one anchored by :data:`NOT_EXTENDED`
    so the claim may not continue into a longer identifier than the thing it
    matched:

    * the display name appears somewhere in the claim;
    * the claim is exactly the display name's stem -- the label with any
      trailing parenthetical (``(Preview)``, ``(Firm)``) removed;
    * the claim opens with that stem, because descriptions sometimes append
      prose after the name;
    * the raw option value is non-empty and appears in the claim, because
      descriptions say ``1:1`` where the choice is named ``Square (1:1)``.

    Notably absent: accepting a claim merely because the display name starts
    with it. That waved through every prefix-superset promotion ("Foo 1" ->
    "Foo 1.5" with a stale "Foo 1" claim). The empty-name/empty-claim guard
    matters for the same reason -- an empty needle matches everything, which
    would accept any text at all for a choice whose value is blank.
    """
    name = (display_name or "").strip().lower()
    value = str(raw_value or "").strip().lower()
    claim = (claimed or "").strip().lower()
    if not name or not claim:
        return False
    stem = re.sub(r"\s*\(.*", "", name).strip()
    if re.search(re.escape(name) + NOT_EXTENDED, claim):
        return True
    if stem and claim == stem:
        return True
    if stem and re.match(re.escape(stem) + NOT_EXTENDED, claim):
        return True
    # Same shape as the branches above, inlined only to satisfy ruff's SIM103;
    # `value and ...` keeps the empty-value guard, so the semantics are identical.
    return bool(value and re.search(re.escape(value) + NOT_EXTENDED, claim))


#: Fixed cases for the matcher itself, independent of what discovery finds:
#: (true display name, raw value, claimed text, expected acceptance, why).
#: The two drift rows marked as holes are mirror images of each other and are
#: the reason this rule is anchored rather than a plain substring test; the
#: sentence-punctuation row is why the anchor stops at ``.``-then-digit instead
#: of rejecting every ``.``.
MATCHER_CASES: list[tuple[str, str, str, bool, str]] = [
    (
        "Gemini 3.7 Flash",
        "gemini-3.7-flash",
        "Gemini 3.7 Flash Pro",
        False,
        "space-extended superset drift: the claim names a longer, different model",
    ),
    ("GPT Image 2", "gpt-image-2", "GPT Image 1.5", False, "real drift"),
    ("GPT Image 1.5", "gpt-image-1.5", "GPT Image 1", False, "prefix-superset drift (v3 hole)"),
    ("Claude Opus 5", "claude-opus-5", "Claude Opus 5.1", False, "SUPERSET drift (v4 hole)"),
    (
        "Claude Opus 5",
        "claude-opus-5",
        "Claude Opus 5. warning: Opus is expensive!",
        True,
        "sentence punctuation after name",
    ),
    (
        "Grok Imagine Video 1.5 (Preview)",
        "grok-imagine-video-1.5-preview",
        "Grok Imagine Video 1.5",
        True,
        "trailing parenthetical trimmed",
    ),
    (
        "Deep Research (Apr 2026)",
        "deep-research-preview-04-2026",
        "Deep Research; Max for best reports",
        True,
        "prose after the stem",
    ),
    ("Square (1:1)", "1:1", "1:1", True, "description uses the raw value"),
    ("Kore (Firm)", "Kore", "Kore", True, "value spelling"),
    ("Gemini 3.7 Flash", "gemini-3.7-flash", "Gemini 3.6 Flash", False, "real drift"),
    ("Anything", "", "total nonsense", False, "empty value must not vacuously accept"),
    (
        "Gemini 3.1 Flash Preview TTS",
        "gemini-3.1-flash-tts-preview",
        "Gemini 2.5 Flash Preview TTS",
        False,
        "real drift",
    ),
]

MATCHER_CASE_IDS = [
    "space-extended-superset-drift",
    "real-drift-newer-name",
    "prefix-superset-drift",
    "superset-drift-dot-version",
    "sentence-punctuation-after-name",
    "trailing-parenthetical-trimmed",
    "prose-after-stem",
    "raw-value-in-description",
    "value-spelling",
    "real-drift-older-name",
    "empty-value-must-not-accept",
    "real-drift-older-tts-name",
]


@pytest.mark.parametrize(
    "display_name, raw_value, claimed, expected, why",
    MATCHER_CASES,
    ids=MATCHER_CASE_IDS,
)
def test_describes_accepts_only_honest_claims(
    display_name: str, raw_value: str, claimed: str, expected: bool, why: str
):
    """The matching rule must accept legitimate prose and reject real drift.

    This runs regardless of what discovery finds, so the rule protecting every
    option is itself always under test and can never go vacuous.
    """
    assert _describes(display_name, raw_value, claimed) is expected, (
        f"_describes({display_name!r}, {raw_value!r}, {claimed!r}) should be "
        f"{expected} -- {why}. Loosening this rule silently disarms the "
        f"per-option guard below for every option in the cog."
    )


def test_discovery_finds_the_documented_options():
    """Discovery must find exactly the recorded population, not a fragment."""
    option_ids = {option_id for option_id, _, _, _ in ASSERTABLE_DEFAULTS}
    assert len(ASSERTABLE_DEFAULTS) == EXPECTED_ASSERTABLE_OPTIONS, (
        f"discovered {len(ASSERTABLE_DEFAULTS)} assertable option(s) on "
        f"GeminiCog, but {EXPECTED_ASSERTABLE_OPTIONS} are recorded. If you "
        f"added or removed a choice-backed option that states a default, update "
        f"EXPECTED_ASSERTABLE_OPTIONS in this file as part of that change. "
        f"Otherwise the discovery walk broke (a renamed group, a py-cord change "
        f"in where options hang off subcommands, an import regression) and every "
        f"option that dropped out is an option this guard stopped protecting. "
        f"Found: {sorted(option_ids)}"
    )
    assert len(UNASSERTABLE_DEFAULTS) == EXPECTED_UNASSERTABLE_OPTIONS, (
        f"discovered {len(UNASSERTABLE_DEFAULTS)} unassertable option(s), but "
        f"{EXPECTED_UNASSERTABLE_OPTIONS} are recorded: {UNASSERTABLE_DEFAULTS}. "
        f"Update EXPECTED_UNASSERTABLE_OPTIONS deliberately if the command "
        f"surface really changed."
    )
    # Known long-standing options; protects against the discovery walk silently
    # emptying out if a group is renamed or the cog stops importing.
    assert "gemini chat --model" in option_ids
    assert "gemini-tools tts --voice_name" in option_ids


def test_no_documented_default_falls_outside_its_own_choices():
    """Options whose default matches no choice cannot be verified -- report them.

    Such an option is skipped by the per-option test through no fault of its
    own, so surface it here instead of letting it pass in silence.
    """
    assert not UNASSERTABLE_DEFAULTS, (
        f"{len(UNASSERTABLE_DEFAULTS)} option(s) declare a default that matches "
        f"none of their own choices, so the '(default: X)' clause cannot be "
        f"checked against anything: {UNASSERTABLE_DEFAULTS}. Point the default "
        f"at a real choice value, or -- if the command body deliberately maps a "
        f"sentinel to the real default -- document that here so the exemption "
        f"is a decision rather than a silent gap."
    )


@pytest.mark.parametrize(
    "option_id, claimed, choice_name, default_value",
    ASSERTABLE_DEFAULTS,
    ids=[option_id for option_id, _, _, _ in ASSERTABLE_DEFAULTS],
)
def test_documented_default_matches_actual_default(
    option_id: str, claimed: str, choice_name: str, default_value: object
):
    """Each '(default: X)' clause must name the choice the code really defaults to."""
    assert _describes(choice_name, default_value, claimed), (
        f"{option_id} tells users '(default: {claimed})' but the code actually "
        f"defaults to {default_value!r} ({choice_name!r}). Every user who leaves "
        f"this option blank sees the wrong value in the Discord UI. Update the "
        f"description to name {choice_name!r} (or its raw value "
        f"{default_value!r})."
    )
