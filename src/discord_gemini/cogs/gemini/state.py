"""Conversation and UI state helpers for the Gemini cog."""

import contextlib
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import discord
from discord import Member, User

if TYPE_CHECKING:
    from .cog import GeminiCog
    from .models import Conversation

MAX_ACTIVE_CONVERSATIONS = 100
CONVERSATION_TTL = timedelta(hours=12)
DAILY_COST_RETENTION_DAYS = 30


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _extract_daily_total(value: float | tuple[float, datetime]) -> float:
    return value[0] if isinstance(value, tuple) else value


def get_conversation(cog: "GeminiCog", conversation_id: int) -> "Conversation | None":
    """Return the tracked conversation for a Discord interaction ID."""

    return cog.conversations.get(conversation_id)


def _track_daily_cost(cog: "GeminiCog", user_id: int, cost: float) -> float:
    """Add a pre-calculated cost to the user's daily total."""

    _prune_daily_costs(cog)
    key = (user_id, date.today().isoformat())
    current_total = _extract_daily_total(cog.daily_costs.get(key, 0.0))
    new_total = current_total + cost
    cog.daily_costs[key] = (new_total, _now_utc())
    return new_total


def _prune_daily_costs(cog: "GeminiCog") -> None:
    cutoff = date.today() - timedelta(days=DAILY_COST_RETENTION_DAYS)
    expired_keys = [key for key in cog.daily_costs if date.fromisoformat(key[1]) < cutoff]
    for key in expired_keys:
        cog.daily_costs.pop(key, None)


async def _strip_previous_view(cog: "GeminiCog", user: Member | User) -> None:
    """Remove the button view from the previous turn's message for a user."""

    prev_view_msg = cog.last_view_messages.pop(user, None)
    if prev_view_msg is None:
        return

    try:
        await prev_view_msg.edit(view=None)
    except discord.NotFound:
        return
    except discord.HTTPException as exc:
        cog.logger.debug(
            "Failed to strip view from message %s: %s",
            prev_view_msg.id,
            exc,
        )


async def _cleanup_conversation(cog: "GeminiCog", user: Member | User) -> None:
    """Remove the last active view and clear view bookkeeping for a user."""

    await _strip_previous_view(cog, user)
    cog.views.pop(user, None)


async def end_conversation(cog: "GeminiCog", conversation_id: int, user: Member | User) -> None:
    """Fully tear down a conversation: delete cache, files, and local state."""

    conversation = cog.conversations.get(conversation_id)
    if conversation is not None:
        await cog._delete_conversation_cache(conversation.params)
        await cog._cleanup_uploaded_files(conversation.params)
        del cog.conversations[conversation_id]

    stale_keys = [
        msg_id
        for msg_id, conv_id in cog.message_to_conversation_id.items()
        if conv_id == conversation_id
    ]
    for key in stale_keys:
        del cog.message_to_conversation_id[key]

    await _cleanup_conversation(cog, user)
    await _prune_runtime_state(cog)


async def _prune_runtime_state(cog: "GeminiCog") -> None:
    """Evict stale conversations and cascade-clean views, msg map, and daily costs."""

    now = _now_utc()

    stale_conversation_ids = [
        cid
        for cid, conversation in cog.conversations.items()
        if now - conversation.updated_at > CONVERSATION_TTL
    ]

    active_conversations = [
        (cid, conversation)
        for cid, conversation in cog.conversations.items()
        if cid not in stale_conversation_ids
    ]
    overflow = len(active_conversations) - MAX_ACTIVE_CONVERSATIONS
    if overflow > 0:
        active_conversations.sort(key=lambda item: item[1].updated_at)
        stale_conversation_ids.extend(cid for cid, _ in active_conversations[:overflow])

    for cid in dict.fromkeys(stale_conversation_ids):
        conversation = cog.conversations.pop(cid, None)
        if conversation is None:
            continue
        with contextlib.suppress(Exception):
            await cog._delete_conversation_cache(conversation.params)
        with contextlib.suppress(Exception):
            await cog._cleanup_uploaded_files(conversation.params)
        starter = conversation.params.conversation_starter
        if starter is not None:
            await _cleanup_conversation(cog, starter)

    stale_msg_keys = [
        msg_id
        for msg_id, conv_id in cog.message_to_conversation_id.items()
        if conv_id not in cog.conversations
    ]
    for key in stale_msg_keys:
        cog.message_to_conversation_id.pop(key, None)

    _prune_daily_costs(cog)


__all__ = [
    "CONVERSATION_TTL",
    "DAILY_COST_RETENTION_DAYS",
    "MAX_ACTIVE_CONVERSATIONS",
    "_cleanup_conversation",
    "_prune_daily_costs",
    "_prune_runtime_state",
    "_strip_previous_view",
    "_track_daily_cost",
    "end_conversation",
    "get_conversation",
]
