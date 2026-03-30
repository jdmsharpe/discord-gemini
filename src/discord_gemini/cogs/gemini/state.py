"""Conversation and UI state helpers for the Gemini cog."""

from datetime import date
from typing import TYPE_CHECKING

import discord
from discord import Member, User

if TYPE_CHECKING:
    from .cog import GeminiCog
    from .models import Conversation


def get_conversation(cog: "GeminiCog", conversation_id: int) -> "Conversation | None":
    """Return the tracked conversation for a Discord interaction ID."""

    return cog.conversations.get(conversation_id)


def _track_daily_cost(cog: "GeminiCog", user_id: int, cost: float) -> float:
    """Add a pre-calculated cost to the user's daily total."""

    key = (user_id, date.today().isoformat())
    cog.daily_costs[key] = cog.daily_costs.get(key, 0.0) + cost
    return cog.daily_costs[key]


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


__all__ = [
    "_cleanup_conversation",
    "_strip_previous_view",
    "_track_daily_cost",
    "end_conversation",
    "get_conversation",
]
