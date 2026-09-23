"""Format the one-line cost summary shown under each response.

The Discord bots in this fleet share this module unchanged, so a request reads the
same way in every bot. Token counts have one meaning in every bot: ``in`` is every
input token including cached ones, and ``out`` is every billed output token including
thinking. A count in parentheses is part of the count before it.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

_CENT = Decimal("0.01")
_SMALLEST_REQUEST_COST = 0.0001


def format_request_cost(amount: float, *, estimated: bool = False) -> str:
    """Format one request's cost with four decimals.

    A nonzero cost below $0.0001 shows as ``<$0.0001`` rather than ``$0.0000``.
    """
    if amount <= 0:
        return "$0.0000"
    text = "<$0.0001" if amount < _SMALLEST_REQUEST_COST else f"${amount:,.4f}"
    return f"est. {text}" if estimated else text


def format_daily_total(amount: float) -> str:
    """Format a running total in whole cents; a nonzero total below a cent shows as ``<$0.01``."""
    value = Decimal(str(amount))
    if value <= 0:
        return "$0.00"
    if value < _CENT:
        return "<$0.01"
    return f"${value.quantize(_CENT, rounding=ROUND_HALF_UP):,}"


def format_tokens(count: int) -> str:
    """Format a token count: exact below 1,000, then ``1.3k``, ``12.5k``, ``184k``, ``1.2M``."""
    if count < 1_000:
        return str(count)
    if count < 999_500:
        thousands = count / 1_000
        text = f"{thousands:.1f}" if thousands < 99.95 else f"{thousands:.0f}"
        return f"{text.removesuffix('.0')}k"
    return f"{f'{count / 1_000_000:.1f}'.removesuffix('.0')}M"


def count_label(count: int, singular: str, plural: str | None = None) -> str:
    """Return a detail such as ``1 search`` or ``2 searches``."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {noun}"


def format_cost_line(
    request_cost: float | None,
    daily_cost: float | None,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_tokens: int = 0,
    thinking_tokens: int = 0,
    details: Iterable[str] = (),
    estimated: bool = False,
) -> str:
    """Build ``$cost · N in (C cached) / M out (T thinking) · details · $X today``.

    Media commands describe the item in ``details`` and pass only the token counts
    the provider reports, so a line can show ``N in``, ``M out``, both, or neither.
    Zero cached or thinking counts and empty details are left out; a ``None`` cost
    or token count is left out.
    """
    parts: list[str] = []
    if request_cost is not None:
        parts.append(format_request_cost(request_cost, estimated=estimated))
    token_parts: list[str] = []
    if input_tokens is not None:
        in_part = f"{format_tokens(input_tokens)} in"
        if cached_tokens:
            in_part += f" ({format_tokens(cached_tokens)} cached)"
        token_parts.append(in_part)
    if output_tokens is not None:
        out_part = f"{format_tokens(output_tokens)} out"
        if thinking_tokens:
            out_part += f" ({format_tokens(thinking_tokens)} thinking)"
        token_parts.append(out_part)
    if token_parts:
        parts.append(" / ".join(token_parts))
    parts.extend(detail for detail in details if detail)
    if daily_cost is not None:
        parts.append(f"{format_daily_total(daily_cost)} today")
    return " · ".join(parts)


__all__ = [
    "count_label",
    "format_cost_line",
    "format_daily_total",
    "format_request_cost",
    "format_tokens",
]
