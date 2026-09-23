"""Pin the fleet-wide cost line format.

Every discord-* bot ships the same ``cost_line`` module and this same test file
(only the import line differs), so a format change must be made in all of them.
"""

import pytest

from discord_gemini.cost_line import (
    count_label,
    format_cost_line,
    format_daily_total,
    format_request_cost,
    format_tokens,
)


@pytest.mark.parametrize(
    ("amount", "estimated", "expected"),
    [
        (0.0, False, "$0.0000"),
        (-0.5, False, "$0.0000"),
        (0.00002, False, "<$0.0001"),
        (0.0001, False, "$0.0001"),
        (0.0871, False, "$0.0871"),
        (12.34567, False, "$12.3457"),
        (1234.5, False, "$1,234.5000"),
        (0.0031, True, "est. $0.0031"),
        (0.00002, True, "est. <$0.0001"),
        (0.0, True, "$0.0000"),
    ],
)
def test_format_request_cost(amount, estimated, expected):
    assert format_request_cost(amount, estimated=estimated) == expected


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (0.0, "$0.00"),
        (0.004, "<$0.01"),
        (0.0099, "<$0.01"),
        (0.01, "$0.01"),
        (0.0871, "$0.09"),
        (0.121, "$0.12"),
        (0.125, "$0.13"),
        (4.87, "$4.87"),
        (1234.5, "$1,234.50"),
    ],
)
def test_format_daily_total(amount, expected):
    assert format_daily_total(amount) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1k"),
        (1_252, "1.3k"),
        (12_457, "12.5k"),
        (99_949, "99.9k"),
        (99_950, "100k"),
        (184_392, "184k"),
        (999_499, "999k"),
        (999_500, "1M"),
        (1_200_000, "1.2M"),
    ],
)
def test_format_tokens(count, expected):
    assert format_tokens(count) == expected


def test_count_label():
    assert count_label(1, "search", "searches") == "1 search"
    assert count_label(2, "search", "searches") == "2 searches"
    assert count_label(3, "image") == "3 images"
    assert count_label(1_500, "code run") == "1,500 code runs"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"request_cost": 0.00002, "daily_cost": 0.003, "input_tokens": 24, "output_tokens": 21},
            "<$0.0001 · 24 in / 21 out · <$0.01 today",
        ),
        (
            {
                "request_cost": 0.0871,
                "daily_cost": 0.0871,
                "input_tokens": 12_465,
                "cached_tokens": 12_457,
                "output_tokens": 405,
                "details": ["2 searches"],
            },
            "$0.0871 · 12.5k in (12.5k cached) / 405 out · 2 searches · $0.09 today",
        ),
        (
            {
                "request_cost": 0.0178,
                "daily_cost": 0.12,
                "input_tokens": 550,
                "cached_tokens": 178,
                "output_tokens": 946,
                "thinking_tokens": 780,
                "details": ["1 search"],
            },
            "$0.0178 · 550 in (178 cached) / 946 out (780 thinking) · 1 search · $0.12 today",
        ),
        (
            {
                "request_cost": 0.0031,
                "daily_cost": 0.004,
                "input_tokens": 2_292,
                "cached_tokens": 512,
                "output_tokens": 441,
                "thinking_tokens": 66,
                "estimated": True,
            },
            "est. $0.0031 · 2.3k in (512 cached) / 441 out (66 thinking) · <$0.01 today",
        ),
        (
            {"request_cost": 0.04, "daily_cost": 0.12, "details": ["1 image", ""]},
            "$0.0400 · 1 image · $0.12 today",
        ),
        (
            {"request_cost": None, "daily_cost": None, "input_tokens": 10, "output_tokens": 5},
            "10 in / 5 out",
        ),
    ],
)
def test_format_cost_line(kwargs, expected):
    request_cost = kwargs.pop("request_cost")
    daily_cost = kwargs.pop("daily_cost")
    assert format_cost_line(request_cost, daily_cost, **kwargs) == expected


def test_one_sided_token_counts():
    assert format_cost_line(0.01, 0.01, input_tokens=1_290, details=["1 image"]) == (
        "$0.0100 · 1.3k in · 1 image · $0.01 today"
    )
    assert format_cost_line(1.0136, 1.01, output_tokens=57_920, details=["1 video"]) == (
        "$1.0136 · 57.9k out · 1 video · $1.01 today"
    )
