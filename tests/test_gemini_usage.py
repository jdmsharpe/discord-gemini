from types import SimpleNamespace

from discord_gemini.cogs.gemini.usage import UsageCounts, extract_usage_counts


class TestExtractUsageCounts:
    def test_generate_content_usage_current_fields(self):
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=120,
                response_token_count=45,
                thoughts_token_count=8,
                cached_content_token_count=60,
                tool_use_prompt_token_count=12,
            )
        )

        assert extract_usage_counts(response) == UsageCounts(
            input_tokens=120,
            output_tokens=45,
            thinking_tokens=8,
            cached_tokens=60,
            tool_use_prompt_tokens=12,
        )

    def test_generate_content_usage_legacy_output_field_fallback(self):
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=90,
                candidates_token_count=33,
                thoughts_token_count=5,
            )
        )

        assert extract_usage_counts(response) == UsageCounts(
            input_tokens=90,
            output_tokens=33,
            thinking_tokens=5,
        )

    def test_interactions_usage_total_field_fallbacks(self):
        interaction = SimpleNamespace(
            usage=SimpleNamespace(
                total_input_tokens=250_000,
                total_output_tokens=60_000,
                total_thought_tokens=5_000,
            )
        )

        assert extract_usage_counts(interaction) == UsageCounts(
            input_tokens=250_000,
            output_tokens=60_000,
            thinking_tokens=5_000,
        )

    def test_interactions_tool_use_tokens_are_read_from_their_own_field(self):
        """Interactions reports tool-use prompt tokens as `total_tool_use_tokens`, apart
        from `total_input_tokens`: a live url_context call reported 25 input + 57 tool-use
        + 78 output + 301 thought = 461 total tokens (2026-09-22)."""
        interaction = SimpleNamespace(
            usage=SimpleNamespace(
                total_input_tokens=25,
                total_tool_use_tokens=57,
                total_output_tokens=78,
                total_thought_tokens=301,
                total_tokens=461,
            )
        )

        assert extract_usage_counts(interaction) == UsageCounts(
            input_tokens=25,
            output_tokens=78,
            thinking_tokens=301,
            tool_use_prompt_tokens=57,
        )

    def test_interactions_cached_tokens_are_read_from_total_cached_tokens(self):
        """Interactions names the cached count `total_cached_tokens`, a subset of
        `total_input_tokens`: the same 9,804-token prompt sent twice to
        `interactions.create` on gemini-3.8-flash reported 0 cached tokens the first time
        and 4,082 the second, with 9,804 input tokens both times (2026-09-22)."""
        first = SimpleNamespace(
            usage=SimpleNamespace(total_input_tokens=9804, total_cached_tokens=0)
        )
        second = SimpleNamespace(
            usage=SimpleNamespace(total_input_tokens=9804, total_cached_tokens=4082)
        )

        assert extract_usage_counts(first) == UsageCounts(input_tokens=9804)
        assert extract_usage_counts(second) == UsageCounts(input_tokens=9804, cached_tokens=4082)

    def test_missing_usage_returns_zero_counts(self):
        assert extract_usage_counts(SimpleNamespace()) == UsageCounts()
        assert extract_usage_counts(None) == UsageCounts()
