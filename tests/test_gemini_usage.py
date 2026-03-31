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

    def test_missing_usage_returns_zero_counts(self):
        assert extract_usage_counts(SimpleNamespace()) == UsageCounts()
        assert extract_usage_counts(None) == UsageCounts()
