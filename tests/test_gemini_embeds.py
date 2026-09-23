from discord import Colour

from discord_gemini.cogs.gemini.embeds import (
    GEMINI_BLUE,
    append_pricing_embed,
    append_response_embeds,
    append_sources_embed,
    append_thinking_embeds,
    error_to_user_description,
)


class TestAppendResponseEmbeds:
    def test_append_response_embeds_short(self):
        """Test append_response_embeds with short text."""
        embeds = []
        append_response_embeds(embeds, "Hello, World!")
        assert len(embeds) == 1
        assert embeds[0].description == "Hello, World!"
        assert embeds[0].title == "Response"

    def test_append_response_embeds_long(self):
        """Test append_response_embeds with long text that needs chunking."""
        embeds = []
        long_text = "A" * 7000
        append_response_embeds(embeds, long_text)
        assert len(embeds) == 2
        assert embeds[0].title == "Response"
        assert embeds[1].title == "Response (Part 2)"

    def test_append_response_embeds_very_long(self):
        """Test append_response_embeds preserves very long text for delivery batching."""
        embeds = []
        very_long_text = "B" * 25000
        append_response_embeds(embeds, very_long_text)
        total_length = sum(len(embed.description) for embed in embeds)
        assert total_length == len(very_long_text)
        assert "".join(embed.description for embed in embeds) == very_long_text


class TestGeminiThinkingEmbeds:
    def test_append_thinking_embeds_with_text(self):
        """Test that thinking embed is created with spoilered text."""
        embeds = []
        append_thinking_embeds(embeds, "My thought process")
        assert len(embeds) == 1
        assert embeds[0].title == "Thinking"
        assert embeds[0].description == "||My thought process||"
        assert embeds[0].color == Colour.light_grey()

    def test_append_thinking_embeds_empty_text(self):
        """Test that no embed is created for empty thinking text."""
        embeds = []
        append_thinking_embeds(embeds, "")
        assert len(embeds) == 0

    def test_append_thinking_embeds_truncates_long_text(self):
        """Test that long thinking text is truncated."""
        embeds = []
        long_text = "A" * 4000
        append_thinking_embeds(embeds, long_text)
        assert len(embeds) == 1
        assert "[thinking truncated]" in embeds[0].description
        assert len(embeds[0].description) <= 3600


class TestAppendSourcesEmbed:
    def test_url_context_uses_label_and_preserves_full_target(self):
        redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/" + "a" * 300
        tool_info = {
            "tools_used": ["url_context"],
            "citations": [],
            "search_queries": ["seven exchange lists", "food exchange lists 7 categories"],
            "url_context_sources": [
                {
                    "retrieved_url": redirect,
                    "display_name": "pressbooks.pub",
                    "status": "URL_RETRIEVAL_STATUS_SUCCESS",
                }
            ],
            "maps_widget_token": None,
        }

        embeds = []
        append_sources_embed(embeds, tool_info)

        assert f"[pressbooks.pub]({redirect})" in embeds[0].description
        assert "URL Context" not in embeds[0].description
        assert "URL_RETRIEVAL_STATUS" not in embeds[0].description
        assert (
            "**Queries:** seven exchange lists, food exchange lists 7 categories"
            in embeds[0].description
        )

    def test_url_context_deduplicates_grounding_citation(self):
        url = "https://example.com/source"
        tool_info = {
            "tools_used": ["google_search", "url_context"],
            "citations": [{"title": "example.com", "uri": url}],
            "search_queries": [],
            "url_context_sources": [
                {
                    "retrieved_url": url,
                    "display_name": "example.com",
                    "status": "URL_RETRIEVAL_STATUS_SUCCESS",
                }
            ],
            "maps_widget_token": None,
        }

        embeds = []
        append_sources_embed(embeds, tool_info)

        assert embeds[0].description.count(f"]({url})") == 1


class TestPricingEmbeds:
    def test_append_pricing_embed(self):
        """Test that append_pricing_embed creates a Gemini Blue embed with the cost line."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash-lite",
            input_tokens=500_000,
            output_tokens=200_000,
            daily_cost=1.25,
        )
        assert len(embeds) == 1
        embed = embeds[0]
        assert embed.color == GEMINI_BLUE
        assert embed.description == "$0.1300 · 500k in / 200k out · $1.25 today"

    def test_append_pricing_embed_zero_tokens(self):
        """Test pricing embed with zero tokens."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-pro",
            input_tokens=0,
            output_tokens=0,
            daily_cost=0.0,
        )
        assert len(embeds) == 1
        assert embeds[0].description == "$0.0000 · 0 in / 0 out · $0.00 today"

    def test_append_pricing_embed_with_thinking_tokens(self):
        """Gemini reports thinking apart from output; the line's output count includes it."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-3-flash-preview",
            input_tokens=100_000,
            output_tokens=50_000,
            daily_cost=0.50,
            thinking_tokens=200_000,
        )
        assert len(embeds) == 1
        assert embeds[0].description == (
            "$0.8000 · 100k in / 250k out (200k thinking) · $0.50 today"
        )

    def test_append_pricing_embed_with_cached_thinking_and_search(self):
        """Cached tokens are shown as part of the input count and thinking tokens as
        part of the output count: 166 output + 780 thinking tokens show as 946 out."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-3.8-flash",
            input_tokens=550,
            output_tokens=166,
            daily_cost=0.12,
            thinking_tokens=780,
            cached_tokens=178,
            google_search_queries=1,
            google_search_grounded=1,
        )
        assert embeds[0].description == (
            "$0.0178 · 550 in (178 cached) / 946 out (780 thinking) · 1 search · $0.12 today"
        )

    def test_append_pricing_embed_zero_thinking_tokens(self):
        """Test pricing embed omits thinking when zero."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash",
            input_tokens=100_000,
            output_tokens=50_000,
            daily_cost=0.10,
            thinking_tokens=0,
        )
        assert len(embeds) == 1
        assert embeds[0].description == "$0.1550 · 100k in / 50k out · $0.10 today"

    def test_append_pricing_embed_with_maps_grounding(self):
        """Test pricing embed includes Maps grounding surcharge."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            daily_cost=0.10,
            google_maps_grounded=True,
        )
        assert len(embeds) == 1
        assert embeds[0].description == "$0.0266 · 1k in / 500 out · maps grounded · $0.10 today"

    def test_append_pricing_embed_with_cached_tokens(self):
        """Cache hits are shown in the footer and billed at the cached rate."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-3.7-flash",
            input_tokens=100_000,
            output_tokens=1_000,
            daily_cost=0.10,
            cached_tokens=80_000,
        )
        assert len(embeds) == 1
        # 20K uncached @ $0.75/M + 80K cached @ $0.075/M + 1K out @ $3.75/M
        expected = (20_000 / 1e6) * 0.75 + (80_000 / 1e6) * 0.075 + (1_000 / 1e6) * 3.75
        assert embeds[0].description == (
            f"${expected:.4f} · 100k in (80k cached) / 1k out · $0.10 today"
        )

    def test_append_pricing_embed_without_cached_tokens_omits_label(self):
        embeds = []
        append_pricing_embed(
            embeds, "gemini-3.7-flash", input_tokens=1000, output_tokens=500, daily_cost=0.10
        )
        assert "cached" not in embeds[0].description

    def test_append_pricing_embed_bills_and_labels_search_queries(self):
        """Gemini 3.x bills each search query at $0.014; the footer names the count."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-3.8-flash",
            input_tokens=1_000_000,
            output_tokens=0,
            daily_cost=0.10,
            google_search_queries=3,
            google_search_grounded=1,
        )
        # 1M input @ $0.75/M + 3 queries @ $0.014
        assert embeds[0].description == "$0.7920 · 1M in / 0 out · 3 searches · $0.10 today"

    def test_append_pricing_embed_bills_a_gemini_2_5_grounded_prompt(self):
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash",
            input_tokens=1_000_000,
            output_tokens=0,
            daily_cost=0.10,
            google_search_queries=1,
            google_search_grounded=True,
        )
        # 1M input @ $0.30/M + one grounded prompt @ $0.035
        assert embeds[0].description == "$0.3350 · 1M in / 0 out · 1 search · $0.10 today"

    def test_append_pricing_embed_without_search_omits_label(self):
        embeds = []
        append_pricing_embed(
            embeds, "gemini-3.8-flash", input_tokens=1000, output_tokens=500, daily_cost=0.10
        )
        assert "search" not in embeds[0].description

    def test_append_pricing_embed_without_maps_grounding(self):
        """Test pricing embed omits Maps label when not grounded."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            daily_cost=0.10,
            google_maps_grounded=False,
        )
        assert len(embeds) == 1
        assert "maps" not in embeds[0].description


class TestErrorToUserDescription:
    def test_error_to_user_description_uses_default_for_empty(self):
        assert error_to_user_description("") == "An unexpected error occurred."

    def test_error_to_user_description_truncates_within_max_length(self):
        output = error_to_user_description("x" * 5000, max_length=40)
        assert len(output) == 40
        assert output.endswith("truncated)")
