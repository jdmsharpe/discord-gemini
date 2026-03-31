from discord import Colour

from discord_gemini.cogs.gemini.embeds import (
    append_pricing_embed,
    append_response_embeds,
    append_thinking_embeds,
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
        """Test append_response_embeds truncates very long text."""
        embeds = []
        very_long_text = "B" * 25000
        append_response_embeds(embeds, very_long_text)
        total_length = sum(len(embed.description) for embed in embeds)
        assert total_length < 21000


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

    def test_append_pricing_embed(self):
        """Test that append_pricing_embed creates a Gemini Blue embed with cost info."""
        embeds = []
        append_pricing_embed(
            embeds,
            "gemini-2.0-flash",
            input_tokens=500_000,
            output_tokens=200_000,
            daily_cost=1.25,
        )
        assert len(embeds) == 1
        embed = embeds[0]
        assert "$" in embed.description
        assert "500,000 tokens in" in embed.description
        assert "200,000 tokens out" in embed.description
        assert "daily $1.25" in embed.description

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
        assert "$0.0000" in embeds[0].description
        assert "0 tokens in" in embeds[0].description
        assert "0 tokens out" in embeds[0].description

    def test_append_pricing_embed_with_thinking_tokens(self):
        """Test pricing embed shows thinking token count."""
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
        assert "200,000 thinking" in embeds[0].description
        assert "100,000 in" in embeds[0].description
        assert "50,000 out" in embeds[0].description

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
        assert "thinking" not in embeds[0].description
        assert "tokens in" in embeds[0].description

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
        assert "Maps grounded" in embeds[0].description

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
        assert "Maps" not in embeds[0].description
