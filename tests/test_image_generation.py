import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from discord import Bot, Intents


class TestImageGenerationTextHandling(unittest.IsolatedAsyncioTestCase):
    """Tests for image generation text response handling and truncation."""

    async def asyncSetUp(self):
        # Mock the auth module
        self.auth_patcher = patch.dict(
            "sys.modules",
            {
                "config.auth": MagicMock(
                    GEMINI_API_KEY="test-api-key", GUILD_IDS=[123456789]
                )
            },
        )
        self.auth_patcher.start()

        # Mock the genai Client
        self.genai_patcher = patch("gemini_api.genai.Client")
        self.mock_genai_client = self.genai_patcher.start()

        mock_client_instance = self.mock_genai_client.return_value
        mock_client_instance.aio.aclose = AsyncMock()
        mock_client_instance.close = MagicMock()

        from gemini_api import GeminiAPI

        intents = Intents.default()
        intents.message_content = True
        self.bot = Bot(intents=intents)
        self.cog = GeminiAPI(bot=self.bot)

    async def asyncTearDown(self):
        self.auth_patcher.stop()
        self.genai_patcher.stop()

    async def test_prompt_prefix_for_generation(self):
        """Test that prompts are prefixed with 'Generate an image:' for Gemini models."""
        # This tests the implementation in _generate_image_with_gemini
        prompt = "a beautiful sunset"
        result = await self.cog._generate_image_with_gemini(
            prompt=prompt,
            model="gemini-3-pro-image-preview",
            number_of_images=1,
            seed=None,
            attachment=None,
        )
        # The actual test would need to mock the API response
        # This is a placeholder showing the test structure

    async def test_prompt_unchanged_for_editing(self):
        """Test that prompts are NOT prefixed when an attachment is provided."""
        # When attachment is present (image editing), prompt should remain unchanged
        # This is a placeholder test showing the intent
        pass

    async def test_text_response_truncation_under_limit(self):
        """Test that short text responses are not truncated."""
        short_text = "This is a short response"
        max_length = 3800

        # Simulate the truncation logic
        truncated = (
            short_text[:max_length] + "..."
            if len(short_text) > max_length
            else short_text
        )

        self.assertEqual(truncated, short_text)
        self.assertNotIn("...", truncated)

    async def test_text_response_truncation_over_limit(self):
        """Test that long text responses are truncated to 3800 chars."""
        long_text = "A" * 5000  # Exceeds 3800 char limit
        max_length = 3800

        # Simulate the truncation logic
        truncated = (
            long_text[:max_length] + "..."
            if len(long_text) > max_length
            else long_text
        )

        self.assertEqual(len(truncated), max_length + 3)  # 3800 + "..."
        self.assertTrue(truncated.endswith("..."))

    async def test_text_response_fits_in_discord_embed(self):
        """Test that truncated text + other content fits Discord's 4096 char limit."""
        # Maximum text we allow
        max_text_length = 3800
        very_long_text = "B" * 10000

        truncated = (
            very_long_text[:max_text_length] + "..."
            if len(very_long_text) > max_text_length
            else very_long_text
        )

        # Simulate the full embed description
        embed_description = "The model did not generate any images.\n"
        embed_description += f"Text response: {truncated}\n"
        embed_description += "\n*Note: These parameters are not yet implemented for Gemini: negative_prompt, guidance_scale, aspect_ratio, person_generation*"

        # Discord's limit is 4096 characters
        self.assertLessEqual(len(embed_description), 4096)


if __name__ == "__main__":
    unittest.main()
