"""Guard tests: no request may leave the SDK's automatic function calling enabled.

Custom tools are executed by ``chat._run_agentic_loop``, never by the SDK. Leaving
AFC nominally on also routes ``generate_content`` through the SDK's AFC wrapper,
which logs a one-shot "direct use of AFC" warning per process.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from discord_gemini.cogs.gemini.chat import handle_new_message_in_conversation
from discord_gemini.cogs.gemini.models import Conversation
from tests.support import AsyncGeminiCogTestCase


def assert_afc_disabled(config) -> None:
    """Assert a GenerateContentConfig opts out of SDK-side function execution."""

    assert config is not None
    assert config.automatic_function_calling is not None
    assert config.automatic_function_calling.disable is True


class TestAutomaticFunctionCallingDisabled(AsyncGeminiCogTestCase):
    """Every generate_content flow must opt out of the SDK's AFC wrapper."""

    async def test_image_generation_disables_afc(self):
        from discord_gemini.util import ImageGenerationParameters

        params = ImageGenerationParameters(prompt="A cat", model="gemini-3.1-flash-image")
        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_image_with_gemini(params, attachment=None)

        assert_afc_disabled(self.cog.client.aio.models.generate_content.call_args.kwargs["config"])

    async def test_music_generation_disables_afc(self):
        from discord_gemini.util import MusicGenerationParameters

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="Lyrics line", inline_data=None)]
                    )
                )
            ]
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)
        params = MusicGenerationParameters(prompts=["Dream pop song"], model="lyria-3-pro-preview")

        await self.cog._generate_music_with_lyria3(params)

        assert_afc_disabled(self.cog.client.aio.models.generate_content.call_args.kwargs["config"])

    async def test_speech_generation_disables_afc(self):
        from discord_gemini.util import SpeechGenerationParameters

        mock_response = MagicMock()
        mock_response.candidates = []
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await self.cog._generate_speech_with_gemini(SpeechGenerationParameters(input_text="hello"))

        assert_afc_disabled(self.cog.client.aio.models.generate_content.call_args.kwargs["config"])

    async def test_chat_command_disables_afc(self):
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.channel = MagicMock()
        ctx.channel.id = 222
        ctx.interaction = MagicMock()
        ctx.interaction.id = 333
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock(return_value=SimpleNamespace(id=444))
        result = SimpleNamespace(
            response=SimpleNamespace(
                text="hi",
                function_calls=[],
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
            ),
            tool_calls_made=[],
            total_input_tokens=10,
            total_output_tokens=20,
            total_thinking_tokens=0,
        )

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ) as run_agentic_loop,
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="hello",
                model="gemini-2.5-flash",
            )

        assert_afc_disabled(run_agentic_loop.await_args.args[3])

    async def test_conversation_continuation_disables_afc(self):
        from discord_gemini.util import ChatCompletionParameters

        author = MagicMock()
        message = MagicMock()
        message.author = author
        message.attachments = []
        message.content = "follow-up"
        message.channel = MagicMock()
        message.reply = AsyncMock(return_value=SimpleNamespace(id=555))

        conversation = Conversation(
            params=ChatCompletionParameters(
                model="gemini-2.5-flash",
                conversation_id=100,
                conversation_starter=author,
            ),
            history=[],
        )
        result = SimpleNamespace(
            response=SimpleNamespace(
                text="hi",
                function_calls=[],
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
            ),
            tool_calls_made=[],
            total_input_tokens=10,
            total_output_tokens=20,
            total_thinking_tokens=0,
        )

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ) as run_agentic_loop,
        ):
            await handle_new_message_in_conversation(self.cog, message, conversation)

        assert_afc_disabled(run_agentic_loop.await_args.args[3])
