from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from discord_gemini.cogs.gemini import tooling as gemini_tooling
from discord_gemini.cogs.gemini.chat import (
    _add_custom_function_tools,
    _configure_tool_context_circulation,
    apply_agentic_video_processing,
)
from discord_gemini.util import AGENTIC_VIDEO_MODELS
from tests.support import AsyncGeminiCogTestCase


class TestGeminiAgenticLoop(AsyncGeminiCogTestCase):
    async def test_run_agentic_loop_executes_function_calls_manually(self):
        function_call = SimpleNamespace(
            name="lookup_time",
            args={"timezone": "UTC"},
            id="call-123",
        )
        first_response = SimpleNamespace(
            text=None,
            function_calls=[function_call],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                response_token_count=4,
                thoughts_token_count=1,
                cached_content_token_count=6,
            ),
            candidates=[],
        )
        final_response = SimpleNamespace(
            text="The current time is 10:00 UTC.",
            function_calls=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=3,
                response_token_count=8,
                thoughts_token_count=0,
            ),
            candidates=[],
        )

        self.cog.client.aio.models.generate_content = AsyncMock(
            side_effect=[first_response, final_response]
        )

        with patch.object(
            gemini_tooling,
            "execute_tool_call",
            AsyncMock(return_value={"result": "10:00 UTC"}),
        ) as execute_tool_call:
            result = await self.cog._run_agentic_loop(
                "gemini-2.5-flash",
                [{"role": "user", "parts": [{"text": "What time is it?"}]}],
                None,
            )

        execute_tool_call.assert_awaited_once_with("lookup_time", {"timezone": "UTC"})
        assert result.response is final_response
        assert result.iterations == 2
        assert result.tool_calls_made == ["lookup_time"]
        assert result.total_input_tokens == 13
        assert result.total_output_tokens == 12
        assert result.total_thinking_tokens == 1
        # Cache hits are summed across iterations (the final turn reports none).
        assert result.total_cached_tokens == 6
        assert result.total_tool_use_prompt_tokens == 0
        second_call_contents = self.cog.client.aio.models.generate_content.call_args_list[1].kwargs[
            "contents"
        ]
        assert second_call_contents[-1]["parts"]
        function_response = second_call_contents[-1]["parts"][0].function_response
        assert function_response is not None
        assert function_response.id == "call-123"

    async def test_run_agentic_loop_sums_tool_use_prompt_tokens(self):
        """Agentic video navigation reports its frames as `tool_use_prompt_token_count`
        (5,847 of 6,472 total tokens on a 10-min clip, probed 2026-09-03); the loop
        must sum it so the chat flows bill it as input instead of discarding it."""
        response = SimpleNamespace(
            text="Done.",
            function_calls=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=197,
                candidates_token_count=171,
                thoughts_token_count=257,
                tool_use_prompt_token_count=5847,
            ),
            candidates=[],
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)

        result = await self.cog._run_agentic_loop(
            "gemini-3.8-flash",
            [{"role": "user", "parts": [{"text": "What happens at the end?"}]}],
            None,
        )

        assert result.total_input_tokens == 197
        assert result.total_tool_use_prompt_tokens == 5847
        assert result.total_thinking_tokens == 257

    async def test_run_agentic_loop_sums_search_queries_per_request(self):
        """Every generate_content call in the loop is billed for its own searches, so
        the loop sums each response's `web_search_queries`, not only the final one's."""

        def grounded(queries):
            return [
                SimpleNamespace(
                    content=SimpleNamespace(parts=[]),
                    grounding_metadata=SimpleNamespace(web_search_queries=queries),
                )
            ]

        first_response = SimpleNamespace(
            text=None,
            function_calls=[SimpleNamespace(name="lookup_time", args={}, id="call-1")],
            usage_metadata=SimpleNamespace(prompt_token_count=10),
            candidates=grounded(["tokyo weather", "tokyo time zone"]),
        )
        middle_response = SimpleNamespace(
            text=None,
            function_calls=[SimpleNamespace(name="lookup_time", args={}, id="call-2")],
            usage_metadata=SimpleNamespace(prompt_token_count=10),
            candidates=[],
        )
        final_response = SimpleNamespace(
            text="Sunny, 10:00 JST.",
            function_calls=[],
            usage_metadata=SimpleNamespace(prompt_token_count=10),
            candidates=grounded(["tokyo forecast"]),
        )
        self.cog.client.aio.models.generate_content = AsyncMock(
            side_effect=[first_response, middle_response, final_response]
        )

        with patch.object(
            gemini_tooling, "execute_tool_call", AsyncMock(return_value={"result": "10:00"})
        ):
            result = await self.cog._run_agentic_loop(
                "gemini-3.8-flash",
                [{"role": "user", "parts": [{"text": "Weather and time in Tokyo?"}]}],
                None,
            )

        assert result.iterations == 3
        assert result.total_search_queries == 3
        assert result.search_grounded_prompts == 2

    async def test_chat_bills_tool_use_prompt_tokens_as_input(self):
        """The cost embed's input count is prompt + tool-use prompt tokens."""
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.channel = MagicMock()
        ctx.channel.id = 222
        ctx.interaction = MagicMock()
        ctx.interaction.id = 333
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock(return_value=SimpleNamespace(id=444))
        response = SimpleNamespace(
            text="At the end a bird lands on the squirrel.",
            function_calls=[],
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
        )
        result = SimpleNamespace(
            response=response,
            tool_calls_made=[],
            total_input_tokens=197,
            total_output_tokens=171,
            total_thinking_tokens=0,
            total_cached_tokens=0,
            total_tool_use_prompt_tokens=5847,
            total_search_queries=0,
            search_grounded_prompts=0,
        )

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ),
            patch("discord_gemini.cogs.gemini.chat.calculate_cost", return_value=0.0) as cost,
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="hello",
                model="gemini-3.8-flash",
            )

        assert cost.call_args.args[1] == 197 + 5847


class TestAgenticVideoProcessing(AsyncGeminiCogTestCase):
    def test_supported_models_are_the_probed_ids(self):
        """These accepted media_processing=AGENTIC in live requests; 3.1 Pro 400s."""
        assert {
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
        } == AGENTIC_VIDEO_MODELS

    def test_the_default_chat_model_uses_agentic_video(self):
        """Argon accepted AGENTIC on a YouTube part, so the default model gets it."""
        import inspect

        from discord_gemini import GeminiCog

        default = inspect.signature(GeminiCog.chat.callback).parameters["model"].default
        assert default in AGENTIC_VIDEO_MODELS

    def test_tags_only_video_parts_on_supported_models(self):
        parts = [
            {"file_data": {"file_uri": "https://youtu.be/x", "mime_type": "video/mp4"}},
            {"inline_data": {"mime_type": "video/webm", "data": b"v"}},
            {"inline_data": {"mime_type": "image/png", "data": b"i"}},
            {"file_data": {"file_uri": "files/abc", "mime_type": "application/pdf"}},
            {"text": "describe"},
        ]

        apply_agentic_video_processing(parts, "gemini-3.8-flash")

        assert parts[0]["media_processing"] == "AGENTIC"
        assert parts[1]["media_processing"] == "AGENTIC"
        assert "media_processing" not in parts[2]
        assert "media_processing" not in parts[3]
        assert "media_processing" not in parts[4]

    def test_leaves_parts_unchanged_on_other_models(self):
        """Sending the field to a model without agentic processing is a 400."""
        parts = [{"file_data": {"file_uri": "https://youtu.be/x", "mime_type": "video/mp4"}}]

        apply_agentic_video_processing(parts, "gemini-3.1-pro-preview")
        apply_agentic_video_processing(parts, "gemini-2.5-flash")

        assert "media_processing" not in parts[0]

    async def test_chat_command_sends_agentic_video_part(self):
        """A YouTube URL on 3.8 Flash reaches the API as a Part with AGENTIC processing."""
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.channel = MagicMock()
        ctx.channel.id = 222
        ctx.interaction = MagicMock()
        ctx.interaction.id = 333
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock(return_value=SimpleNamespace(id=444))
        response = SimpleNamespace(
            text="A bunny.",
            function_calls=[],
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
        )
        result = SimpleNamespace(
            response=response,
            tool_calls_made=[],
            total_input_tokens=1,
            total_output_tokens=1,
            total_thinking_tokens=0,
            total_cached_tokens=0,
            total_tool_use_prompt_tokens=0,
            total_search_queries=0,
            search_grounded_prompts=0,
        )
        loop = AsyncMock(return_value=result)

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch("discord_gemini.cogs.gemini.chat._run_agentic_loop", loop),
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="What happens at the end?",
                model="gemini-3.8-flash",
                url="https://www.youtube.com/watch?v=aqz-KE-bpKQ",
            )

        contents = loop.await_args.args[2]
        video_part = contents[0]["parts"][0]
        assert isinstance(video_part, types.Part)
        assert video_part.file_data is not None
        assert video_part.file_data.mime_type == "video/mp4"
        assert video_part.media_processing == "AGENTIC"
        assert contents[0]["parts"][1].media_processing is None

    async def test_chat_long_response_with_sidecars_uses_embed_batches(self):
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.channel = MagicMock()
        ctx.channel.id = 222
        ctx.interaction = MagicMock()
        ctx.interaction.id = 333
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock(return_value=SimpleNamespace(id=444))
        response = SimpleNamespace(
            text="R" * 8000,
            function_calls=[],
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
        )
        result = SimpleNamespace(
            response=response,
            tool_calls_made=[],
            total_input_tokens=10,
            total_output_tokens=20,
            total_thinking_tokens=0,
            total_cached_tokens=0,
            total_tool_use_prompt_tokens=0,
            total_search_queries=0,
            search_grounded_prompts=0,
        )

        with (
            patch(
                "discord_gemini.cogs.gemini.chat.keep_typing",
                AsyncMock(),
            ),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ),
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="hello",
                model="gemini-2.5-flash",
            )

        assert ctx.send_followup.await_count > 1
        for call in ctx.send_followup.await_args_list:
            assert "embeds" in call.kwargs
            assert not str(call.kwargs.get("content", "")).startswith("**Response:**")


def _chat_ctx() -> AsyncMock:
    ctx = AsyncMock()
    ctx.author = MagicMock()
    ctx.author.id = 111
    ctx.channel = MagicMock()
    ctx.channel.id = 222
    ctx.interaction = MagicMock()
    ctx.interaction.id = 333
    ctx.defer = AsyncMock()
    ctx.send_followup = AsyncMock(return_value=SimpleNamespace(id=444))
    return ctx


def _loop_result(text: str = "ok") -> SimpleNamespace:
    response = SimpleNamespace(
        text=text,
        function_calls=[],
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
    )
    return SimpleNamespace(
        response=response,
        tool_calls_made=[],
        total_input_tokens=38,
        total_output_tokens=5,
        total_thinking_tokens=0,
        total_cached_tokens=0,
        total_tool_use_prompt_tokens=0,
        total_search_queries=0,
        search_grounded_prompts=0,
    )


class TestGemini4ArgonDefault(AsyncGeminiCogTestCase):
    async def test_default_chat_request_sends_only_what_argon_accepts(self):
        """With every option left blank the request goes to Argon with no thinking
        config (minimal would 400), no sampling parameters, no tools and AFC off."""
        ctx = _chat_ctx()
        loop = AsyncMock(return_value=_loop_result())

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch("discord_gemini.cogs.gemini.chat._run_agentic_loop", loop),
        ):
            await self.cog.chat.callback(self.cog, ctx=ctx, prompt="hello")

        model = loop.await_args.args[1]
        config = loop.await_args.args[3]
        assert model == "gemini-4-argon"
        assert config.thinking_config is None
        assert config.temperature is None
        assert config.top_p is None
        assert config.tools is None
        assert config.tool_config is None
        assert config.media_resolution is None
        assert config.max_output_tokens is None
        assert config.cached_content is None
        assert config.automatic_function_calling.disable is True

    async def test_argon_marks_temperature_and_top_p_as_ignored(self):
        ctx = _chat_ctx()
        loop = AsyncMock(return_value=_loop_result())

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch("discord_gemini.cogs.gemini.chat._run_agentic_loop", loop),
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="hello",
                model="gemini-4-argon",
                temperature=0.5,
                top_p=0.9,
            )

        description = ctx.send_followup.await_args_list[0].kwargs["embeds"][0].description
        assert "**Temperature:** 0.5 (ignored: Gemini 4 models do not use temperature)" in (
            description
        )
        assert "**Nucleus Sampling:** 0.9 (ignored: Gemini 4 models do not use top_p)" in (
            description
        )
        assert "looping" not in description

    async def test_gemini_3_keeps_the_temperature_warning(self):
        ctx = _chat_ctx()
        loop = AsyncMock(return_value=_loop_result())

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch("discord_gemini.cogs.gemini.chat._run_agentic_loop", loop),
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="hello",
                model="gemini-3.8-flash",
                temperature=0.5,
                top_p=0.9,
            )

        description = ctx.send_followup.await_args_list[0].kwargs["embeds"][0].description
        assert "(warning: Gemini 3 recommends 1.0; lower values may cause looping)" in description
        assert "**Nucleus Sampling:** 0.9\n" in description

    async def test_argon_rejects_minimal_thinking_before_the_request(self):
        ctx = _chat_ctx()
        loop = AsyncMock(return_value=_loop_result())

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch("discord_gemini.cogs.gemini.chat._run_agentic_loop", loop),
        ):
            await self.cog.chat.callback(
                self.cog, ctx=ctx, prompt="hello", thinking_level="minimal"
            )

        loop.assert_not_awaited()
        embed = ctx.send_followup.await_args_list[0].kwargs["embed"]
        assert "gemini-4-argon" in embed.description
        assert "Minimal" in embed.description

    def test_argon_combines_builtin_and_custom_tools(self):
        """The combination gate checked startswith("gemini-3") and refused Gemini 4."""
        config_args = {"tools": [{"google_search": {}}]}

        _configure_tool_context_circulation(
            config_args,
            model="gemini-4-argon",
            custom_functions_enabled=True,
        )

        tool_config = config_args["tool_config"]
        assert tool_config.include_server_side_tool_invocations is True
        assert tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.VALIDATED


class TestGeminiChatCachedTokenBilling(AsyncGeminiCogTestCase):
    async def test_chat_passes_cached_tokens_to_calculate_cost(self):
        """Cache hits summed by the agentic loop must reach the cost split, not be dropped."""
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
            total_input_tokens=5_000,
            total_output_tokens=20,
            total_thinking_tokens=0,
            total_cached_tokens=4_000,
            total_tool_use_prompt_tokens=0,
            total_search_queries=0,
            search_grounded_prompts=0,
        )

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ),
            patch(
                "discord_gemini.cogs.gemini.chat.calculate_cost", return_value=0.0
            ) as calculate_cost,
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="hello",
                model="gemini-3.8-flash",
            )

        calculate_cost.assert_called_once()
        assert calculate_cost.call_args.args[:2] == ("gemini-3.8-flash", 5_000)
        assert calculate_cost.call_args.kwargs["cached_tokens"] == 4_000


class TestGeminiChatSearchBilling(AsyncGeminiCogTestCase):
    async def test_chat_bills_search_queries_summed_by_the_loop(self):
        """The loop's search counts reach the cost, the log and the pricing footer."""
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
                text="Spain won Euro 2024.",
                function_calls=[],
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
            ),
            tool_calls_made=[],
            total_input_tokens=1_000_000,
            total_output_tokens=0,
            total_thinking_tokens=0,
            total_cached_tokens=0,
            total_tool_use_prompt_tokens=0,
            total_search_queries=2,
            search_grounded_prompts=1,
        )

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ),
            patch("discord_gemini.cogs.gemini.chat.SHOW_COST_EMBEDS", True),
            patch.object(self.cog, "_log_cost") as log_cost,
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="Who won Euro 2024?",
                model="gemini-3.8-flash",
                google_search=True,
            )

        # 1M input @ $0.75/M + 2 search queries @ $0.014
        assert log_cost.call_args.args[3] == pytest.approx(0.778)
        assert log_cost.call_args.kwargs["google_search_queries"] == 2
        footer = ctx.send_followup.call_args.kwargs["embeds"][-1].description
        assert footer == "$0.7780 · 1M in / 0 out · 2 searches · $0.78 today"

    async def test_maps_queries_are_not_billed_as_search_queries(self):
        """A Maps-only response lists its Maps query in `web_search_queries`; it is
        billed once as Maps grounding, not again as a Google Search query."""
        ctx = _chat_ctx()
        result = _loop_result("Red Bay Coffee and Philz Coffee.")
        result.total_input_tokens = 1_000_000
        result.total_output_tokens = 0
        result.total_search_queries = 1
        result.search_grounded_prompts = 1
        result.response.candidates[0].grounding_metadata = SimpleNamespace(
            grounding_chunks=[
                SimpleNamespace(
                    web=None,
                    maps=SimpleNamespace(uri="https://maps.google.com/?cid=1", title="Red Bay"),
                    retrieved_context=None,
                )
            ],
            web_search_queries=["coffee shops near Ferry Building"],
            image_search_queries=None,
            grounding_supports=None,
            search_entry_point=None,
        )

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ),
            patch("discord_gemini.cogs.gemini.chat.SHOW_COST_EMBEDS", True),
            patch.object(self.cog, "_log_cost") as log_cost,
        ):
            await self.cog.chat.callback(
                self.cog,
                ctx=ctx,
                prompt="Coffee near the Ferry Building?",
                model="gemini-4-argon",
                google_maps=True,
            )

        assert log_cost.call_args.kwargs["google_search_queries"] == 0
        assert log_cost.call_args.kwargs["google_maps_grounded"] is True
        # 1M input @ $2.00/M + one Maps grounding charge @ $0.014, no Search charge.
        assert log_cost.call_args.args[3] == pytest.approx(2.014)
        footer = ctx.send_followup.call_args.kwargs["embeds"][-1].description
        assert "search" not in footer

    @pytest.mark.parametrize(("tool", "billed_queries"), [("google_search", 2), ("google_maps", 0)])
    async def test_follow_up_bills_search_queries_only_with_the_search_tool(
        self, tool, billed_queries
    ):
        from discord_gemini.cogs.gemini.chat import handle_new_message_in_conversation
        from discord_gemini.cogs.gemini.models import Conversation
        from discord_gemini.util import ChatCompletionParameters

        author = MagicMock()
        message = MagicMock()
        message.author = author
        message.attachments = []
        message.content = "follow-up"
        message.channel = MagicMock()
        message.reply = AsyncMock(return_value=MagicMock(id=555, edit=AsyncMock()))
        conversation = Conversation(
            params=ChatCompletionParameters(
                model="gemini-4-argon",
                conversation_id=100,
                conversation_starter=author,
                tools=[{tool: {}}],
            ),
            history=[],
        )
        result = _loop_result()
        result.total_search_queries = 2
        result.search_grounded_prompts = 1

        with (
            patch("discord_gemini.cogs.gemini.chat.keep_typing", AsyncMock()),
            patch(
                "discord_gemini.cogs.gemini.chat._run_agentic_loop",
                AsyncMock(return_value=result),
            ),
            patch.object(self.cog, "_log_cost") as log_cost,
        ):
            await handle_new_message_in_conversation(self.cog, message, conversation)

        assert log_cost.call_args.kwargs["google_search_queries"] == billed_queries

    def test_search_tool_detection(self):
        from discord_gemini.cogs.gemini.chat import _search_tool_enabled

        assert _search_tool_enabled([{"google_search": {}}]) is True
        assert _search_tool_enabled([{"google_maps": {}}]) is False
        assert _search_tool_enabled([{"url_context": {}}, {"function_declarations": []}]) is False
        assert _search_tool_enabled([]) is False


class TestGeminiToolCombinationConfig:
    def test_configure_tool_context_circulation_for_builtin_and_custom_tools(self):
        config_args = {"tools": [{"google_search": {}}]}

        _configure_tool_context_circulation(
            config_args,
            model="gemini-3-flash-preview",
            custom_functions_enabled=True,
        )

        tool_config = config_args["tool_config"]
        assert tool_config.include_server_side_tool_invocations is True
        assert tool_config.function_calling_config is not None
        assert tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.VALIDATED

    def test_configure_tool_context_circulation_for_builtin_tools_only(self):
        config_args = {"tools": [{"google_search": {}}]}

        _configure_tool_context_circulation(
            config_args,
            model="gemini-3.1-pro-preview",
            custom_functions_enabled=False,
        )

        tool_config = config_args["tool_config"]
        assert tool_config.include_server_side_tool_invocations is True
        assert tool_config.function_calling_config is None

    def test_configure_tool_context_circulation_skips_unsupported_models(self):
        config_args = {"tools": [{"google_search": {}}]}

        _configure_tool_context_circulation(
            config_args,
            model="gemini-2.5-flash",
            custom_functions_enabled=True,
        )

        assert "tool_config" not in config_args

    def test_add_custom_function_tools_disables_sdk_auto_execution(self):
        def lookup_time() -> str:
            return "10:00 UTC"

        config_args = {"tools": [{"google_search": {}}]}
        with patch.object(gemini_tooling, "get_tool_callables", return_value=[lookup_time]):
            _add_custom_function_tools(config_args, custom_functions_enabled=True)

        assert config_args["tools"] == [{"google_search": {}}, lookup_time]
        assert config_args["automatic_function_calling"] is not None
        assert config_args["automatic_function_calling"].disable is True

    def test_add_custom_function_tools_noop_without_callables(self):
        config_args = {}
        with patch.object(gemini_tooling, "get_tool_callables", return_value=[]):
            _add_custom_function_tools(config_args, custom_functions_enabled=True)

        assert config_args == {}
