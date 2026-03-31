from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from discord_gemini.cogs.gemini import tooling as gemini_tooling
from discord_gemini.cogs.gemini.chat import _add_custom_function_tools
from tests.support import AsyncGeminiCogTestCase


class TestGeminiAgenticLoop(AsyncGeminiCogTestCase):
    async def test_run_agentic_loop_executes_function_calls_manually(self):
        function_call = SimpleNamespace(name="lookup_time", args={"timezone": "UTC"})
        first_response = SimpleNamespace(
            text=None,
            function_calls=[function_call],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                response_token_count=4,
                thoughts_token_count=1,
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

        second_call_contents = self.cog.client.aio.models.generate_content.call_args_list[1].kwargs[
            "contents"
        ]
        assert second_call_contents[-1]["parts"]

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
