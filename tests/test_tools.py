import pytest

from discord_gemini.cogs.gemini.tooling import (
    BuiltinGeminiToolProvider,
    McpToolProvider,
    ToolEntry,
    clear_registry,
    execute_tool_call,
    get_tool_providers,
    get_registered_tools,
    get_tool_callables,
    namespace_tool_name,
    split_namespaced_tool_name,
    tool,
)


class TestToolDecorator:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Save original registry, then clear for isolated tests
        self._original = dict(get_registered_tools())
        clear_registry()
        yield
        # Restore original registry (starter tools)
        clear_registry()
        from discord_gemini.cogs.gemini.tooling import _TOOL_REGISTRY

        _TOOL_REGISTRY.update(self._original)

    def test_registers_sync_function(self):
        @tool
        def my_sync_tool(x: int) -> str:
            """A sync tool."""
            return str(x)

        registry = get_registered_tools()
        assert "my_sync_tool" in registry
        entry = registry["my_sync_tool"]
        assert isinstance(entry, ToolEntry)
        assert entry.name == "my_sync_tool"
        assert entry.description == "A sync tool."
        assert entry.is_async is False

    def test_registers_async_function(self):
        @tool
        async def my_async_tool(name: str) -> str:
            """An async tool."""
            return f"hello {name}"

        registry = get_registered_tools()
        assert "my_async_tool" in registry
        assert registry["my_async_tool"].is_async is True

    def test_decorator_returns_original_function(self):
        @tool
        def passthrough(x: int) -> int:
            """Identity."""
            return x

        assert passthrough(42) == 42

    def test_get_tool_callables(self):
        @tool
        def tool_a() -> str:
            """A."""
            return "a"

        @tool
        def tool_b() -> str:
            """B."""
            return "b"

        callables = get_tool_callables()
        assert len(callables) == 2
        assert tool_a in callables
        assert tool_b in callables

    def test_clear_registry(self):
        @tool
        def temp_tool() -> str:
            """Temp."""
            return ""

        assert "temp_tool" in get_registered_tools()
        clear_registry()
        assert get_registered_tools() == {}


class TestExecuteToolCall:
    @pytest.fixture(autouse=True)
    def setup(self):
        self._original = dict(get_registered_tools())
        clear_registry()
        yield
        clear_registry()
        from discord_gemini.cogs.gemini.tooling import _TOOL_REGISTRY

        _TOOL_REGISTRY.update(self._original)

    async def test_execute_sync_tool(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await execute_tool_call("add", {"a": 3, "b": 4})
        assert result == {"result": 7}

    async def test_execute_async_tool(self):
        @tool
        async def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        result = await execute_tool_call("greet", {"name": "World"})
        assert result == {"result": "Hello, World!"}

    async def test_execute_unknown_tool(self):
        result = await execute_tool_call("nonexistent", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    async def test_execute_tool_with_exception(self):
        @tool
        def failing_tool() -> str:
            """Always fails."""
            raise ValueError("intentional error")

        result = await execute_tool_call("failing_tool", {})
        assert "error" in result
        assert "ValueError" in result["error"]

    async def test_execute_with_no_args(self):
        @tool
        def no_args_tool() -> str:
            """No args needed."""
            return "done"

        result = await execute_tool_call("no_args_tool", None)
        assert result == {"result": "done"}

    async def test_non_serializable_result_converted_to_string(self):
        @tool
        def returns_set() -> set:
            """Returns a set (not JSON-serializable)."""
            return {1, 2, 3}

        result = await execute_tool_call("returns_set", {})
        assert "result" in result
        assert isinstance(result["result"], str)

    async def test_execute_namespaced_tool_call_routes_to_provider(self):
        class FakeProvider:
            provider_id = "fake"

            def list_declarations(self, model: str) -> list[object]:
                return []

            async def execute(self, name: str, args: dict[str, object]) -> dict[str, object]:
                return {"provider": self.provider_id, "name": name, "args": args}

            def supports(self, model: str) -> bool:
                return True

        result = await execute_tool_call(
            namespace_tool_name("fake", "lookup"),
            {"query": "time"},
            providers=[FakeProvider()],
        )

        assert result == {"provider": "fake", "name": "lookup", "args": {"query": "time"}}

    async def test_execute_namespaced_tool_call_unknown_provider_error(self):
        result = await execute_tool_call(namespace_tool_name("missing", "lookup"), {}, providers=[])

        assert result == {"error": "Unknown tool provider: missing"}

    async def test_execute_namespaced_tool_call_prefers_provider_namespace(self):
        @tool
        def lookup() -> str:
            """Local lookup tool."""
            return "local"

        class FakeProvider:
            provider_id = "fake"

            def list_declarations(self, model: str) -> list[object]:
                return []

            async def execute(self, name: str, args: dict[str, object]) -> dict[str, object]:
                return {"result": f"provider:{name}"}

            def supports(self, model: str) -> bool:
                return True

        result = await execute_tool_call(
            namespace_tool_name("fake", "lookup"),
            {},
            providers=[FakeProvider()],
        )

        assert result == {"result": "provider:lookup"}


class TestToolProviders:
    def test_get_tool_providers_returns_expected_order(self):
        provider_ids = [provider.provider_id for provider in get_tool_providers()]

        assert provider_ids == ["local", "builtin", "mcp"]

    @pytest.mark.asyncio
    async def test_builtin_provider_execute_returns_server_side_error(self):
        provider = BuiltinGeminiToolProvider()

        result = await provider.execute("google_search", {})

        assert "server-side" in result["error"]

    def test_mcp_provider_is_disabled_stub(self):
        provider = McpToolProvider()

        assert provider.supports("gemini-2.5-pro") is False


class TestNamespacedToolHelpers:
    def test_namespace_tool_name_round_trips(self):
        namespaced = namespace_tool_name("local", "roll_dice")

        assert namespaced == "local.roll_dice"
        assert split_namespaced_tool_name(namespaced) == ("local", "roll_dice")

    def test_split_namespaced_tool_name_rejects_invalid_values(self):
        assert split_namespaced_tool_name("missing_separator") is None
        assert split_namespaced_tool_name(".tool") is None
        assert split_namespaced_tool_name("provider.") is None


class TestStarterTools:
    async def test_get_current_time_default(self):
        result = await execute_tool_call("get_current_time", {})
        assert "result" in result
        assert "UTC" in result["result"]

    async def test_get_current_time_invalid_timezone(self):
        result = await execute_tool_call("get_current_time", {"timezone": "Invalid/Zone"})
        assert "result" in result
        assert "Unknown timezone" in result["result"]

    async def test_roll_dice_default(self):
        result = await execute_tool_call("roll_dice", {})
        assert "result" in result
        assert "d6" in result["result"]

    async def test_roll_dice_multiple(self):
        result = await execute_tool_call("roll_dice", {"sides": 20, "count": 3})
        assert "result" in result
        assert "3d20" in result["result"]
        assert "total" in result["result"]
