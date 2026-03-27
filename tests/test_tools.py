import unittest

from tools import (
    ToolEntry,
    clear_registry,
    execute_tool_call,
    get_registered_tools,
    get_tool_callables,
    tool,
)


class TestToolDecorator(unittest.TestCase):
    def setUp(self):
        # Save original registry, then clear for isolated tests
        self._original = dict(get_registered_tools())
        clear_registry()

    def tearDown(self):
        # Restore original registry (starter tools)
        clear_registry()
        from tools import _TOOL_REGISTRY

        _TOOL_REGISTRY.update(self._original)

    def test_registers_sync_function(self):
        @tool
        def my_sync_tool(x: int) -> str:
            """A sync tool."""
            return str(x)

        registry = get_registered_tools()
        self.assertIn("my_sync_tool", registry)
        entry = registry["my_sync_tool"]
        self.assertIsInstance(entry, ToolEntry)
        self.assertEqual(entry.name, "my_sync_tool")
        self.assertEqual(entry.description, "A sync tool.")
        self.assertFalse(entry.is_async)

    def test_registers_async_function(self):
        @tool
        async def my_async_tool(name: str) -> str:
            """An async tool."""
            return f"hello {name}"

        registry = get_registered_tools()
        self.assertIn("my_async_tool", registry)
        self.assertTrue(registry["my_async_tool"].is_async)

    def test_decorator_returns_original_function(self):
        @tool
        def passthrough(x: int) -> int:
            """Identity."""
            return x

        self.assertEqual(passthrough(42), 42)

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
        self.assertEqual(len(callables), 2)
        self.assertIn(tool_a, callables)
        self.assertIn(tool_b, callables)

    def test_clear_registry(self):
        @tool
        def temp_tool() -> str:
            """Temp."""
            return ""

        self.assertIn("temp_tool", get_registered_tools())
        clear_registry()
        self.assertEqual(get_registered_tools(), {})


class TestExecuteToolCall(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original = dict(get_registered_tools())
        clear_registry()

    def tearDown(self):
        clear_registry()
        from tools import _TOOL_REGISTRY

        _TOOL_REGISTRY.update(self._original)

    async def test_execute_sync_tool(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await execute_tool_call("add", {"a": 3, "b": 4})
        self.assertEqual(result, {"result": 7})

    async def test_execute_async_tool(self):
        @tool
        async def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        result = await execute_tool_call("greet", {"name": "World"})
        self.assertEqual(result, {"result": "Hello, World!"})

    async def test_execute_unknown_tool(self):
        result = await execute_tool_call("nonexistent", {})
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    async def test_execute_tool_with_exception(self):
        @tool
        def failing_tool() -> str:
            """Always fails."""
            raise ValueError("intentional error")

        result = await execute_tool_call("failing_tool", {})
        self.assertIn("error", result)
        self.assertIn("ValueError", result["error"])

    async def test_execute_with_no_args(self):
        @tool
        def no_args_tool() -> str:
            """No args needed."""
            return "done"

        result = await execute_tool_call("no_args_tool", None)
        self.assertEqual(result, {"result": "done"})

    async def test_non_serializable_result_converted_to_string(self):
        @tool
        def returns_set() -> set:
            """Returns a set (not JSON-serializable)."""
            return {1, 2, 3}

        result = await execute_tool_call("returns_set", {})
        self.assertIn("result", result)
        self.assertIsInstance(result["result"], str)


class TestStarterTools(unittest.IsolatedAsyncioTestCase):
    async def test_get_current_time_default(self):
        result = await execute_tool_call("get_current_time", {})
        self.assertIn("result", result)
        self.assertIn("UTC", result["result"])

    async def test_get_current_time_invalid_timezone(self):
        result = await execute_tool_call("get_current_time", {"timezone": "Invalid/Zone"})
        self.assertIn("result", result)
        self.assertIn("Unknown timezone", result["result"])

    async def test_roll_dice_default(self):
        result = await execute_tool_call("roll_dice", {})
        self.assertIn("result", result)
        self.assertIn("d6", result["result"])

    async def test_roll_dice_multiple(self):
        result = await execute_tool_call("roll_dice", {"sides": 20, "count": 3})
        self.assertIn("result", result)
        self.assertIn("3d20", result["result"])
        self.assertIn("total", result["result"])


if __name__ == "__main__":
    unittest.main()
