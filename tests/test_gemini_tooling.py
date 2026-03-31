from types import SimpleNamespace

from discord_gemini.cogs.gemini import tooling as gemini_tooling
from tests.support import AsyncGeminiCogTestCase


class TestGeminiTooling(AsyncGeminiCogTestCase):
    async def test_enrich_file_search_tools_injects_store_ids(self):
        """Test that enrich_file_search_tools injects store IDs."""
        tools = [{"file_search": {}}]
        error = self.cog.enrich_file_search_tools(tools)
        assert error is None
        assert tools[0] == {"file_search": {"file_search_store_names": ["store-1", "store-2"]}}

    async def test_enrich_file_search_tools_no_file_search(self):
        """Test that enrich_file_search_tools is a no-op without file_search."""
        tools = [{"google_search": {}}]
        error = self.cog.enrich_file_search_tools(tools)
        assert error is None
        assert tools == [{"google_search": {}}]

    async def test_enrich_file_search_tools_no_store_ids(self):
        """Test that enrich_file_search_tools returns error when store IDs not configured."""
        original = gemini_tooling.GEMINI_FILE_SEARCH_STORE_IDS
        gemini_tooling.GEMINI_FILE_SEARCH_STORE_IDS = []
        try:
            tools = [{"file_search": {}}]
            error = self.cog.enrich_file_search_tools(tools)
            assert error is not None
            assert "GEMINI_FILE_SEARCH_STORE_IDS" in error
        finally:
            gemini_tooling.GEMINI_FILE_SEARCH_STORE_IDS = original

    async def test_resolve_tools_for_view_rejects_unsupported_builtin_custom_combo(self):
        conversation = SimpleNamespace(
            params=SimpleNamespace(
                model="gemini-2.5-flash",
                tools=[],
                custom_functions_enabled=False,
            )
        )

        active, message = gemini_tooling._resolve_tools_for_view(
            self.cog,
            ["google_search"],
            True,
            conversation,
        )

        assert active == set()
        assert message is not None
        assert "cannot be combined" in message
        assert conversation.params.tools == []
        assert conversation.params.custom_functions_enabled is False
