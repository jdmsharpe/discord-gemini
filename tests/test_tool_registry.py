from discord_gemini.cogs.gemini.tool_registry import (
    build_runtime_tool_config,
    get_tool_metadata,
    get_tool_registry,
    iter_tool_registry,
)


def test_registry_keys_match_canonical_ids():
    for key, metadata in get_tool_registry().items():
        assert metadata.canonical_id == key


def test_iter_tool_registry_includes_custom_functions_by_default():
    tool_ids = [tool.canonical_id for tool in iter_tool_registry()]

    assert "custom_functions" in tool_ids


def test_iter_tool_registry_can_exclude_custom_functions():
    tool_ids = [tool.canonical_id for tool in iter_tool_registry(include_custom_functions=False)]

    assert "custom_functions" not in tool_ids


def test_build_runtime_tool_config_returns_expected_payload():
    assert build_runtime_tool_config("google_search") == {"google_search": {}}
    assert build_runtime_tool_config("custom_functions") is None
    assert build_runtime_tool_config("missing") is None


def test_registry_metadata_exposes_model_constraints():
    file_search = get_tool_metadata("file_search")

    assert file_search is not None
    assert "gemini-2.5-pro" in file_search.model_allowlist
