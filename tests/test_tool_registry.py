import inspect

from discord_gemini.cogs.gemini.tool_registry import (
    build_runtime_tool_config,
    get_tool_metadata,
    get_tool_registry,
    iter_tool_registry,
)
from discord_gemini.util import MODEL_PRICING

#: Tools whose availability is gated by a per-model allowlist.
MODEL_GATED_TOOLS = ("google_maps", "url_context", "file_search")


def _default_chat_model() -> str:
    """Read the live `/gemini chat` default so this guard cannot drift from cog.py."""
    from discord_gemini import GeminiCog

    return inspect.signature(GeminiCog.chat.callback).parameters["model"].default


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


def test_default_chat_model_allowlisted_for_grounding_tools():
    """gemini-3.6-flash is the default chat model and must be allowed its grounding tools."""
    for tool_id in MODEL_GATED_TOOLS:
        metadata = get_tool_metadata(tool_id)
        assert metadata is not None
        assert "gemini-3.6-flash" in metadata.model_allowlist


def test_default_chat_model_is_priced_and_fully_tool_enabled():
    """Both misses this guards against are silent.

    ``MODEL_PRICING.get(model, UNKNOWN_CHAT_MODEL_PRICING)`` bills an unpriced model
    at the fallback rate with no error, and a model missing from an allowlist just
    loses the tool. Anchored to the live default so promoting a new one re-checks both.
    """
    default_model = _default_chat_model()

    assert default_model in MODEL_PRICING, (
        f"{default_model} is the default chat model but is missing from MODEL_PRICING, "
        f"so every request would be billed at the unknown-model fallback rate."
    )
    for tool_id in MODEL_GATED_TOOLS:
        metadata = get_tool_metadata(tool_id)
        assert metadata is not None
        assert metadata.model_allowlist is not None
        assert default_model in metadata.model_allowlist, (
            f"{default_model} is the default chat model but is missing from the "
            f"{tool_id} allowlist, so the tool silently disappears."
        )
