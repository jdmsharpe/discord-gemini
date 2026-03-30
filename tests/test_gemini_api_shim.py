import importlib
import warnings


def test_gemini_api_shim_warns_and_reexports():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("gemini_api")

    assert module.GeminiCog.__name__ == "GeminiCog"
    assert module.Conversation.__name__ == "Conversation"
    assert not hasattr(module, "GeminiAPI")
    assert any(item.category is DeprecationWarning for item in caught)
