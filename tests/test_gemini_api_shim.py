import importlib
import warnings


def test_gemini_api_shim_warns_and_reexports():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("gemini_api")

    assert module.GeminiAPI.__name__ == "GeminiAPI"
    assert module.Conversation.__name__ == "Conversation"
    assert any(item.category is DeprecationWarning for item in caught)
