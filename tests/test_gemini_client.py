from importlib import reload
from unittest.mock import patch

from discord_gemini.cogs.gemini import client as gemini_client
from discord_gemini.config import auth as auth_module


class TestGeminiClientBuilders:
    def test_build_gemini_client_without_api_version_configures_retries(self):
        with (
            patch.object(gemini_client.genai, "Client") as mock_client,
            patch.object(gemini_client, "GEMINI_API_VERSION", None),
        ):
            gemini_client.build_gemini_client()

        mock_client.assert_called_once()
        kwargs = mock_client.call_args.kwargs
        assert kwargs["api_key"] == gemini_client.GEMINI_API_KEY
        http_options = kwargs["http_options"]
        assert http_options.retry_options is not None
        assert http_options.retry_options.attempts == gemini_client.MAX_API_ATTEMPTS
        assert http_options.api_version is None

    def test_build_gemini_client_with_api_version_carries_retries(self):
        with (
            patch.object(gemini_client.genai, "Client") as mock_client,
            patch.object(gemini_client, "GEMINI_API_VERSION", "v1"),
        ):
            gemini_client.build_gemini_client()

        mock_client.assert_called_once()
        http_options = mock_client.call_args.kwargs["http_options"]
        assert http_options.api_version == "v1"
        assert http_options.retry_options.attempts == gemini_client.MAX_API_ATTEMPTS
        assert 429 in http_options.retry_options.http_status_codes

    def test_auth_module_reads_gemini_api_version(self):
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_VERSION": "v1alpha",
            },
            clear=False,
        ):
            reload(auth_module)
            assert auth_module.GEMINI_API_VERSION == "v1alpha"

        reload(auth_module)
