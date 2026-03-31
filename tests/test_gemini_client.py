from importlib import reload
from unittest.mock import patch

from discord_gemini.cogs.gemini import client as gemini_client
from discord_gemini.config import auth as auth_module


class TestGeminiClientBuilders:
    def test_build_gemini_client_without_api_version(self):
        with patch.object(gemini_client.genai, "Client") as mock_client:
            with patch.object(gemini_client, "GEMINI_API_VERSION", None):
                gemini_client.build_gemini_client()

        mock_client.assert_called_once_with(api_key=gemini_client.GEMINI_API_KEY)

    def test_build_gemini_client_with_api_version(self):
        with (
            patch.object(gemini_client.types, "HttpOptions", return_value="http-options") as http_opts,
            patch.object(gemini_client.genai, "Client") as mock_client,
            patch.object(gemini_client, "GEMINI_API_VERSION", "v1"),
        ):
            gemini_client.build_gemini_client()

        http_opts.assert_called_once_with(api_version="v1")
        mock_client.assert_called_once_with(
            api_key=gemini_client.GEMINI_API_KEY,
            http_options="http-options",
        )

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
