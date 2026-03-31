from types import SimpleNamespace

from discord_gemini.cogs.gemini.responses import (
    _build_thinking_config,
    _get_response_content_parts,
    extract_thinking_text,
    extract_tool_info,
)


class TestExtractToolInfo:
    def test_extract_tool_info_empty_output(self):
        """Test extract_tool_info with empty candidates."""
        response = SimpleNamespace(candidates=[])
        tool_info = extract_tool_info(response)
        assert tool_info["tools_used"] == []
        assert tool_info["citations"] == []
        assert tool_info["search_queries"] == []

    def test_extract_tool_info_google_search(self):
        """Test extract_tool_info detects google_search and citations."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=["who won euro 2024"],
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    uri="https://example.com/source",
                                    title="Example Source",
                                )
                            )
                        ],
                        search_entry_point=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                )
            ]
        )

        tool_info = extract_tool_info(response)
        assert "google_search" in tool_info["tools_used"]
        assert tool_info["search_queries"] == ["who won euro 2024"]
        assert tool_info["citations"] == [
            {"title": "Example Source", "uri": "https://example.com/source"}
        ]

    def test_extract_tool_info_code_execution(self):
        """Test extract_tool_info detects code_execution from response parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=None,
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                executable_code=SimpleNamespace(code="print(2 + 2)"),
                                code_execution_result=None,
                            )
                        ]
                    ),
                )
            ]
        )

        tool_info = extract_tool_info(response)
        assert "code_execution" in tool_info["tools_used"]
        assert tool_info["citations"] == []
        assert tool_info["search_queries"] == []

    def test_extract_tool_info_google_maps(self):
        """Test extract_tool_info detects google_maps citations and widget token."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=[],
                        grounding_chunks=[
                            SimpleNamespace(
                                maps=SimpleNamespace(
                                    uri="https://maps.google.com/?cid=123",
                                    title="Test Place",
                                )
                            )
                        ],
                        google_maps_widget_context_token="widgetcontent/token",
                        search_entry_point=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=None,
                )
            ]
        )

        tool_info = extract_tool_info(response)
        assert "google_maps" in tool_info["tools_used"]
        assert tool_info["citations"] == [
            {"title": "Test Place", "uri": "https://maps.google.com/?cid=123"}
        ]
        assert tool_info["maps_widget_token"] == "widgetcontent/token"

    def test_extract_tool_info_url_context(self):
        """Test extract_tool_info detects url_context metadata."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=None,
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=SimpleNamespace(
                        url_metadata=[
                            SimpleNamespace(
                                retrieved_url="https://example.com/a",
                                url_retrieval_status="URL_RETRIEVAL_STATUS_SUCCESS",
                            )
                        ]
                    ),
                )
            ]
        )

        tool_info = extract_tool_info(response)
        assert "url_context" in tool_info["tools_used"]
        assert tool_info["url_context_sources"] == [
            {
                "retrieved_url": "https://example.com/a",
                "status": "URL_RETRIEVAL_STATUS_SUCCESS",
            }
        ]

    def test_extract_tool_info_file_search_via_retrieval_metadata(self):
        """Test extract_tool_info detects file_search via retrieval_metadata."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=[],
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    uri="fileSearchStores/store1/documents/doc1",
                                    title="uploaded-doc.pdf",
                                ),
                                maps=None,
                            )
                        ],
                        search_entry_point=None,
                        google_maps_widget_context_token=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=None,
                    retrieval_metadata=SimpleNamespace(data="retrieval info"),
                )
            ]
        )

        tool_info = extract_tool_info(response)
        assert "file_search" in tool_info["tools_used"]

    def test_extract_tool_info_file_search_fallback(self):
        """Test extract_tool_info detects file_search when grounding chunks present without search/maps."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=[],
                        grounding_chunks=[SimpleNamespace(web=None, maps=None)],
                        search_entry_point=None,
                        google_maps_widget_context_token=None,
                    ),
                    content=SimpleNamespace(parts=[]),
                    url_context_metadata=None,
                )
            ]
        )

        tool_info = extract_tool_info(response)
        assert "file_search" in tool_info["tools_used"]


class TestGeminiThinkingResponses:
    def test_build_thinking_config_none_when_no_params(self):
        """Test that _build_thinking_config returns None when both params are None."""
        result = _build_thinking_config(None, None)
        assert result is None

    def test_build_thinking_config_with_level(self):
        """Test that _build_thinking_config sets thinking_level and include_thoughts."""
        result = _build_thinking_config("high", None)
        assert result is not None
        assert "HIGH" in str(result.thinking_level)
        assert result.include_thoughts is True

    def test_build_thinking_config_with_budget(self):
        """Test that _build_thinking_config sets thinking_budget and include_thoughts."""
        result = _build_thinking_config(None, 1024)
        assert result is not None
        assert result.thinking_budget == 1024
        assert result.include_thoughts is True

    def test_build_thinking_config_with_both(self):
        """Test that _build_thinking_config sets both level and budget."""
        result = _build_thinking_config("low", 512)
        assert result is not None
        assert "LOW" in str(result.thinking_level)
        assert result.thinking_budget == 512
        assert result.include_thoughts is True

    def test_build_thinking_config_budget_zero(self):
        """Test that _build_thinking_config handles budget=0 (disable thinking)."""
        result = _build_thinking_config(None, 0)
        assert result is not None
        assert result.thinking_budget == 0

    def test_build_thinking_config_budget_dynamic(self):
        """Test that _build_thinking_config handles budget=-1 (dynamic)."""
        result = _build_thinking_config(None, -1)
        assert result is not None
        assert result.thinking_budget == -1

    def test_extract_thinking_text_with_thoughts(self):
        """Test extracting thought summaries from response parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(thought=True, text="Let me think..."),
                            SimpleNamespace(thought=False, text="The answer is 42."),
                        ]
                    )
                )
            ]
        )
        result = extract_thinking_text(response)
        assert result == "Let me think..."

    def test_extract_thinking_text_multiple_thought_parts(self):
        """Test extracting multiple thought summary parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(thought=True, text="Step 1: analyze"),
                            SimpleNamespace(thought=True, text="Step 2: compute"),
                            SimpleNamespace(thought=False, text="Result: 7"),
                        ]
                    )
                )
            ]
        )
        result = extract_thinking_text(response)
        assert result == "Step 1: analyze\n\nStep 2: compute"

    def test_extract_thinking_text_no_thoughts(self):
        """Test that no thinking text is returned when there are no thought parts."""
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(thought=False, text="Hello!")])
                )
            ]
        )
        result = extract_thinking_text(response)
        assert result == ""

    def test_extract_thinking_text_empty_response(self):
        """Test extracting thinking text from an empty response."""
        response = SimpleNamespace(candidates=[])
        result = extract_thinking_text(response)
        assert result == ""

    def test_extract_thinking_text_no_candidates(self):
        """Test extracting thinking text when candidates is None."""
        response = SimpleNamespace(candidates=None)
        result = extract_thinking_text(response)
        assert result == ""

    def test_get_response_content_parts_returns_parts(self):
        """Test that response parts are extracted correctly."""
        part1 = SimpleNamespace(text="Hello", thought=False)
        part2 = SimpleNamespace(text="Thinking...", thought=True)
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part1, part2]))]
        )
        result = _get_response_content_parts(response)
        assert result is not None
        assert len(result) == 2
        assert result[0] is part1
        assert result[1] is part2

    def test_get_response_content_parts_empty_candidates(self):
        """Test that None is returned for empty candidates."""
        response = SimpleNamespace(candidates=[])
        result = _get_response_content_parts(response)
        assert result is None

    def test_get_response_content_parts_no_content(self):
        """Test that None is returned when content is None."""
        response = SimpleNamespace(candidates=[SimpleNamespace(content=None)])
        result = _get_response_content_parts(response)
        assert result is None

    def test_get_response_content_parts_no_parts(self):
        """Test that None is returned when parts is empty."""
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])
        result = _get_response_content_parts(response)
        assert result is None
