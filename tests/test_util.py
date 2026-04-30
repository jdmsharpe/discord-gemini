import pytest

from discord_gemini.util import (
    ATTACHMENT_FILE_API_MAX_SIZE,
    ATTACHMENT_FILE_API_THRESHOLD,
    ATTACHMENT_MAX_INLINE_SIZE,
    ATTACHMENT_PDF_MAX_INLINE_SIZE,
    CACHE_MIN_TOKEN_COUNT,
    CACHE_TTL,
    DEFAULT_MUSIC_MODEL,
    FILE_SEARCH_INCOMPATIBLE_TOOLS,
    IMAGE_PRICING,
    LYRIA_3_MODELS,
    LYRIA_REALTIME_MODEL,
    MAPS_GROUNDING_COST_PER_REQUEST,
    MAX_AGENTIC_ITERATIONS,
    MODEL_PRICING,
    MUTUALLY_EXCLUSIVE_TOOLS,
    TOOL_CODE_EXECUTION,
    TOOL_FILE_SEARCH,
    TOOL_GOOGLE_MAPS,
    TOOL_GOOGLE_SEARCH,
    TOOL_URL_CONTEXT,
    TTS_PRICING,
    VIDEO_PRICING,
    AgenticResult,
    ChatCompletionParameters,
    ImageGenerationParameters,
    MusicGenerationParameters,
    ResearchParameters,
    SpeechGenerationParameters,
    VideoGenerationParameters,
    calculate_cost,
    calculate_image_cost,
    calculate_tts_cost,
    calculate_video_cost,
    check_mutually_exclusive_tools,
    chunk_text,
    filter_file_search_incompatible_tools,
    filter_supported_tools_for_model,
    has_server_side_tools,
    model_supports_tool_combinations,
    resolve_tool_name,
    validate_builtin_custom_tool_combination,
)


class TestChatCompletionParameters:
    def test_default_values(self):
        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        assert params.model == "gemini-3-flash-preview"
        assert params.system_instruction is None
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.seed is None
        assert params.temperature is None
        assert params.top_p is None
        assert params.conversation_starter is None
        assert params.conversation_id is None
        assert params.channel_id is None
        assert params.paused is False
        assert params.history == []
        assert params.tools == []
        assert params.media_resolution is None
        assert params.thinking_level is None
        assert params.thinking_budget is None
        assert params.cache_name is None
        assert params.cached_history_length == 0
        assert params.uploaded_file_names == []
        assert params.custom_functions_enabled is False

    def test_all_parameters(self):
        params = ChatCompletionParameters(
            model="gemini-2.5-flash",
            system_instruction="You are a helpful assistant.",
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=42,
            temperature=0.8,
            top_p=0.9,
            media_resolution="MEDIA_RESOLUTION_HIGH",
            conversation_id=123456,
            channel_id=789012,
            paused=True,
            tools=[
                TOOL_GOOGLE_SEARCH,
                TOOL_CODE_EXECUTION,
                TOOL_GOOGLE_MAPS,
                TOOL_URL_CONTEXT,
            ],
        )
        assert params.model == "gemini-2.5-flash"
        assert params.system_instruction == "You are a helpful assistant."
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.3
        assert params.seed == 42
        assert params.temperature == 0.8
        assert params.top_p == 0.9
        assert params.media_resolution == "MEDIA_RESOLUTION_HIGH"
        assert params.conversation_id == 123456
        assert params.channel_id == 789012
        assert params.paused is True
        assert params.tools == [
            TOOL_GOOGLE_SEARCH,
            TOOL_CODE_EXECUTION,
            TOOL_GOOGLE_MAPS,
            TOOL_URL_CONTEXT,
        ]

    def test_uploaded_file_names_default_isolated(self):
        """Test that uploaded_file_names list is isolated between instances."""
        params_one = ChatCompletionParameters(model="gemini-3-flash-preview")
        params_one.uploaded_file_names.append("files/abc123")
        params_two = ChatCompletionParameters(model="gemini-3-flash-preview")
        assert params_two.uploaded_file_names == []
        assert params_one.uploaded_file_names is not params_two.uploaded_file_names

    def test_history_default_isolated(self):
        """Test that history list is isolated between instances."""
        params_one = ChatCompletionParameters(model="gemini-3-flash-preview")
        params_one.history.append({"role": "user", "parts": [{"text": "hello"}]})
        params_two = ChatCompletionParameters(model="gemini-3-flash-preview")
        assert params_two.history == []
        assert params_one.history is not params_two.history

    def test_tools_default_isolated(self):
        """Test that tools list is isolated between instances."""
        params_one = ChatCompletionParameters(model="gemini-3-flash-preview")
        params_one.tools.append(TOOL_GOOGLE_SEARCH)
        params_two = ChatCompletionParameters(model="gemini-3-flash-preview")
        assert params_two.tools == []
        assert params_one.tools is not params_two.tools

    def test_filter_supported_tools_for_model(self):
        """Test model-based filtering for tool compatibility."""
        tools = [
            TOOL_GOOGLE_SEARCH,
            TOOL_CODE_EXECUTION,
            TOOL_GOOGLE_MAPS,
            TOOL_URL_CONTEXT,
        ]
        # gemini-2.0-flash-lite does not support google_maps or url_context
        supported, unsupported = filter_supported_tools_for_model("gemini-2.0-flash-lite", tools)
        assert supported == [TOOL_GOOGLE_SEARCH, TOOL_CODE_EXECUTION]
        assert "google_maps" in unsupported
        assert "url_context" in unsupported

    def test_filter_supported_tools_file_search_supported_model(self):
        """Test that file_search is supported on compatible models."""
        tools = [TOOL_FILE_SEARCH]
        supported, unsupported = filter_supported_tools_for_model("gemini-2.5-pro", tools)
        assert supported == [TOOL_FILE_SEARCH]
        assert unsupported == []

    def test_filter_supported_tools_file_search_unsupported_model(self):
        """Test that file_search is filtered out for unsupported models."""
        tools = [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION]
        supported, unsupported = filter_supported_tools_for_model("gemini-2.0-flash", tools)
        assert supported == [TOOL_CODE_EXECUTION]
        assert unsupported == ["file_search"]


class TestResolveToolName:
    def test_resolve_standard_tools(self):
        """Test resolving standard tool configs to names."""
        assert resolve_tool_name(TOOL_GOOGLE_SEARCH) == "google_search"
        assert resolve_tool_name(TOOL_CODE_EXECUTION) == "code_execution"
        assert resolve_tool_name(TOOL_FILE_SEARCH) == "file_search"

    def test_resolve_enriched_file_search(self):
        """Test resolving file_search config with injected store IDs."""
        enriched = {"file_search": {"file_search_store_names": ["store1", "store2"]}}
        assert resolve_tool_name(enriched) == "file_search"

    def test_resolve_unknown_tool(self):
        """Test that unknown tool configs return None."""
        assert resolve_tool_name({"unknown_tool": {}}) is None

    def test_resolve_callable(self):
        """Test that a Python callable resolves to 'custom_functions'."""

        def my_func():
            pass

        assert resolve_tool_name(my_func) == "custom_functions"

    def test_resolve_function_declarations_dict(self):
        """Test that a function_declarations dict resolves to 'custom_functions'."""
        tool = {"function_declarations": [{"name": "test", "description": "test"}]}
        assert resolve_tool_name(tool) == "custom_functions"


class TestAgenticResult:
    def test_default_values(self):
        result = AgenticResult(response=None, contents=[])
        assert result.response is None
        assert result.contents == []
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.total_thinking_tokens == 0
        assert result.iterations == 0
        assert result.tool_calls_made == []

    def test_tool_calls_made_isolation(self):
        """Test that tool_calls_made list is isolated between instances."""
        r1 = AgenticResult(response=None, contents=[])
        r1.tool_calls_made.append("get_time")
        r2 = AgenticResult(response=None, contents=[])
        assert r2.tool_calls_made == []

    def test_max_agentic_iterations_constant(self):
        assert MAX_AGENTIC_ITERATIONS == 10


class TestFilterSupportedToolsWithCallables:
    def test_callables_pass_through(self):
        """Test that Python callables pass through model filtering unchanged."""

        def my_func():
            pass

        tools = [TOOL_GOOGLE_SEARCH, my_func]
        supported, unsupported = filter_supported_tools_for_model("gemini-2.0-flash-lite", tools)
        assert my_func in supported
        assert unsupported == []

    def test_callables_not_deepcopied(self):
        """Test that callables are the same object (not deepcopied)."""

        def my_func():
            pass

        tools = [my_func]
        supported, _ = filter_supported_tools_for_model("gemini-3-flash-preview", tools)
        assert supported[0] is my_func


class TestToolCombinationSupport:
    def test_has_server_side_tools_detects_builtin_tools(self):
        assert has_server_side_tools([TOOL_GOOGLE_SEARCH]) is True

    def test_has_server_side_tools_ignores_custom_functions(self):
        def my_func():
            pass

        assert has_server_side_tools([my_func]) is False

    def test_model_supports_tool_combinations_for_gemini_3(self):
        assert model_supports_tool_combinations("gemini-3-flash-preview") is True
        assert model_supports_tool_combinations("gemini-3.1-pro-preview") is True
        assert model_supports_tool_combinations("gemini-2.5-flash") is False

    def test_validate_builtin_custom_tool_combination_rejects_unsupported_model(self):
        result = validate_builtin_custom_tool_combination(
            "gemini-2.5-flash",
            [TOOL_GOOGLE_SEARCH],
            custom_functions_enabled=True,
        )
        assert result is not None
        assert "cannot be combined" in result

    def test_validate_builtin_custom_tool_combination_allows_gemini_3(self):
        result = validate_builtin_custom_tool_combination(
            "gemini-3-flash-preview",
            [TOOL_GOOGLE_SEARCH],
            custom_functions_enabled=True,
        )
        assert result is None


class TestFilterFileSearchIncompatibleTools:
    def test_no_file_search_returns_unchanged(self):
        """Test that tools are unchanged when file_search is not present."""
        tools = [TOOL_GOOGLE_SEARCH, TOOL_CODE_EXECUTION]
        filtered, removed = filter_file_search_incompatible_tools(tools)
        assert filtered == tools
        assert removed == []

    def test_file_search_removes_incompatible_tools(self):
        """Test that incompatible tools are removed when file_search is present."""
        tools = [
            TOOL_FILE_SEARCH,
            TOOL_GOOGLE_SEARCH,
            TOOL_CODE_EXECUTION,
            TOOL_URL_CONTEXT,
        ]
        filtered, removed = filter_file_search_incompatible_tools(tools)
        assert filtered == [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION]
        assert "google_search" in removed
        assert "url_context" in removed

    def test_file_search_alone_no_removals(self):
        """Test file_search with only compatible tools."""
        tools = [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION]
        filtered, removed = filter_file_search_incompatible_tools(tools)
        assert filtered == [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION]
        assert removed == []

    def test_file_search_incompatible_tools_constant(self):
        """Test that the incompatible tools set contains expected tools."""
        assert "google_search" in FILE_SEARCH_INCOMPATIBLE_TOOLS
        assert "google_maps" in FILE_SEARCH_INCOMPATIBLE_TOOLS
        assert "url_context" in FILE_SEARCH_INCOMPATIBLE_TOOLS
        assert "code_execution" not in FILE_SEARCH_INCOMPATIBLE_TOOLS
        assert "file_search" not in FILE_SEARCH_INCOMPATIBLE_TOOLS


class TestMutuallyExclusiveTools:
    def test_no_conflict_returns_none(self):
        """No error when non-conflicting tools are selected."""
        assert check_mutually_exclusive_tools({"google_search", "code_execution"}) is None

    def test_google_search_and_maps_conflict(self):
        """Error returned when google_search and google_maps are both selected."""
        result = check_mutually_exclusive_tools({"google_search", "google_maps"})
        assert result is not None
        assert "google_search" in result
        assert "google_maps" in result

    def test_single_tool_no_conflict(self):
        """No error when only one tool from a conflicting pair is selected."""
        assert check_mutually_exclusive_tools({"google_search"}) is None
        assert check_mutually_exclusive_tools({"google_maps"}) is None

    def test_empty_set_no_conflict(self):
        """No error when no tools are selected."""
        assert check_mutually_exclusive_tools(set()) is None

    def test_constant_contains_search_maps_pair(self):
        """The MUTUALLY_EXCLUSIVE_TOOLS constant includes the search/maps pair."""
        assert ("google_search", "google_maps") in MUTUALLY_EXCLUSIVE_TOOLS


class TestImageGenerationParameters:
    def test_to_dict_basic(self):
        params = ImageGenerationParameters(
            prompt="A house in the woods",
            model="imagen-3.0-generate-001",
            number_of_images=2,
            aspect_ratio="16:9",
        )
        result = params.to_dict()
        assert result["number_of_images"] == 2
        assert result["aspect_ratio"] == "16:9"

    def test_to_dict_with_negative_prompt(self):
        params = ImageGenerationParameters(
            prompt="A sunset",
            model="imagen-4.0-generate-001",
            negative_prompt="blurry, low quality",
        )
        result = params.to_dict()
        assert result["negative_prompt"] == "blurry, low quality"

    def test_to_dict_with_seed_and_guidance(self):
        params = ImageGenerationParameters(
            prompt="A mountain",
            model="imagen-3.0-generate-001",
            seed=42,
            guidance_scale=7.5,
        )
        result = params.to_dict()
        assert result["seed"] == 42
        assert result["guidance_scale"] == 7.5

    def test_person_generation_mapping(self):
        """Test that person_generation values are properly mapped."""
        # Test dont_allow
        params = ImageGenerationParameters(
            prompt="Test",
            model="imagen-3.0-generate-001",
            person_generation="dont_allow",
        )
        result = params.to_dict()
        assert result["person_generation"] == "DONT_ALLOW"

        # Test allow_all
        params = ImageGenerationParameters(
            prompt="Test",
            model="imagen-3.0-generate-001",
            person_generation="allow_all",
        )
        result = params.to_dict()
        assert result["person_generation"] == "ALLOW_ALL"

    def test_person_generation_allow_adult_excluded(self):
        """Test that allow_adult (default) is not included in output."""
        params = ImageGenerationParameters(
            prompt="Test",
            model="imagen-3.0-generate-001",
            person_generation="allow_adult",
        )
        result = params.to_dict()
        assert "person_generation" not in result

    def test_none_values_excluded(self):
        """Test that None values are not included in to_dict output."""
        params = ImageGenerationParameters(
            prompt="Test",
            model="gemini-3-pro-image-preview",
        )
        result = params.to_dict()
        assert "aspect_ratio" not in result
        assert "negative_prompt" not in result
        assert "seed" not in result
        assert "guidance_scale" not in result

    def test_image_size_default_none(self):
        """Test that image_size defaults to None."""
        params = ImageGenerationParameters(prompt="Test", model="gemini-3.1-flash-image-preview")
        assert params.image_size is None

    def test_image_size_set(self):
        """Test that image_size can be set to a valid value."""
        params = ImageGenerationParameters(
            prompt="Test", model="gemini-3.1-flash-image-preview", image_size="2k"
        )
        assert params.image_size == "2k"

    def test_google_image_search_default_false(self):
        """Test that google_image_search defaults to False."""
        params = ImageGenerationParameters(prompt="Test", model="gemini-3.1-flash-image-preview")
        assert params.google_image_search is False

    def test_google_image_search_enabled(self):
        """Test that google_image_search can be enabled."""
        params = ImageGenerationParameters(
            prompt="Test", model="gemini-3.1-flash-image-preview", google_image_search=True
        )
        assert params.google_image_search is True

    def test_new_fields_isolation(self):
        """Test that image_size and google_image_search are independent across instances."""
        params1 = ImageGenerationParameters(
            prompt="A",
            model="gemini-3.1-flash-image-preview",
            image_size="1k",
            google_image_search=True,
        )
        params2 = ImageGenerationParameters(prompt="B", model="gemini-3.1-flash-image-preview")
        assert params1.image_size == "1k"
        assert params1.google_image_search is True
        assert params2.image_size is None
        assert params2.google_image_search is False


class TestVideoGenerationParameters:
    def test_to_dict_basic(self):
        params = VideoGenerationParameters(
            prompt="A cat playing piano",
            model="veo-2.0-generate-001",
            aspect_ratio="16:9",
            resolution="720p",
            number_of_videos=2,
        )
        result = params.to_dict()
        assert result["aspect_ratio"] == "16:9"
        assert result["resolution"] == "720p"
        assert result["number_of_videos"] == 2

    def test_to_dict_with_duration(self):
        params = VideoGenerationParameters(
            prompt="A sunset timelapse",
            model="veo-3.0-generate-001",
            duration_seconds=8,
        )
        result = params.to_dict()
        assert result["duration_seconds"] == 8

    def test_to_dict_with_negative_prompt(self):
        params = VideoGenerationParameters(
            prompt="A dog running",
            model="veo-2.0-generate-001",
            negative_prompt="blurry, distorted",
        )
        result = params.to_dict()
        assert result["negative_prompt"] == "blurry, distorted"

    def test_enhance_prompt(self):
        params = VideoGenerationParameters(
            prompt="Test video",
            model="veo-3.1-generate-preview",
            enhance_prompt=True,
        )
        result = params.to_dict()
        assert result["enhance_prompt"] is True

        params_disabled = VideoGenerationParameters(
            prompt="Test video",
            model="veo-3.1-generate-preview",
            enhance_prompt=False,
        )
        result_disabled = params_disabled.to_dict()
        assert result_disabled["enhance_prompt"] is False

    def test_person_generation_mapping(self):
        """Test that person_generation values are properly mapped for Veo."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-2.0-generate-001",
            person_generation="dont_allow",
        )
        result = params.to_dict()
        assert result["person_generation"] == "dont_allow"

    def test_person_generation_allow_adult_excluded(self):
        """Test that allow_adult (default) is not included in output."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-2.0-generate-001",
            person_generation="allow_adult",
        )
        result = params.to_dict()
        assert "person_generation" not in result

    def test_none_values_excluded(self):
        """Test that None values are not included in to_dict output."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-2.0-generate-001",
        )
        result = params.to_dict()
        assert "aspect_ratio" not in result
        assert "resolution" not in result
        assert "negative_prompt" not in result
        assert "number_of_videos" not in result
        assert "duration_seconds" not in result
        assert "enhance_prompt" not in result

    def test_has_last_frame_default(self):
        """Test that has_last_frame defaults to False."""
        params = VideoGenerationParameters(prompt="Test", model="veo-3.1-generate-preview")
        assert params.has_last_frame is False

    def test_has_last_frame_set(self):
        """Test that has_last_frame can be set to True."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        assert params.has_last_frame is True

    def test_has_last_frame_not_in_to_dict(self):
        """Test that has_last_frame is not included in to_dict output (display-only field)."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        result = params.to_dict()
        assert "has_last_frame" not in result

    def test_has_last_frame_isolation(self):
        """Test that has_last_frame is independent between instances."""
        params1 = VideoGenerationParameters(
            prompt="A", model="veo-3.1-generate-preview", has_last_frame=True
        )
        params2 = VideoGenerationParameters(prompt="B", model="veo-3.1-generate-preview")
        assert params1.has_last_frame is True
        assert params2.has_last_frame is False


class TestSpeechGenerationParameters:
    def test_defaults(self):
        params = SpeechGenerationParameters(input_text="Hello world")
        assert params.input_text == "Hello world"
        assert params.model == "gemini-2.5-flash-preview-tts"
        assert params.voice_name == "Kore"
        assert params.multi_speaker is False
        assert params.speaker_configs is None
        assert params.style_prompt is None

    def test_to_dict_single_speaker(self):
        params = SpeechGenerationParameters(
            input_text="Hello world",
            voice_name="Puck",
        )
        result = params.to_dict()
        assert result["response_modalities"] == ["AUDIO"]
        assert "speech_config" in result
        assert "voice_config" in result["speech_config"]
        assert (
            result["speech_config"]["voice_config"]["prebuilt_voice_config"]["voice_name"] == "Puck"
        )

    def test_to_dict_multi_speaker(self):
        speaker_configs = [
            {"speaker": "Joe", "voice_name": "Kore"},
            {"speaker": "Jane", "voice_name": "Puck"},
        ]
        params = SpeechGenerationParameters(
            input_text="Joe: Hello! Jane: Hi there!",
            multi_speaker=True,
            speaker_configs=speaker_configs,
        )
        result = params.to_dict()
        assert result["response_modalities"] == ["AUDIO"]
        assert "speech_config" in result
        assert "multi_speaker_voice_config" in result["speech_config"]
        speaker_voice_configs = result["speech_config"]["multi_speaker_voice_config"][
            "speaker_voice_configs"
        ]
        assert len(speaker_voice_configs) == 2
        assert speaker_voice_configs[0]["speaker"] == "Joe"
        assert (
            speaker_voice_configs[0]["voice_config"]["prebuilt_voice_config"]["voice_name"]
            == "Kore"
        )


class TestMusicGenerationParameters:
    def test_defaults(self):
        params = MusicGenerationParameters(prompts=["upbeat jazz"])
        assert params.prompts == ["upbeat jazz"]
        assert params.model == DEFAULT_MUSIC_MODEL
        assert params.prompt_weights is None
        assert params.duration == 30
        assert params.bpm is None
        assert params.scale is None
        assert params.guidance == 4.0
        assert params.density is None
        assert params.brightness is None
        assert params.temperature == 1.1
        assert params.top_k == 40
        assert params.seed is None
        assert params.mute_bass is False
        assert params.mute_drums is False
        assert params.only_bass_and_drums is False

    def test_supported_music_model_constants(self):
        """Test the supported Lyria model constants."""
        assert DEFAULT_MUSIC_MODEL == "lyria-3-clip-preview"
        assert LYRIA_REALTIME_MODEL == "lyria-realtime-exp"
        assert "lyria-3-pro-preview" in LYRIA_3_MODELS
        assert "lyria-3-clip-preview" in LYRIA_3_MODELS

    def test_to_weighted_prompts_without_weights(self):
        params = MusicGenerationParameters(prompts=["jazz", "piano"])
        result = params.to_weighted_prompts()
        assert len(result) == 2
        assert result[0] == {"text": "jazz", "weight": 1.0}
        assert result[1] == {"text": "piano", "weight": 1.0}

    def test_to_weighted_prompts_with_weights(self):
        params = MusicGenerationParameters(
            prompts=["jazz", "piano"],
            prompt_weights=[0.7, 0.3],
        )
        result = params.to_weighted_prompts()
        assert len(result) == 2
        assert result[0] == {"text": "jazz", "weight": 0.7}
        assert result[1] == {"text": "piano", "weight": 0.3}

    def test_to_weighted_prompts_mismatched_weights(self):
        """Test that mismatched weights fall back to default 1.0."""
        params = MusicGenerationParameters(
            prompts=["jazz", "piano", "drums"],
            prompt_weights=[0.7, 0.3],  # Only 2 weights for 3 prompts
        )
        result = params.to_weighted_prompts()
        # Should fall back to default weights
        assert result[0]["weight"] == 1.0
        assert result[1]["weight"] == 1.0
        assert result[2]["weight"] == 1.0

    def test_to_music_config_basic(self):
        params = MusicGenerationParameters(prompts=["test"])
        result = params.to_music_config()
        assert result["guidance"] == 4.0
        assert result["temperature"] == 1.1
        assert result["top_k"] == 40
        assert result["mute_bass"] is False
        assert result["mute_drums"] is False
        assert result["only_bass_and_drums"] is False

    def test_to_music_config_with_optional_params(self):
        params = MusicGenerationParameters(
            prompts=["jazz"],
            bpm=120,
            scale="C_MAJOR_A_MINOR",
            density=0.5,
            brightness=0.7,
            seed=42,
        )
        result = params.to_music_config()
        assert result["bpm"] == 120
        assert result["scale"] == "C_MAJOR_A_MINOR"
        assert result["density"] == 0.5
        assert result["brightness"] == 0.7
        assert result["seed"] == 42

    def test_to_music_config_mute_options(self):
        params = MusicGenerationParameters(
            prompts=["test"],
            mute_bass=True,
            mute_drums=True,
        )
        result = params.to_music_config()
        assert result["mute_bass"] is True
        assert result["mute_drums"] is True

    def test_to_music_config_only_bass_and_drums(self):
        params = MusicGenerationParameters(
            prompts=["test"],
            only_bass_and_drums=True,
        )
        result = params.to_music_config()
        assert result["only_bass_and_drums"] is True


class TestResearchParameters:
    def test_defaults(self):
        params = ResearchParameters(prompt="Research quantum computing")
        assert params.prompt == "Research quantum computing"
        assert params.agent == "deep-research-preview-04-2026"
        assert params.file_search is False
        assert params.google_maps is False

    def test_all_parameters(self):
        params = ResearchParameters(
            prompt="Compare EV batteries",
            agent="deep-research-max-preview-04-2026",
            file_search=True,
            google_maps=True,
        )
        assert params.prompt == "Compare EV batteries"
        assert params.agent == "deep-research-max-preview-04-2026"
        assert params.file_search is True
        assert params.google_maps is True


class TestChunkText:
    def test_chunk_text_small(self):
        text = "This is a test."
        size = 4
        result = chunk_text(text, size)
        assert result == ["This", " is ", "a te", "st."]

    def test_chunk_text_exact_size(self):
        text = "abcd"
        size = 4
        result = chunk_text(text, size)
        assert result == ["abcd"]

    def test_chunk_text_larger_than_text(self):
        text = "Hello"
        size = 100
        result = chunk_text(text, size)
        assert result == ["Hello"]

    def test_chunk_text_empty(self):
        text = ""
        size = 4
        result = chunk_text(text, size)
        assert result == []

    def test_chunk_text_default_size(self):
        text = "a" * 8192
        result = chunk_text(text)  # Default size is 4096
        assert len(result) == 2
        assert len(result[0]) == 4096
        assert len(result[1]) == 4096

    def test_chunk_text_long(self):
        text = "This is a test. " * 64  # len(text) * 64 = 1024
        size = 1024
        result = chunk_text(text, size)
        assert len(result[0]) == size


class TestTruncateText:
    """Tests for the truncate_text utility function."""

    def test_truncate_text_under_limit(self):
        """Test that short text is not truncated."""
        from discord_gemini.util import truncate_text

        text = "Short text"
        result = truncate_text(text, 100)
        assert result == text
        assert "..." not in result

    def test_truncate_text_over_limit(self):
        """Test that long text is truncated with suffix."""
        from discord_gemini.util import truncate_text

        text = "A" * 100
        result = truncate_text(text, 50)
        assert len(result) == 50
        assert result.endswith("...")
        assert result == "A" * 47 + "..."

    def test_truncate_text_at_limit(self):
        """Test that text exactly at limit is not truncated."""
        from discord_gemini.util import truncate_text

        text = "B" * 100
        result = truncate_text(text, 100)
        assert result == text
        assert "..." not in result

    def test_truncate_text_custom_suffix(self):
        """Test truncation with custom suffix."""
        from discord_gemini.util import truncate_text

        text = "Hello, world!"
        result = truncate_text(text, 5, suffix="[...]")
        assert result == "[...]"
        assert len(result) == 5

    def test_truncate_text_none(self):
        """Test that None input returns None."""
        from discord_gemini.util import truncate_text

        result = truncate_text(None, 100)
        assert result is None

    def test_truncate_text_empty_string(self):
        """Test that empty string is handled correctly."""
        from discord_gemini.util import truncate_text

        result = truncate_text("", 100)
        assert result == ""

    def test_truncate_text_empty_suffix(self):
        """Test truncation with empty suffix."""
        from discord_gemini.util import truncate_text

        text = "Hello, world!"
        result = truncate_text(text, 5, suffix="")
        assert result == "Hello"
        assert len(result) == 5

    def test_truncate_text_prompt_limit(self):
        """Test standard 2000 char prompt truncation."""
        from discord_gemini.util import truncate_text

        long_prompt = "A" * 3000
        result = truncate_text(long_prompt, 2000)
        assert len(result) == 2000
        assert result.endswith("...")

    def test_truncate_text_response_limit(self):
        """Test standard 3500 char response truncation."""
        from discord_gemini.util import truncate_text

        long_response = "B" * 5000
        result = truncate_text(long_response, 3500)
        assert len(result) == 3500
        assert result.endswith("...")

    def test_truncate_text_max_length_zero_or_negative(self):
        """Test that non-positive max_length returns empty string."""
        from discord_gemini.util import truncate_text

        assert truncate_text("Hello", 0) == ""
        assert truncate_text("Hello", -1) == ""

    def test_truncate_text_suffix_longer_than_limit(self):
        """Test truncation when suffix is longer than max_length."""
        from discord_gemini.util import truncate_text

        text = "A" * 100
        result = truncate_text(text, 2, suffix="[...]")
        assert result == "[."
        assert len(result) == 2


class TestCacheConstants:
    """Tests for explicit caching constants."""

    def test_cache_min_token_count_contains_expected_models(self):
        """Test that CACHE_MIN_TOKEN_COUNT includes selected 3.x and 2.5 models."""
        assert "gemini-3.1-pro-preview" in CACHE_MIN_TOKEN_COUNT
        assert "gemini-3-flash-preview" in CACHE_MIN_TOKEN_COUNT
        assert "gemini-2.5-pro" in CACHE_MIN_TOKEN_COUNT
        assert "gemini-2.5-flash" in CACHE_MIN_TOKEN_COUNT

    def test_cache_min_token_count_values(self):
        """Test that token thresholds are correct per model tier."""
        assert CACHE_MIN_TOKEN_COUNT["gemini-3.1-pro-preview"] == 4096
        assert CACHE_MIN_TOKEN_COUNT["gemini-3-flash-preview"] == 1024
        assert CACHE_MIN_TOKEN_COUNT["gemini-2.5-pro"] == 4096
        assert CACHE_MIN_TOKEN_COUNT["gemini-2.5-flash"] == 1024

    def test_cache_min_token_count_excludes_unsupported_models(self):
        """Test that older models still rely on implicit-only/no explicit caching here."""
        assert "gemini-2.0-flash" not in CACHE_MIN_TOKEN_COUNT
        assert "gemini-2.0-flash-lite" not in CACHE_MIN_TOKEN_COUNT

    def test_cache_ttl_is_valid(self):
        """Test that CACHE_TTL is a valid duration string."""
        assert CACHE_TTL == "3600s"


class TestChatCompletionParametersCaching:
    """Tests for cache-related fields on ChatCompletionParameters."""

    def test_cache_fields_with_values(self):
        """Test that cache fields can be set."""
        params = ChatCompletionParameters(
            model="gemini-2.5-flash",
            cache_name="cachedContents/abc123",
            cached_history_length=4,
        )
        assert params.cache_name == "cachedContents/abc123"
        assert params.cached_history_length == 4

    def test_cache_fields_reset(self):
        """Test that cache fields can be cleared."""
        params = ChatCompletionParameters(
            model="gemini-2.5-flash",
            cache_name="cachedContents/abc123",
            cached_history_length=4,
        )
        params.cache_name = None
        params.cached_history_length = 0
        assert params.cache_name is None
        assert params.cached_history_length == 0


class TestAttachmentSizeConstants:
    """Tests for attachment size limit constants."""

    def test_inline_max_size(self):
        """Test that inline data max is 100 MB."""
        assert ATTACHMENT_MAX_INLINE_SIZE == 100 * 1024 * 1024

    def test_pdf_inline_max_size(self):
        """Test that PDF inline max is 50 MB."""
        assert ATTACHMENT_PDF_MAX_INLINE_SIZE == 50 * 1024 * 1024

    def test_file_api_threshold(self):
        """Test that File API threshold is 20 MB."""
        assert ATTACHMENT_FILE_API_THRESHOLD == 20 * 1024 * 1024

    def test_file_api_max_size(self):
        """Test that File API max is 2 GB."""
        assert ATTACHMENT_FILE_API_MAX_SIZE == 2 * 1024 * 1024 * 1024

    def test_threshold_ordering(self):
        """Test that size limits are in correct ascending order."""
        assert ATTACHMENT_FILE_API_THRESHOLD < ATTACHMENT_PDF_MAX_INLINE_SIZE
        assert ATTACHMENT_PDF_MAX_INLINE_SIZE < ATTACHMENT_MAX_INLINE_SIZE
        assert ATTACHMENT_MAX_INLINE_SIZE < ATTACHMENT_FILE_API_MAX_SIZE


class TestModelPricing:
    """Tests for MODEL_PRICING and calculate_cost."""

    def test_pricing_contains_all_chat_models(self):
        """Test that MODEL_PRICING includes all chat models."""
        expected_models = [
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        for model in expected_models:
            assert model in MODEL_PRICING, f"{model} missing from MODEL_PRICING"

    def test_pricing_values_are_positive(self):
        """Test that all pricing values are positive."""
        for model, (input_price, output_price) in MODEL_PRICING.items():
            assert input_price > 0, f"{model} input price should be positive"
            assert output_price > 0, f"{model} output price should be positive"

    def test_pricing_output_greater_than_input(self):
        """Test that output pricing is always >= input pricing."""
        for model, (input_price, output_price) in MODEL_PRICING.items():
            assert output_price >= input_price, f"{model} output price should be >= input price"

    def test_calculate_cost_known_model(self):
        """Test cost calculation for a known model."""
        # gemini-2.0-flash: $0.10/M input, $0.40/M output
        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.50)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_cost("gemini-2.5-pro", 0, 0)
        assert cost == pytest.approx(0.0)

    def test_calculate_cost_unknown_model_uses_default(self):
        """Test that unknown models fall back to default pricing."""
        cost = calculate_cost("unknown-model", 1_000_000, 1_000_000)
        # Default is (2.0, 12.0)
        assert cost == pytest.approx(14.0)

    def test_calculate_cost_small_token_count(self):
        """Test cost calculation with realistic small token counts."""
        # gemini-3.1-pro-preview: $2.00/M input, $12.00/M output
        cost = calculate_cost("gemini-3.1-pro-preview", 500, 200)
        expected = (500 / 1_000_000) * 2.0 + (200 / 1_000_000) * 12.0
        assert cost == pytest.approx(expected)

    def test_calculate_cost_with_thinking_tokens(self):
        """Test that thinking tokens are billed at the output token rate."""
        # gemini-2.5-flash: $0.30/M input, $2.50/M output
        cost = calculate_cost("gemini-2.5-flash", 1_000_000, 500_000, thinking_tokens=500_000)
        # 1M * $0.30 + (500K + 500K) * $2.50 = $0.30 + $2.50 = $2.80
        assert cost == pytest.approx(2.80)

    def test_calculate_cost_with_zero_thinking_tokens(self):
        """Test that zero thinking tokens doesn't affect cost."""
        cost_without = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
        cost_with = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000, thinking_tokens=0)
        assert cost_without == pytest.approx(cost_with)

    def test_calculate_cost_thinking_only(self):
        """Test cost when output is mostly thinking tokens."""
        # gemini-3-flash-preview: $0.50/M input, $3.00/M output
        cost = calculate_cost("gemini-3-flash-preview", 100_000, 50_000, thinking_tokens=1_000_000)
        expected = (100_000 / 1_000_000) * 0.50 + ((50_000 + 1_000_000) / 1_000_000) * 3.0
        assert cost == pytest.approx(expected)

    def test_calculate_cost_with_maps_grounding(self):
        """Test that Maps grounding adds the per-request surcharge."""
        base_cost = calculate_cost("gemini-2.5-flash", 1000, 500)
        maps_cost = calculate_cost("gemini-2.5-flash", 1000, 500, google_maps_grounded=True)
        assert maps_cost - base_cost == pytest.approx(MAPS_GROUNDING_COST_PER_REQUEST)

    def test_calculate_cost_without_maps_grounding(self):
        """Test that Maps surcharge is not applied when grounding is False."""
        cost_default = calculate_cost("gemini-2.5-flash", 1000, 500)
        cost_explicit = calculate_cost("gemini-2.5-flash", 1000, 500, google_maps_grounded=False)
        assert cost_default == pytest.approx(cost_explicit)

    def test_calculate_cost_maps_grounding_with_thinking(self):
        """Test that Maps surcharge stacks with thinking token cost."""
        base = calculate_cost("gemini-3-flash-preview", 1000, 500, thinking_tokens=2000)
        with_maps = calculate_cost(
            "gemini-3-flash-preview", 1000, 500, thinking_tokens=2000, google_maps_grounded=True
        )
        assert with_maps - base == pytest.approx(MAPS_GROUNDING_COST_PER_REQUEST)


class TestImagePricing:
    """Tests for IMAGE_PRICING and calculate_image_cost."""

    def test_pricing_contains_all_image_models(self):
        """Test that IMAGE_PRICING includes all image generation models."""
        expected_models = [
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
            "gemini-2.5-flash-image",
            "imagen-4.0-generate-001",
            "imagen-4.0-ultra-generate-001",
            "imagen-4.0-fast-generate-001",
        ]
        for model in expected_models:
            assert model in IMAGE_PRICING, f"{model} missing from IMAGE_PRICING"

    def test_pricing_values_are_non_negative(self):
        """Test that all pricing values are non-negative."""
        for model, (input_rate, size_prices) in IMAGE_PRICING.items():
            assert input_rate >= 0, f"{model} input rate should be >= 0"
            for size, price in size_prices.items():
                assert price > 0, f"{model} size={size} cost should be > 0"

    def test_imagen_models_have_zero_input_cost(self):
        """Test that Imagen models have zero input token cost (flat per-image pricing)."""
        imagen_models = [
            "imagen-4.0-generate-001",
            "imagen-4.0-ultra-generate-001",
            "imagen-4.0-fast-generate-001",
        ]
        for model in imagen_models:
            input_rate, _ = IMAGE_PRICING[model]
            assert input_rate == 0.0, f"{model} should have zero input rate"

    def test_calculate_image_cost_gemini_model(self):
        """Test cost calculation for a Gemini image model with input tokens."""
        # gemini-3.1-flash-image-preview: $0.50/M input, $0.067/image at 1K
        cost = calculate_image_cost(
            "gemini-3.1-flash-image-preview", num_images=2, input_tokens=1_000_000
        )
        expected = 0.50 + 2 * 0.067  # input cost + 2 images
        assert cost == pytest.approx(expected)

    def test_calculate_image_cost_2k_resolution(self):
        """Test that 2K resolution uses higher per-image cost."""
        cost_1k = calculate_image_cost(
            "gemini-3.1-flash-image-preview", num_images=1, image_size="1k"
        )
        cost_2k = calculate_image_cost(
            "gemini-3.1-flash-image-preview", num_images=1, image_size="2k"
        )
        assert cost_1k == pytest.approx(0.067)
        assert cost_2k == pytest.approx(0.101)
        assert cost_2k > cost_1k

    def test_calculate_image_cost_case_insensitive(self):
        """Test that image_size lookup is case-insensitive."""
        cost_lower = calculate_image_cost(
            "gemini-3.1-flash-image-preview", num_images=1, image_size="2k"
        )
        cost_upper = calculate_image_cost(
            "gemini-3.1-flash-image-preview", num_images=1, image_size="2K"
        )
        assert cost_lower == pytest.approx(cost_upper)

    def test_calculate_image_cost_imagen_model(self):
        """Test cost calculation for an Imagen model (no input tokens)."""
        cost = calculate_image_cost("imagen-4.0-generate-001", num_images=4)
        assert cost == pytest.approx(4 * 0.04)

    def test_calculate_image_cost_zero_images(self):
        """Test cost calculation when no images are generated."""
        cost = calculate_image_cost("gemini-3.1-flash-image-preview", num_images=0)
        assert cost == pytest.approx(0.0)

    def test_calculate_image_cost_unknown_model(self):
        """Test that unknown models use default pricing."""
        cost = calculate_image_cost("unknown-image-model", num_images=1, input_tokens=0)
        # Default: (0.50, {None: 0.067}) -> $0.067
        assert cost == pytest.approx(0.067)

    def test_calculate_image_cost_with_input_tokens_only(self):
        """Test cost when images=0 but input tokens are charged."""
        cost = calculate_image_cost(
            "gemini-3.1-flash-image-preview", num_images=0, input_tokens=1_000_000
        )
        assert cost == pytest.approx(0.50)  # input cost only


class TestVideoPricing:
    """Tests for VIDEO_PRICING and calculate_video_cost."""

    def test_pricing_contains_all_video_models(self):
        """Test that VIDEO_PRICING includes all video generation models."""
        expected_models = [
            "veo-3.1-lite-generate-preview",
            "veo-3.1-generate-preview",
            "veo-3.1-fast-generate-preview",
            "veo-3.0-generate-001",
            "veo-3.0-fast-generate-001",
            "veo-2.0-generate-001",
        ]
        for model in expected_models:
            assert model in VIDEO_PRICING, f"{model} missing from VIDEO_PRICING"

    def test_pricing_values_are_positive(self):
        """Test that all per-second rates are positive."""
        for model, resolution_prices in VIDEO_PRICING.items():
            for resolution, rate in resolution_prices.items():
                assert rate > 0, f"{model} {resolution} rate should be positive"

    def test_fast_models_cheaper_than_standard(self):
        """Test that fast variants are cheaper than standard."""
        assert (
            VIDEO_PRICING["veo-3.1-lite-generate-preview"]["720p"]
            < VIDEO_PRICING["veo-3.1-fast-generate-preview"]["720p"]
        )
        assert (
            VIDEO_PRICING["veo-3.1-fast-generate-preview"]["720p"]
            < VIDEO_PRICING["veo-3.1-generate-preview"]["720p"]
        )
        assert (
            VIDEO_PRICING["veo-3.0-fast-generate-001"]["720p"]
            < VIDEO_PRICING["veo-3.0-generate-001"]["720p"]
        )

    def test_calculate_video_cost_lite_default_resolution(self):
        """Test Lite pricing uses the default 720p per-second rate."""
        cost = calculate_video_cost("veo-3.1-lite-generate-preview", duration_seconds=8)
        assert cost == pytest.approx(0.40)

    def test_calculate_video_cost_lite_1080p(self):
        """Test Lite pricing uses the 1080p rate when requested."""
        cost = calculate_video_cost(
            "veo-3.1-lite-generate-preview", duration_seconds=8, resolution="1080p"
        )
        assert cost == pytest.approx(0.64)

    def test_calculate_video_cost_basic(self):
        """Test basic video cost calculation."""
        # veo-3.1-generate-preview: $0.40/sec, 8 seconds
        cost = calculate_video_cost("veo-3.1-generate-preview", duration_seconds=8)
        assert cost == pytest.approx(3.20)

    def test_calculate_video_cost_3_1_fast_4k(self):
        """Test 4k pricing for Veo 3.1 Fast."""
        cost = calculate_video_cost(
            "veo-3.1-fast-generate-preview", duration_seconds=8, resolution="4k"
        )
        assert cost == pytest.approx(2.40)

    def test_calculate_video_cost_multiple_videos(self):
        """Test cost for multiple videos."""
        cost = calculate_video_cost("veo-2.0-generate-001", duration_seconds=5, num_videos=2)
        assert cost == pytest.approx(5 * 2 * 0.35)

    def test_calculate_video_cost_zero_duration(self):
        """Test cost with zero duration."""
        cost = calculate_video_cost("veo-3.1-generate-preview", duration_seconds=0)
        assert cost == pytest.approx(0.0)

    def test_calculate_video_cost_unknown_model(self):
        """Test that unknown models use default pricing ($0.35/sec)."""
        cost = calculate_video_cost("unknown-veo", duration_seconds=10)
        assert cost == pytest.approx(3.50)


class TestTtsPricing:
    """Tests for TTS_PRICING and calculate_tts_cost."""

    def test_pricing_contains_all_tts_models(self):
        """Test that TTS_PRICING includes all TTS models."""
        expected_models = [
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        ]
        for model in expected_models:
            assert model in TTS_PRICING, f"{model} missing from TTS_PRICING"

    def test_pricing_values_are_positive(self):
        """Test that all pricing values are positive."""
        for model, (input_price, output_price) in TTS_PRICING.items():
            assert input_price > 0, f"{model} input price should be positive"
            assert output_price > 0, f"{model} output price should be positive"

    def test_pro_more_expensive_than_flash(self):
        """Test that Pro TTS is more expensive than Flash TTS."""
        flash_in, flash_out = TTS_PRICING["gemini-2.5-flash-preview-tts"]
        pro_in, pro_out = TTS_PRICING["gemini-2.5-pro-preview-tts"]
        assert pro_in > flash_in
        assert pro_out > flash_out

    def test_calculate_tts_cost_basic(self):
        """Test basic TTS cost calculation."""
        # gemini-2.5-flash-preview-tts: $0.50/M input, $10.00/M output
        cost = calculate_tts_cost(
            "gemini-2.5-flash-preview-tts", input_tokens=1_000_000, output_tokens=1_000_000
        )
        assert cost == pytest.approx(10.50)

    def test_calculate_tts_cost_zero_tokens(self):
        """Test cost with zero tokens."""
        cost = calculate_tts_cost("gemini-2.5-flash-preview-tts", 0, 0)
        assert cost == pytest.approx(0.0)

    def test_calculate_tts_cost_unknown_model(self):
        """Test that unknown models use default pricing."""
        cost = calculate_tts_cost("unknown-tts", input_tokens=1_000_000, output_tokens=1_000_000)
        # Default: (0.50, 10.00)
        assert cost == pytest.approx(10.50)

    def test_calculate_tts_cost_small_token_count(self):
        """Test cost calculation with realistic small token counts."""
        # gemini-2.5-pro-preview-tts: $1.00/M input, $20.00/M output
        cost = calculate_tts_cost(
            "gemini-2.5-pro-preview-tts", input_tokens=500, output_tokens=10_000
        )
        expected = (500 / 1_000_000) * 1.00 + (10_000 / 1_000_000) * 20.00
        assert cost == pytest.approx(expected)


class TestChatCompletionParametersThinking:
    """Tests for thinking-related fields on ChatCompletionParameters."""

    def test_thinking_fields_default_none(self):
        """Test that thinking fields default to None."""
        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        assert params.thinking_level is None
        assert params.thinking_budget is None

    def test_thinking_level_set(self):
        """Test setting thinking_level."""
        params = ChatCompletionParameters(model="gemini-3-flash-preview", thinking_level="high")
        assert params.thinking_level == "high"
        assert params.thinking_budget is None

    def test_thinking_budget_set(self):
        """Test setting thinking_budget."""
        params = ChatCompletionParameters(model="gemini-2.5-flash", thinking_budget=1024)
        assert params.thinking_level is None
        assert params.thinking_budget == 1024

    def test_thinking_budget_zero(self):
        """Test thinking_budget=0 (disable thinking)."""
        params = ChatCompletionParameters(model="gemini-2.5-flash", thinking_budget=0)
        assert params.thinking_budget == 0

    def test_thinking_budget_dynamic(self):
        """Test thinking_budget=-1 (dynamic)."""
        params = ChatCompletionParameters(model="gemini-2.5-flash", thinking_budget=-1)
        assert params.thinking_budget == -1

    def test_both_thinking_params(self):
        """Test setting both thinking_level and thinking_budget."""
        params = ChatCompletionParameters(
            model="gemini-3-flash-preview", thinking_level="low", thinking_budget=512
        )
        assert params.thinking_level == "low"
        assert params.thinking_budget == 512
