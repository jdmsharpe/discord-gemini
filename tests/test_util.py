import unittest
from util import (
    ATTACHMENT_FILE_API_MAX_SIZE,
    ATTACHMENT_FILE_API_THRESHOLD,
    ATTACHMENT_MAX_INLINE_SIZE,
    ATTACHMENT_PDF_MAX_INLINE_SIZE,
    CACHE_MIN_TOKEN_COUNT,
    CACHE_TTL,
    FILE_SEARCH_INCOMPATIBLE_TOOLS,
    MODEL_PRICING,
    TOOL_CODE_EXECUTION,
    TOOL_FILE_SEARCH,
    TOOL_GOOGLE_MAPS,
    TOOL_GOOGLE_SEARCH,
    TOOL_URL_CONTEXT,
    ChatCompletionParameters,
    calculate_cost,
    filter_file_search_incompatible_tools,
    filter_supported_tools_for_model,
    resolve_tool_name,
    ImageGenerationParameters,
    VideoGenerationParameters,
    SpeechGenerationParameters,
    MusicGenerationParameters,
    ResearchParameters,
    EmbeddingParameters,
    chunk_text,
)


class TestChatCompletionParameters(unittest.TestCase):
    def test_default_values(self):
        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        self.assertEqual(params.model, "gemini-3-flash-preview")
        self.assertIsNone(params.system_instruction)
        self.assertIsNone(params.frequency_penalty)
        self.assertIsNone(params.presence_penalty)
        self.assertIsNone(params.seed)
        self.assertIsNone(params.temperature)
        self.assertIsNone(params.top_p)
        self.assertIsNone(params.conversation_starter)
        self.assertIsNone(params.conversation_id)
        self.assertIsNone(params.channel_id)
        self.assertFalse(params.paused)
        self.assertEqual(params.history, [])
        self.assertEqual(params.tools, [])
        self.assertIsNone(params.media_resolution)
        self.assertIsNone(params.thinking_level)
        self.assertIsNone(params.thinking_budget)
        self.assertIsNone(params.cache_name)
        self.assertEqual(params.cached_history_length, 0)
        self.assertEqual(params.uploaded_file_names, [])

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
        self.assertEqual(params.model, "gemini-2.5-flash")
        self.assertEqual(params.system_instruction, "You are a helpful assistant.")
        self.assertEqual(params.frequency_penalty, 0.5)
        self.assertEqual(params.presence_penalty, 0.3)
        self.assertEqual(params.seed, 42)
        self.assertEqual(params.temperature, 0.8)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.media_resolution, "MEDIA_RESOLUTION_HIGH")
        self.assertEqual(params.conversation_id, 123456)
        self.assertEqual(params.channel_id, 789012)
        self.assertTrue(params.paused)
        self.assertEqual(
            params.tools,
            [
                TOOL_GOOGLE_SEARCH,
                TOOL_CODE_EXECUTION,
                TOOL_GOOGLE_MAPS,
                TOOL_URL_CONTEXT,
            ],
        )

    def test_uploaded_file_names_default_isolated(self):
        """Test that uploaded_file_names list is isolated between instances."""
        params_one = ChatCompletionParameters(model="gemini-3-flash-preview")
        params_one.uploaded_file_names.append("files/abc123")
        params_two = ChatCompletionParameters(model="gemini-3-flash-preview")
        self.assertEqual(params_two.uploaded_file_names, [])
        self.assertIsNot(params_one.uploaded_file_names, params_two.uploaded_file_names)

    def test_history_default_isolated(self):
        """Test that history list is isolated between instances."""
        params_one = ChatCompletionParameters(model="gemini-3-flash-preview")
        params_one.history.append({"role": "user", "parts": [{"text": "hello"}]})
        params_two = ChatCompletionParameters(model="gemini-3-flash-preview")
        self.assertEqual(params_two.history, [])
        self.assertIsNot(params_one.history, params_two.history)

    def test_tools_default_isolated(self):
        """Test that tools list is isolated between instances."""
        params_one = ChatCompletionParameters(model="gemini-3-flash-preview")
        params_one.tools.append(TOOL_GOOGLE_SEARCH)
        params_two = ChatCompletionParameters(model="gemini-3-flash-preview")
        self.assertEqual(params_two.tools, [])
        self.assertIsNot(params_one.tools, params_two.tools)

    def test_filter_supported_tools_for_model(self):
        """Test model-based filtering for tool compatibility."""
        tools = [
            TOOL_GOOGLE_SEARCH,
            TOOL_CODE_EXECUTION,
            TOOL_GOOGLE_MAPS,
            TOOL_URL_CONTEXT,
        ]
        supported, unsupported = filter_supported_tools_for_model(
            "gemini-3-flash-preview", tools
        )
        self.assertEqual(
            supported,
            [TOOL_GOOGLE_SEARCH, TOOL_CODE_EXECUTION, TOOL_URL_CONTEXT],
        )
        self.assertEqual(unsupported, ["google_maps"])

    def test_filter_supported_tools_file_search_supported_model(self):
        """Test that file_search is supported on compatible models."""
        tools = [TOOL_FILE_SEARCH]
        supported, unsupported = filter_supported_tools_for_model(
            "gemini-2.5-pro", tools
        )
        self.assertEqual(supported, [TOOL_FILE_SEARCH])
        self.assertEqual(unsupported, [])

    def test_filter_supported_tools_file_search_unsupported_model(self):
        """Test that file_search is filtered out for unsupported models."""
        tools = [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION]
        supported, unsupported = filter_supported_tools_for_model(
            "gemini-2.0-flash", tools
        )
        self.assertEqual(supported, [TOOL_CODE_EXECUTION])
        self.assertEqual(unsupported, ["file_search"])


class TestResolveToolName(unittest.TestCase):
    def test_resolve_standard_tools(self):
        """Test resolving standard tool configs to names."""
        self.assertEqual(resolve_tool_name(TOOL_GOOGLE_SEARCH), "google_search")
        self.assertEqual(resolve_tool_name(TOOL_CODE_EXECUTION), "code_execution")
        self.assertEqual(resolve_tool_name(TOOL_FILE_SEARCH), "file_search")

    def test_resolve_enriched_file_search(self):
        """Test resolving file_search config with injected store IDs."""
        enriched = {
            "file_search": {
                "file_search_store_names": ["store1", "store2"]
            }
        }
        self.assertEqual(resolve_tool_name(enriched), "file_search")

    def test_resolve_unknown_tool(self):
        """Test that unknown tool configs return None."""
        self.assertIsNone(resolve_tool_name({"unknown_tool": {}}))


class TestFilterFileSearchIncompatibleTools(unittest.TestCase):
    def test_no_file_search_returns_unchanged(self):
        """Test that tools are unchanged when file_search is not present."""
        tools = [TOOL_GOOGLE_SEARCH, TOOL_CODE_EXECUTION]
        filtered, removed = filter_file_search_incompatible_tools(tools)
        self.assertEqual(filtered, tools)
        self.assertEqual(removed, [])

    def test_file_search_removes_incompatible_tools(self):
        """Test that incompatible tools are removed when file_search is present."""
        tools = [
            TOOL_FILE_SEARCH,
            TOOL_GOOGLE_SEARCH,
            TOOL_CODE_EXECUTION,
            TOOL_URL_CONTEXT,
        ]
        filtered, removed = filter_file_search_incompatible_tools(tools)
        self.assertEqual(filtered, [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION])
        self.assertIn("google_search", removed)
        self.assertIn("url_context", removed)

    def test_file_search_alone_no_removals(self):
        """Test file_search with only compatible tools."""
        tools = [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION]
        filtered, removed = filter_file_search_incompatible_tools(tools)
        self.assertEqual(filtered, [TOOL_FILE_SEARCH, TOOL_CODE_EXECUTION])
        self.assertEqual(removed, [])

    def test_file_search_incompatible_tools_constant(self):
        """Test that the incompatible tools set contains expected tools."""
        self.assertIn("google_search", FILE_SEARCH_INCOMPATIBLE_TOOLS)
        self.assertIn("google_maps", FILE_SEARCH_INCOMPATIBLE_TOOLS)
        self.assertIn("url_context", FILE_SEARCH_INCOMPATIBLE_TOOLS)
        self.assertNotIn("code_execution", FILE_SEARCH_INCOMPATIBLE_TOOLS)
        self.assertNotIn("file_search", FILE_SEARCH_INCOMPATIBLE_TOOLS)


class TestImageGenerationParameters(unittest.TestCase):
    def test_to_dict_basic(self):
        params = ImageGenerationParameters(
            prompt="A house in the woods",
            model="imagen-3.0-generate-001",
            number_of_images=2,
            aspect_ratio="16:9",
        )
        result = params.to_dict()
        self.assertEqual(result["number_of_images"], 2)
        self.assertEqual(result["aspect_ratio"], "16:9")

    def test_to_dict_with_negative_prompt(self):
        params = ImageGenerationParameters(
            prompt="A sunset",
            model="imagen-4.0-generate-001",
            negative_prompt="blurry, low quality",
        )
        result = params.to_dict()
        self.assertEqual(result["negative_prompt"], "blurry, low quality")

    def test_to_dict_with_seed_and_guidance(self):
        params = ImageGenerationParameters(
            prompt="A mountain",
            model="imagen-3.0-generate-001",
            seed=42,
            guidance_scale=7.5,
        )
        result = params.to_dict()
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["guidance_scale"], 7.5)

    def test_person_generation_mapping(self):
        """Test that person_generation values are properly mapped."""
        # Test dont_allow
        params = ImageGenerationParameters(
            prompt="Test",
            model="imagen-3.0-generate-001",
            person_generation="dont_allow",
        )
        result = params.to_dict()
        self.assertEqual(result["person_generation"], "DONT_ALLOW")

        # Test allow_all
        params = ImageGenerationParameters(
            prompt="Test",
            model="imagen-3.0-generate-001",
            person_generation="allow_all",
        )
        result = params.to_dict()
        self.assertEqual(result["person_generation"], "ALLOW_ALL")

    def test_person_generation_allow_adult_excluded(self):
        """Test that allow_adult (default) is not included in output."""
        params = ImageGenerationParameters(
            prompt="Test",
            model="imagen-3.0-generate-001",
            person_generation="allow_adult",
        )
        result = params.to_dict()
        self.assertNotIn("person_generation", result)

    def test_none_values_excluded(self):
        """Test that None values are not included in to_dict output."""
        params = ImageGenerationParameters(
            prompt="Test",
            model="gemini-3-pro-image-preview",
        )
        result = params.to_dict()
        self.assertNotIn("aspect_ratio", result)
        self.assertNotIn("negative_prompt", result)
        self.assertNotIn("seed", result)
        self.assertNotIn("guidance_scale", result)

    def test_image_size_default_none(self):
        """Test that image_size defaults to None."""
        params = ImageGenerationParameters(prompt="Test", model="gemini-3.1-flash-image-preview")
        self.assertIsNone(params.image_size)

    def test_image_size_set(self):
        """Test that image_size can be set to a valid value."""
        params = ImageGenerationParameters(
            prompt="Test", model="gemini-3.1-flash-image-preview", image_size="2k"
        )
        self.assertEqual(params.image_size, "2k")

    def test_google_image_search_default_false(self):
        """Test that google_image_search defaults to False."""
        params = ImageGenerationParameters(prompt="Test", model="gemini-3.1-flash-image-preview")
        self.assertFalse(params.google_image_search)

    def test_google_image_search_enabled(self):
        """Test that google_image_search can be enabled."""
        params = ImageGenerationParameters(
            prompt="Test", model="gemini-3.1-flash-image-preview", google_image_search=True
        )
        self.assertTrue(params.google_image_search)

    def test_new_fields_isolation(self):
        """Test that image_size and google_image_search are independent across instances."""
        params1 = ImageGenerationParameters(
            prompt="A", model="gemini-3.1-flash-image-preview", image_size="1k", google_image_search=True
        )
        params2 = ImageGenerationParameters(prompt="B", model="gemini-3.1-flash-image-preview")
        self.assertEqual(params1.image_size, "1k")
        self.assertTrue(params1.google_image_search)
        self.assertIsNone(params2.image_size)
        self.assertFalse(params2.google_image_search)


class TestVideoGenerationParameters(unittest.TestCase):
    def test_to_dict_basic(self):
        params = VideoGenerationParameters(
            prompt="A cat playing piano",
            model="veo-2.0-generate-001",
            aspect_ratio="16:9",
            number_of_videos=2,
        )
        result = params.to_dict()
        self.assertEqual(result["aspect_ratio"], "16:9")
        self.assertEqual(result["number_of_videos"], 2)

    def test_to_dict_with_duration(self):
        params = VideoGenerationParameters(
            prompt="A sunset timelapse",
            model="veo-3.0-generate-001",
            duration_seconds=8,
        )
        result = params.to_dict()
        self.assertEqual(result["duration_seconds"], 8)

    def test_to_dict_with_negative_prompt(self):
        params = VideoGenerationParameters(
            prompt="A dog running",
            model="veo-2.0-generate-001",
            negative_prompt="blurry, distorted",
        )
        result = params.to_dict()
        self.assertEqual(result["negative_prompt"], "blurry, distorted")

    def test_enhance_prompt(self):
        params = VideoGenerationParameters(
            prompt="Test video",
            model="veo-3.1-generate-preview",
            enhance_prompt=True,
        )
        result = params.to_dict()
        self.assertTrue(result["enhance_prompt"])

        params_disabled = VideoGenerationParameters(
            prompt="Test video",
            model="veo-3.1-generate-preview",
            enhance_prompt=False,
        )
        result_disabled = params_disabled.to_dict()
        self.assertFalse(result_disabled["enhance_prompt"])

    def test_person_generation_mapping(self):
        """Test that person_generation values are properly mapped for Veo."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-2.0-generate-001",
            person_generation="dont_allow",
        )
        result = params.to_dict()
        self.assertEqual(result["person_generation"], "dont_allow")

    def test_person_generation_allow_adult_excluded(self):
        """Test that allow_adult (default) is not included in output."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-2.0-generate-001",
            person_generation="allow_adult",
        )
        result = params.to_dict()
        self.assertNotIn("person_generation", result)

    def test_none_values_excluded(self):
        """Test that None values are not included in to_dict output."""
        params = VideoGenerationParameters(
            prompt="Test",
            model="veo-2.0-generate-001",
        )
        result = params.to_dict()
        self.assertNotIn("aspect_ratio", result)
        self.assertNotIn("negative_prompt", result)
        self.assertNotIn("number_of_videos", result)
        self.assertNotIn("duration_seconds", result)
        self.assertNotIn("enhance_prompt", result)


class TestSpeechGenerationParameters(unittest.TestCase):
    def test_defaults(self):
        params = SpeechGenerationParameters(input_text="Hello world")
        self.assertEqual(params.input_text, "Hello world")
        self.assertEqual(params.model, "gemini-2.5-flash-preview-tts")
        self.assertEqual(params.voice_name, "Kore")
        self.assertFalse(params.multi_speaker)
        self.assertIsNone(params.speaker_configs)
        self.assertIsNone(params.style_prompt)

    def test_to_dict_single_speaker(self):
        params = SpeechGenerationParameters(
            input_text="Hello world",
            voice_name="Puck",
        )
        result = params.to_dict()
        self.assertEqual(result["response_modalities"], ["AUDIO"])
        self.assertIn("speech_config", result)
        self.assertIn("voice_config", result["speech_config"])
        self.assertEqual(
            result["speech_config"]["voice_config"]["prebuilt_voice_config"][
                "voice_name"
            ],
            "Puck",
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
        self.assertEqual(result["response_modalities"], ["AUDIO"])
        self.assertIn("speech_config", result)
        self.assertIn("multi_speaker_voice_config", result["speech_config"])
        speaker_voice_configs = result["speech_config"]["multi_speaker_voice_config"][
            "speaker_voice_configs"
        ]
        self.assertEqual(len(speaker_voice_configs), 2)
        self.assertEqual(speaker_voice_configs[0]["speaker"], "Joe")
        self.assertEqual(
            speaker_voice_configs[0]["voice_config"]["prebuilt_voice_config"][
                "voice_name"
            ],
            "Kore",
        )


class TestMusicGenerationParameters(unittest.TestCase):
    def test_defaults(self):
        params = MusicGenerationParameters(prompts=["upbeat jazz"])
        self.assertEqual(params.prompts, ["upbeat jazz"])
        self.assertIsNone(params.prompt_weights)
        self.assertEqual(params.duration, 30)
        self.assertIsNone(params.bpm)
        self.assertIsNone(params.scale)
        self.assertEqual(params.guidance, 4.0)
        self.assertIsNone(params.density)
        self.assertIsNone(params.brightness)
        self.assertEqual(params.temperature, 1.1)
        self.assertEqual(params.top_k, 40)
        self.assertIsNone(params.seed)
        self.assertFalse(params.mute_bass)
        self.assertFalse(params.mute_drums)
        self.assertFalse(params.only_bass_and_drums)

    def test_to_weighted_prompts_without_weights(self):
        params = MusicGenerationParameters(prompts=["jazz", "piano"])
        result = params.to_weighted_prompts()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"text": "jazz", "weight": 1.0})
        self.assertEqual(result[1], {"text": "piano", "weight": 1.0})

    def test_to_weighted_prompts_with_weights(self):
        params = MusicGenerationParameters(
            prompts=["jazz", "piano"],
            prompt_weights=[0.7, 0.3],
        )
        result = params.to_weighted_prompts()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"text": "jazz", "weight": 0.7})
        self.assertEqual(result[1], {"text": "piano", "weight": 0.3})

    def test_to_weighted_prompts_mismatched_weights(self):
        """Test that mismatched weights fall back to default 1.0."""
        params = MusicGenerationParameters(
            prompts=["jazz", "piano", "drums"],
            prompt_weights=[0.7, 0.3],  # Only 2 weights for 3 prompts
        )
        result = params.to_weighted_prompts()
        # Should fall back to default weights
        self.assertEqual(result[0]["weight"], 1.0)
        self.assertEqual(result[1]["weight"], 1.0)
        self.assertEqual(result[2]["weight"], 1.0)

    def test_to_music_config_basic(self):
        params = MusicGenerationParameters(prompts=["test"])
        result = params.to_music_config()
        self.assertEqual(result["guidance"], 4.0)
        self.assertEqual(result["temperature"], 1.1)
        self.assertEqual(result["top_k"], 40)
        self.assertFalse(result["mute_bass"])
        self.assertFalse(result["mute_drums"])
        self.assertFalse(result["only_bass_and_drums"])

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
        self.assertEqual(result["bpm"], 120)
        self.assertEqual(result["scale"], "C_MAJOR_A_MINOR")
        self.assertEqual(result["density"], 0.5)
        self.assertEqual(result["brightness"], 0.7)
        self.assertEqual(result["seed"], 42)

    def test_to_music_config_mute_options(self):
        params = MusicGenerationParameters(
            prompts=["test"],
            mute_bass=True,
            mute_drums=True,
        )
        result = params.to_music_config()
        self.assertTrue(result["mute_bass"])
        self.assertTrue(result["mute_drums"])

    def test_to_music_config_only_bass_and_drums(self):
        params = MusicGenerationParameters(
            prompts=["test"],
            only_bass_and_drums=True,
        )
        result = params.to_music_config()
        self.assertTrue(result["only_bass_and_drums"])


class TestResearchParameters(unittest.TestCase):
    def test_defaults(self):
        params = ResearchParameters(prompt="Research quantum computing")
        self.assertEqual(params.prompt, "Research quantum computing")
        self.assertEqual(params.agent, "deep-research-pro-preview-12-2025")
        self.assertFalse(params.file_search)

    def test_all_parameters(self):
        params = ResearchParameters(
            prompt="Compare EV batteries",
            agent="deep-research-pro-preview-12-2025",
            file_search=True,
        )
        self.assertEqual(params.prompt, "Compare EV batteries")
        self.assertEqual(params.agent, "deep-research-pro-preview-12-2025")
        self.assertTrue(params.file_search)


class TestEmbeddingParameters(unittest.TestCase):
    def test_defaults(self):
        params = EmbeddingParameters(prompt="Test text for embedding")
        self.assertEqual(params.prompt, "Test text for embedding")
        self.assertEqual(params.model, "gemini-embedding-exp")
        self.assertEqual(params.model_options, "gemini-embedding-exp")


class TestChunkText(unittest.TestCase):
    def test_chunk_text_small(self):
        text = "This is a test."
        size = 4
        result = chunk_text(text, size)
        self.assertEqual(
            result,
            ["This", " is ", "a te", "st."],
        )

    def test_chunk_text_exact_size(self):
        text = "abcd"
        size = 4
        result = chunk_text(text, size)
        self.assertEqual(result, ["abcd"])

    def test_chunk_text_larger_than_text(self):
        text = "Hello"
        size = 100
        result = chunk_text(text, size)
        self.assertEqual(result, ["Hello"])

    def test_chunk_text_empty(self):
        text = ""
        size = 4
        result = chunk_text(text, size)
        self.assertEqual(result, [])

    def test_chunk_text_default_size(self):
        text = "a" * 8192
        result = chunk_text(text)  # Default size is 4096
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 4096)
        self.assertEqual(len(result[1]), 4096)

    def test_chunk_text_long(self):
        text = "This is a test. " * 64  # len(text) * 64 = 1024
        size = 1024
        result = chunk_text(text, size)
        self.assertEqual(len(result[0]), size)


class TestTruncateText(unittest.TestCase):
    """Tests for the truncate_text utility function."""

    def test_truncate_text_under_limit(self):
        """Test that short text is not truncated."""
        from util import truncate_text

        text = "Short text"
        result = truncate_text(text, 100)
        self.assertEqual(result, text)
        self.assertNotIn("...", result)

    def test_truncate_text_over_limit(self):
        """Test that long text is truncated with suffix."""
        from util import truncate_text

        text = "A" * 100
        result = truncate_text(text, 50)
        self.assertEqual(len(result), 53)  # 50 + "..."
        self.assertTrue(result.endswith("..."))
        self.assertEqual(result, "A" * 50 + "...")

    def test_truncate_text_at_limit(self):
        """Test that text exactly at limit is not truncated."""
        from util import truncate_text

        text = "B" * 100
        result = truncate_text(text, 100)
        self.assertEqual(result, text)
        self.assertNotIn("...", result)

    def test_truncate_text_custom_suffix(self):
        """Test truncation with custom suffix."""
        from util import truncate_text

        text = "Hello, world!"
        result = truncate_text(text, 5, suffix="[...]")
        self.assertEqual(result, "Hello[...]")

    def test_truncate_text_none(self):
        """Test that None input returns None."""
        from util import truncate_text

        result = truncate_text(None, 100)
        self.assertIsNone(result)

    def test_truncate_text_empty_string(self):
        """Test that empty string is handled correctly."""
        from util import truncate_text

        result = truncate_text("", 100)
        self.assertEqual(result, "")

    def test_truncate_text_empty_suffix(self):
        """Test truncation with empty suffix."""
        from util import truncate_text

        text = "Hello, world!"
        result = truncate_text(text, 5, suffix="")
        self.assertEqual(result, "Hello")
        self.assertEqual(len(result), 5)

    def test_truncate_text_prompt_limit(self):
        """Test standard 2000 char prompt truncation."""
        from util import truncate_text

        long_prompt = "A" * 3000
        result = truncate_text(long_prompt, 2000)
        self.assertEqual(len(result), 2003)  # 2000 + "..."
        self.assertTrue(result.endswith("..."))

    def test_truncate_text_response_limit(self):
        """Test standard 3500 char response truncation."""
        from util import truncate_text

        long_response = "B" * 5000
        result = truncate_text(long_response, 3500)
        self.assertEqual(len(result), 3503)  # 3500 + "..."
        self.assertTrue(result.endswith("..."))


class TestCacheConstants(unittest.TestCase):
    """Tests for explicit caching constants."""

    def test_cache_min_token_count_contains_expected_models(self):
        """Test that CACHE_MIN_TOKEN_COUNT includes only Gemini 3.x models."""
        self.assertIn("gemini-3.1-pro-preview", CACHE_MIN_TOKEN_COUNT)
        self.assertIn("gemini-3-flash-preview", CACHE_MIN_TOKEN_COUNT)

    def test_cache_min_token_count_values(self):
        """Test that token thresholds are correct per model tier."""
        self.assertEqual(CACHE_MIN_TOKEN_COUNT["gemini-3.1-pro-preview"], 4096)
        self.assertEqual(CACHE_MIN_TOKEN_COUNT["gemini-3-flash-preview"], 1024)

    def test_cache_min_token_count_excludes_implicit_only_models(self):
        """Test that 2.5 and below models rely on implicit caching."""
        self.assertNotIn("gemini-2.5-pro", CACHE_MIN_TOKEN_COUNT)
        self.assertNotIn("gemini-2.5-flash", CACHE_MIN_TOKEN_COUNT)
        self.assertNotIn("gemini-2.0-flash", CACHE_MIN_TOKEN_COUNT)
        self.assertNotIn("gemini-2.0-flash-lite", CACHE_MIN_TOKEN_COUNT)

    def test_cache_ttl_is_valid(self):
        """Test that CACHE_TTL is a valid duration string."""
        self.assertEqual(CACHE_TTL, "1800s")


class TestChatCompletionParametersCaching(unittest.TestCase):
    """Tests for cache-related fields on ChatCompletionParameters."""

    def test_cache_fields_with_values(self):
        """Test that cache fields can be set."""
        params = ChatCompletionParameters(
            model="gemini-2.5-flash",
            cache_name="cachedContents/abc123",
            cached_history_length=4,
        )
        self.assertEqual(params.cache_name, "cachedContents/abc123")
        self.assertEqual(params.cached_history_length, 4)

    def test_cache_fields_reset(self):
        """Test that cache fields can be cleared."""
        params = ChatCompletionParameters(
            model="gemini-2.5-flash",
            cache_name="cachedContents/abc123",
            cached_history_length=4,
        )
        params.cache_name = None
        params.cached_history_length = 0
        self.assertIsNone(params.cache_name)
        self.assertEqual(params.cached_history_length, 0)


class TestAttachmentSizeConstants(unittest.TestCase):
    """Tests for attachment size limit constants."""

    def test_inline_max_size(self):
        """Test that inline data max is 100 MB."""
        self.assertEqual(ATTACHMENT_MAX_INLINE_SIZE, 100 * 1024 * 1024)

    def test_pdf_inline_max_size(self):
        """Test that PDF inline max is 50 MB."""
        self.assertEqual(ATTACHMENT_PDF_MAX_INLINE_SIZE, 50 * 1024 * 1024)

    def test_file_api_threshold(self):
        """Test that File API threshold is 20 MB."""
        self.assertEqual(ATTACHMENT_FILE_API_THRESHOLD, 20 * 1024 * 1024)

    def test_file_api_max_size(self):
        """Test that File API max is 2 GB."""
        self.assertEqual(ATTACHMENT_FILE_API_MAX_SIZE, 2 * 1024 * 1024 * 1024)

    def test_threshold_ordering(self):
        """Test that size limits are in correct ascending order."""
        self.assertLess(ATTACHMENT_FILE_API_THRESHOLD, ATTACHMENT_PDF_MAX_INLINE_SIZE)
        self.assertLess(ATTACHMENT_PDF_MAX_INLINE_SIZE, ATTACHMENT_MAX_INLINE_SIZE)
        self.assertLess(ATTACHMENT_MAX_INLINE_SIZE, ATTACHMENT_FILE_API_MAX_SIZE)


class TestModelPricing(unittest.TestCase):
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
            self.assertIn(model, MODEL_PRICING, f"{model} missing from MODEL_PRICING")

    def test_pricing_values_are_positive(self):
        """Test that all pricing values are positive."""
        for model, (input_price, output_price) in MODEL_PRICING.items():
            self.assertGreater(input_price, 0, f"{model} input price should be positive")
            self.assertGreater(output_price, 0, f"{model} output price should be positive")

    def test_pricing_output_greater_than_input(self):
        """Test that output pricing is always >= input pricing."""
        for model, (input_price, output_price) in MODEL_PRICING.items():
            self.assertGreaterEqual(
                output_price, input_price,
                f"{model} output price should be >= input price"
            )

    def test_calculate_cost_known_model(self):
        """Test cost calculation for a known model."""
        # gemini-2.0-flash: $0.10/M input, $0.40/M output
        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.50)  # $0.10 + $0.40

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_cost("gemini-2.5-pro", 0, 0)
        self.assertAlmostEqual(cost, 0.0)

    def test_calculate_cost_unknown_model_uses_default(self):
        """Test that unknown models fall back to default pricing."""
        cost = calculate_cost("unknown-model", 1_000_000, 1_000_000)
        # Default is (2.0, 12.0)
        self.assertAlmostEqual(cost, 14.0)

    def test_calculate_cost_small_token_count(self):
        """Test cost calculation with realistic small token counts."""
        # gemini-3.1-pro-preview: $2.00/M input, $12.00/M output
        cost = calculate_cost("gemini-3.1-pro-preview", 500, 200)
        expected = (500 / 1_000_000) * 2.0 + (200 / 1_000_000) * 12.0
        self.assertAlmostEqual(cost, expected)

    def test_calculate_cost_with_thinking_tokens(self):
        """Test that thinking tokens are billed at the output token rate."""
        # gemini-2.5-flash: $0.30/M input, $2.50/M output
        cost = calculate_cost("gemini-2.5-flash", 1_000_000, 500_000, thinking_tokens=500_000)
        # 1M * $0.30 + (500K + 500K) * $2.50 = $0.30 + $2.50 = $2.80
        self.assertAlmostEqual(cost, 2.80)

    def test_calculate_cost_with_zero_thinking_tokens(self):
        """Test that zero thinking tokens doesn't affect cost."""
        cost_without = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
        cost_with = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000, thinking_tokens=0)
        self.assertAlmostEqual(cost_without, cost_with)

    def test_calculate_cost_thinking_only(self):
        """Test cost when output is mostly thinking tokens."""
        # gemini-3-flash-preview: $0.50/M input, $3.00/M output
        cost = calculate_cost("gemini-3-flash-preview", 100_000, 50_000, thinking_tokens=1_000_000)
        expected = (100_000 / 1_000_000) * 0.50 + ((50_000 + 1_000_000) / 1_000_000) * 3.0
        self.assertAlmostEqual(cost, expected)


class TestChatCompletionParametersThinking(unittest.TestCase):
    """Tests for thinking-related fields on ChatCompletionParameters."""

    def test_thinking_fields_default_none(self):
        """Test that thinking fields default to None."""
        params = ChatCompletionParameters(model="gemini-3-flash-preview")
        self.assertIsNone(params.thinking_level)
        self.assertIsNone(params.thinking_budget)

    def test_thinking_level_set(self):
        """Test setting thinking_level."""
        params = ChatCompletionParameters(
            model="gemini-3-flash-preview", thinking_level="high"
        )
        self.assertEqual(params.thinking_level, "high")
        self.assertIsNone(params.thinking_budget)

    def test_thinking_budget_set(self):
        """Test setting thinking_budget."""
        params = ChatCompletionParameters(
            model="gemini-2.5-flash", thinking_budget=1024
        )
        self.assertIsNone(params.thinking_level)
        self.assertEqual(params.thinking_budget, 1024)

    def test_thinking_budget_zero(self):
        """Test thinking_budget=0 (disable thinking)."""
        params = ChatCompletionParameters(
            model="gemini-2.5-flash", thinking_budget=0
        )
        self.assertEqual(params.thinking_budget, 0)

    def test_thinking_budget_dynamic(self):
        """Test thinking_budget=-1 (dynamic)."""
        params = ChatCompletionParameters(
            model="gemini-2.5-flash", thinking_budget=-1
        )
        self.assertEqual(params.thinking_budget, -1)

    def test_both_thinking_params(self):
        """Test setting both thinking_level and thinking_budget."""
        params = ChatCompletionParameters(
            model="gemini-3-flash-preview", thinking_level="low", thinking_budget=512
        )
        self.assertEqual(params.thinking_level, "low")
        self.assertEqual(params.thinking_budget, 512)


if __name__ == "__main__":
    unittest.main()
