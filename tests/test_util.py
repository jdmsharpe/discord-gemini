import unittest
from util import (
    ChatCompletionParameters,
    ImageGenerationParameters,
    VideoGenerationParameters,
    SpeechGenerationParameters,
    MusicGenerationParameters,
    EmbeddingParameters,
    chunk_text,
)


class TestChatCompletionParameters(unittest.TestCase):
    def test_default_values(self):
        params = ChatCompletionParameters(model="gemini-3-pro-preview")
        self.assertEqual(params.model, "gemini-3-pro-preview")
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

    def test_all_parameters(self):
        params = ChatCompletionParameters(
            model="gemini-2.5-flash",
            system_instruction="You are a helpful assistant.",
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=42,
            temperature=0.8,
            top_p=0.9,
            conversation_id=123456,
            channel_id=789012,
            paused=True,
        )
        self.assertEqual(params.model, "gemini-2.5-flash")
        self.assertEqual(params.system_instruction, "You are a helpful assistant.")
        self.assertEqual(params.frequency_penalty, 0.5)
        self.assertEqual(params.presence_penalty, 0.3)
        self.assertEqual(params.seed, 42)
        self.assertEqual(params.temperature, 0.8)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.conversation_id, 123456)
        self.assertEqual(params.channel_id, 789012)
        self.assertTrue(params.paused)

    def test_history_default_isolated(self):
        """Test that history list is isolated between instances."""
        params_one = ChatCompletionParameters(model="gemini-3-pro-preview")
        params_one.history.append({"role": "user", "parts": [{"text": "hello"}]})
        params_two = ChatCompletionParameters(model="gemini-3-pro-preview")
        self.assertEqual(params_two.history, [])
        self.assertIsNot(params_one.history, params_two.history)


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


if __name__ == "__main__":
    unittest.main()
