from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from discord_gemini.cogs.gemini.music import (
    _build_lyria3_prompt,
    _build_music_notes_file,
    _music_file_suffix_for_mime_type,
    music_command,
)
from discord_gemini.cogs.gemini.responses import MusicGenerationError
from tests.support import AsyncGeminiCogTestCase


class TestLyriaHelpers:
    def test_build_lyria3_prompt_for_pro_omits_duration_guidance(self):
        from discord_gemini.util import MusicGenerationParameters

        params = MusicGenerationParameters(
            prompts=["Dreamy synthpop with warm vocals"],
            model="lyria-3-pro-preview",
            duration=90,
            bpm=120,
            scale="C_MAJOR_A_MINOR",
            density=0.4,
            brightness=0.6,
            guidance=5.0,
        )

        prompt = _build_lyria3_prompt(params)

        assert "Dreamy synthpop with warm vocals" in prompt
        assert "Target duration" not in prompt
        assert "Tempo: 120 BPM." in prompt
        assert "Musical key or scale: C MAJOR A MINOR." in prompt
        assert "Density: 0.4 on a 0 to 1 scale." in prompt
        assert "Brightness: 0.6 on a 0 to 1 scale." in prompt
        assert "Prompt adherence target: 5.0 on a 0 to 6 scale." in prompt

    def test_build_lyria3_prompt_for_clip_forces_30_second_note(self):
        from discord_gemini.util import MusicGenerationParameters

        params = MusicGenerationParameters(
            prompts=["Lo-fi beat"],
            model="lyria-3-clip-preview",
            duration=75,
        )

        prompt = _build_lyria3_prompt(params)

        assert "Generate a 30-second music clip." in prompt
        assert "Target duration" not in prompt

    def test_music_file_suffix_for_mime_type(self):
        assert _music_file_suffix_for_mime_type("audio/mpeg") == "mp3"
        assert _music_file_suffix_for_mime_type("audio/wav") == "wav"
        assert _music_file_suffix_for_mime_type("audio/opus") == "opus"
        assert _music_file_suffix_for_mime_type("audio/ogg") == "ogg"
        assert _music_file_suffix_for_mime_type("audio/alaw") == "alaw"
        assert _music_file_suffix_for_mime_type("audio/mulaw") == "mulaw"
        assert _music_file_suffix_for_mime_type(None) == "mp3"

    def test_build_music_notes_file_returns_none_when_short(self):
        assert _build_music_notes_file("Short notes") is None
        assert _build_music_notes_file(None) is None

    def test_build_music_notes_file_returns_attachment_when_truncated(self):
        notes_file = _build_music_notes_file("A" * 501)

        assert notes_file is not None
        assert notes_file.filename == "music_notes.txt"


class TestLyria3Generation(AsyncGeminiCogTestCase):
    async def test_generate_music_with_lyria3_uses_generate_content(self):
        from discord_gemini.util import MusicGenerationParameters

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="Lyrics line", inline_data=None),
                            SimpleNamespace(
                                text=None,
                                inline_data=SimpleNamespace(
                                    data=b"audio-bytes",
                                    mime_type="audio/mpeg",
                                ),
                            ),
                        ]
                    )
                )
            ]
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)

        params = MusicGenerationParameters(
            prompts=["Dream pop song"],
            model="lyria-3-pro-preview",
            duration=75,
            bpm=110,
        )

        audio_data, text_response, mime_type = await self.cog._generate_music_with_lyria3(params)

        assert audio_data == b"audio-bytes"
        assert text_response == "Lyrics line"
        assert mime_type == "audio/mpeg"

        call_kwargs = self.cog.client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "lyria-3-pro-preview"
        assert call_kwargs["config"].response_modalities == ["AUDIO", "TEXT"]
        assert "Target duration" not in call_kwargs["contents"]
        assert "Tempo: 110 BPM." in call_kwargs["contents"]

    async def test_music_command_for_lyria3_pro_omits_duration_in_embed_and_log(self):
        ctx = MagicMock()
        ctx.author.id = 123
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()

        self.cog._log_cost = MagicMock()
        self.cog._send_error_followup = AsyncMock()

        with patch(
            "discord_gemini.cogs.gemini.music._generate_music_with_lyria3",
            AsyncMock(return_value=(b"audio-bytes", "Lyrics line", "audio/mpeg")),
        ):
            await music_command(
                self.cog,
                ctx,
                prompt="Dream pop song",
                attachment=None,
                model="lyria-3-pro-preview",
                duration=75,
                bpm=110,
            )

        send_kwargs = ctx.send_followup.await_args.kwargs
        embed = send_kwargs["embed"]

        assert "**Target Duration:**" not in embed.description
        assert "**Mode:** Song generation" in embed.description
        assert "duration_seconds" not in self.cog._log_cost.call_args.kwargs

    async def test_generate_music_with_lyria3_with_attachment_uses_multimodal_contents(self):
        from discord_gemini.util import MusicGenerationParameters

        image = Image.new("RGB", (2, 2), color="blue")
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")

        response = SimpleNamespace(candidates=[])
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=image_bytes.getvalue())

        attachment = MagicMock()
        attachment.filename = "reference.png"
        attachment.content_type = "image/png"

        params = MusicGenerationParameters(
            prompts=["Dream pop song"],
            model="lyria-3-pro-preview",
        )

        await self.cog._generate_music_with_lyria3(params, attachment)

        call_kwargs = self.cog.client.aio.models.generate_content.call_args.kwargs
        assert isinstance(call_kwargs["contents"], list)
        assert len(call_kwargs["contents"]) == 2
        assert isinstance(call_kwargs["contents"][0], str)
        assert isinstance(call_kwargs["contents"][1], Image.Image)

    async def test_generate_music_with_lyria3_returns_text_without_audio(self):
        from discord_gemini.util import MusicGenerationParameters

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="Only lyrics", inline_data=None)]
                    )
                )
            ]
        )
        self.cog.client.aio.models.generate_content = AsyncMock(return_value=response)

        params = MusicGenerationParameters(
            prompts=["Minimal piano interlude"],
            model="lyria-3-clip-preview",
        )

        audio_data, text_response, mime_type = await self.cog._generate_music_with_lyria3(params)

        assert audio_data is None
        assert text_response == "Only lyrics"
        assert mime_type is None

    async def test_build_lyria3_music_contents_invalid_attachment_raises(self):
        from discord_gemini.util import MusicGenerationParameters

        self.cog._fetch_attachment_bytes = AsyncMock(return_value=b"not-an-image")
        attachment = MagicMock()
        attachment.filename = "bad.png"
        attachment.content_type = "image/png"

        params = MusicGenerationParameters(
            prompts=["Dream pop song"],
            model="lyria-3-pro-preview",
        )

        with pytest.raises(MusicGenerationError):
            await self.cog._build_lyria3_music_contents(params, attachment)


class TestMusicAttachmentValidation(AsyncGeminiCogTestCase):
    async def test_validate_music_attachment_accepts_lyria3_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "image/png"
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-3-pro-preview", attachment)
        assert result is None

    async def test_validate_music_attachment_rejects_realtime_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "image/png"
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-realtime-exp", attachment)
        assert result is not None
        assert "Lyria 3 Pro Preview" in result

    async def test_validate_music_attachment_rejects_non_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "audio/mpeg"
        attachment.filename = "clip.mp3"

        result = self.cog._validate_music_attachment("lyria-3-pro-preview", attachment)
        assert result == "Music reference attachments must be image files."

    async def test_validate_music_attachment_uses_filename_when_content_type_missing(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = None
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-3-clip-preview", attachment)
        assert result is None
