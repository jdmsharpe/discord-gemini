import base64
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
        embed = send_kwargs["embeds"][0]

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


class TestMusicCostTracking(AsyncGeminiCogTestCase):
    """/music is billed per song; every generation must reach the daily cost ledger."""

    def _ctx(self):
        ctx = MagicMock()
        ctx.author.id = 123
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        return ctx

    async def _run_lyria3(self, model, audio=b"audio-bytes"):
        ctx = self._ctx()
        self.cog._log_cost = MagicMock()
        self.cog._send_error_followup = AsyncMock()

        with patch(
            "discord_gemini.cogs.gemini.music._generate_music_with_lyria3",
            AsyncMock(return_value=(audio, None, "audio/mpeg")),
        ):
            await music_command(
                self.cog,
                ctx,
                prompt="Dream pop song",
                attachment=None,
                model=model,
            )
        return ctx, self.cog._log_cost.call_args

    async def test_lyria3_clip_is_billed_per_song(self):
        _, call_args = await self._run_lyria3("lyria-3-clip-preview")

        assert call_args.args[3] == pytest.approx(0.04)
        assert call_args.args[4] == pytest.approx(0.04)
        assert "unpriced" not in call_args.kwargs

    async def test_lyria3_pro_is_billed_per_song(self):
        _, call_args = await self._run_lyria3("lyria-3-pro-preview")

        assert call_args.args[3] == pytest.approx(0.08)
        assert call_args.args[4] == pytest.approx(0.08)
        assert "unpriced" not in call_args.kwargs

    async def test_music_cost_reaches_the_daily_ledger(self):
        await self._run_lyria3("lyria-3-pro-preview")

        totals = [total for total, _ in self.cog.daily_costs.values()]
        assert totals == [pytest.approx(0.08)]

    async def test_text_only_response_is_not_billed(self):
        """Lyria 3 can answer with text and no audio — nothing was generated to bill."""
        _, call_args = await self._run_lyria3("lyria-3-clip-preview", audio=None)

        assert call_args.args[3] == pytest.approx(0.0)
        assert "unpriced" not in call_args.kwargs

    async def test_realtime_is_logged_as_unpriced(self):
        """lyria-realtime-exp has no published per-song price; never invent one."""
        ctx = self._ctx()
        self.cog._log_cost = MagicMock()
        self.cog._send_error_followup = AsyncMock()

        with patch(
            "discord_gemini.cogs.gemini.music._generate_music_with_lyria_realtime",
            AsyncMock(return_value=b"audio-bytes"),
        ):
            await music_command(
                self.cog,
                ctx,
                prompt="Ambient drone",
                attachment=None,
                model="lyria-realtime-exp",
                duration=30,
            )

        call_args = self.cog._log_cost.call_args
        assert call_args.args[3] == pytest.approx(0.0)
        assert call_args.kwargs["unpriced"] is True
        assert call_args.kwargs["duration_seconds"] == 30


class TestMusicAttachmentValidation(AsyncGeminiCogTestCase):
    async def test_validate_music_attachment_accepts_lyria3_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "image/png"
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-3-pro-preview", attachment)
        assert result is None

    async def test_validate_music_attachment_accepts_lyria35_image(self):
        attachment = MagicMock()
        attachment.size = 1024
        attachment.content_type = "image/png"
        attachment.filename = "cover.png"

        result = self.cog._validate_music_attachment("lyria-3.5", attachment)
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


class TestLyria35(AsyncGeminiCogTestCase):
    """Lyria 3.5 runs on the Interactions API and returns base64 MP3 audio plus text."""

    @staticmethod
    def _interaction(audio: bytes | None, text: str | None):
        output_audio = (
            SimpleNamespace(
                data=base64.b64encode(audio).decode("ascii"),
                mime_type="audio/mpeg",
                type="audio",
            )
            if audio is not None
            else None
        )
        return SimpleNamespace(output_audio=output_audio, output_text=text)

    async def test_generate_music_with_lyria35_uses_interactions_create(self):
        from discord_gemini.util import MusicGenerationParameters

        audio = b"ID3\x03fake-mp3"
        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=self._interaction(audio, "[[A0]]\nVerse one")
        )
        params = MusicGenerationParameters(prompts=["Dream pop song"], model="lyria-3.5", bpm=110)

        audio_data, text_response, mime_type = await self.cog._generate_music_with_lyria35(params)

        assert audio_data == audio
        assert text_response == "[[A0]]\nVerse one"
        assert mime_type == "audio/mpeg"
        call_kwargs = self.cog.client.aio.interactions.create.call_args.kwargs
        assert call_kwargs["model"] == "lyria-3.5"
        assert isinstance(call_kwargs["input"], str)
        assert "Tempo: 110 BPM." in call_kwargs["input"]
        assert "30-second" not in call_kwargs["input"]
        self.cog.client.aio.models.generate_content.assert_not_called()

    async def test_generate_music_with_lyria35_with_attachment_sends_an_image_part(self):
        from discord_gemini.util import MusicGenerationParameters

        image = Image.new("RGB", (2, 2), color="blue")
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        png = image_bytes.getvalue()

        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=self._interaction(None, None)
        )
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=png)
        attachment = MagicMock()
        attachment.filename = "reference.png"
        attachment.content_type = "image/png"
        params = MusicGenerationParameters(prompts=["Dream pop song"], model="lyria-3.5")

        await self.cog._generate_music_with_lyria35(params, attachment)

        parts = self.cog.client.aio.interactions.create.call_args.kwargs["input"]
        assert isinstance(parts, list) and len(parts) == 2
        assert parts[0]["type"] == "text" and "Dream pop song" in parts[0]["text"]
        assert parts[1] == {
            "type": "image",
            "mime_type": "image/png",
            "data": base64.b64encode(png).decode("ascii"),
        }

    async def test_generate_music_with_lyria35_returns_text_without_audio(self):
        from discord_gemini.util import MusicGenerationParameters

        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=self._interaction(None, "Lyrics only")
        )
        params = MusicGenerationParameters(prompts=["Dream pop song"], model="lyria-3.5")

        assert await self.cog._generate_music_with_lyria35(params) == (None, "Lyrics only", None)

    async def test_generate_music_with_lyria35_wraps_api_errors(self):
        from discord_gemini.util import MusicGenerationParameters

        self.cog.client.aio.interactions.create = AsyncMock(side_effect=RuntimeError("boom"))
        params = MusicGenerationParameters(prompts=["Dream pop song"], model="lyria-3.5")

        with pytest.raises(MusicGenerationError, match="Music generation failed: boom"):
            await self.cog._generate_music_with_lyria35(params)

    async def test_music_command_for_lyria35_bills_per_song_and_reports_song_mode(self):
        ctx = MagicMock()
        ctx.author.id = 123
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        self.cog._log_cost = MagicMock()
        self.cog._send_error_followup = AsyncMock()

        with (
            patch(
                "discord_gemini.cogs.gemini.music._generate_music_with_lyria35",
                AsyncMock(return_value=(b"audio-bytes", "Lyrics line", "audio/mpeg")),
            ) as lyria35,
            patch(
                "discord_gemini.cogs.gemini.music._generate_music_with_lyria3", AsyncMock()
            ) as lyria3,
            patch("discord_gemini.cogs.gemini.music.SHOW_COST_EMBEDS", True),
        ):
            await music_command(
                self.cog, ctx, prompt="Dream pop song", attachment=None, model="lyria-3.5"
            )

        lyria35.assert_awaited_once()
        lyria3.assert_not_called()
        sent_embeds = ctx.send_followup.await_args.kwargs["embeds"]
        embed = sent_embeds[0]
        assert "**Mode:** Song generation" in embed.description
        assert "**Format:** MP3" in embed.description
        assert sent_embeds[1].description == "$0.0800 · 1 song · $0.08 today"
        call_args = self.cog._log_cost.call_args
        assert call_args.args[3] == pytest.approx(0.08)
        assert "unpriced" not in call_args.kwargs
        assert "duration_seconds" not in call_args.kwargs

    def test_lyria35_is_priced_per_song(self):
        from discord_gemini.util import LYRIA_INTERACTIONS_MODELS, calculate_music_cost

        assert frozenset({"lyria-3.5"}) == LYRIA_INTERACTIONS_MODELS
        assert calculate_music_cost("lyria-3.5") == pytest.approx(0.08)


class TestMusicCostLine:
    def test_song_models_show_one_song(self):
        from discord_gemini.cogs.gemini.music import _music_cost_line

        assert _music_cost_line("lyria-3.5", 0.08, 0.12, 30) == "$0.0800 · 1 song · $0.12 today"

    def test_clip_model_shows_one_clip(self):
        from discord_gemini.cogs.gemini.music import _music_cost_line

        assert _music_cost_line("lyria-3-clip-preview", 0.04, 0.04, 30) == (
            "$0.0400 · 1 clip · $0.04 today"
        )

    def test_realtime_has_no_cost(self):
        from discord_gemini.cogs.gemini.music import _music_cost_line
        from discord_gemini.util import LYRIA_REALTIME_MODEL

        assert _music_cost_line(LYRIA_REALTIME_MODEL, None, 0.12, 45) == (
            "45s · no published price · $0.12 today"
        )
