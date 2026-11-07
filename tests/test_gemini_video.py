import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from discord_gemini.cogs.gemini.cog import GeminiCog
from discord_gemini.cogs.gemini.command_options import VIDEO_MODEL_CHOICES
from discord_gemini.cogs.gemini.responses import APICallError
from discord_gemini.cogs.gemini.video import (
    DEFAULT_OMNI_VIDEO_MODEL,
    OMNI_VIDEO_MODELS,
    OMNI_VIDEO_POLL_INTERVAL,
    VIDEO_SUPPORTED_RESOLUTIONS,
    OmniVideoUsage,
    _attachment_image_mime_type,
    _create_video_response_embed,
    _estimate_omni_video_bytes,
    _extract_omni_video_usage,
    _generate_video_with_omni,
    _validate_omni_video_request,
)
from discord_gemini.cost_line import format_daily_total, format_request_cost
from discord_gemini.util import (
    VIDEO_GENERATION_TIMEOUT,
    VIDEO_TOKEN_PRICING,
    VideoGenerationParameters,
    calculate_omni_video_cost,
)
from tests.support import AsyncGeminiCogTestCase, build_mock_bot

OMNI_11 = "gemini-omni-1.1-flash"
OMNI_12 = "gemini-omni-1.2-flash"


def _encoded_image(fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(buf, format=fmt)
    return buf.getvalue()


class TestAttachmentImageMimeType:
    """Frame images retain their declared or detected MIME types."""

    @staticmethod
    def _attachment(content_type):
        attachment = MagicMock()
        attachment.content_type = content_type
        return attachment

    @pytest.mark.parametrize(
        ("fmt", "content_type", "expected_mime"),
        [
            ("WEBP", "image/webp", "image/webp"),
            ("PNG", "image/png", "image/png"),
            ("JPEG", "image/jpeg; charset=binary", "image/jpeg"),
        ],
    )
    def test_uses_attachment_content_type(self, fmt, content_type, expected_mime):
        data = _encoded_image(fmt)
        image = _attachment_image_mime_type(data, self._attachment(content_type))
        assert image == expected_mime

    def test_falls_back_to_detected_format(self):
        data = _encoded_image("WEBP")
        image = _attachment_image_mime_type(data, self._attachment(None))
        assert image == "image/webp"

    def test_ignores_non_image_content_type(self):
        data = _encoded_image("PNG")
        image = _attachment_image_mime_type(data, self._attachment("application/octet-stream"))
        assert image == "image/png"


class TestVideoResponseEmbed(AsyncGeminiCogTestCase):
    async def _assert_video_mode(self, params, expected_mode, attachment=None):
        embed, files = await self.cog._create_video_response_embed(
            video_params=params,
            generated_videos=[b"fake-video-bytes"],
            attachment=attachment,
        )
        for file in files:
            file.close()
        assert expected_mode in embed.description
        assert len(files) == 1

    async def test_mode_text_to_video(self):
        """Test embed shows Text-to-Video mode when no attachments."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(prompt="A sunset", model=DEFAULT_OMNI_VIDEO_MODEL)
        await self._assert_video_mode(params, "Text-to-Video")

    async def test_mode_image_to_video(self):
        """Test embed shows Image-to-Video mode when attachment provided."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(prompt="A sunset", model=DEFAULT_OMNI_VIDEO_MODEL)
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Image-to-Video", attachment=mock_attachment)

    async def test_mode_interpolation(self):
        """Test embed shows Interpolation mode when both attachment and last_frame."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model=DEFAULT_OMNI_VIDEO_MODEL,
            has_last_frame=True,
        )
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Interpolation", attachment=mock_attachment)

    async def test_embed_includes_resolution(self):
        """Test embed includes the selected output resolution."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model=DEFAULT_OMNI_VIDEO_MODEL,
            resolution="1080p",
        )
        embed, files = await self.cog._create_video_response_embed(
            video_params=params,
            generated_videos=[b"fake-video-bytes"],
            attachment=None,
        )
        for file in files:
            file.close()

        assert "**Resolution:** 1080p" in embed.description


def _fake_omni_interaction(
    uri: str | None = "https://x/v1beta/files/abc123:download?alt=media",
    video_tokens: int = 57920,
    status: str = "completed",
    errors=None,
    data: str | None = None,
    input_tokens: int = 0,
    thought_tokens: int = 0,
):
    usage = SimpleNamespace(
        output_tokens_by_modality=[SimpleNamespace(modality="video", tokens=video_tokens)],
        total_input_tokens=input_tokens,
        total_output_tokens=video_tokens,
        total_thought_tokens=thought_tokens,
    )
    interaction = MagicMock()
    interaction.id = "omni-1"
    interaction.status = status
    interaction.errors = errors
    interaction.usage = usage
    if uri is None and data is None:
        interaction.output_video = None
    else:
        interaction.output_video = SimpleNamespace(uri=uri, data=data, mime_type="video/mp4")
    return interaction


def _in_progress():
    return _fake_omni_interaction(uri=None, video_tokens=0, status="in_progress")


class TestOmniVideoGeneration(AsyncGeminiCogTestCase):
    """Omni must run in background mode and poll, like research.

    A synchronous `interactions.create` on the GA gemini-omni-1.1-flash was closed
    server-side after 60.26 s twice ("Server disconnected without sending a
    response", no id returned); `background=True` returned an id in 0.79 s and
    completed after 55.7 s of `interactions.get` polling (probe 2026-08-28).
    """

    SLEEP = "discord_gemini.cogs.gemini.video.asyncio.sleep"

    @staticmethod
    def _params(**kwargs):
        base = {
            "prompt": "A red ball rolls across a table",
            "model": DEFAULT_OMNI_VIDEO_MODEL,
            "aspect_ratio": "16:9",
        }
        base.update(kwargs)
        return VideoGenerationParameters(**base)

    async def test_downloads_video_and_returns_tokens(self):
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_in_progress())
        self.cog.client.aio.interactions.get = AsyncMock(
            side_effect=[_in_progress(), _fake_omni_interaction()]
        )
        self.cog.client.files.download = MagicMock(return_value=b"mp4-bytes")

        with patch(self.SLEEP, new_callable=AsyncMock) as sleep:
            videos, usage = await _generate_video_with_omni(self.cog, self._params())

        assert videos == [b"mp4-bytes"]
        assert usage.video_output_tokens == 57920
        _, kwargs = self.cog.client.aio.interactions.create.call_args
        assert kwargs["model"] == DEFAULT_OMNI_VIDEO_MODEL
        assert kwargs["input"] == "A red ball rolls across a table"
        assert kwargs["background"] is True
        # No resolution requested -> the key is absent, so the API default (720p) applies.
        assert kwargs["response_format"] == {
            "type": "video",
            "aspect_ratio": "16:9",
            "delivery": "uri",
        }
        # Polled by id every OMNI_VIDEO_POLL_INTERVAL seconds until `completed`.
        assert self.cog.client.aio.interactions.get.await_args_list == [
            call("omni-1"),
            call("omni-1"),
        ]
        assert sleep.await_args_list == [call(OMNI_VIDEO_POLL_INTERVAL)] * 2
        assert OMNI_VIDEO_POLL_INTERVAL == 5
        # File name is parsed from the URI, not the raw URI.
        _, dl_kwargs = self.cog.client.files.download.call_args
        assert dl_kwargs["file"] == "files/abc123"

    async def test_requested_resolution_is_passed_through(self):
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_fake_omni_interaction())
        self.cog.client.files.download = MagicMock(return_value=b"mp4-bytes")

        await _generate_video_with_omni(self.cog, self._params(resolution="1080p"))

        _, kwargs = self.cog.client.aio.interactions.create.call_args
        assert kwargs["response_format"]["resolution"] == "1080p"
        self.cog.client.aio.interactions.get.assert_not_called()

    @pytest.mark.parametrize(
        ("status", "expected"),
        [("failed", "failed: Upstream rendering error"), ("cancelled", "cancelled: Upstream")],
    )
    async def test_terminal_failure_surfaces_the_interaction_error(self, status, expected):
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_in_progress())
        self.cog.client.aio.interactions.get = AsyncMock(
            return_value=_fake_omni_interaction(
                uri=None,
                status=status,
                errors=[SimpleNamespace(code="internal", message="Upstream rendering error")],
            )
        )
        self.cog.client.files.download = MagicMock()

        with (
            patch(self.SLEEP, new_callable=AsyncMock),
            pytest.raises(APICallError, match=expected),
        ):
            await _generate_video_with_omni(self.cog, self._params())

        self.cog.client.files.download.assert_not_called()

    async def test_failed_without_error_detail_still_raises(self):
        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(uri=None, status="failed", errors=None)
        )

        with pytest.raises(APICallError, match="Omni video generation failed"):
            await _generate_video_with_omni(self.cog, self._params())

    async def test_times_out_after_video_generation_timeout(self):
        """The Omni poller is bounded by the same VIDEO_GENERATION_TIMEOUT as Veo."""
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_in_progress())
        self.cog.client.aio.interactions.get = AsyncMock(return_value=_in_progress())

        with (
            patch(self.SLEEP, new_callable=AsyncMock),
            patch("discord_gemini.cogs.gemini.video.time") as fake_time,
            pytest.raises(TimeoutError, match="timed out"),
        ):
            fake_time.time.side_effect = [0, 1, VIDEO_GENERATION_TIMEOUT + 1]
            await _generate_video_with_omni(self.cog, self._params())

        # One poll ran before the clock crossed the deadline; then it gave up.
        self.cog.client.aio.interactions.get.assert_awaited_once()

    async def test_missing_uri_returns_no_bytes(self):
        from discord_gemini.util import VideoGenerationParameters

        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(uri=None)
        )
        self.cog.client.files.download = MagicMock(return_value=b"unused")

        params = VideoGenerationParameters(prompt="x", model=DEFAULT_OMNI_VIDEO_MODEL)
        videos, usage = await _generate_video_with_omni(self.cog, params)

        assert videos == []
        assert usage.video_output_tokens == 57920
        self.cog.client.files.download.assert_not_called()

    async def test_polled_base64_data_is_decoded_when_there_is_no_uri(self):
        """`interactions.get` returns the video as base64 `data`; the `uri` is only
        guaranteed on the creation response (Omni docs)."""
        import base64

        self.cog.client.aio.interactions.create = AsyncMock(return_value=_in_progress())
        self.cog.client.aio.interactions.get = AsyncMock(
            return_value=_fake_omni_interaction(
                uri=None, data=base64.b64encode(b"inline-mp4").decode("ascii")
            )
        )
        self.cog.client.files.download = MagicMock()

        with patch(self.SLEEP, new_callable=AsyncMock):
            videos, usage = await _generate_video_with_omni(self.cog, self._params())

        assert videos == [b"inline-mp4"]
        assert usage.video_output_tokens == 57920
        self.cog.client.files.download.assert_not_called()

    async def test_failed_uri_download_falls_back_to_base64_data(self):
        import base64

        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(
                data=base64.b64encode(b"inline-mp4").decode("ascii")
            )
        )
        self.cog.client.files.download = MagicMock(side_effect=RuntimeError("403"))

        videos, _ = await _generate_video_with_omni(self.cog, self._params())

        assert videos == [b"inline-mp4"]
        self.cog.client.files.download.assert_called_once()

    async def test_uri_is_preferred_when_both_are_present(self):
        import base64

        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(
                data=base64.b64encode(b"inline-mp4").decode("ascii")
            )
        )
        self.cog.client.files.download = MagicMock(return_value=b"downloaded-mp4")

        videos, _ = await _generate_video_with_omni(self.cog, self._params())

        assert videos == [b"downloaded-mp4"]

    async def test_undecodable_data_returns_no_bytes(self):
        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(uri=None, data="not base64!")
        )

        videos, _ = await _generate_video_with_omni(self.cog, self._params())

        assert videos == []

    async def test_base64_data_with_line_breaks_is_decoded_leniently(self):
        """Strict decoding rejects line breaks; the lenient decoder still recovers the clip."""
        import base64

        payload = bytes(range(256)) * 4
        wrapped = base64.encodebytes(payload).decode("ascii")
        assert "\n" in wrapped
        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(uri=None, data=wrapped)
        )

        videos, _ = await _generate_video_with_omni(self.cog, self._params())

        assert videos == [payload]

    async def test_failed_uri_download_is_logged_with_its_error(self):
        """With a URI whose download failed and no inline data, the log names the URI
        and the download error rather than saying there was no URI."""
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_fake_omni_interaction())
        self.cog.client.files.download = MagicMock(side_effect=RuntimeError("403 Forbidden"))

        with patch.object(self.cog.logger, "error") as log_error:
            videos, usage = await _generate_video_with_omni(self.cog, self._params())

        assert videos == []
        assert usage.video_output_tokens == 57920
        message = log_error.call_args.args[0] % log_error.call_args.args[1:]
        assert "could not be downloaded" in message
        assert "download failed: 403 Forbidden" in message
        assert "files/abc123" in message
        assert "no video URI" not in message

    async def test_missing_uri_and_data_is_logged_as_such(self):
        self.cog.client.aio.interactions.create = AsyncMock(
            return_value=_fake_omni_interaction(uri=None)
        )

        with patch.object(self.cog.logger, "error") as log_error:
            await _generate_video_with_omni(self.cog, self._params())

        message = log_error.call_args.args[0] % log_error.call_args.args[1:]
        assert "no video URI and no decodable inline data" in message

    @pytest.mark.parametrize(
        ("prompt", "has_last_frame", "expected"),
        [
            ("ball rolls", True, "<FIRST_FRAME> <LAST_FRAME> ball rolls"),
            ("<FIRST_FRAME> ball rolls", True, "<LAST_FRAME> <FIRST_FRAME> ball rolls"),
            ("ball rolls to <LAST_FRAME>", True, "<FIRST_FRAME> ball rolls to <LAST_FRAME>"),
            (
                "<FIRST_FRAME> <LAST_FRAME> ball rolls",
                True,
                "<FIRST_FRAME> <LAST_FRAME> ball rolls",
            ),
            ("ball rolls", False, "<FIRST_FRAME> ball rolls"),
            ("<FIRST_FRAME> ball rolls", False, "<FIRST_FRAME> ball rolls"),
        ],
    )
    def test_each_frame_tag_is_added_only_when_missing(self, prompt, has_last_frame, expected):
        from discord_gemini.cogs.gemini.video import _tag_omni_frame_prompt

        tagged = _tag_omni_frame_prompt(prompt, has_last_frame)

        assert tagged == expected
        assert tagged.count("<FIRST_FRAME>") == 1
        assert tagged.count("<LAST_FRAME>") == (1 if has_last_frame else 0)

    @staticmethod
    def _image_attachment(fmt="PNG", content_type="image/png"):
        attachment = MagicMock()
        attachment.content_type = content_type
        return attachment, _encoded_image(fmt)

    async def test_first_frame_is_sent_as_an_image_part_before_the_tagged_prompt(self):
        import base64

        from google.genai import interactions
        from pydantic import TypeAdapter

        attachment, data = self._image_attachment()
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=data)
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_fake_omni_interaction())
        self.cog.client.files.download = MagicMock(return_value=b"mp4")

        await _generate_video_with_omni(self.cog, self._params(), attachment)

        sent = self.cog.client.aio.interactions.create.call_args.kwargs["input"]
        assert sent == [
            {
                "type": "image",
                "data": base64.b64encode(data).decode("ascii"),
                "mime_type": "image/png",
            },
            {"type": "text", "text": "<FIRST_FRAME> A red ball rolls across a table"},
        ]
        # The same client-side validation interactions.create applies.
        TypeAdapter(interactions.InteractionsInput).validate_python(sent)

    async def test_first_and_last_frames_are_sent_in_order_with_both_tags(self):
        import base64

        from google.genai import interactions
        from pydantic import TypeAdapter

        first, first_data = self._image_attachment("PNG", "image/png")
        last, last_data = self._image_attachment("JPEG", None)
        self.cog._fetch_attachment_bytes = AsyncMock(side_effect=[first_data, last_data])
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_fake_omni_interaction())
        self.cog.client.files.download = MagicMock(return_value=b"mp4")

        await _generate_video_with_omni(self.cog, self._params(has_last_frame=True), first, last)

        sent = self.cog.client.aio.interactions.create.call_args.kwargs["input"]
        assert [part["type"] for part in sent] == ["image", "image", "text"]
        assert sent[0]["data"] == base64.b64encode(first_data).decode("ascii")
        assert sent[1]["data"] == base64.b64encode(last_data).decode("ascii")
        # No usable content type: the format PIL detects is used.
        assert sent[1]["mime_type"] == "image/jpeg"
        assert sent[2]["text"] == "<FIRST_FRAME> <LAST_FRAME> A red ball rolls across a table"
        TypeAdapter(interactions.InteractionsInput).validate_python(sent)

    async def test_a_prompt_that_places_its_own_tags_is_sent_unchanged(self):
        attachment, data = self._image_attachment()
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=data)
        self.cog.client.aio.interactions.create = AsyncMock(return_value=_fake_omni_interaction())
        self.cog.client.files.download = MagicMock(return_value=b"mp4")
        prompt = "[# Sources <FIRST_FRAME>@Image1] the ball rolls"

        await _generate_video_with_omni(self.cog, self._params(prompt=prompt), attachment)

        sent = self.cog.client.aio.interactions.create.call_args.kwargs["input"]
        assert sent[-1] == {"type": "text", "text": prompt}

    @pytest.mark.parametrize(
        ("fetched", "message"),
        [(None, "Could not download the `attachment` image"), (b"not an image", "not a readable")],
    )
    async def test_an_unreadable_frame_stops_before_the_paid_request(self, fetched, message):
        attachment = MagicMock()
        attachment.content_type = None
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=fetched)
        self.cog.client.aio.interactions.create = AsyncMock()

        with pytest.raises(ValueError, match=message):
            await _generate_video_with_omni(self.cog, self._params(), attachment)

        self.cog.client.aio.interactions.create.assert_not_called()

    def test_usage_splits_input_video_and_text_output_tokens(self):
        """`total_output_tokens` includes the video tokens; thinking is reported apart."""
        usage = SimpleNamespace(
            total_input_tokens=1_300,
            total_output_tokens=58_000,
            total_thought_tokens=420,
            output_tokens_by_modality=[
                SimpleNamespace(modality="video", tokens=57_920),
                SimpleNamespace(modality="text", tokens=80),
            ],
        )

        result = _extract_omni_video_usage(SimpleNamespace(usage=usage))

        assert result == OmniVideoUsage(
            input_tokens=1_300,
            video_output_tokens=57_920,
            text_output_tokens=80 + 420,
            thinking_tokens=420,
        )
        assert result.output_tokens == 58_000 + 420

    def test_usage_of_a_real_image_to_video_response_bills_every_token(self):
        """Usage shape returned by a first-frame request: image 1,100 + text 24 input
        tokens, a single video modality entry, 691 unlabelled output tokens and 384
        thinking tokens reported apart (total_tokens 60,119 = 1,124 + 58,611 + 384)."""
        usage = SimpleNamespace(
            input_tokens_by_modality=[
                SimpleNamespace(modality="image", tokens=1_100),
                SimpleNamespace(modality="text", tokens=24),
            ],
            output_tokens_by_modality=[SimpleNamespace(modality="video", tokens=57_920)],
            total_input_tokens=1_124,
            total_output_tokens=58_611,
            total_thought_tokens=384,
            total_tokens=60_119,
        )

        result = _extract_omni_video_usage(SimpleNamespace(usage=usage))

        assert result == OmniVideoUsage(
            input_tokens=1_124,
            video_output_tokens=57_920,
            text_output_tokens=691 + 384,
            thinking_tokens=384,
        )
        cost = calculate_omni_video_cost(
            OMNI_11,
            result.video_output_tokens,
            input_tokens=result.input_tokens,
            text_output_tokens=result.text_output_tokens,
        )
        assert cost == pytest.approx(1.0136 + 0.001686 + 0.009675)
        assert format_request_cost(cost) == "$1.0250"

    def test_usage_without_counts_is_all_zero(self):
        assert _extract_omni_video_usage(SimpleNamespace(usage=None)) == OmniVideoUsage()

    def test_omni_is_default_and_first_choice(self):
        """gemini-omni-1.2-flash is the default and heads the picker."""
        assert DEFAULT_OMNI_VIDEO_MODEL == OMNI_12
        assert VIDEO_MODEL_CHOICES[0].value == DEFAULT_OMNI_VIDEO_MODEL
        default = inspect.signature(self.cog.video.callback).parameters["model"].default
        assert default == DEFAULT_OMNI_VIDEO_MODEL


class TestOmniVideoModels:
    """Every id in OMNI_VIDEO_MODELS must route through the Omni path."""

    def test_1_2_and_1_1_are_the_omni_ids(self):
        """gemini-omni-flash-preview was removed from routing."""
        assert {OMNI_12, OMNI_11} == OMNI_VIDEO_MODELS
        assert DEFAULT_OMNI_VIDEO_MODEL in OMNI_VIDEO_MODELS
        for veo in ("veo-3.1-generate-preview", "veo-3.1-lite-generate-preview"):
            assert veo not in OMNI_VIDEO_MODELS

    def test_omni_choices_match_the_routed_set(self):
        """A menu Omni id outside the set would be sent down the Veo path."""
        omni_choices = {
            choice.value for choice in VIDEO_MODEL_CHOICES if choice.value.startswith("gemini-omni")
        }
        assert omni_choices == OMNI_VIDEO_MODELS
        assert "gemini-omni-flash-preview" not in omni_choices

    def test_retired_preview_keeps_its_pricing_row(self):
        """Retired ids keep their pricing row so historical costs stay correct."""
        assert VIDEO_TOKEN_PRICING["gemini-omni-flash-preview"] == 17.50

    @pytest.mark.parametrize(("model", "rate"), [(OMNI_12, 12.00), (OMNI_11, 17.50)])
    def test_each_omni_id_is_priced_at_its_own_video_rate(self, model, rate):
        # The unknown-model fallback is 17.50, so pin the explicit row's presence as
        # well as its rate: a dropped row must fail here.
        assert model in VIDEO_TOKEN_PRICING
        assert VIDEO_TOKEN_PRICING[model] == rate
        assert calculate_omni_video_cost(model, 1_000_000) == pytest.approx(rate)

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    async def test_video_command_routes_each_omni_id_to_the_interactions_path(self, model):
        """`is_omni` must key off the set, so every Omni id reaches the Interactions path."""
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        with patch("discord_gemini.cogs.gemini.client.build_gemini_client"):
            cog = GeminiCog(bot=bot)
        cog._send_error_followup = AsyncMock()

        with (
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([], OmniVideoUsage())),
            ) as omni,
        ):
            await cog.video.callback(cog, ctx=ctx, prompt="a red ball", model=model)

        cog._send_error_followup.assert_not_awaited()
        omni.assert_awaited_once()
        assert omni.await_args.args[1].model == model

    @pytest.mark.parametrize(("resolution", "shown"), [(None, "720p"), ("1080p", "1080p")])
    async def test_omni_cost_embed_shows_the_requested_resolution(self, resolution, shown):
        """The label is the REQUESTED resolution (720p by default), and no duration is
        derived from the token count; the duration shown is the default length."""
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        with patch("discord_gemini.cogs.gemini.client.build_gemini_client"):
            cog = GeminiCog(bot=bot)
        cog._send_error_followup = AsyncMock()

        with (
            patch("discord_gemini.cogs.gemini.video.SHOW_COST_EMBEDS", True),
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([b"mp4"], OmniVideoUsage(video_output_tokens=57920))),
            ),
            patch.object(cog, "_log_cost") as log_cost,
        ):
            await cog.video.callback(
                cog,
                ctx=ctx,
                prompt="a red ball",
                model=DEFAULT_OMNI_VIDEO_MODEL,
                resolution=resolution,
            )

        cog._send_error_followup.assert_not_awaited()
        send_kwargs = ctx.send_followup.call_args.kwargs
        for file in send_kwargs.get("files", []):
            file.close()
        pricing = send_kwargs["embeds"][-1].description
        cost = calculate_omni_video_cost(DEFAULT_OMNI_VIDEO_MODEL, 57920)
        assert pricing == (
            f"{format_request_cost(cost)} · 57.9k out · 1 video · {shown} · 10s"
            f" · {format_daily_total(cost)} today"
        )
        assert log_cost.call_args.kwargs["resolution"] == shown
        assert log_cost.call_args.kwargs["video_tokens"] == 57920
        assert log_cost.call_args.kwargs["duration_seconds"] == 10

    async def test_omni_image_to_video_forwards_frames_and_bills_every_token(self):
        """Both images reach the Omni path, and the input image tokens and the
        thinking tokens are billed with the video tokens and shown on the cost line."""
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        with patch("discord_gemini.cogs.gemini.client.build_gemini_client"):
            cog = GeminiCog(bot=bot)
        cog._send_error_followup = AsyncMock()
        first, last = MagicMock(size=1_000), MagicMock(size=1_000)
        usage = OmniVideoUsage(
            input_tokens=2_600,
            video_output_tokens=57_920,
            text_output_tokens=500,
            thinking_tokens=500,
        )

        with (
            patch("discord_gemini.cogs.gemini.video.SHOW_COST_EMBEDS", True),
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([b"mp4"], usage)),
            ) as omni,
            patch.object(cog, "_log_cost") as log_cost,
        ):
            await cog.video.callback(
                cog,
                ctx=ctx,
                prompt="the ball rolls",
                model=DEFAULT_OMNI_VIDEO_MODEL,
                attachment=first,
                last_frame=last,
            )

        cog._send_error_followup.assert_not_awaited()
        assert omni.await_args.args[2:] == (first, last)
        send_kwargs = ctx.send_followup.call_args.kwargs
        for file in send_kwargs.get("files", []):
            file.close()
        assert (
            "**Mode:** Interpolation (First + Last Frame)" in send_kwargs["embeds"][0].description
        )
        cost = 57_920 * 12.00 / 1e6 + 2_600 * 1.50 / 1e6 + 500 * 9.00 / 1e6
        assert pricing_equal(send_kwargs["embeds"][-1].description, cost)
        assert log_cost.call_args.args[3] == pytest.approx(cost)
        assert log_cost.call_args.kwargs["input_tokens"] == 2_600
        assert log_cost.call_args.kwargs["text_output_tokens"] == 500


def pricing_equal(line: str, cost: float) -> bool:
    return line == (
        f"{format_request_cost(cost)} · 2.6k in / 58.4k out (500 thinking) · 1 video · 720p · 10s"
        f" · {format_daily_total(cost)} today"
    )


class TestOmniUndeliveredVideo:
    """A completed Omni clip is billed even when the bot cannot retrieve the file."""

    @staticmethod
    async def _run(usage):
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        with patch("discord_gemini.cogs.gemini.client.build_gemini_client"):
            cog = GeminiCog(bot=bot)
        cog._send_error_followup = AsyncMock()
        with (
            patch("discord_gemini.cogs.gemini.video.SHOW_COST_EMBEDS", True),
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([], usage)),
            ),
            patch.object(cog, "_log_cost") as log_cost,
        ):
            await cog.video.callback(cog, ctx=ctx, prompt="a red ball")
        cog._send_error_followup.assert_not_awaited()
        return cog, ctx, log_cost

    async def test_billed_clip_that_was_not_delivered_is_still_charged(self):
        usage = OmniVideoUsage(
            input_tokens=24, video_output_tokens=57_920, text_output_tokens=700, thinking_tokens=300
        )

        cog, ctx, log_cost = await self._run(usage)

        cost = calculate_omni_video_cost(
            DEFAULT_OMNI_VIDEO_MODEL, 57_920, input_tokens=24, text_output_tokens=700
        )
        log_cost.assert_called_once()
        assert log_cost.call_args.args[3] == pytest.approx(cost)
        assert log_cost.call_args.kwargs["videos"] == 0
        assert log_cost.call_args.kwargs["delivered"] is False
        assert cog.daily_costs, "the charge must reach the daily total"
        send_kwargs = ctx.send_followup.call_args.kwargs
        assert send_kwargs["embeds"][0].title == "Video Not Delivered"
        assert "billed" in send_kwargs["embeds"][0].description
        assert "safety filters" not in send_kwargs["embeds"][0].description
        assert send_kwargs["embeds"][-1].description == (
            f"{format_request_cost(cost)} · 24 in / 58.6k out (300 thinking) · 0 videos · 720p · 10s"
            f" · {format_daily_total(cost)} today"
        )
        assert not send_kwargs.get("files")

    async def test_nothing_generated_and_nothing_billed_is_not_charged(self):
        _cog, ctx, log_cost = await self._run(OmniVideoUsage())

        log_cost.assert_not_called()
        embed = ctx.send_followup.call_args.kwargs["embed"]
        assert embed.title == "No Videos Generated"


class TestOmniVideoValidation:
    def _params(self, **kwargs):
        from discord_gemini.util import VideoGenerationParameters

        base = {"prompt": "x", "model": DEFAULT_OMNI_VIDEO_MODEL, "aspect_ratio": "16:9"}
        base.update(kwargs)
        return VideoGenerationParameters(**base)

    def test_accepts_bare_text_to_video(self):
        assert _validate_omni_video_request(self._params(), None, None) is None

    @pytest.mark.parametrize("resolution", ["720p", "1080p"])
    def test_accepts_probed_resolutions_on_the_ga_id(self, resolution):
        assert _validate_omni_video_request(self._params(resolution=resolution), None, None) is None

    @pytest.mark.parametrize("resolution", ["360p", "4k"])
    def test_refuses_360p_and_4k_on_omni_1_1(self, resolution):
        params = self._params(model=OMNI_11, resolution=resolution)
        error = _validate_omni_video_request(params, None, None)
        assert error and f"`{resolution}`" in error and "not yet supported for Gemini Omni" in error
        assert "720p, 1080p" in error

    def test_supported_resolutions_per_model(self):
        assert VIDEO_SUPPORTED_RESOLUTIONS[OMNI_11] == {"720p", "1080p"}
        assert VIDEO_SUPPORTED_RESOLUTIONS[OMNI_12] == {"360p", "720p", "1080p", "4k"}

    def test_menu_values_equal_the_default_models_supported_resolutions(self):
        from discord_gemini.cogs.gemini.command_options import VIDEO_RESOLUTION_CHOICES

        menu = {choice.value for choice in VIDEO_RESOLUTION_CHOICES}
        assert menu == VIDEO_SUPPORTED_RESOLUTIONS[DEFAULT_OMNI_VIDEO_MODEL]

    @pytest.mark.parametrize(
        ("model", "duration", "has_last_frame", "expected"),
        [
            (OMNI_12, 3, False, None),
            (OMNI_12, 15, False, None),
            (OMNI_12, 2, False, "at least 3 seconds"),
            (OMNI_12, 16, False, "at most 15 seconds on `gemini-omni-1.2-flash`"),
            (OMNI_11, 10, False, None),
            (OMNI_11, 11, False, "at most 10 seconds on `gemini-omni-1.1-flash`"),
            (OMNI_12, 10, True, None),
            (OMNI_12, 11, True, "With `last_frame`, `duration` can be at most 10 seconds"),
        ],
    )
    def test_duration_bounds(self, model, duration, has_last_frame, expected):
        params = self._params(model=model, duration=duration, has_last_frame=has_last_frame)
        attachment = MagicMock() if has_last_frame else None
        error = _validate_omni_video_request(params, attachment, attachment)
        if expected is None:
            assert error is None
        else:
            assert error and expected in error

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    def test_every_omni_id_declares_its_supported_resolutions(self, model):
        """`_validate_omni_video_request` lists the entry in its error message, so an
        Omni id without one would print an empty list of supported values."""
        assert VIDEO_SUPPORTED_RESOLUTIONS.get(model)

    def test_accepts_a_first_frame_attachment(self):
        assert _validate_omni_video_request(self._params(), MagicMock(), None) is None

    def test_accepts_first_and_last_frames(self):
        params = self._params(has_last_frame=True)
        assert _validate_omni_video_request(params, MagicMock(), MagicMock()) is None

    def test_rejects_last_frame_without_a_first_frame(self):
        """`<LAST_FRAME>` must be used with `<FIRST_FRAME>` (Omni prompt guide)."""
        error = _validate_omni_video_request(self._params(has_last_frame=True), None, MagicMock())
        assert error and "`last_frame` needs an `attachment`" in error

    def test_refuses_a_frame_image_over_the_inline_limit(self):
        from discord_gemini.cogs.gemini.video import OMNI_MAX_FRAME_IMAGE_BYTES

        assert OMNI_MAX_FRAME_IMAGE_BYTES == 20 * 1024 * 1024
        small = MagicMock(size=OMNI_MAX_FRAME_IMAGE_BYTES)
        large = MagicMock(size=OMNI_MAX_FRAME_IMAGE_BYTES + 1)
        params = self._params(has_last_frame=True)

        assert _validate_omni_video_request(params, small, small) is None
        error = _validate_omni_video_request(params, large, small)
        assert error and "`attachment` image is 20.0 MB" in error and "20 MB or smaller" in error
        error = _validate_omni_video_request(params, small, large)
        assert error and "`last_frame` image" in error

    async def test_an_oversized_frame_is_refused_before_any_download_or_request(self):
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        cog = GeminiCog(bot=bot)
        cog._client = MagicMock()
        cog._fetch_attachment_bytes = AsyncMock()
        cog.client.aio.interactions.create = AsyncMock()

        await cog.video.callback(
            cog, ctx=ctx, prompt="a red ball", attachment=MagicMock(size=25 * 1024 * 1024)
        )

        cog._fetch_attachment_bytes.assert_not_called()
        cog.client.aio.interactions.create.assert_not_called()
        assert "25.0 MB" in ctx.send_followup.call_args.kwargs["embed"].description


class TestOmniVideoCost:
    def test_cost_is_exact_token_based(self):
        from discord_gemini.util import calculate_omni_video_cost

        assert calculate_omni_video_cost(OMNI_11, 57920) == pytest.approx(57920 * 17.5 / 1e6)
        assert calculate_omni_video_cost(OMNI_12, 17_376) == pytest.approx(17_376 * 12.0 / 1e6)

    def test_omni_12_720p_three_second_clip_cost(self):
        """Probe usage: 17,376 video + 13 input + 537 thinking tokens."""
        cost = calculate_omni_video_cost(OMNI_12, 17_376, input_tokens=13, text_output_tokens=537)
        assert cost == pytest.approx(17_376 * 12.0 / 1e6 + 13 * 1.5 / 1e6 + 537 * 9.0 / 1e6)

    def test_input_and_text_output_tokens_bill_at_their_own_rates(self):
        """Pricing page: $1.50 input (text / image / video / audio), $9.00 text output
        including thinking, $17.50 video output, per 1M tokens."""
        cost = calculate_omni_video_cost(
            OMNI_11, 57_920, input_tokens=1_000_000, text_output_tokens=1_000_000
        )
        assert cost == pytest.approx(57_920 * 17.50 / 1e6 + 1.50 + 9.00)

    def test_rates_are_declared_for_every_omni_id(self):
        # Read through util: test_config_pricing reloads the pricing module with
        # override files, while util keeps the bundled tables it imported.
        from discord_gemini.util import (
            VIDEO_TOKEN_INPUT_PRICING,
            VIDEO_TOKEN_TEXT_OUTPUT_PRICING,
        )

        for model in OMNI_VIDEO_MODELS | {"gemini-omni-flash-preview"}:
            assert VIDEO_TOKEN_INPUT_PRICING[model] == 1.50
            assert VIDEO_TOKEN_TEXT_OUTPUT_PRICING[model] == 9.00

    def test_unknown_tokenized_model_bills_the_fallback_rates(self):
        cost = calculate_omni_video_cost(
            "gemini-omni-9", 1_000_000, input_tokens=1_000_000, text_output_tokens=1_000_000
        )
        assert cost == pytest.approx(17.50 + 1.50 + 9.00)


class TestOmniDuration:
    """The duration option is sent to the API only when it is set."""

    @staticmethod
    def _params(**kwargs):
        return VideoGenerationParameters(
            prompt="a red ball", model=OMNI_12, aspect_ratio="16:9", **kwargs
        )

    @pytest.mark.parametrize(("duration", "expected"), [(7, "7s"), (None, None)])
    async def test_duration_is_sent_only_when_set(self, duration, expected):
        cog = MagicMock()
        cog.client.aio.interactions.create = AsyncMock(return_value=_fake_omni_interaction())
        cog.logger = MagicMock()
        with patch(
            "discord_gemini.cogs.gemini.video._download_omni_video_uri",
            AsyncMock(return_value=(b"mp4", None)),
        ):
            await _generate_video_with_omni(cog, self._params(duration=duration))

        response_format = cog.client.aio.interactions.create.call_args.kwargs["response_format"]
        assert response_format.get("duration") == expected
        assert ("duration" in response_format) is (duration is not None)

    async def test_embed_shows_requested_and_default_duration(self):
        cog = MagicMock()
        cog.logger = MagicMock()
        for duration, shown in ((7, "7s"), (None, "10s")):
            embed, files = await _create_video_response_embed(
                cog, self._params(duration=duration), [b"mp4"], None
            )
            for file in files:
                file.close()
            assert f"**Duration:** {shown}" in embed.description

    def test_duration_option_is_bounded_in_the_command(self):
        option = next(
            opt
            for opt in GeminiCog.video.options
            if opt.name == "duration"  # type: ignore[attr-defined]
        )
        assert (option.min_value, option.max_value) == (3, 15)
        assert len(option.description) <= 100


class TestOmniUploadLimit:
    """Refuse before a paid call when the file cannot fit; explain after if it did not."""

    @staticmethod
    def _ctx(limit: int | None):
        ctx = AsyncMock()
        ctx.author = MagicMock()
        ctx.author.id = 111
        ctx.defer = AsyncMock()
        ctx.send_followup = AsyncMock()
        ctx.interaction = SimpleNamespace(attachment_size_limit=limit)
        return ctx

    @staticmethod
    def _cog():
        bot = build_mock_bot()
        bot.loop = asyncio.get_running_loop()
        with patch("discord_gemini.cogs.gemini.client.build_gemini_client"):
            cog = GeminiCog(bot=bot)
        cog._send_error_followup = AsyncMock()
        return cog

    def test_estimate_uses_the_measured_rates_and_the_default_length(self):
        assert _estimate_omni_video_bytes("720p", 3) == 3 * 756_522
        assert _estimate_omni_video_bytes(None, None) == 10 * 756_522
        assert _estimate_omni_video_bytes("4k", 15) == 15 * 2_705_897

    async def test_estimate_over_the_limit_is_refused_without_a_request(self):
        cog, ctx = self._cog(), self._ctx(10 * 1024 * 1024)
        omni = AsyncMock()
        with patch("discord_gemini.cogs.gemini.video._generate_video_with_omni", omni):
            await cog.video.callback(cog, ctx=ctx, prompt="x", resolution="1080p", duration=15)

        omni.assert_not_awaited()
        description = ctx.send_followup.call_args.kwargs["embed"].description
        # 15 s x 1,018,427 B = 14.6 MB; 10 MiB / 1,018,427 B = 10 whole seconds.
        assert "15-second 1080p video is estimated at about 14.6 MB" in description
        assert "upload limit of 10 MB" in description
        assert "10 seconds or less at 1080p" in description

    async def test_a_limit_too_small_for_three_seconds_suggests_a_lower_resolution(self):
        cog, ctx = self._cog(), self._ctx(5 * 1024 * 1024)
        omni = AsyncMock()
        with patch("discord_gemini.cogs.gemini.video._generate_video_with_omni", omni):
            await cog.video.callback(cog, ctx=ctx, prompt="x", resolution="4k", duration=3)

        omni.assert_not_awaited()
        description = ctx.send_followup.call_args.kwargs["embed"].description
        assert "Even 3 seconds at 4k would not fit" in description
        assert "Try a lower resolution." in description

    async def test_estimate_within_the_limit_runs(self):
        cog, ctx = self._cog(), self._ctx(10 * 1024 * 1024)
        omni = AsyncMock(return_value=([], OmniVideoUsage()))
        with patch("discord_gemini.cogs.gemini.video._generate_video_with_omni", omni):
            await cog.video.callback(cog, ctx=ctx, prompt="x", duration=10)
        omni.assert_awaited_once()

    async def test_generated_video_over_the_limit_is_billed_but_not_attached(self):
        cog, ctx = self._cog(), self._ctx(10 * 1024 * 1024)
        usage = OmniVideoUsage(input_tokens=13, video_output_tokens=17_376, text_output_tokens=537)
        big = b"\0" * (12 * 1024 * 1024)
        with (
            patch("discord_gemini.cogs.gemini.video.SHOW_COST_EMBEDS", True),
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([big], usage)),
            ),
            patch.object(cog, "_log_cost") as log_cost,
        ):
            await cog.video.callback(cog, ctx=ctx, prompt="x", duration=4)

        cost = calculate_omni_video_cost(OMNI_12, 17_376, input_tokens=13, text_output_tokens=537)
        send_kwargs = ctx.send_followup.call_args.kwargs
        assert not send_kwargs.get("files")
        title_embed = send_kwargs["embeds"][0]
        assert title_embed.title == "Video Too Large to Upload"
        assert title_embed.description == (
            "Gemini Omni generated a 12.0 MB video and Google billed it, but this server's "
            "upload limit is 10 MB. Try a shorter duration or a lower resolution."
        )
        assert len(send_kwargs["embeds"]) == 2
        assert send_kwargs["embeds"][-1].description.startswith(format_request_cost(cost))
        assert "1 video · 720p · 4s" in send_kwargs["embeds"][-1].description
        assert log_cost.call_args.args[3] == pytest.approx(cost)
        assert log_cost.call_args.kwargs["delivered"] is False
        assert log_cost.call_args.kwargs["too_large"] is True
        assert cog.daily_costs

    async def test_generated_video_within_the_limit_is_attached(self):
        cog, ctx = self._cog(), self._ctx(10 * 1024 * 1024)
        usage = OmniVideoUsage(video_output_tokens=17_376)
        with (
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_omni",
                AsyncMock(return_value=([b"mp4"], usage)),
            ),
            patch.object(cog, "_log_cost") as log_cost,
        ):
            await cog.video.callback(cog, ctx=ctx, prompt="x")

        send_kwargs = ctx.send_followup.call_args.kwargs
        for file in send_kwargs.get("files") or []:
            file.close()
        assert len(send_kwargs["files"]) == 1
        assert log_cost.call_args.kwargs["too_large"] is False
        assert log_cost.call_args.kwargs["delivered"] is True
