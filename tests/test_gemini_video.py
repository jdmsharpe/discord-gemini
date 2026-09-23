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
    _build_veo_image,
    _generate_video_with_omni,
    _generate_video_with_veo,
    _validate_omni_video_request,
    _validate_video_request,
)
from discord_gemini.cost_line import format_daily_total, format_request_cost
from discord_gemini.util import (
    VIDEO_GENERATION_TIMEOUT,
    VIDEO_TOKEN_PRICING,
    VideoGenerationParameters,
    calculate_omni_video_cost,
)
from tests.support import AsyncGeminiCogTestCase, build_mock_bot


def _encoded_image(fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(buf, format=fmt)
    return buf.getvalue()


class TestBuildVeoImage:
    """Veo image inputs must carry raw bytes + a mime type.

    A `PIL.Image` silently validates into an all-None `types.Image`, which the API
    rejects with "should contain both bytesBase64Encoded and mimeType".
    """

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
        image = _build_veo_image(data, self._attachment(content_type))
        assert image.image_bytes == data
        assert image.mime_type == expected_mime

    def test_falls_back_to_detected_format(self):
        data = _encoded_image("WEBP")
        image = _build_veo_image(data, self._attachment(None))
        assert image.mime_type == "image/webp"
        assert image.image_bytes == data

    def test_ignores_non_image_content_type(self):
        data = _encoded_image("PNG")
        image = _build_veo_image(data, self._attachment("application/octet-stream"))
        assert image.mime_type == "image/png"

    def test_never_serializes_to_an_empty_struct(self):
        """The regression: the API 400s when image_bytes/mime_type are absent."""
        image = _build_veo_image(_encoded_image("WEBP"), self._attachment("image/webp"))
        payload = image.model_dump(exclude_none=True)
        assert payload, "types.Image serialized to {} — the API rejects this"
        assert "image_bytes" in payload
        assert "mime_type" in payload


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

        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        await self._assert_video_mode(params, "Text-to-Video")

    async def test_mode_image_to_video(self):
        """Test embed shows Image-to-Video mode when attachment provided."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Image-to-Video", attachment=mock_attachment)

    async def test_mode_interpolation(self):
        """Test embed shows Interpolation mode when both attachment and last_frame."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        mock_attachment = MagicMock()
        await self._assert_video_mode(params, "Interpolation", attachment=mock_attachment)

    async def test_mode_last_frame_only(self):
        """Test embed shows Last Frame Constrained mode when only last_frame."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-generate-preview",
            has_last_frame=True,
        )
        await self._assert_video_mode(params, "Last Frame Constrained")

    async def test_embed_includes_resolution(self):
        """Test embed includes the selected output resolution."""
        from discord_gemini.util import VideoGenerationParameters

        params = VideoGenerationParameters(
            prompt="A sunset",
            model="veo-3.1-lite-generate-preview",
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


class TestVideoValidation:
    def test_rejects_multiple_videos(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="16:9",
            resolution=None,
            number_of_videos=2,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert "number_of_videos" in error

    def test_rejects_4k_on_lite(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="16:9",
            resolution="4k",
            number_of_videos=1,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert "`4k`" in error

    def test_rejects_non_8s_for_1080p(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="9:16",
            resolution="1080p",
            number_of_videos=1,
            duration_seconds=6,
            has_last_frame=False,
        )
        assert error == "The `1080p` resolution only supports 8 second videos."

    def test_rejects_interpolation_without_8s(self):
        error = _validate_video_request(
            model="veo-3.1-generate-preview",
            aspect_ratio="16:9",
            resolution=None,
            number_of_videos=1,
            duration_seconds=6,
            has_last_frame=True,
        )
        assert "last_frame" in error

    def test_accepts_lite_1080p_portrait_at_8s(self):
        error = _validate_video_request(
            model="veo-3.1-lite-generate-preview",
            aspect_ratio="9:16",
            resolution="1080p",
            number_of_videos=1,
            duration_seconds=8,
            has_last_frame=False,
        )
        assert error is None


class TestVeoGenerateVideosSource(AsyncGeminiCogTestCase):
    """google-genai 2.14.0 deprecated the prompt/image arguments in favour of `source`."""

    @staticmethod
    def _attachment():
        attachment = MagicMock()
        attachment.content_type = "image/png"
        return attachment

    async def _call_veo(self, attachment=None, last_frame_attachment=None):
        from discord_gemini.util import VideoGenerationParameters

        self.cog.client.aio.models.generate_videos = AsyncMock(
            return_value=SimpleNamespace(done=True, name="operations/1", response=None)
        )
        params = VideoGenerationParameters(prompt="A sunset", model="veo-3.1-generate-preview")
        await _generate_video_with_veo(self.cog, params, attachment, last_frame_attachment)
        return self.cog.client.aio.models.generate_videos.call_args.kwargs

    async def test_prompt_is_passed_through_source(self):
        kwargs = await self._call_veo()

        assert "prompt" not in kwargs
        assert "image" not in kwargs
        assert kwargs["source"].prompt == "A sunset"
        assert kwargs["source"].image is None

    async def test_image_is_passed_through_source(self):
        data = _encoded_image("PNG")
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=data)

        kwargs = await self._call_veo(attachment=self._attachment())

        assert "image" not in kwargs
        assert kwargs["source"].image.image_bytes == data
        assert kwargs["source"].image.mime_type == "image/png"

    async def test_last_frame_stays_on_the_config(self):
        """`last_frame` is a GenerateVideosConfig field, not a source field."""
        data = _encoded_image("PNG")
        self.cog._fetch_attachment_bytes = AsyncMock(return_value=data)

        kwargs = await self._call_veo(last_frame_attachment=self._attachment())

        assert kwargs["config"].last_frame.image_bytes == data
        # `hasattr` is always False on a pydantic model for an undeclared field,
        # so assert on the declared field set instead — that is what would
        # actually change if last_frame ever migrated onto the source.
        assert "last_frame" not in type(kwargs["source"]).model_fields


def _fake_omni_interaction(
    uri: str | None = "https://x/v1beta/files/abc123:download?alt=media",
    video_tokens: int = 57920,
    status: str = "completed",
    errors=None,
):
    modality = MagicMock()
    modality.modality = "video"
    modality.tokens = video_tokens
    usage = MagicMock()
    usage.output_tokens_by_modality = [modality]
    interaction = MagicMock()
    interaction.id = "omni-1"
    interaction.status = status
    interaction.errors = errors
    interaction.usage = usage
    if uri is None:
        interaction.output_video = None
    else:
        output_video = MagicMock()
        output_video.uri = uri
        output_video.mime_type = "video/mp4"
        interaction.output_video = output_video
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
            videos, tokens = await _generate_video_with_omni(self.cog, self._params())

        assert videos == [b"mp4-bytes"]
        assert tokens == 57920
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
        videos, tokens = await _generate_video_with_omni(self.cog, params)

        assert videos == []
        assert tokens == 57920
        self.cog.client.files.download.assert_not_called()

    def test_omni_is_default_and_first_choice(self):
        """gemini-omni-1.1-flash (GA 2026-08-27) is the default and heads the picker."""
        assert DEFAULT_OMNI_VIDEO_MODEL == "gemini-omni-1.1-flash"
        assert VIDEO_MODEL_CHOICES[0].value == DEFAULT_OMNI_VIDEO_MODEL
        default = inspect.signature(self.cog.video.callback).parameters["model"].default
        assert default == DEFAULT_OMNI_VIDEO_MODEL


class TestOmniVideoModels:
    """Every id in OMNI_VIDEO_MODELS must route through the Omni path."""

    def test_ga_id_is_the_only_omni_id(self):
        """gemini-omni-flash-preview shuts down 2026-09-30 and was removed from routing."""
        assert {"gemini-omni-1.1-flash"} == OMNI_VIDEO_MODELS
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

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    def test_each_omni_id_is_priced_at_17_50_per_million(self, model):
        # The unknown-model fallback happens to be 17.50 too, so pin the explicit
        # row's presence as well as its rate: a dropped row must fail here.
        assert model in VIDEO_TOKEN_PRICING
        assert VIDEO_TOKEN_PRICING[model] == 17.50
        assert calculate_omni_video_cost(model, 1_000_000) == pytest.approx(17.50)

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
                AsyncMock(return_value=([], 0)),
            ) as omni,
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_veo",
                AsyncMock(return_value=[]),
            ) as veo,
        ):
            await cog.video.callback(cog, ctx=ctx, prompt="a red ball", model=model)

        cog._send_error_followup.assert_not_awaited()
        omni.assert_awaited_once()
        assert omni.await_args.args[1].model == model
        veo.assert_not_awaited()

    async def test_video_command_routes_veo_to_generate_videos(self):
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
                AsyncMock(return_value=([], 0)),
            ) as omni,
            patch(
                "discord_gemini.cogs.gemini.video._generate_video_with_veo",
                AsyncMock(return_value=[]),
            ) as veo,
        ):
            await cog.video.callback(
                cog, ctx=ctx, prompt="a red ball", model="veo-3.1-generate-preview"
            )

        cog._send_error_followup.assert_not_awaited()
        veo.assert_awaited_once()
        omni.assert_not_awaited()

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    def test_each_omni_id_rejects_veo_only_options(self, model):
        params = VideoGenerationParameters(
            prompt="x", model=model, duration_seconds=8, negative_prompt="no cats"
        )
        error = _validate_omni_video_request(params, None, None)
        assert error and "duration" in error and "negative_prompt" in error

    @pytest.mark.parametrize(("resolution", "shown"), [(None, "720p"), ("1080p", "1080p")])
    async def test_omni_cost_embed_shows_the_requested_resolution(self, resolution, shown):
        """The label is the REQUESTED resolution (720p by default), and no duration is
        derived from the token count: a 3 s 1080p clip billed the same 57,920 tokens
        as a default 720p clip, so the old `~10s 720p` label was simply wrong."""
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
                AsyncMock(return_value=([b"mp4"], 57920)),
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
            f"{format_request_cost(cost)} · 57.9k out · 1 video · {shown}"
            f" · {format_daily_total(cost)} today"
        )
        assert log_cost.call_args.kwargs["resolution"] == shown
        assert log_cost.call_args.kwargs["video_tokens"] == 57920
        assert "duration_seconds" not in log_cost.call_args.kwargs


class TestVeoCostEmbed:
    @pytest.mark.parametrize(
        ("resolution", "duration_seconds", "expected"),
        [
            (None, None, "$0.4000 · 1 video · 8s · $0.40 today"),
            ("720p", 4, "$0.2000 · 1 video · 4s · 720p · $0.20 today"),
            ("1080p", 8, "$0.6400 · 1 video · 8s · 1080p · $0.64 today"),
        ],
    )
    async def test_veo_cost_embed_shows_count_duration_and_resolution(
        self, resolution, duration_seconds, expected
    ):
        """Veo is billed per second: the line shows the video count, the duration
        (8 s when none is requested) and the requested resolution."""
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
                "discord_gemini.cogs.gemini.video._generate_video_with_veo",
                AsyncMock(return_value=[b"mp4"]),
            ),
        ):
            await cog.video.callback(
                cog,
                ctx=ctx,
                prompt="a red ball",
                model="veo-3.1-lite-generate-preview",
                resolution=resolution,
                duration_seconds=duration_seconds,
            )

        cog._send_error_followup.assert_not_awaited()
        send_kwargs = ctx.send_followup.call_args.kwargs
        for file in send_kwargs.get("files", []):
            file.close()
        assert send_kwargs["embeds"][-1].description == expected


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

    def test_refuses_4k_on_the_ga_id_as_not_yet_supported(self):
        """4k and 360p are documented for Omni but were never probed and have no price
        row, so the 1.1 id refuses them explicitly instead of passing them through."""
        error = _validate_omni_video_request(self._params(resolution="4k"), None, None)
        assert error and "`4k`" in error and "not yet supported for Gemini Omni" in error
        assert "720p, 1080p" in error

    def test_ga_id_supported_resolutions_are_exactly_the_probed_pair(self):
        assert VIDEO_SUPPORTED_RESOLUTIONS[DEFAULT_OMNI_VIDEO_MODEL] == {"720p", "1080p"}

    @pytest.mark.parametrize("model", sorted(OMNI_VIDEO_MODELS))
    def test_every_omni_id_declares_its_supported_resolutions(self, model):
        """`_validate_omni_video_request` lists the entry in its error message, so an
        Omni id without one would print an empty list of supported values."""
        assert VIDEO_SUPPORTED_RESOLUTIONS.get(model)

    def test_rejected_option_message_names_resolution_as_accepted_on_the_ga_id(self):
        error = _validate_omni_video_request(self._params(duration_seconds=8), None, None)
        assert error and "an `aspect_ratio` and a `resolution`" in error

    def test_rejects_duration(self):
        error = _validate_omni_video_request(self._params(duration_seconds=8), None, None)
        assert error and "duration" in error

    def test_rejects_multiple_videos(self):
        error = _validate_omni_video_request(self._params(number_of_videos=2), None, None)
        assert error and "number_of_videos" in error

    def test_rejects_image_attachment(self):
        error = _validate_omni_video_request(self._params(), MagicMock(), None)
        assert error and "attachment" in error

    def test_rejects_last_frame(self):
        error = _validate_omni_video_request(self._params(has_last_frame=True), None, MagicMock())
        assert error and "last_frame" in error


class TestOmniVideoCost:
    def test_cost_is_exact_token_based(self):
        from discord_gemini.util import calculate_omni_video_cost

        cost = calculate_omni_video_cost(DEFAULT_OMNI_VIDEO_MODEL, 57920)
        assert cost == pytest.approx(57920 * 17.5 / 1_000_000)
