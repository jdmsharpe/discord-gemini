from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from discord import Embed

from discord_gemini.cogs.gemini.embed_delivery import (
    DISCORD_EMBED_TOTAL_LIMIT,
    count_embed_chars,
    pack_embeds,
    send_embed_batches,
)


def test_count_embed_chars_includes_visible_text_fields():
    embed = Embed(title="Title", description="Description")
    embed.add_field(name="Field", value="Value")
    embed.set_footer(text="Footer")
    embed.set_author(name="Author")

    assert count_embed_chars(embed) == len("TitleDescriptionFieldValueFooterAuthor")


def test_pack_embeds_splits_aggregate_overflow():
    embeds = [
        Embed(title="One", description="a" * 4000),
        Embed(title="Two", description="b" * 2500),
        Embed(title="Three", description="c" * 100),
    ]

    batches = pack_embeds(embeds)

    assert len(batches) == 2
    assert batches[0] == [embeds[0]]
    assert batches[1] == [embeds[1], embeds[2]]
    assert all(
        sum(count_embed_chars(embed) for embed in batch) <= DISCORD_EMBED_TOTAL_LIMIT
        for batch in batches
    )


def test_pack_embeds_splits_more_than_ten_embeds():
    embeds = [Embed(description=str(index)) for index in range(11)]

    batches = pack_embeds(embeds)

    assert [len(batch) for batch in batches] == [10, 1]


@pytest.mark.asyncio
async def test_send_embed_batches_attaches_view_only_to_final_batch():
    send = AsyncMock(side_effect=["first", "second"])
    view = object()
    embeds = [
        Embed(description="a" * 4000),
        Embed(description="b" * 2500),
    ]

    result = await send_embed_batches(send, embeds=embeds, view=view)

    assert result == "second"
    assert send.await_count == 2
    first_kwargs = send.await_args_list[0].kwargs
    second_kwargs = send.await_args_list[1].kwargs
    assert "view" not in first_kwargs
    assert second_kwargs["view"] is view


@pytest.mark.asyncio
async def test_send_embed_batches_attaches_files_to_referencing_batch():
    send = AsyncMock(side_effect=["first", "second"])
    first_embed = Embed(description="a" * 4000)
    second_embed = Embed(description="b" * 2100)
    second_embed.set_image(url="attachment://image.png")
    unreferenced = SimpleNamespace(filename="report.txt")
    referenced = SimpleNamespace(filename="image.png")

    result = await send_embed_batches(
        send,
        embeds=[first_embed, second_embed],
        files=[unreferenced, referenced],
    )

    assert result == "second"
    first_kwargs = send.await_args_list[0].kwargs
    second_kwargs = send.await_args_list[1].kwargs
    assert first_kwargs["files"] == [unreferenced]
    assert second_kwargs["files"] == [referenced]


@pytest.mark.asyncio
async def test_send_embed_batches_preserves_single_file_kwarg_when_possible():
    send = AsyncMock(return_value="message")
    file = SimpleNamespace(filename="audio.mp3")

    result = await send_embed_batches(
        send,
        embeds=[Embed(description="done")],
        file=file,
    )

    assert result == "message"
    assert send.await_args.kwargs["file"] is file
    assert "files" not in send.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
async def test_failed_upload_retries_complete_bytes_and_metadata(tmp_path, owned):
    from io import BytesIO

    from discord import File, HTTPException

    payload = b"the complete attachment contents"
    if owned:
        path = tmp_path / "report.txt"
        path.write_bytes(payload)
        upload = File(path, filename="report.txt", spoiler=True, description="Report")
    else:
        stream = BytesIO(b"prefix" + payload)
        stream.seek(6)
        upload = File(stream, filename="report.txt", spoiler=True, description="Report")
    attempts = []

    async def send(**kwargs):
        file = kwargs["file"]
        attempts.append(file.fp.read())
        assert file.filename == "SPOILER_report.txt"
        assert file.spoiler is True
        assert file.description == "Report"
        file.close()
        if len(attempts) == 1:
            raise HTTPException(SimpleNamespace(status=400, reason="Bad Request"), "bad embed")
        assert "embeds" not in kwargs
        assert kwargs["content"] == "done"
        return "fallback"

    try:
        result = await send_embed_batches(send, embeds=[Embed(description="done")], file=upload)
        assert result == "fallback"
        assert attempts == [payload, payload]
    finally:
        upload.close()
        upload.fp.close()
