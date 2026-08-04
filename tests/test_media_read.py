from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from mcp.types import ResourceLink, TextContent

from media_read import MEDIA_LINK_TTL_SECONDS, MediaReader, MediaReadError


@pytest.mark.asyncio
async def test_media_read_is_advertised_as_strictly_read_only():
    from server import mcp

    listed = next(item for item in await mcp.list_tools() if item.name == "media_read")
    annotations = listed.annotations

    assert annotations is not None
    assert annotations.readOnlyHint is True
    assert annotations.destructiveHint is False
    assert annotations.openWorldHint is False
    assert annotations.idempotentHint is True
    assert listed.meta["ui"]["resourceUri"] == "ui://ombre-brain/media-viewer-v1.html"
    assert listed.meta["openai/outputTemplate"] == "ui://ombre-brain/media-viewer-v1.html"


@pytest.mark.asyncio
async def test_media_read_returns_structured_preview_urls(tmp_path, monkeypatch):
    from web import media as media_web

    bucket = {"metadata": {"media": ["_media/bucket-1/photo.png"]}}
    test_reader, vault = _reader(tmp_path, {"bucket": bucket})
    (vault / "_media" / "bucket-1" / "photo.png").write_bytes(b"png")
    monkeypatch.setattr(media_web, "_reader_instance", test_reader)

    result = await media_web.tool_result(bucket_id="bucket")

    assert result.structuredContent["items"][0]["mime_type"] == "image/png"
    assert result.structuredContent["items"][0]["url"].startswith(
        "https://brain.example/media/read?token="
    )
    assert isinstance(result.content[0], TextContent)
    assert isinstance(result.content[1], ResourceLink)


class _BucketManager:
    def __init__(self, buckets=None):
        self.buckets = buckets or {}

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)


def _reader(tmp_path: Path, buckets=None):
    vault = tmp_path / "buckets"
    (vault / "_media" / "bucket-1").mkdir(parents=True)
    config = {
        "buckets_dir": str(vault),
        "media_dir": str(vault / "_media"),
        "deployment": {"public_url": "https://brain.example/mcp"},
    }
    return MediaReader(config, _BucketManager(buckets)), vault


def test_media_read_mints_resource_link_and_verifies_original_file(tmp_path):
    reader, vault = _reader(tmp_path)
    original = b"\x89PNG\r\n\x1a\noriginal-bytes"
    media = vault / "_media" / "bucket-1" / "photo.png"
    media.write_bytes(original)

    resolved = reader.resolve_path("_media/bucket-1/photo.png", title="照片")
    link = reader.resource_links([resolved])[0]

    assert isinstance(link, ResourceLink)
    assert link.mimeType == "image/png"
    assert link.size == len(original)
    assert str(link.uri).startswith("https://brain.example/media/read?token=")
    token = parse_qs(urlsplit(str(link.uri)).query)["token"][0]
    verified = reader.verify(token)
    assert verified.path.read_bytes() == original


@pytest.mark.parametrize(
    "value",
    [
        "../secret.png",
        "_media/../../secret.png",
        "/tmp/secret.png",
        "permanent/memory.md",
        "_media\\..\\secret.png",
    ],
)
def test_media_read_rejects_escape_paths(tmp_path, value):
    reader, _vault = _reader(tmp_path)
    with pytest.raises(MediaReadError):
        reader.resolve_path(value)


def test_media_read_rejects_symlinks(tmp_path):
    reader, vault = _reader(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    link = vault / "_media" / "bucket-1" / "linked.png"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(MediaReadError, match="符号链接"):
        reader.resolve_path("_media/bucket-1/linked.png")


def test_media_read_rejects_tampered_and_expired_tokens(tmp_path):
    reader, vault = _reader(tmp_path)
    (vault / "_media" / "bucket-1" / "photo.jpg").write_bytes(b"jpeg")
    token = reader.sign("_media/bucket-1/photo.jpg", now=1000)

    assert reader.verify(token, now=1000).mime_type == "image/jpeg"
    with pytest.raises(MediaReadError, match="签名"):
        reader.verify(token[:-1] + ("A" if token[-1] != "A" else "B"), now=1000)
    with pytest.raises(MediaReadError, match="过期"):
        reader.verify(token, now=1000 + MEDIA_LINK_TTL_SECONDS + 1)


def test_media_read_resolves_all_media_from_bucket(tmp_path):
    bucket = {
        "metadata": {
            "media": [
                {"path": "_media/bucket-1/a.png", "title": "A"},
                {"path": "_media/bucket-1/b.jpg", "title": "B"},
            ]
        }
    }
    reader, vault = _reader(tmp_path, {"15263670e3a7": bucket})
    (vault / "_media" / "bucket-1" / "a.png").write_bytes(b"a")
    (vault / "_media" / "bucket-1" / "b.jpg").write_bytes(b"b")

    items = asyncio.run(reader.resolve_many(bucket_id="15263670e3a7"))
    assert [item.title for item in items] == ["A", "B"]


def test_media_read_requires_exactly_one_locator(tmp_path):
    reader, _vault = _reader(tmp_path)
    with pytest.raises(MediaReadError):
        asyncio.run(reader.resolve_many())
    with pytest.raises(MediaReadError):
        asyncio.run(reader.resolve_many(bucket_id="x", media_path="_media/x.png"))


def test_media_read_refuses_external_configured_media_directory(tmp_path):
    vault = tmp_path / "buckets"
    vault.mkdir()
    with pytest.raises(MediaReadError, match="_media"):
        MediaReader(
            {
                "buckets_dir": str(vault),
                "media_dir": str(tmp_path / "elsewhere"),
            },
            _BucketManager(),
        )
