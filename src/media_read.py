"""Secure read-only access to media persisted under ``buckets/_media``.

The MCP tool returns short-lived HTTP resource links.  The HTTP route validates
the signature and resolves the path again before streaming the original bytes;
media is never copied into an MCP response or converted to base64.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

from mcp.types import ResourceLink
from public_origin import configured_public_origin


MEDIA_LINK_TTL_SECONDS = 300
_MAX_TOKEN_CHARS = 8192
_SAFE_INLINE_IMAGE_TYPES = {
    "image/avif",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


class MediaReadError(ValueError):
    """A requested media reference is absent, ambiguous, or unsafe."""


@dataclass(frozen=True)
class ResolvedMedia:
    path: Path
    relative_path: str
    mime_type: str
    size: int
    title: str


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


class MediaReader:
    """Resolve media references and mint/verify short-lived signed links."""

    def __init__(self, config: dict[str, Any], bucket_manager: Any) -> None:
        self.config = config
        self.bucket_manager = bucket_manager
        self.vault_dir = Path(str(config["buckets_dir"])).resolve()
        configured_media = config.get("media_dir") or self.vault_dir / "_media"
        self.media_dir = Path(str(configured_media)).resolve()
        if self.media_dir != self.vault_dir / "_media":
            raise MediaReadError("media_read 只允许使用库内 _media/ 目录。")
        self._secret = secrets.token_bytes(32)

    def _normalize_reference(self, value: object) -> str:
        raw = str(value or "").strip().replace("\\", "/")
        if not raw or "\x00" in raw:
            raise MediaReadError("媒体路径不能为空。")
        pure = PurePosixPath(raw)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise MediaReadError("媒体路径非法或包含路径穿越。")
        parts = list(pure.parts)
        if parts and parts[0] == self.vault_dir.name:
            parts = parts[1:]
        if not parts or parts[0] != "_media":
            raise MediaReadError("media_read 只允许访问 _media/ 目录。")
        return PurePosixPath(*parts).as_posix()

    def resolve_path(self, value: object, *, title: str = "") -> ResolvedMedia:
        relative = self._normalize_reference(value)
        candidate = self.vault_dir.joinpath(*PurePosixPath(relative).parts)

        # Reject every symlink component, even if its final resolved target would
        # remain below _media.  Persisted Ombre media never needs symlinks.
        cursor = self.vault_dir
        for part in PurePosixPath(relative).parts:
            cursor = cursor / part
            try:
                mode = cursor.lstat().st_mode
            except OSError as exc:
                raise MediaReadError("媒体文件不存在。") from exc
            if stat.S_ISLNK(mode):
                raise MediaReadError("媒体路径包含符号链接，已拒绝。")

        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise MediaReadError("媒体文件不存在。") from exc
        if self.media_dir not in resolved.parents or not resolved.is_file():
            raise MediaReadError("media_read 只允许访问 _media/ 中的普通文件。")
        mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return ResolvedMedia(
            path=resolved,
            relative_path=relative,
            mime_type=mime_type,
            size=resolved.stat().st_size,
            title=(str(title or "").strip()[:200] or resolved.name),
        )

    async def resolve_many(self, *, bucket_id: str = "", media_path: str = "") -> list[ResolvedMedia]:
        bucket_id = str(bucket_id or "").strip()
        media_path = str(media_path or "").strip()
        if bool(bucket_id) == bool(media_path):
            raise MediaReadError("请且只请提供 bucket_id 或 media_path 其中一个。")
        if media_path:
            return [self.resolve_path(media_path)]

        bucket = await self.bucket_manager.get(bucket_id)
        if not bucket:
            raise MediaReadError("没有找到该记忆桶。")
        items = (bucket.get("metadata") or {}).get("media") or []
        if not isinstance(items, list):
            items = [items]
        resolved: list[ResolvedMedia] = []
        for item in items:
            if isinstance(item, dict):
                path = item.get("path")
                title = item.get("title") or ""
            else:
                path, title = item, ""
            resolved.append(self.resolve_path(path, title=str(title)))
        if not resolved:
            raise MediaReadError("该记忆桶没有媒体文件。")
        return resolved

    def sign(self, relative_path: str, *, now: int | None = None) -> str:
        expires = int(now if now is not None else time.time()) + MEDIA_LINK_TTL_SECONDS
        payload = json.dumps(
            {"exp": expires, "path": self._normalize_reference(relative_path)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        encoded = _b64encode(payload)
        signature = _b64encode(hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest())
        return f"{encoded}.{signature}"

    def verify(self, token: str, *, now: int | None = None) -> ResolvedMedia:
        if not token or len(token) > _MAX_TOKEN_CHARS or token.count(".") != 1:
            raise MediaReadError("媒体链接无效。")
        encoded, supplied = token.split(".", 1)
        expected = _b64encode(hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied, expected):
            raise MediaReadError("媒体链接签名无效。")
        try:
            payload = json.loads(_b64decode(encoded).decode("utf-8"))
            expires = int(payload["exp"])
            relative_path = str(payload["path"])
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise MediaReadError("媒体链接载荷无效。") from exc
        current = int(now if now is not None else time.time())
        if expires < current or expires > current + MEDIA_LINK_TTL_SECONDS + 5:
            raise MediaReadError("媒体链接已过期或时间无效。")
        return self.resolve_path(relative_path)

    def resource_links(self, media: Iterable[ResolvedMedia]) -> list[ResourceLink]:
        public_origin = configured_public_origin(self.config)
        if not public_origin:
            raise MediaReadError("未配置 deployment.public_url，无法生成可访问的媒体短链。")
        links: list[ResourceLink] = []
        for item in media:
            token = quote(self.sign(item.relative_path), safe="")
            links.append(
                ResourceLink(
                    type="resource_link",
                    uri=f"{public_origin}/media/read?token={token}",
                    name=item.path.name,
                    title=item.title,
                    description="Ombre Brain 原始媒体文件（短时只读链接）",
                    mimeType=item.mime_type,
                    size=item.size,
                )
            )
        return links

    @staticmethod
    def may_render_inline(mime_type: str) -> bool:
        return mime_type in _SAFE_INLINE_IMAGE_TYPES
