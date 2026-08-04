"""Signed, read-only HTTP delivery for persisted Ombre media."""

from __future__ import annotations

from mcp.types import CallToolResult, TextContent
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from media_read import MEDIA_LINK_TTL_SECONDS, MediaReader, MediaReadError
from . import _shared as sh


_reader_instance: MediaReader | None = None


def reader() -> MediaReader:
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = MediaReader(sh.config, sh.bucket_mgr)
    return _reader_instance


async def tool_result(*, bucket_id: str = "", media_path: str = "") -> CallToolResult:
    try:
        items = await reader().resolve_many(bucket_id=bucket_id, media_path=media_path)
        return CallToolResult(content=reader().resource_links(items))
    except MediaReadError as exc:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"media_read 拒绝：{exc}")],
        )


def register(mcp) -> None:
    @mcp.custom_route("/media/read", methods=["GET"])
    async def media_read_http(request: Request):
        try:
            item = reader().verify(str(request.query_params.get("token") or ""))
        except MediaReadError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)

        headers = {
            "Cache-Control": f"private, max-age={MEDIA_LINK_TTL_SECONDS}",
            "X-Content-Type-Options": "nosniff",
        }
        options = {}
        if not reader().may_render_inline(item.mime_type):
            options = {"filename": item.path.name, "content_disposition_type": "attachment"}
        return FileResponse(
            item.path,
            media_type=item.mime_type,
            headers=headers,
            **options,
        )
