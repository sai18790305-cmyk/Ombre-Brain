"""Signed, read-only HTTP delivery for persisted Ombre media."""

from __future__ import annotations

from mcp.types import CallToolResult, TextContent
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from media_read import MEDIA_LINK_TTL_SECONDS, MediaReader, MediaReadError
from public_origin import configured_public_origin
from . import _shared as sh


_reader_instance: MediaReader | None = None
MEDIA_VIEWER_URI = "ui://ombre-brain/media-viewer-v2.html"

_MEDIA_VIEWER_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { margin: 0; font: 14px system-ui, sans-serif; color: #202124; }
    #gallery { display: grid; gap: 10px; }
    figure { margin: 0; }
    img { display: block; width: 100%; max-height: 560px; object-fit: contain;
          border-radius: 10px; background: #f3f4f6; }
    figcaption { padding: 6px 2px 0; color: #5f6368; overflow-wrap: anywhere; }
    #empty { padding: 12px; color: #5f6368; }
  </style>
</head>
<body>
  <div id="gallery"></div><div id="empty">正在读取照片…</div>
  <script>
    const gallery = document.getElementById("gallery");
    const empty = document.getElementById("empty");
    function render(output) {
      const items = Array.isArray(output?.items) ? output.items : [];
      gallery.replaceChildren();
      empty.hidden = items.length > 0;
      for (const item of items) {
        if (!String(item.mime_type || "").startsWith("image/")) continue;
        const figure = document.createElement("figure");
        const image = document.createElement("img");
        image.src = String(item.url || "");
        image.alt = String(item.title || item.name || "Ombre Brain 照片");
        image.loading = "eager";
        const caption = document.createElement("figcaption");
        caption.textContent = image.alt;
        figure.append(image, caption);
        gallery.append(figure);
      }
      empty.hidden = gallery.childElementCount > 0;
      if (!empty.hidden) empty.textContent = "没有可预览的图片。";
    }
    function receive(message) {
      if (message?.jsonrpc !== "2.0") return;
      if (message.method === "ui/notifications/tool-result") {
        render(message.params?.structuredContent);
      }
      if (message.id === initializeId && message.result) {
        window.parent.postMessage({
          jsonrpc: "2.0",
          method: "ui/notifications/initialized"
        }, "*");
      }
    }
    const initializeId = 1;
    window.addEventListener("message", (event) => {
      if (event.source !== window.parent) return;
      receive(event.data);
    }, { passive: true });
    window.parent.postMessage({
      jsonrpc: "2.0",
      id: initializeId,
      method: "ui/initialize",
      params: {
        appInfo: { name: "Ombre Brain media viewer", version: "2.0.0" },
        appCapabilities: {},
        protocolVersion: "2026-01-26"
      }
    }, "*");
  </script>
</body>
</html>
""".strip()


def reader() -> MediaReader:
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = MediaReader(sh.config, sh.bucket_mgr)
    return _reader_instance


async def tool_result(*, bucket_id: str = "", media_path: str = "") -> CallToolResult:
    try:
        items = await reader().resolve_many(bucket_id=bucket_id, media_path=media_path)
        links = reader().resource_links(items)
        preview_items = [
            {
                "url": str(link.uri),
                "name": link.name,
                "title": link.title or link.name,
                "mime_type": link.mimeType or "application/octet-stream",
                "size": link.size,
            }
            for link in links
        ]
        return CallToolResult(
            structuredContent={"items": preview_items},
            content=[
                TextContent(
                    type="text",
                    text=f"已读取 {len(links)} 个媒体文件；图片已在预览组件中显示。",
                ),
                *links,
            ],
        )
    except MediaReadError as exc:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"media_read 拒绝：{exc}")],
        )


def register(mcp) -> None:
    public_origin = configured_public_origin(sh.config)

    @mcp.resource(
        MEDIA_VIEWER_URI,
        name="Ombre Brain media viewer",
        description="在 ChatGPT 中预览 media_read 返回的图片。",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "prefersBorder": True,
                "domain": public_origin,
                "csp": {"resourceDomains": [public_origin] if public_origin else []},
            },
            "openai/widgetDescription": "显示 Ombre Brain 中读取到的照片。",
            "openai/widgetDomain": public_origin,
            "openai/widgetPrefersBorder": True,
            "openai/widgetCSP": {
                "resource_domains": [public_origin] if public_origin else [],
                "connect_domains": [],
            },
        },
    )
    async def media_viewer() -> str:
        return _MEDIA_VIEWER_HTML

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
