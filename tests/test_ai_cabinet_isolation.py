import pytest

from bucket_manager import BucketManager
from identity_context import current_ai_identity, use_ai_identity
from server_app import AICabinetMiddleware


@pytest.mark.asyncio
async def test_legacy_is_susu_and_new_writes_are_scoped(test_config):
    manager = BucketManager(test_config, embedding_engine=None)
    legacy_path = manager.dynamic_dir + "/legacy.md"
    with open(legacy_path, "w", encoding="utf-8") as handle:
        handle.write("---\nid: legacy\nimportance: 5\ntype: dynamic\n---\nold\n")

    with use_ai_identity("shenyan"):
        shenyan_id = await manager.create("new", bucket_type="dynamic")
        visible = await manager.list_all()
        assert [bucket["id"] for bucket in visible] == [shenyan_id]
        assert visible[0]["metadata"]["ai_identity"] == "shenyan"
        assert await manager.get("legacy") is None

    visible = await manager.list_all()
    assert [bucket["id"] for bucket in visible] == ["legacy"]
    assert await manager.get(shenyan_id) is None


@pytest.mark.asyncio
async def test_cabinet_middleware_rewrites_only_shenyan_endpoint():
    seen = []

    async def app(scope, receive, send):
        seen.append((scope["path"], current_ai_identity()))

    middleware = AICabinetMiddleware(app)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    await middleware({"type": "http", "path": "/mcp", "raw_path": b"/mcp"}, receive, send)
    await middleware(
        {"type": "http", "path": "/mcp/shenyan", "raw_path": b"/mcp/shenyan"},
        receive,
        send,
    )

    assert seen == [("/mcp", "susu"), ("/mcp", "shenyan")]
    assert current_ai_identity() == "susu"
