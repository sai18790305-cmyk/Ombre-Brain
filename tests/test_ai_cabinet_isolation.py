import pytest

from bucket_manager import BucketManager
from identity_context import (
    cabinet_writes_allowed,
    current_ai_identity,
    use_ai_identity,
    use_read_identity,
)
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
        stats = await manager.get_stats()
        assert stats["dynamic_count"] == 1

    visible = await manager.list_all()
    assert [bucket["id"] for bucket in visible] == ["legacy"]
    assert await manager.get(shenyan_id) is None
    stats = await manager.get_stats()
    assert stats["dynamic_count"] == 1


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


def test_explicit_cross_cabinet_context_is_read_only():
    with use_ai_identity("shenyan"):
        assert cabinet_writes_allowed() is True
        with use_read_identity("susu"):
            assert current_ai_identity() == "susu"
            assert cabinet_writes_allowed() is False
        assert current_ai_identity() == "shenyan"
        assert cabinet_writes_allowed() is True
