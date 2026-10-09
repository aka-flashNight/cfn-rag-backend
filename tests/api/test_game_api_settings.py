"""/api/game/ask 的请求级设置项转发测试。

只验一件事：NPCChatRequest 的 agent_enabled / summarize_interval 确实被送进
TurnOrchestrator（编排层的行为语义在 tests/orchestrator/test_settings_switches.py）。
用桩替换 orchestrator，不触发任何 LLM 调用与真实 NPC 状态文件读写。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.game_api as game_api
from services.npc.manager import get_npc_manager
from services.orchestrator.events import content_event, done_event, meta_event


class _StubOrchestrator:
    """记录构造参数，run() 只吐一条最小合法事件流。"""

    kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).kwargs = kwargs

    async def run(self):
        yield meta_event(
            emotion="普通", favorability_change=0, favorability=42, relationship_level="熟悉",
        )
        yield content_event("嗯。")
        yield done_event(self.kwargs["session_id"])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(game_api, "TurnOrchestrator", _StubOrchestrator)
    app = FastAPI()
    app.include_router(game_api.router, prefix="/api/game")
    app.dependency_overrides[get_npc_manager] = lambda: object()
    return TestClient(app)


def _ask(client, **extra) -> dict:
    payload = {"query": "你好", "npc_name": "铁匠", "session_id": "s-settings", **extra}
    resp = client.post("/api/game/ask", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_omitted_settings_keep_agent_on_and_no_tier(client):
    """老客户端不传这两个字段：Agent 按启用处理，档位交回服务端默认。"""
    body = _ask(client)
    assert body["reply"] == "嗯。"
    assert _StubOrchestrator.kwargs["agent_enabled"] is True
    assert _StubOrchestrator.kwargs["summarize_interval"] is None


def test_explicit_settings_are_forwarded(client):
    _ask(client, agent_enabled=False, summarize_interval=100)
    assert _StubOrchestrator.kwargs["agent_enabled"] is False
    assert _StubOrchestrator.kwargs["summarize_interval"] == 100


def test_agent_enabled_true_is_forwarded(client):
    _ask(client, agent_enabled=True, summarize_interval=10)
    assert _StubOrchestrator.kwargs["agent_enabled"] is True
    assert _StubOrchestrator.kwargs["summarize_interval"] == 10


def test_out_of_whitelist_tier_still_reaches_orchestrator(client):
    """白名单收敛在编排层做（非法值回落默认），API 层不静默改写请求。"""
    _ask(client, summarize_interval=7)
    assert _StubOrchestrator.kwargs["summarize_interval"] == 7
