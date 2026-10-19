"""端点契约（离线假流）：SSE 形状、session 复用、finalize、请求校验、未接线降级。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.dialogue_interject import api as interject_api
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import parse_sse


def _client(primary=None, repair=None) -> TestClient:
    interject_api.set_stream_factories(primary, repair)
    app = FastAPI()
    app.include_router(interject_api.router, prefix="/game")
    return TestClient(app)


def _factory_for(mapping: dict[str, list[str]]):
    async def _stream(request):
        for chunk in mapping.get(request.turn_id, []):
            yield chunk

    return _stream


@pytest.fixture(autouse=True)
def _reset_stream_factories():
    yield
    interject_api.set_stream_factories(None, None)


def test_endpoint_streams_f01_and_returns_session(memory_store):
    client = _client(_factory_for({fx.DAILY_TURN_A: fx.F01_JSONL}))

    response = client.post("/game/dialogue-interject", json=fx.daily_request())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "line", "line", "line", "done"]
    assert [data["seq"] for name, data in events if name == "line"] == [0, 1, 2]
    done = events[-1][1]
    assert done["dialogue_instance_id"] == fx.DAILY_DIALOGUE_ID
    assert done["session_id"]
    assert done["usage"] == {}

    # 请求阶段只种入 played（seq0）；本轮 $PC 与 AI 行要等 finalize
    info = memory_store.dialogue_session_get_sync(fx.DAILY_DIALOGUE_ID)
    assert info is not None
    assert info.session_id == done["session_id"]
    assert [row.seq for row in memory_store.dialogue_ledger_read_sync(info.session_id)] == [0]


def test_two_turns_share_one_session(memory_store):
    turn_b = [
        '{"add":"任务是吧，我这儿正好有个跑腿的活儿。","emo":"普通"}\n',
        '{"closing":"感兴趣就接着往下听。","emo":"普通"}\n',
    ]
    client = _client(_factory_for({fx.DAILY_TURN_A: fx.F01_JSONL, fx.DAILY_TURN_B: turn_b}))

    first = parse_sse(client.post("/game/dialogue-interject", json=fx.daily_request()).text)
    second = parse_sse(
        client.post("/game/dialogue-interject", json=fx.daily_turn_b_request()).text
    )
    first_done = next(data for name, data in first if name == "done")
    second_done = next(data for name, data in second if name == "done")

    assert first_done["session_id"] == second_done["session_id"]
    ledger = memory_store.dialogue_ledger_read_sync(first_done["session_id"])
    assert [row.seq for row in ledger] == [0, 1, 2]


def test_finalize_endpoint_is_idempotent(memory_store):
    client = _client(_factory_for({fx.DAILY_TURN_A: fx.F01_JSONL}))
    client.post("/game/dialogue-interject", json=fx.daily_request())

    first = client.post("/game/dialogue-interject/finalize", json=fx.F01_FINALIZE)
    assert first.status_code == 200
    body = first.json()
    assert set(body) == {
        "ok", "dialogue_instance_id", "session_id", "appended", "total",
    }
    assert body["ok"] is True
    assert body["appended"] == 4
    assert body["total"] == 5

    second = client.post("/game/dialogue-interject/finalize", json=fx.F01_FINALIZE)
    assert second.json()["appended"] == 0
    assert second.json()["total"] == 5


def test_unconfigured_endpoint_returns_error_and_creates_no_session(memory_store, monkeypatch):
    # B1：默认走真 LLM 路径，但测试必须离线——强制判定为未配置
    monkeypatch.setattr(interject_api, "llm_is_configured", lambda *a, **k: False)
    client = _client(None)
    response = client.post("/game/dialogue-interject", json=fx.daily_request())
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["error"]
    assert events[0][1]["code"] == "llm_not_configured"
    assert memory_store.dialogue_session_get_sync(fx.DAILY_DIALOGUE_ID) is None


def test_task_mode_without_api_key_returns_llm_not_configured(memory_store, monkeypatch):
    """B4 任务模式接真模型；未配置 key 时明确 error，不建 session、不产 plan。"""
    monkeypatch.setattr(interject_api, "llm_is_configured", lambda *a, **k: False)
    client = _client(None)
    response = client.post("/game/dialogue-interject", json=fx.task_request())
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["error"]
    assert events[0][1]["code"] == "llm_not_configured"
    assert memory_store.dialogue_session_get_sync(fx.TASK_DIALOGUE_ID) is None


@pytest.mark.parametrize("payload", [
    # daily 模式配 task_pre
    {**fx.daily_request(), "dialogue": {**fx.daily_request()["dialogue"], "kind": "task_pre"}},
    # 玩家输入超 150 字
    {**fx.daily_request(), "player_input": "啊" * 151},
    # remaining 编号跳号
    {**fx.daily_request(), "dialogue": {
        **fx.daily_request()["dialogue"],
        "remaining": [
            {"index": 1, "name": fx.NPC, "text": "一"},
            {"index": 3, "name": fx.NPC, "text": "三"},
        ],
    }},
    # dialogue_instance_id 含路径字符
    {**fx.daily_request(), "dialogue_instance_id": "../evil"},
    # task 模式缺 task_context
    {**fx.task_request(), "task_context": None},
    # played seq 重复
    {**fx.daily_request(), "dialogue": {
        **fx.daily_request()["dialogue"],
        "played": [
            {"seq": 0, "role": "npc", "name": fx.NPC, "text": "一"},
            {"seq": 0, "role": "npc", "name": fx.NPC, "text": "二"},
        ],
    }},
])
def test_request_validation_rejects_bad_contract(memory_store, payload):
    client = _client(_factory_for({}))
    assert client.post("/game/dialogue-interject", json=payload).status_code == 422


def test_endpoint_keeps_error_after_partial_output(memory_store):
    async def failing(request):
        yield '{"add":"你先听我说。","emo":"普通"}\n'
        raise RuntimeError("connection reset")

    client = _client(failing)
    response = client.post("/game/dialogue-interject", json=fx.daily_request())
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "line", "error"]
    assert events[-1][1]["code"] == "stream_failed"
    assert events[-1][1]["valid_lines"] == 1
    assert "plan" not in [name for name, _ in events]
    assert "done" not in [name for name, _ in events]
