"""integration API 测试（B1-4）：bind/unbind/status 幂等、换槽、模式口径、slot_key 校验。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.integration.state import get_integration_state


def _make_client() -> TestClient:
    from api.integration_api import router

    app = FastAPI()
    app.include_router(router, prefix="/integration")
    return TestClient(app)


@pytest.fixture
def state():
    st = get_integration_state()
    original_launch = st.launch_mode
    st.unbind()
    st.launch_mode = "standalone"
    yield st
    st.unbind()
    st.launch_mode = original_launch


def test_bind_status_unbind_flow(state):
    client = _make_client()
    payload = {
        "slot_key": "cf7_0123456789abcdef0123456789ab",
        "character": {"name": "林三", "level": 12},
        "progress": {"tasks_finished": [200001], "task_chains_progress": {"主线甲": 3}},
        "scene": {"stageFlag": "基地1层", "stageName": "", "entrance": "出生地", "inCombat": False},
        "player": {"gender": "男", "level": 12, "title": {"final": "契约者"}},
    }

    resp = client.post("/integration/bind", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["mode"] == "embedded"  # bind 后按 embedded 口径（需求 §3）
    assert body["frontend_url"] == "http://127.0.0.1:7077/?embedded=1"

    st = client.get("/integration/status").json()
    assert st["bound_slot"] == payload["slot_key"]
    assert st["character"]["name"] == "林三"
    assert st["progress"]["tasks_finished"] == [200001]
    assert st["scene"] == payload["scene"]
    assert st["player"] == payload["player"]
    assert st["mode"] == "embedded"
    # 非「主线」链值不可判 → stage 空（前端手选退居兜底）
    assert st["stage"] is None
    assert st["stage_name"] is None

    resp = client.post("/integration/unbind")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "standalone"
    st2 = client.get("/integration/status").json()
    assert st2["bound_slot"] is None
    assert st2["character"] is None
    assert st2["scene"] is None
    assert st2["player"] is None
    assert st2["frontend_url"] == "http://127.0.0.1:7077/"


def test_status_derives_stage_from_mainline_and_level(state):
    """阶段推导接线：主线链值定 1-5，全完成后按等级模拟 5/6/7；解绑后回到 null。"""
    client = _make_client()

    def stage_with(chain: int, level: int):
        client.post("/integration/bind", json={
            "slot_key": "slot_stage",
            "progress": {"tasks_finished": [], "task_chains_progress": {"主线": chain}},
            "player": {"level": level},
        })
        st = client.get("/integration/status").json()
        return st["stage"], st["stage_name"]

    assert stage_with(21, 10) == (1, "废城")
    assert stage_with(22, 10) == (2, "堕落城")
    assert stage_with(36, 15) == (3, "荒漠军阀")
    assert stage_with(37, 20) == (4, "黑铁会总堂")
    assert stage_with(68, 30) == (5, "诺亚")
    assert stage_with(780, 39) == (5, "诺亚")
    assert stage_with(780, 45) == (6, "雪山")
    assert stage_with(780, 55) == (7, "已通关")

    client.post("/integration/unbind")
    st = client.get("/integration/status").json()
    assert st["stage"] is None
    assert st["stage_name"] is None


def test_bind_scene_defaults_when_partial(state):
    """scene 段字段缺省容忍（Host 侧已验证形状；此处保证对缺键对象的兼容）。"""
    client = _make_client()
    resp = client.post("/integration/bind", json={"slot_key": "slot_scene", "scene": {}})
    assert resp.status_code == 200
    st = client.get("/integration/status").json()
    assert st["scene"] == {"stageFlag": "", "stageName": "", "entrance": "", "inCombat": False}


def test_bind_same_slot_is_idempotent_refresh(state):
    client = _make_client()
    assert (
        client.post("/integration/bind", json={"slot_key": "crazyflasher7_saves", "character": {"name": "甲"}}).status_code
        == 200
    )
    assert (
        client.post("/integration/bind", json={"slot_key": "crazyflasher7_saves", "character": {"name": "乙"}}).status_code
        == 200
    )
    st = client.get("/integration/status").json()
    assert st["bound_slot"] == "crazyflasher7_saves"
    assert st["character"]["name"] == "乙"  # 同槽位重复 bind＝投影刷新（覆盖式）


def test_bind_switch_slot(state):
    client = _make_client()
    client.post("/integration/bind", json={"slot_key": "slot_a"})
    client.post("/integration/bind", json={"slot_key": "slot_b"})
    assert client.get("/integration/status").json()["bound_slot"] == "slot_b"


def test_bind_calls_ensure_agent_save_files(state, monkeypatch):
    """B2-4 接线：bind 成功后按槽位键调 ensure（落盘行为见 tests/test_startup_ensure.py）。"""
    from api import integration_api

    calls: list[str] = []
    monkeypatch.setattr(integration_api, "ensure_agent_save_files", calls.append)

    resp = _make_client().post("/integration/bind", json={"slot_key": "slot_ensure"})
    assert resp.status_code == 200
    assert calls == ["slot_ensure"]


def test_bind_checks_index_but_unbind_does_not(state, monkeypatch):
    """bind 是一次语义节点，复核语料指纹；unbind 不触发——索引与存档无关，
    覆盖层也不再进语料（后台重建行为见 tests/retrieval/test_refresh.py）。"""
    from api import integration_api

    calls: list[int] = []
    monkeypatch.setattr(integration_api, "refresh_index_if_stale", lambda: calls.append(1))

    client = _make_client()
    assert client.post("/integration/bind", json={"slot_key": "slot_refresh"}).status_code == 200
    assert calls == [1]
    assert client.post("/integration/unbind").status_code == 200
    assert calls == [1]


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", "a" * 129, "a b", "存档"])
def test_bind_rejects_invalid_slot_key(state, bad):
    client = _make_client()
    resp = client.post("/integration/bind", json={"slot_key": bad})
    assert resp.status_code == 422
    assert client.get("/integration/status").json()["bound_slot"] is None


def test_launch_mode_embedded_without_bind(state):
    state.launch_mode = "embedded"
    client = _make_client()
    st = client.get("/integration/status").json()
    assert st["mode"] == "embedded"
    assert st["launch_mode"] == "embedded"
    assert st["bound_slot"] is None
    assert st["frontend_url"] == "http://127.0.0.1:7077/?embedded=1"


def test_shutdown_rejected_for_standalone(state):
    from services.integration.state import set_process_shutdown_callback

    calls: list[int] = []
    set_process_shutdown_callback(lambda: calls.append(1))
    try:
        assert state.launch_mode == "standalone"
        resp = _make_client().post("/integration/shutdown")
        assert resp.status_code == 409
        assert calls == []  # 用户手动启动的进程即使被 bind 也绝不触发退出
    finally:
        set_process_shutdown_callback(None)


def test_shutdown_embedded_requests_process_exit(state):
    from services.integration.state import set_process_shutdown_callback

    calls: list[int] = []
    set_process_shutdown_callback(lambda: calls.append(1))
    try:
        state.launch_mode = "embedded"
        resp = _make_client().post("/integration/shutdown")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "status": "shutting_down"}
        assert calls == [1]
    finally:
        set_process_shutdown_callback(None)


def test_shutdown_without_callback_unavailable(state):
    from services.integration.state import set_process_shutdown_callback

    set_process_shutdown_callback(None)
    state.launch_mode = "embedded"
    resp = _make_client().post("/integration/shutdown")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# task-completed：与游戏 Host 的线格式对齐（阶段四触发一）
# ---------------------------------------------------------------------------

# C# 侧 RagTerminalService.NotifyTaskCompletedAsync 实际发出的字节（JObject 键序＋Formatting.None），
# 见 launcher/tests/RagTerminal/RagTerminalServiceTests.cs::TaskCompletedPostsSingleEventAndToleratesFailure。
HOST_WIRE_BODY = '{"slot_key":"save1","task_id":213501,"npc":"Andy Law"}'


def test_task_completed_forwards_host_wire_body(state, monkeypatch):
    """跨仓线格式锁：Host 原样发出的 body 必须被解析成同一组字段，并回 Host 认识的响应形状。"""
    from types import SimpleNamespace

    from api import integration_api

    calls: list[dict] = []

    async def _recorder(*, slot_key, task_id, npc):
        calls.append({"slot_key": slot_key, "task_id": task_id, "npc": npc})
        return SimpleNamespace(accepted=True, detail="已受理", job_id="job-1")

    monkeypatch.setattr(integration_api, "handle_task_completed", _recorder)

    resp = _make_client().post(
        "/integration/task-completed",
        content=HOST_WIRE_BODY,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "accepted": True, "detail": "已受理", "job_id": "job-1"}
    assert calls == [{"slot_key": "save1", "task_id": 213501, "npc": "Andy Law"}]


def test_task_completed_npc_defaults_to_empty(state, monkeypatch):
    """npc 缺省＝空串（Host 侧读不到发布 NPC 时发空；后端以任务记录的 get_npc 为权威）。"""
    from types import SimpleNamespace

    from api import integration_api

    calls: list[dict] = []

    async def _recorder(*, slot_key, task_id, npc):
        calls.append({"slot_key": slot_key, "task_id": task_id, "npc": npc})
        return SimpleNamespace(accepted=False, detail="未挂标记", job_id="")

    monkeypatch.setattr(integration_api, "handle_task_completed", _recorder)

    resp = _make_client().post(
        "/integration/task-completed",
        content='{"slot_key":"save1","task_id":213501}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert calls == [{"slot_key": "save1", "task_id": 213501, "npc": ""}]


def test_task_completed_unbound_slot_is_not_accepted(state):
    """未绑定槽位时走真处理器：200＋accepted=False，且不触碰委派存储（换档后旧事件一律忽略）。"""
    resp = _make_client().post(
        "/integration/task-completed",
        content=HOST_WIRE_BODY,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["accepted"] is False
    assert body["job_id"] == ""
    assert "槽位" in body["detail"]


@pytest.mark.parametrize("payload", [
    '{"slot_key":"a/b","task_id":213501}',      # 槽位键含路径分隔符
    '{"slot_key":"","task_id":213501}',         # 空槽位
    '{"slot_key":"save1","task_id":0}',         # 非正整数 id
    '{"slot_key":"save1","task_id":"abc"}',     # id 非数字（"213501" 会被 pydantic 强转，不算畸形）
    '{"slot_key":"save1"}',                    # 缺 task_id
    '{"task_id":213501}',                      # 缺 slot_key
])
def test_task_completed_rejects_invalid_shape(state, payload):
    resp = _make_client().post(
        "/integration/task-completed",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
