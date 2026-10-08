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
    assert st["mode"] == "embedded"

    resp = client.post("/integration/unbind")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "standalone"
    st2 = client.get("/integration/status").json()
    assert st2["bound_slot"] is None
    assert st2["character"] is None
    assert st2["frontend_url"] == "http://127.0.0.1:7077/"


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
