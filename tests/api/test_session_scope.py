"""B2-5 会话作用域：/api/game/sessions 的创建落槽与列表过滤。

- 创建时记录当前绑定槽位；未绑定＝NULL（standalone）；
- 列表按当前绑定过滤：bind 态只回该槽会话，standalone 只回 NULL 会话。
NpcManager 用桩（all_states→{}），MemoryStore 用 tmp 库，不碰真实数据。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.game_api as game_api
from services.integration.state import get_integration_state
from services.memory.store import MemoryStore
from services.npc.manager import get_npc_manager


class _StubNpcManager:
    async def all_states(self) -> dict:
        return {}


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "memory.db")


@pytest.fixture
def client(store, monkeypatch) -> TestClient:
    monkeypatch.setattr(game_api, "get_memory_store", lambda: store)
    app = FastAPI()
    app.include_router(game_api.router, prefix="/api/game")
    app.dependency_overrides[get_npc_manager] = lambda: _StubNpcManager()
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


def _create(client: TestClient, title: str) -> str:
    resp = client.post("/api/game/sessions", json={"npc_name": "铁匠", "title": title})
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def _titles(client: TestClient) -> list[str]:
    resp = client.get("/api/game/sessions")
    assert resp.status_code == 200, resp.text
    return sorted(s["title"] for s in resp.json()["sessions"])


def test_create_stamps_current_bound_slot(client, state, store):
    state.bind(slot_key="slot_a")
    sid = _create(client, "绑定槽会话")
    assert [x.session_id for x in store.list_sessions_sync("slot_a")] == [sid]
    assert store.list_sessions_sync(None) == []


def test_unbound_create_leaves_slot_null(client, state, store):
    _create(client, "standalone 会话")
    assert store.list_sessions_sync(None)[0].slot_key is None


def test_list_filters_by_current_scope(client, state):
    _create(client, "standalone-1")
    _create(client, "standalone-2")

    state.bind(slot_key="slot_a")
    _create(client, "a-1")

    state.bind(slot_key="slot_b")
    _create(client, "b-1")

    assert _titles(client) == ["b-1"]
    state.bind(slot_key="slot_a")
    assert _titles(client) == ["a-1"]
    state.unbind()
    assert _titles(client) == ["standalone-1", "standalone-2"]


def test_creation_slot_does_not_follow_later_bind(client, state):
    """归属看创建当刻：standalone 建的会话在之后 bind 的槽位列表里不可见。"""
    _create(client, "先建后绑")
    state.bind(slot_key="slot_a")
    assert _titles(client) == []
