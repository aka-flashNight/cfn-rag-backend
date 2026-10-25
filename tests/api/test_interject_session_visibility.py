"""B3：插话 session 在终端 `/api/game/sessions` 可见，且按绑定槽位隔离。

NpcManager 用桩，MemoryStore 指向 tmp 库；不调用模型、不碰真实数据。
"""

from __future__ import annotations

import asyncio
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.game_api as game_api
import services.memory.store as store_module
from services.dialogue_interject.schemas import InterjectRequest
from services.dialogue_interject.sessions import ensure_session
from services.integration.state import get_integration_state
from services.memory.store import MemoryStore
from services.npc.manager import get_npc_manager
from tests.dialogue_interject import fixtures as fx


class _StubNpcManager:
    async def all_states(self) -> dict:
        return {}


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    previous = store_module._MEMORY_STORE
    memory = MemoryStore(db_path=tmp_path / "memory.db")
    store_module.set_memory_store(memory)
    try:
        yield memory
    finally:
        store_module.set_memory_store(previous)
        memory.close()


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


def _interject_session() -> None:
    """按默认标题建一条插话 session（session_title 留空走 `日常插话 · NPC · 时间`）。"""
    payload = fx.daily_request(session_title="")
    request = InterjectRequest.model_validate(payload)
    asyncio.run(ensure_session(request))


def test_interject_session_listed_with_default_title(client, state, store):
    state.bind(slot_key="slot_a")
    _interject_session()

    sessions = client.get("/api/game/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["npc_name"] == fx.NPC
    assert re.fullmatch(
        rf"日常插话 · {re.escape(fx.NPC)} · \d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}",
        sessions[0]["title"],
    )
    # 终端拿到的 session_id 就是 MemoryStore 里那条（继续聊用的是同一把钥匙）
    assert sessions[0]["session_id"] == store.dialogue_session_get_sync(
        fx.DAILY_DIALOGUE_ID
    ).session_id


def test_interject_session_is_slot_scoped(client, state):
    state.bind(slot_key="slot_a")
    _interject_session()
    state.bind(slot_key="slot_b")
    assert client.get("/api/game/sessions").json()["sessions"] == []
    state.unbind()
    assert client.get("/api/game/sessions").json()["sessions"] == []
    state.bind(slot_key="slot_a")
    assert len(client.get("/api/game/sessions").json()["sessions"]) == 1


def test_same_dialogue_instance_does_not_duplicate_in_list(client, state):
    state.bind(slot_key="slot_a")
    _interject_session()
    _interject_session()
    assert len(client.get("/api/game/sessions").json()["sessions"]) == 1
