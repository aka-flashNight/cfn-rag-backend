"""集成测试共享夹具：临时 MemoryStore + 假流工厂复位。"""

from __future__ import annotations

import pytest

import services.memory.store as store_module
from services.dialogue_interject.api import set_stream_factories
from services.memory.store import MemoryStore


@pytest.fixture
def fake_memory(tmp_path):
    previous = store_module._MEMORY_STORE
    store = MemoryStore(tmp_path / "memory.db")
    store_module.set_memory_store(store)
    try:
        yield store
    finally:
        store_module.set_memory_store(previous)
        store.close()


@pytest.fixture(autouse=True)
def _reset_stream_factories():
    yield
    set_stream_factories(None, None)
    # 假服务允许内存 bind；测试间复位，避免 slot_key 串味
    from services.integration.state import get_integration_state

    get_integration_state().unbind()


@pytest.fixture(autouse=True)
def _forbid_real_llm(monkeypatch):
    """集成测试一律走假流：真 LLM 客户端构造直接失败。"""
    from services.llm.client import LLMClient

    def _boom(cls, cfg):
        raise RuntimeError("集成测试禁止调用真实 LLM：请使用假流")

    monkeypatch.setattr(LLMClient, "for_config", classmethod(_boom))
    yield
