"""对话内嵌插话测试的共享夹具。

所有测试离线：假流回放 JSONL，MemoryStore 指向 tmp_path，不碰生产 data/。
"""

from __future__ import annotations

import pytest

import services.memory.store as store_module
from services.memory.store import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    """临时 memory.db；替换全局单例，teardown 关闭并还原（不创建生产库）。"""
    previous = store_module._MEMORY_STORE
    store = MemoryStore(tmp_path / "memory.db")
    store_module.set_memory_store(store)
    try:
        yield store
    finally:
        store_module.set_memory_store(previous)
        store.close()


@pytest.fixture(autouse=True)
def _forbid_real_llm(monkeypatch):
    """本目录测试一律离线：任何未注入假客户端的真 LLM 调用直接失败。"""
    from services.llm.client import LLMClient

    def _boom(cls, cfg):
        raise RuntimeError("测试禁止调用真实 LLM：请注入假流/假客户端")

    monkeypatch.setattr(LLMClient, "for_config", classmethod(_boom))
    yield
