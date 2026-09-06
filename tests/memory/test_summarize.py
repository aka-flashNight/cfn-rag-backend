"""摘要 worker 测试（06 §1.3）：有界队列、串行执行、超时重试放弃、走统一 LLMClient。"""

from __future__ import annotations

import asyncio

import pytest

from services.llm import LLMConfig
from services.memory.store import MemoryStore
from services.memory.summarize import (
    SummaryRequest,
    SummaryWorker,
    _build_summary_prompt,
    should_summarize,
)

from tests.conftest import FakeOpenAI, FakeResponse


def test_should_summarize_interval():
    assert should_summarize(30, 30)
    assert should_summarize(60, 30)
    assert not should_summarize(31, 30)
    assert not should_summarize(0, 30)


def test_summary_prompt_rolls_existing():
    from services.memory.store import ChatMessage

    msgs = [
        ChatMessage(id=1, role="user", content="你好", timestamp=0.0),
        ChatMessage(id=2, role="assistant", content="欢迎", timestamp=0.0),
    ]
    p1 = _build_summary_prompt("铁匠", msgs, None)
    assert "铁匠" in p1 and "你好" in p1
    p2 = _build_summary_prompt("铁匠", msgs, "旧摘要内容")
    assert "旧摘要内容" in p2 and "补充" in p2


class _FakeLLM:
    """替换 LLMClient.for_config 的假客户端（for_config 共享单实例，模拟真实客户端缓存）。"""

    instances: list["_FakeLLM"] = []
    behavior = "ok"

    def __init__(self, cfg=None):
        self.prompts: list[str] = []
        self.calls = 0
        type(self).instances.append(self)

    @classmethod
    def for_config(cls, cfg):
        if not cls.instances:
            cls(cfg)
        return cls.instances[-1]

    async def chat(self, req):
        self.calls += 1
        self.prompts.append(req.messages[0]["content"])
        assert req.purpose == "summary"
        assert req.send_image is False
        if self.behavior == "empty":
            from services.llm import ChatResult

            return ChatResult(content="  ")
        from services.llm import ChatResult

        return ChatResult(content="整合后的滚动摘要")

    async def chat_stream(self, req):  # pragma: no cover - 摘要不用流式
        raise AssertionError("摘要不应走流式")


@pytest.fixture
def patched_llm(monkeypatch):
    _FakeLLM.instances.clear()

    import services.memory.summarize as summarize_mod

    monkeypatch.setattr(summarize_mod, "LLMClient", _FakeLLM)
    yield _FakeLLM


async def test_worker_summarizes_and_saves(tmp_path, patched_llm):
    store = MemoryStore(db_path=tmp_path / "m.db")
    for i in range(4):
        store.add_message_sync("s1", "user" if i % 2 == 0 else "assistant", f"msg{i}")

    worker = SummaryWorker(store)
    assert worker.submit(SummaryRequest(session_id="s1", npc_name="铁匠", llm_config=LLMConfig()))
    worker.start()
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()

    assert store.get_summary_sync("s1") == "整合后的滚动摘要"
    fake = patched_llm.instances[0]
    assert fake.calls == 1
    assert "msg0" in fake.prompts[0]


async def test_worker_retries_then_gives_up(tmp_path, patched_llm, monkeypatch):
    """失败重试 1 次后放弃本轮（不崩溃、不丢队列）。"""
    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "你好")

    class FailingLLM(_FakeLLM):
        async def chat(self, req):
            self.calls += 1
            raise RuntimeError("LLM 挂了")

    monkeypatch.setattr("services.memory.summarize.LLMClient", FailingLLM)
    FailingLLM.instances.clear()

    worker = SummaryWorker(store)
    worker.start()
    worker.submit(SummaryRequest(session_id="s1", npc_name="铁匠", llm_config=LLMConfig()))
    await asyncio.wait_for(worker._queue.join(), timeout=10)
    await worker.stop()

    fake = FailingLLM.instances[0]
    assert fake.calls == 2  # 重试 1 次
    assert store.get_summary_sync("s1") is None  # 放弃本轮


async def test_worker_empty_response_not_saved(tmp_path, patched_llm, monkeypatch):
    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "你好")

    class EmptyLLM(_FakeLLM):
        behavior = "empty"

    import services.memory.summarize as summarize_mod

    monkeypatch.setattr(summarize_mod, "LLMClient", EmptyLLM)
    EmptyLLM.instances.clear()
    worker = SummaryWorker(store)
    worker.start()
    worker.submit(SummaryRequest(session_id="s1", npc_name="铁匠", llm_config=LLMConfig()))
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()
    assert store.get_summary_sync("s1") is None


async def test_bounded_queue_drops_when_full(tmp_path, patched_llm):
    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "你好")
    worker = SummaryWorker(store)
    # 不启动 worker，把队列塞满
    accepted = 0
    for i in range(20):
        if worker.submit(SummaryRequest(session_id=f"s{i}", npc_name="铁匠", llm_config=LLMConfig())):
            accepted += 1
    assert accepted == 16  # 队列上限 16，超出丢弃
    await worker.stop()
