"""摘要 worker 测试（06 §1.3）：有界队列、串行执行、超时重试放弃、走统一 LLMClient；
以及基础长期记忆与摘要同轮产出（B3-2）。"""

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


# ---------------------------------------------------------------------------
# 基础长期记忆与摘要同轮产出（B3-2）
# ---------------------------------------------------------------------------

from services.memory.long_term import (  # noqa: E402
    SUMMARY_TOOL_NAME,
    LongTermSnapshot,
    format_summary_block,
)
from services.memory.store import ChatMessage  # noqa: E402

MSGS = [
    ChatMessage(id=1, role="user", content="帮我把铁锤找回来了", timestamp=0.0),
    ChatMessage(id=2, role="assistant", content="这份情我记着", timestamp=0.0),
]


class _RecordingLLM(_FakeLLM):
    """记录 tools 与 prompt，并可脚本化返回 tool_calls。"""

    tool_calls_to_return: list[dict] = []

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.tools_seen: list = []

    async def chat(self, req):
        self.calls += 1
        self.prompts.append(req.messages[0]["content"])
        self.tools_seen.append(req.tools)
        from services.llm import ChatResult

        return ChatResult(content="整合后的滚动摘要", tool_calls=list(type(self).tool_calls_to_return))


@pytest.fixture
def recording_llm(monkeypatch):
    _RecordingLLM.instances.clear()
    _RecordingLLM.tool_calls_to_return = []
    monkeypatch.setattr("services.memory.summarize.LLMClient", _RecordingLLM)
    yield _RecordingLLM


async def _run_worker(store, req: SummaryRequest) -> None:
    worker = SummaryWorker(store)
    worker.start()
    worker.submit(req)
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()


def test_prompt_byte_identical_when_long_term_off():
    """未启用长期记忆＝提示词逐字不变，且不出现任何长期记忆字样。"""
    plain = _build_summary_prompt("铁匠", MSGS, "旧摘要")
    explicit_off = _build_summary_prompt("铁匠", MSGS, "旧摘要", long_term_block="")
    assert plain == explicit_off
    assert "长期记忆" not in plain
    assert "基础印象" not in plain


def test_prompt_adds_isolated_long_term_block():
    """启用时长期记忆是独立块，并写清它不参与摘要撰写。"""
    block = format_summary_block(
        LongTermSnapshot(slot_key="slot_1", npc="铁匠", base="整体印象。")
    )
    prompt = _build_summary_prompt("铁匠", MSGS, None, long_term_block=block)

    assert "【长期记忆（跨对话窗口保留）】" in prompt
    assert "长期记忆不参与对话摘要的撰写" in prompt
    assert "无论是否更新基础印象，都必须先输出摘要正文。" in prompt
    # 对话记录本身照旧在
    assert "帮我把铁锤找回来了" in prompt


async def test_worker_without_scope_sends_no_tools(tmp_path, recording_llm):
    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "你好")
    await _run_worker(store, SummaryRequest(session_id="s1", npc_name="铁匠", llm_config=LLMConfig()))

    fake = recording_llm.instances[0]
    assert fake.tools_seen == [None]
    assert "长期记忆" not in fake.prompts[0]
    assert store.get_summary_sync("s1") == "整合后的滚动摘要"


async def test_worker_with_scope_produces_base_same_round(tmp_path, recording_llm):
    """同一次调用：摘要走 content，基础印象走 tool_calls。"""
    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "帮我把铁锤找回来了")
    recording_llm.tool_calls_to_return = [{
        "id": "b1", "type": "function",
        "function": {"name": SUMMARY_TOOL_NAME, "arguments": '{"content": "你是个靠得住的老主顾。"}'},
    }]

    await _run_worker(store, SummaryRequest(
        session_id="s1", npc_name="铁匠", llm_config=LLMConfig(), long_term_scope="slot_1",
    ))

    fake = recording_llm.instances[0]
    assert fake.calls == 1
    tool_names = [t["function"]["name"] for t in fake.tools_seen[0]]
    assert tool_names == [SUMMARY_TOOL_NAME]
    assert store.get_summary_sync("s1") == "整合后的滚动摘要"
    assert store.ltm_get_base_sync("slot_1", "铁匠") == "你是个靠得住的老主顾。"


async def test_worker_scope_comes_from_request_not_live_state(tmp_path, recording_llm):
    """作用域在提交时就解析好：worker 延后执行期间换存档也不会写错地方。"""
    from services.integration.state import get_integration_state

    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "你好")
    recording_llm.tool_calls_to_return = [{
        "id": "b1", "type": "function",
        "function": {"name": SUMMARY_TOOL_NAME, "arguments": '{"content": "印象。"}'},
    }]

    state = get_integration_state()
    state.bind("slot_9")
    try:
        await _run_worker(store, SummaryRequest(
            session_id="s1", npc_name="铁匠", llm_config=LLMConfig(), long_term_scope="slot_1",
        ))
    finally:
        state.unbind()

    assert store.ltm_get_base_sync("slot_1", "铁匠") == "印象。"
    assert store.ltm_get_base_sync("slot_9", "铁匠") is None


async def test_worker_without_tool_call_keeps_base(tmp_path, recording_llm):
    """近期对话没有明显变化＝模型不发调用，基础印象保持原样，摘要照旧更新。"""
    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "你好")
    store.ltm_save_base_sync("slot_1", "铁匠", "原有印象。", "s0")

    await _run_worker(store, SummaryRequest(
        session_id="s1", npc_name="铁匠", llm_config=LLMConfig(), long_term_scope="slot_1",
    ))

    assert store.ltm_get_base_sync("slot_1", "铁匠") == "原有印象。"
    assert store.get_summary_sync("s1") == "整合后的滚动摘要"


async def test_base_write_failure_does_not_block_summary(tmp_path, recording_llm, monkeypatch):
    store = MemoryStore(db_path=tmp_path / "m.db")
    store.add_message_sync("s1", "user", "你好")
    recording_llm.tool_calls_to_return = [{
        "id": "b1", "type": "function",
        "function": {"name": SUMMARY_TOOL_NAME, "arguments": '{"content": "印象。"}'},
    }]

    async def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "ltm_save_base", _boom)
    await _run_worker(store, SummaryRequest(
        session_id="s1", npc_name="铁匠", llm_config=LLMConfig(), long_term_scope="slot_1",
    ))

    assert store.get_summary_sync("s1") == "整合后的滚动摘要"
