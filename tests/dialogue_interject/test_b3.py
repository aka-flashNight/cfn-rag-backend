"""B3：confirm 同源、草案关闭保留、委派降权、提示词纯净化与 status。

全部离线：假 LLM/假工具/临时 MemoryStore；不写真实任务、存档、覆盖层。
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from types import SimpleNamespace

import pytest

import services.dialogue_interject.tools as tools_module
from services.dialogue_interject.context import DailyBlocks, assemble_daily_blocks
from services.dialogue_interject.llm import stream_daily_llm
from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.prompts import build_daily_system_prompt
from services.dialogue_interject.sessions import ensure_session
from services.dialogue_interject.tools import DailyToolRuntime
from services.llm import LLMConfig, StreamEvent
from services.npc.manager import NPCState
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import collect_events, make_request


DRAFT = {
    "draft_id": "d1",
    "title": "城西补给线",
    "publish_npc": fx.NPC,
    "finish_requirements": [],
    "finish_submit_items": [],
    "rewards": [],
}
DRAFT_ROW = SimpleNamespace(draft=DRAFT, draft_id="d1", bargain_count=0)
STATE = NPCState(
    favorability=3,
    relationship_level="熟悉",
    sex="男",
    emotions=["普通", "微笑"],
    faction="A兵团",
    titles=["军需官"],
)


def _config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key", api_base="http://unit.test/v1", model_name="test-model",
    )


class FakeChatClient:
    """只实现 confirm 用到的 chat()。"""

    def __init__(self, content: str = "", error: Exception | None = None):
        self.content = content
        self.error = error
        self.requests: list = []

    async def chat(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content, usage={"total_tokens": 12})


class FakeMemory:
    def __init__(self):
        self.deleted: list[tuple[str, str]] = []

    async def delete_draft(self, session_id: str, *, draft_id: str) -> None:
        self.deleted.append((session_id, draft_id))


def _blocks(**overrides) -> DailyBlocks:
    payload = dict(
        system_prompt="SYS",
        npc_state=STATE,
        npc_states={fx.NPC: STATE},
        game_data=None,
        registry=None,
        memory=FakeMemory(),
        session_id=fx.DAILY_DIALOGUE_ID,
        pending_draft_row=DRAFT_ROW,
        pending_draft_summary="城西补给线：去城西看一眼，带回货物；报酬 2400 金币。",
        progress_stage=2,
    )
    payload.update(overrides)
    return DailyBlocks(**payload)


def _runtime(blocks: DailyBlocks, llm) -> DailyToolRuntime:
    return DailyToolRuntime(make_request(fx.daily_request()), blocks, llm=llm)


def _capture_confirm(monkeypatch, payload: dict | None = None) -> dict:
    captured: dict = {}

    def _fake(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(payload=payload or {"status": "confirmed", "task_id": 7})

    monkeypatch.setattr(tools_module, "execute_confirm_agent_task", _fake)
    return captured


CONFIRM_JSON = (
    '{"title":"夜行城西","description":"去城西看一眼，把货带回来。",'
    '"get_dialogue":[{"name":"Andy Law","emotion":"认真","text":"这活我交给你。"}],'
    '"finish_dialogue":[{"name":"Andy Law","emotion":"满意","text":"干得不错。"}],'
    '"return_for_reward":false}'
)


async def test_confirm_uses_generated_args_and_records_generated(monkeypatch, caplog):
    captured = _capture_confirm(monkeypatch)
    client = FakeChatClient(CONFIRM_JSON)
    blocks = _blocks()
    with caplog.at_level("INFO", logger="services.dialogue_interject.tools"):
        result = await _runtime(blocks, client)._run_confirm()

    assert captured["args"]["title"] == "夜行城西"
    assert captured["args"]["description"].startswith("去城西看一眼")
    assert captured["args"]["draft_id"] == "d1"
    assert captured["kwargs"]["generation_warnings"] is None
    assert blocks.memory.deleted == [(fx.DAILY_DIALOGUE_ID, "d1")]
    assert "已经正式派给玩家" in result[0]
    assert "confirm 文案" in caplog.text and "source=generated" in caplog.text
    assert "confirm applied" in caplog.text
    # 生成走的是插话现场的 system+user 前缀
    assert client.requests[0].messages[0] == {"role": "system", "content": "SYS"}
    assert "最近怎么样？" in client.requests[0].messages[1]["content"]


async def test_confirm_falls_back_to_template_when_generation_fails(monkeypatch, caplog):
    captured = _capture_confirm(monkeypatch)
    client = FakeChatClient(error=RuntimeError("llm down"))
    blocks = _blocks()
    with caplog.at_level("INFO", logger="services.dialogue_interject.tools"):
        result = await _runtime(blocks, client)._run_confirm()

    assert captured["args"]["title"] == "城西补给线"  # 模板取草案标题
    assert captured["args"]["draft_id"] == "d1"
    assert captured["args"]["get_dialogue"]
    assert "source=fallback" in caplog.text
    assert "已经正式派给玩家" in result[0]


async def test_confirm_publish_failure_keeps_draft_and_logs_rejected(monkeypatch, caplog):
    _capture_confirm(
        monkeypatch, payload={"status": "error", "message": "草案校验未通过"},
    )
    client = FakeChatClient(CONFIRM_JSON)
    blocks = _blocks()
    with caplog.at_level("INFO", logger="services.dialogue_interject.tools"):
        result = await _runtime(blocks, client)._run_confirm()

    assert "发布失败" in result[0]
    assert "草案校验未通过" in result[0]
    assert blocks.memory.deleted == []
    assert "confirm rejected" in caplog.text


# ---------------------------------------------------------------------------
# 草案关闭保留 + 终端继续
# ---------------------------------------------------------------------------


class _StubNpcManager:
    async def get(self, _name: str) -> NPCState:
        return STATE

    async def all_states(self) -> dict[str, NPCState]:
        return {fx.NPC: STATE}


class _FakeGameData:
    map_registry = None


async def test_disconnect_keeps_draft_and_terminal_can_cancel(memory_store, monkeypatch):
    from services.integration.state import get_integration_state

    get_integration_state().unbind()
    request = make_request(fx.daily_request())
    binding = await ensure_session(request, store=memory_store)

    async def _factory(_request):
        await memory_store.upsert_draft(binding.session_id, DRAFT, bargain_count=1)
        yield '{"add":"这活给你。"}\n'
        raise asyncio.CancelledError()

    orchestrator = InterjectOrchestrator(
        request, session_id=binding.session_id, stream_factory=_factory,
    )
    with pytest.raises(asyncio.CancelledError):
        await collect_events(orchestrator)

    # 草案不被删除，且落在终端能读到的那把 session 钥匙上
    row = memory_store.get_draft_sync(binding.session_id)
    assert row is not None
    assert row.draft["draft_id"] == "d1"
    assert row.bargain_count == 1
    # 未播不写：台账里只有请求带进来的 seq0
    assert [item.seq for item in memory_store.dialogue_ledger_read_sync(binding.session_id)] == [0]

    # 终端继续聊：按 session_id 装配上下文能看到这份草案
    blocks = await assemble_daily_blocks(
        request,
        npc_manager=_StubNpcManager(),
        game_data=_FakeGameData(),
        memory=memory_store,
        rag_enabled=False,
        session_id=binding.session_id,
    )
    assert blocks.pending_draft_row is not None
    assert blocks.session_id == binding.session_id

    # 终端（同一把 key）可取消这份草案
    monkeypatch.setattr(tools_module, "execute_cancel_agent_task", lambda *a, **k: None)
    await _runtime(blocks, None)._run_cancel()
    assert memory_store.get_draft_sync(binding.session_id) is None


async def test_legacy_draft_under_dialogue_instance_id_still_readable(memory_store):
    from services.integration.state import get_integration_state

    get_integration_state().unbind()
    request = make_request(fx.daily_request())
    await memory_store.upsert_draft(fx.DAILY_DIALOGUE_ID, DRAFT, bargain_count=0)

    blocks = await assemble_daily_blocks(
        request,
        npc_manager=_StubNpcManager(),
        game_data=_FakeGameData(),
        memory=memory_store,
        rag_enabled=False,
    )
    assert blocks.pending_draft_row is not None
    assert blocks.session_id == fx.DAILY_DIALOGUE_ID


# ---------------------------------------------------------------------------
# 委派降权
# ---------------------------------------------------------------------------

DELEGATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_dialogue_rewrite",
        "description": "delegate",
        "parameters": {"type": "object", "properties": {}},
    },
}
DELEGATE_CALL = {
    "type": "function",
    "id": "call_delegate_1",
    "function": {
        "name": "delegate_dialogue_rewrite",
        "arguments": '{"change_type":"relationship_change","reason":"玩家救了整队"}',
    },
}


class _FakeRegistry:
    def get_openai_tools(self, names=None):
        return [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
            for name in (names or [])
        ]


class _ScriptedLLM:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        items = self.rounds.pop(0) if self.rounds else []
        for text, tool_calls in items:
            if text:
                yield StreamEvent(kind="content", text=text)
            if tool_calls is not None:
                yield StreamEvent(kind="finish", tool_calls=tool_calls)


async def test_delegate_round_returns_to_final_dialogue_without_blocking(monkeypatch):
    worker_started = asyncio.Event()
    worker_release = asyncio.Event()
    calls: list[dict] = []

    async def _fake_apply(tool_calls, *, npc_name, session_id, llm_config=None):
        calls.append({"tool_calls": tool_calls, "npc": npc_name, "session": session_id})
        worker_started.set()
        # 模拟后台 worker：只起任务，不 await（受理方绝不阻塞对话）
        asyncio.get_running_loop().create_task(worker_release.wait())
        return ["relationship_change：已受理"]

    monkeypatch.setattr(tools_module, "apply_delegate_calls", _fake_apply)
    monkeypatch.setattr(
        tools_module, "delegate_tool_or_none",
        lambda **_kwargs: DELEGATE_SCHEMA,
    )

    client = _ScriptedLLM([
        [(None, [DELEGATE_CALL])],
        [('{"add":"这次算我欠你的。","emo":"感激"}\n', None),
         ('{"closing":"以后有事找我。","emo":"感激"}\n', None)],
    ])
    blocks = _blocks(
        registry=_FakeRegistry(),
        memory=FakeMemory(),
        pending_draft_row=None,
        pending_draft_summary="",
        delegate_enabled=True,
    )
    started = time.perf_counter()
    chunks = [
        chunk async for chunk in stream_daily_llm(
            make_request(fx.daily_request()),
            client=client,
            blocks=blocks,
            llm_config=_config(),
        )
    ]
    elapsed = time.perf_counter() - started
    text = "".join(chunks)

    assert await asyncio.wait_for(worker_started.wait(), 1.0)
    assert '{"add":"这次算我欠你的。","emo":"感激"}' in text
    assert '{"closing":"以后有事找我。","emo":"感激"}' in text
    assert "已受理" not in text
    assert elapsed < 2.0  # 后台 worker 未完成也不影响最终对话
    assert calls[0]["session"] == fx.DAILY_DIALOGUE_ID
    assert calls[0]["npc"] == fx.NPC
    assert client.requests[0].tools[-1]["function"]["name"] == "delegate_dialogue_rewrite"
    assert client.requests[1].tools is None
    worker_release.set()


def test_delegate_tool_only_registered_when_enabled(monkeypatch):
    monkeypatch.setattr(
        tools_module, "delegate_tool_or_none",
        lambda **_kwargs: DELEGATE_SCHEMA,
    )
    off = _runtime(_blocks(delegate_enabled=False, registry=_FakeRegistry()), None)
    assert all(
        item["function"]["name"] != "delegate_dialogue_rewrite"
        for item in (off.tools() or [])
    )
    on = _runtime(_blocks(delegate_enabled=True, registry=_FakeRegistry()), None)
    assert any(
        item["function"]["name"] == "delegate_dialogue_rewrite"
        for item in (on.tools() or [])
    )


# ---------------------------------------------------------------------------
# 提示词纯净化与 status
# ---------------------------------------------------------------------------


def test_daily_prompt_is_mechanism_free_and_delegate_is_opt_in():
    system = build_daily_system_prompt(npc_name=fx.NPC, state=STATE)
    for forbidden in ("后台", "Host", "参数与结果不会出现在台词里", "会保持等待"):
        assert forbidden not in system
    assert "delegate_dialogue_rewrite" not in system
    assert "调用工具的那一轮不要输出 add/closing" in system
    assert "草案只是拟定，不是已发布" in system

    with_delegate = build_daily_system_prompt(
        npc_name=fx.NPC, state=STATE, delegate_enabled=True,
    )
    assert "delegate_dialogue_rewrite" in with_delegate
    assert "只有非常必要时才调用" in with_delegate


async def test_search_tool_round_emits_searching_status_before_lines():
    from services.dialogue_interject.tools import ToolExecutionResult

    search_call = {
        "type": "function",
        "id": "call_search",
        "function": {"name": "route_search", "arguments": '{"keywords":"北边仓库"}'},
    }

    class _Executor:
        def tools(self):
            return [{
                "type": "function",
                "function": {"name": "route_search", "description": "", "parameters": {}},
            }]

        async def execute(self, tool_calls, *, round_text=""):
            return ToolExecutionResult(
                requires_followup=True,
                messages=[],
                result_block="【办理结果】找到资料",
                instruction="用 JSONL 回答",
                route_kind="search",
            )

    client = _ScriptedLLM([
        [(None, [search_call])],
        [('{"add":"北边仓库的事我查过了。","emo":"普通"}\n', None)],
    ])
    factory = partial(
        stream_daily_llm,
        client=client,
        blocks=DailyBlocks(system_prompt="SYS"),
        tool_executor=_Executor(),
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)
    names = [event.event for event in events]
    assert names == ["status", "status", "line", "done"]
    assert events[0].data["stage"] == "thinking"
    assert events[1].data["stage"] == "searching"
    assert events[2].data["text"] == "北边仓库的事我查过了。"
