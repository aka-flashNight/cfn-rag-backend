"""B2：工具轮、工具内容不污染台词、参考续写、finalize 前缀与取消。"""

from __future__ import annotations

import asyncio
from functools import partial

import pytest

from services.dialogue_interject.context import DailyBlocks, build_user_context
from services.dialogue_interject.llm import stream_daily_llm
from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.schemas import FinalizeRequest
from services.dialogue_interject.sessions import ensure_session, finalize
from services.dialogue_interject.tools import ToolExecutionResult
from services.llm import LLMConfig, StreamEvent
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import collect_events, make_orchestrator, make_request


TOOL_CALL_DRAFT = {
    "type": "function",
    "id": "call_draft_1",
    "function": {
        "name": "route_task_draft",
        "arguments": '{"note":"玩家想要跑腿的活儿，参数不应出现"}',
    },
}
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "route_task_draft",
        "description": "draft",
        "parameters": {"type": "object", "properties": {}},
    },
}]


class ScriptedLLM:
    """按轮回放脚本：每轮是 (content_text, tool_calls) 列表。"""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        round_items = self.rounds.pop(0) if self.rounds else []
        for text, tool_calls in round_items:
            if text:
                yield StreamEvent(kind="content", text=text)
            if tool_calls is not None:
                yield StreamEvent(kind="finish", tool_calls=tool_calls)


class FakeExecutor:
    def __init__(self, result: ToolExecutionResult, tools=None):
        self.result = result
        self._tools = tools if tools is not None else TOOL_SCHEMA
        self.calls = []

    def tools(self):
        return self._tools

    async def execute(self, tool_calls, *, round_text=""):
        self.calls.append((tool_calls, round_text))
        return self.result


def _config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key", api_base="http://unit.test/v1", model_name="test-model",
    )


async def test_tool_round_then_final_lines_only():
    client = ScriptedLLM([
        [(None, [TOOL_CALL_DRAFT])],
        [('{"add":"你倒是爽快。","emo":"普通"}\n', None),
         ('{"closing":"接不接你自己想。","emo":"普通"}\n', None)],
    ])
    executor = FakeExecutor(ToolExecutionResult(
        requires_followup=True,
        messages=[
            {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL_DRAFT]},
            {"role": "tool", "tool_call_id": "call_draft_1", "content": '{"status":"draft_created"}'},
            {"role": "user", "content": "【后台结果：草案已拟定】\n城西补给线\n\n现在只输出最终确认对话"},
        ],
        result_block="【后台结果：草案已拟定】\n城西补给线",
        instruction="现在只输出最终确认对话",
        route_kind="task_draft",
    ))
    blocks = DailyBlocks(system_prompt="SYS", npc_emotions=("普通",))

    chunks = [
        chunk async for chunk in stream_daily_llm(
            make_request(fx.daily_request()),
            client=client,
            blocks=blocks,
            tool_executor=executor,
            llm_config=_config(),
        )
    ]
    text = "".join(chunks)
    assert '{"add":"你倒是爽快。","emo":"普通"}' in text
    assert '{"closing":"接不接你自己想。","emo":"普通"}' in text
    assert "参数不应出现" not in text
    assert "draft_created" not in text
    assert len(executor.calls) == 1
    assert executor.calls[0][1] == ""  # 工具轮没有正文
    assert client.requests[0].tools == TOOL_SCHEMA
    assert client.requests[1].tools is None
    assert any("草案已拟定" in str(m.get("content")) for m in client.requests[1].messages)


async def test_tool_result_and_args_never_pollute_sse_lines():
    client = ScriptedLLM([
        [(None, [TOOL_CALL_DRAFT])],
        [('{"add":"真正台词。","emo":"普通"}\n', None)],
    ])
    executor = FakeExecutor(ToolExecutionResult(
        requires_followup=True,
        messages=[],
        result_block='{"add":"后台结果不应出现"}',
        instruction="请输出最终台词",
        route_kind="task_draft",
    ))
    blocks = DailyBlocks(system_prompt="SYS")
    factory = partial(
        stream_daily_llm,
        client=client,
        blocks=blocks,
        tool_executor=executor,
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)
    assert [event.event for event in events] == ["status", "line", "done"]
    assert events[1].data["text"] == "真正台词。"
    assert "后台结果不应出现" not in str(events)


async def test_memory_tool_can_coexist_with_spoken_lines_without_followup():
    memory_call = {
        "type": "function",
        "id": "call_mem_1",
        "function": {"name": "update_long_term_memory", "arguments": '{"content":"记住他"}'},
    }
    client = ScriptedLLM([
        [('{"add":"我记下了。","emo":"普通"}\n', [memory_call])],
    ])
    executor = FakeExecutor(ToolExecutionResult(requires_followup=False))
    blocks = DailyBlocks(system_prompt="SYS")
    chunks = [
        chunk async for chunk in stream_daily_llm(
            make_request(fx.daily_request()),
            client=client,
            blocks=blocks,
            tool_executor=executor,
            llm_config=_config(),
        )
    ]
    assert "我记下了。" in "".join(chunks)
    assert len(client.requests) == 1
    assert len(executor.calls) == 1


async def test_tool_execution_is_cancelled_when_stream_is_closed():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowExecutor(FakeExecutor):
        async def execute(self, tool_calls, *, round_text=""):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return self.result

    client = ScriptedLLM([[(None, [TOOL_CALL_DRAFT])], []])
    executor = SlowExecutor(ToolExecutionResult(requires_followup=True))
    blocks = DailyBlocks(system_prompt="SYS")
    factory = partial(
        stream_daily_llm,
        client=client,
        blocks=blocks,
        tool_executor=executor,
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )

    async def consume():
        async for _event in orchestrator.run():
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


def test_reference_continuation_is_rendered_as_reference_only():
    context = build_user_context(make_request(fx.daily_turn_b_request()))
    assert "【参考续写（上一轮未播，仅供参考，可不采用）】" in context
    assert "上一轮未播新增：你倒是难得过来一趟。" in context
    assert "上一轮未播结束语：说吧，想聊什么？" in context
    assert "先别说这个，我想问问任务。" in context


@pytest.mark.asyncio
async def test_f04_request_with_reference_streams_first_line():
    events = await collect_events(
        make_orchestrator(fx.daily_turn_b_request(), chunks=fx.F04_TURN_B_JSONL)
    )
    names = [event.event for event in events]
    assert names == ["status", "line", "line", "done"]
    assert events[1].data["text"] == "任务是吧，我这儿正好有个跑腿的活儿。"
    assert names.index("line") < names.index("done")


async def test_finalize_accepts_growing_full_prefix_idempotently(memory_store):
    await ensure_session(make_request(fx.daily_request()), store=memory_store)
    prefix = {"dialogue_instance_id": fx.DAILY_DIALOGUE_ID, "transcript": fx.F01_FINALIZE["transcript"][:3]}
    first = await finalize(FinalizeRequest.model_validate(prefix), store=memory_store)
    assert first.appended == 2 and first.total == 3

    second = await finalize(FinalizeRequest.model_validate(fx.F01_FINALIZE), store=memory_store)
    assert second.appended == 2 and second.total == 5

    third = await finalize(FinalizeRequest.model_validate(fx.F01_FINALIZE), store=memory_store)
    assert third.appended == 0 and third.total == 5


async def test_no_ai_reply_turn_keeps_played_pc(memory_store):
    await ensure_session(make_request(fx.daily_request()), store=memory_store)
    transcript = {
        "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
        "transcript": [
            {"seq": 0, "role": "npc", "name": fx.NPC, "text": "又是你。"},
            {"seq": 1, "role": "player", "name": "$PC", "text": "最近怎么样？"},
        ],
    }
    result = await finalize(FinalizeRequest.model_validate(transcript), store=memory_store)
    assert result.appended == 1
    assert result.total == 2
    ledger = memory_store.dialogue_ledger_read_sync(result.session_id)
    assert [(row.seq, row.role, row.content) for row in ledger] == [
        (0, "assistant", "又是你。"),
        (1, "user", "最近怎么样？"),
    ]


async def test_draft_runtime_uses_task_runner_and_returns_final_dialogue_instruction(monkeypatch):
    """草案链路：复用 TaskRunner 的 final payload，写草案并给出最终对话指令。"""
    from types import SimpleNamespace

    import services.dialogue_interject.tools as tools_module
    from services.dialogue_interject.tools import DailyToolRuntime

    class FakeMemory:
        def __init__(self):
            self.drafts = []

        async def upsert_draft(self, session_id, draft, bargain_count=0):
            self.drafts.append((session_id, draft, bargain_count))

    class FakeHandle:
        done = True

        async def wait_final(self, timeout):
            return SimpleNamespace(
                kind="final",
                payload={
                    "draft": {"draft_id": "d1", "title": "城西补给线"},
                    "draft_summary": "城西补给线：去看一眼，带回货物；报酬 2400 金币。",
                    "bargain_count": 0,
                },
            )

        def cancel(self):
            pass

    captured = {}

    def _launch(cls, **kwargs):
        captured.update(kwargs)
        return FakeHandle()

    monkeypatch.setattr(tools_module.TaskRunner, "launch", classmethod(_launch))

    memory = FakeMemory()
    blocks = DailyBlocks(
        system_prompt="SYS",
        npc_state=SimpleNamespace(faction="A兵团", titles=["军需官"], favorability=3, challenge=None),
        npc_states={},
        game_data=object(),
        registry=object(),
        memory=memory,
        session_id=fx.DAILY_DIALOGUE_ID,
        progress_stage=2,
        progress_desc="阶段 2",
        recent_dialogue="Andy Law：又来？",
        long_term_block="",
    )
    runtime = DailyToolRuntime(make_request(fx.daily_request()), blocks)
    result = await runtime.execute([TOOL_CALL_DRAFT])

    assert result.requires_followup is True
    assert captured["kind"] == "task_draft"
    assert captured["player_query"] == "最近怎么样？"
    assert captured["npc_name"] == fx.NPC
    assert memory.drafts[0][0] == fx.DAILY_DIALOGUE_ID
    assert memory.drafts[0][1]["title"] == "城西补给线"
    assert "城西补给线" in result.result_block
    assert "最终确认对话" in result.instruction
    assert "不接就算了" in result.instruction


async def test_route_failure_degrades_to_result_block_not_exception(monkeypatch):
    """工具失败/超时：记录后给最终轮一个降级结果块，不拖垮整轮。"""
    from types import SimpleNamespace

    from services.dialogue_interject.tools import DailyToolRuntime

    blocks = DailyBlocks(
        system_prompt="SYS",
        npc_state=SimpleNamespace(faction="A兵团", titles=[], favorability=0, challenge=None),
        npc_states={},
        game_data=object(),
        registry=object(),
        memory=None,
        session_id=fx.DAILY_DIALOGUE_ID,
    )
    runtime = DailyToolRuntime(make_request(fx.daily_request()), blocks)

    async def _boom(_route):
        raise RuntimeError("tool exploded")

    monkeypatch.setattr(runtime, "_execute_route", _boom)
    result = await runtime.execute([TOOL_CALL_DRAFT])
    assert result.requires_followup is True
    assert "未能完成" in result.result_block
    assert "不要编造" in result.instruction


async def test_route_failure_degrades_to_result_block_not_exception(monkeypatch):
    """工具失败/超时：记录后给最终轮一个降级结果块，不拖垮整轮。"""
    from types import SimpleNamespace

    from services.dialogue_interject.tools import DailyToolRuntime

    blocks = DailyBlocks(
        system_prompt="SYS",
        npc_state=SimpleNamespace(faction="A兵团", titles=[], favorability=0, challenge=None),
        npc_states={},
        game_data=object(),
        registry=object(),
        memory=None,
        session_id=fx.DAILY_DIALOGUE_ID,
    )
    runtime = DailyToolRuntime(make_request(fx.daily_request()), blocks)

    async def _boom(_route):
        raise RuntimeError("tool exploded")

    monkeypatch.setattr(runtime, "_execute_route", _boom)
    result = await runtime.execute([TOOL_CALL_DRAFT])
    assert result.requires_followup is True
    assert "未能完成" in result.result_block
    assert "不要编造" in result.instruction


def test_daily_tool_runtime_schema_matrix():
    from types import SimpleNamespace

    from services.dialogue_interject.tools import DailyToolRuntime

    class FakeRegistry:
        def __init__(self):
            self.names = None

        def get_openai_tools(self, names=None):
            self.names = list(names or [])
            return [
                {"type": "function", "function": {"name": name, "parameters": {}}}
                for name in self.names
            ]

    registry = FakeRegistry()
    blocks = DailyBlocks(
        system_prompt="SYS",
        registry=registry,
        npc_state=SimpleNamespace(faction="A兵团", titles=[], favorability=0, challenge=None),
        npc_states={},
        pending_draft_row=None,
    )
    tools = DailyToolRuntime(make_request(fx.daily_request()), blocks).tools()
    names = [item["function"]["name"] for item in tools]
    assert "route_task_draft" in names
    assert "route_search" in names
    assert "route_task_confirm" not in names

    blocks_with_draft = DailyBlocks(
        system_prompt="SYS",
        registry=registry,
        npc_state=blocks.npc_state,
        npc_states={},
        pending_draft_row=SimpleNamespace(draft={"draft_id": "d1"}, bargain_count=0),
    )
    names = [
        item["function"]["name"]
        for item in DailyToolRuntime(make_request(fx.daily_request()), blocks_with_draft).tools()
    ]
    assert {"route_task_confirm", "route_task_cancel", "route_task_update"} <= set(names)


async def test_confirm_tool_publishes_and_clears_draft(monkeypatch):
    """confirm 接线：走既有 handler（发布文本用模板兜底），成功后清草案。"""
    from types import SimpleNamespace

    import services.dialogue_interject.tools as tools_module
    from services.dialogue_interject.tools import DailyToolRuntime

    class FakeMemory:
        def __init__(self):
            self.deleted = []

        async def delete_draft(self, session_id, draft_id):
            self.deleted.append((session_id, draft_id))

    memory = FakeMemory()
    draft_row = SimpleNamespace(
        draft={"draft_id": "d1", "title": "城西补给线"},
        draft_id="d1",
        bargain_count=0,
    )
    blocks = DailyBlocks(
        system_prompt="SYS",
        npc_state=SimpleNamespace(faction="A兵团", titles=[], favorability=0, challenge=None),
        npc_states={},
        game_data=object(),
        registry=object(),
        memory=memory,
        session_id=fx.DAILY_DIALOGUE_ID,
        pending_draft_row=draft_row,
    )

    monkeypatch.setattr(
        tools_module,
        "execute_confirm_agent_task",
        lambda *_args, **_kwargs: SimpleNamespace(payload={"status": "confirmed"}),
    )
    runtime = DailyToolRuntime(make_request(fx.daily_request()), blocks)
    result = await runtime.execute([{
        "type": "function",
        "id": "call_confirm",
        "function": {"name": "route_task_confirm", "arguments": "{}"},
    }])
    assert result.requires_followup is True
    assert "正式派给玩家" in result.result_block
    assert memory.deleted == [(fx.DAILY_DIALOGUE_ID, "d1")]
