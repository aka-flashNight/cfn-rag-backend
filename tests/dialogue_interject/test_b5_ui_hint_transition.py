"""B5：插话 status.ui_hint、任务草案过渡话与最终确认的顺序。"""

from __future__ import annotations

import asyncio

import pytest

from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.llm import stream_daily_llm
from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.schemas import StreamStatus, status_event
from services.dialogue_interject.tools import (
    DailyToolRuntime,
    ToolExecutionResult,
    ToolProgress,
    _unwrap_transition_text,
)
from services.llm import LLMConfig, StreamEvent
from services.subagents.base import SubagentEvent
from services.subagents.task_runner import TaskRunner
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import collect_events, make_request

TOOL_CALL_DRAFT = {
    "type": "function",
    "id": "call_draft_1",
    "function": {
        "name": "route_task_draft",
        "arguments": '{"note":"玩家想要跑腿的活儿"}',
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


def _config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key", api_base="http://unit.test/v1", model_name="test-model",
    )


class ScriptedLLM:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        for text, tool_calls in (self.rounds.pop(0) if self.rounds else []):
            if text:
                yield StreamEvent(kind="content", text=text)
            if tool_calls is not None:
                yield StreamEvent(kind="finish", tool_calls=tool_calls)


class ProgressExecutor:
    """生产执行器的最小替身：暴露 progress_queue 并先推进度再返回结果。"""

    def __init__(self, result: ToolExecutionResult):
        self.result = result
        self.progress_queue: asyncio.Queue[ToolProgress] = asyncio.Queue()

    def tools(self):
        return TOOL_SCHEMA

    async def execute(self, tool_calls, *, round_text=""):
        self.progress_queue.put_nowait(ToolProgress(ui_hint="准备候选中"))
        self.progress_queue.put_nowait(
            ToolProgress(line="这事我记下了。", emotion="普通")
        )
        await asyncio.sleep(0.02)
        return self.result


def test_status_event_carries_ui_hint():
    event = status_event("drafting", "校验修正中")
    assert event.data == {"stage": "drafting", "ui_hint": "校验修正中"}


@pytest.mark.asyncio
async def test_orchestrator_forwards_ui_hint_and_dedupes_identical_pairs():
    async def factory(_request):
        yield StreamStatus("drafting", "准备候选中")
        yield StreamStatus("drafting", "准备候选中")
        yield StreamStatus("drafting", "拟定委托中")
        yield '{"add":"成了。","emo":"普通"}\n'

    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)
    statuses = [event.data for event in events if event.event == "status"]
    assert statuses == [
        {"stage": "thinking", "ui_hint": ""},
        {"stage": "drafting", "ui_hint": "准备候选中"},
        {"stage": "drafting", "ui_hint": "拟定委托中"},
    ]


@pytest.mark.asyncio
async def test_progress_queue_emits_transition_line_before_final_round():
    client = ScriptedLLM([
        [(None, [TOOL_CALL_DRAFT])],
        [('{"add":"这活你接吗？","emo":"普通"}\n', None)],
    ])
    executor = ProgressExecutor(ToolExecutionResult(
        requires_followup=True,
        messages=[
            {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL_DRAFT]},
            {"role": "user", "content": "【办理结果】草案已拟定"},
        ],
        result_block="【办理结果】草案已拟定",
        instruction="输出最终确认对话",
        route_kind="task_draft",
    ))
    factory = lambda request: stream_daily_llm(  # noqa: E731 - 测试内联工厂
        request,
        client=client,
        blocks=DailyBlocks(system_prompt="SYS", npc_emotions=("普通",)),
        tool_executor=executor,
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)

    assert [event.event for event in events] == [
        "status", "status", "status", "line", "line", "done",
    ]
    assert events[1].data == {"stage": "drafting", "ui_hint": ""}
    assert events[2].data == {"stage": "drafting", "ui_hint": "准备候选中"}
    assert events[3].data["text"] == "这事我记下了。"
    assert events[3].data["interim"] is True
    assert "interim" not in events[4].data
    assert events[4].data["text"] == "这活你接吗？"


class FakeHandle:
    def __init__(self, events):
        self.events = list(events)
        self.done = False
        self.cancelled = False

    async def next_event(self, timeout):
        await asyncio.sleep(0.01)
        if not self.events:
            raise asyncio.TimeoutError()
        event = self.events.pop(0)
        if event.kind in ("final", "failed"):
            self.done = True
        return event

    def cancel(self):
        self.cancelled = True
        self.done = True


class ImmediateLLM:
    def __init__(self, text: str):
        self.text = text
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        yield StreamEvent(kind="content", text=self.text)


@pytest.mark.asyncio
async def test_task_runner_progress_pushes_ui_hints_then_transition(monkeypatch):
    handle = FakeHandle([
        SubagentEvent(kind="progress", ui_hint="准备候选中"),
        SubagentEvent(kind="intermediate", ui_hint="拟定委托中", vague_note="跑腿的活"),
        SubagentEvent(kind="progress", ui_hint="拟定委托中"),
        SubagentEvent(kind="final", payload={
            "status": "draft_created",
            "draft_id": "d1",
            "draft_summary": "去城西送一封信，报酬 100 金币。",
            "draft": {"draft_id": "d1", "title": "送信"},
            "bargain_count": 0,
        }),
    ])
    monkeypatch.setattr(TaskRunner, "launch", staticmethod(lambda **_kwargs: handle))

    llm = ImmediateLLM("这事我记下了。")
    runtime = DailyToolRuntime(
        make_request(fx.daily_request()),
        DailyBlocks(system_prompt="SYS", npc_emotions=("普通",)),
        llm=llm,
    )
    result = await runtime.execute([TOOL_CALL_DRAFT])

    progress = []
    while not runtime.progress_queue.empty():
        progress.append(runtime.progress_queue.get_nowait())
    assert [item.ui_hint for item in progress if item.ui_hint] == [
        "准备候选中", "拟定委托中", "拟定委托中",
    ]
    assert [item.line for item in progress if item.line] == ["这事我记下了。"]
    assert "草案已拟定" in result.result_block
    assert "这事我记下了。" in result.instruction
    assert "不要重复这句" in result.instruction
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_task_runner_cancels_transition_when_final_arrives_first(monkeypatch):
    handle = FakeHandle([
        SubagentEvent(kind="intermediate", ui_hint="拟定委托中", vague_note="跑腿的活"),
        SubagentEvent(kind="final", payload={
            "status": "draft_created",
            "draft_id": "d1",
            "draft_summary": "去城西送一封信。",
            "draft": {"draft_id": "d1", "title": "送信"},
            "bargain_count": 0,
        }),
    ])
    monkeypatch.setattr(TaskRunner, "launch", staticmethod(lambda **_kwargs: handle))

    class SlowLLM:
        def __init__(self):
            self.cancelled = False

        async def chat_stream(self, request):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            yield StreamEvent(kind="content", text="不该出现")

    llm = SlowLLM()
    runtime = DailyToolRuntime(
        make_request(fx.daily_request()),
        DailyBlocks(system_prompt="SYS", npc_emotions=("普通",)),
        llm=llm,
    )
    await runtime.execute([TOOL_CALL_DRAFT])

    assert llm.cancelled
    lines = []
    while not runtime.progress_queue.empty():
        item = runtime.progress_queue.get_nowait()
        if item.line:
            lines.append(item.line)
    assert lines == []


def test_unwrap_transition_text_accepts_jsonl_but_keeps_plain_text():
    assert _unwrap_transition_text('{"add":"我这儿有个活。","emo":"普通"}') == "我这儿有个活。"
    assert _unwrap_transition_text(
        '{"add":"我这儿有个活。","emo":"普通"}\n这句不该吞掉'
    ) == "我这儿有个活。"
    assert _unwrap_transition_text("我这儿有个活。") == "我这儿有个活。"
