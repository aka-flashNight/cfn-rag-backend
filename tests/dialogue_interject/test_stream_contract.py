"""假流 SSE 契约：F-01/F-02/F-05/F-05b/F-06/F-07/F-08/F-09/F-10/F-11。

断言事件顺序与形状、首句早于 done/plan、seq 单调、split、mute 先于 line、
零有效/部分有效后不再产出台词。全部离线，不调真实模型。
"""

from __future__ import annotations

import asyncio

import pytest

from services.dialogue_interject.orchestrator import InterjectOrchestrator
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import (
    collect_events,
    make_orchestrator,
    make_request,
)


def _names(events) -> list[str]:
    return [event.event for event in events]


def _index(events, name: str, occurrence: int = 0) -> int:
    seen = 0
    for index, event in enumerate(events):
        if event.event == name:
            if seen == occurrence:
                return index
            seen += 1
    raise AssertionError(f"事件 {name} 未出现")


@pytest.mark.asyncio
async def test_f01_daily_add_add_closing_in_order():
    orchestrator = make_orchestrator(fx.daily_request(), chunks=fx.F01_JSONL)
    events = await collect_events(orchestrator)

    assert _names(events) == ["status", "line", "line", "line", "done"]
    lines = [event.data for event in events if event.event == "line"]
    assert [line["seq"] for line in lines] == [0, 1, 2]
    assert [line["kind"] for line in lines] == ["add", "add", "closing"]
    assert [line["text"] for line in lines] == [
        "还能怎么样，老样子。",
        "你倒是难得过来一趟。",
        "说吧，想聊什么？",
    ]
    assert all(line["speaker"] == fx.NPC for line in lines)
    assert all(line["emotion"] == "普通" for line in lines)
    assert all("split" not in line for line in lines)

    # 首条有效 line 早于 done
    assert _index(events, "line") < _index(events, "done")
    done = events[-1].data
    assert done["dialogue_instance_id"] == fx.DAILY_DIALOGUE_ID
    assert done["session_id"] == "session-test"
    assert done["usage"] == {}


@pytest.mark.asyncio
async def test_f02_long_add_is_split_with_marks_and_reassembles():
    payload = fx.daily_request(player_input="北边那条路还能走吗？", turn_id=fx.DAILY_TURN_B)
    events = await collect_events(make_orchestrator(payload, chunks=fx.F02_JSONL))

    assert _names(events) == ["status", "line", "line", "line", "done"]
    lines = [event.data for event in events if event.event == "line"]

    split_lines = lines[:2]
    assert [line["seq"] for line in lines] == [0, 1, 2]
    assert [line["split"] for line in split_lines] == [
        {"index": 0, "count": 2},
        {"index": 1, "count": 2},
    ]
    assert "".join(line["text"] for line in split_lines) == fx.F02_LONG_ADD
    assert all(len(line["text"]) <= 165 for line in split_lines)
    assert split_lines[0]["text"].endswith("。")

    closing = lines[2]
    assert closing["kind"] == "closing"
    assert "split" not in closing


@pytest.mark.asyncio
async def test_f05_task_end_plan_after_first_line():
    events = await collect_events(
        make_orchestrator(fx.task_request(), chunks=fx.F05_JSONL, session_id="task-session")
    )

    assert _names(events) == ["status", "line", "plan", "done"]
    assert _index(events, "line") < _index(events, "plan")

    line = events[1].data
    assert line == {
        "kind": "add",
        "text": "你倒是爽快。",
        "emotion": "普通",
        "speaker": fx.NPC,
        "seq": 0,
    }
    plan = events[2].data
    assert plan == {
        "mute": False,
        "end": 3,
        "edits": [
            {
                "n": 2,
                "kind": "edit",
                "lines": [{"text": "北边仓库昨晚丢了一批货。", "emotion": "普通"}],
            }
        ],
    }
    assert events[-1].data["session_id"] == "task-session"


@pytest.mark.asyncio
async def test_f05b_task_edit_split_inside_plan():
    events = await collect_events(
        make_orchestrator(fx.task_request(), chunks=fx.F05B_JSONL)
    )
    assert _names(events) == ["status", "line", "plan", "done"]

    plan = events[2].data
    assert plan["end"] == 4
    assert plan["mute"] is False
    edit = plan["edits"][0]
    assert edit["n"] == 3 and edit["kind"] == "edit"
    assert [line["split"] for line in edit["lines"]] == [
        {"index": 0, "count": 2},
        {"index": 1, "count": 2},
    ]
    assert "".join(line["text"] for line in edit["lines"]) == fx.F05B_LONG_EDIT
    assert all(len(line["text"]) <= 165 for line in edit["lines"])


@pytest.mark.asyncio
async def test_f06_task_remove_plan():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F06_JSONL))
    plan = next(event.data for event in events if event.event == "plan")
    assert plan == {"mute": False, "end": 3, "edits": [{"n": 2, "kind": "remove"}]}


@pytest.mark.asyncio
async def test_f07_task_missing_nochange_only_lists_explicit_edit():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F07_JSONL))
    plan = next(event.data for event in events if event.event == "plan")
    assert plan == {
        "mute": False,
        "end": 4,
        "edits": [
            {"n": 3, "kind": "edit", "lines": [{"text": "清单就压在柜台底下。", "emotion": "普通"}]}
        ],
    }


@pytest.mark.asyncio
async def test_f08_task_whole_group_replacement_has_null_end():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F08_JSONL))
    plan = next(event.data for event in events if event.event == "plan")
    assert plan["end"] is None
    assert [edit["kind"] for edit in plan["edits"]] == ["edit", "remove"]


@pytest.mark.asyncio
async def test_f09_mute_event_precedes_line_and_plan_keeps_mute():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F09_JSONL))
    assert _names(events) == ["status", "mute", "line", "plan", "done"]
    assert _index(events, "mute") < _index(events, "line")
    assert events[1].data == {"mute": True}

    line = events[2].data
    assert line["text"] == "行了，这事不用你操心。"
    assert line["seq"] == 0

    plan = events[3].data
    assert plan == {"mute": True, "end": 1, "edits": []}


@pytest.mark.asyncio
async def test_f10_zero_valid_retries_once_then_error_without_lines():
    calls: list[str] = []

    async def primary(_request):
        calls.append("primary")
        for chunk in fx.F10_TEXT:
            yield chunk

    async def repair(_request):
        calls.append("repair")
        for chunk in fx.F10_TEXT:
            yield chunk

    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()),
        session_id="session-f10",
        stream_factory=primary,
        repair_factory=repair,
    )
    events = await collect_events(orchestrator)

    assert calls == ["primary", "repair"]  # 只修复一次
    assert _names(events) == ["status", "error"]
    error = events[-1].data
    assert error["code"] == "no_valid_lines"
    assert error["valid_lines"] == 0
    assert "line" not in _names(events) and "plan" not in _names(events)


@pytest.mark.asyncio
async def test_f11_partial_then_stream_failed_keeps_line_but_no_plan():
    orchestrator = make_orchestrator(
        fx.daily_request(turn_id=fx.DAILY_TURN_B),
        chunks=fx.F11_JSONL,
        error=RuntimeError("connection reset"),
    )
    events = await collect_events(orchestrator)

    assert _names(events) == ["status", "line", "error"]
    assert events[1].data["text"] == "你先听我说。"
    assert events[1].data["seq"] == 0
    error = events[-1].data
    assert error["code"] == "stream_failed"
    assert error["valid_lines"] == 1
    assert "plan" not in _names(events) and "done" not in _names(events)


@pytest.mark.asyncio
async def test_task_partial_stream_failure_never_emits_plan():
    orchestrator = make_orchestrator(
        fx.task_request(),
        chunks=['{"n":2,"line":"改写到一半","emo":"普通"}\n'],
        error=RuntimeError("boom"),
    )
    events = await collect_events(orchestrator)
    assert _names(events) == ["status", "error"]
    assert events[-1].data["code"] == "stream_failed"
    assert events[-1].data["valid_lines"] == 1
    assert "plan" not in _names(events)


@pytest.mark.asyncio
async def test_first_line_is_emitted_before_stream_finishes():
    """首句不等整轮：生成器在第二块之前被 gate 挡住，消费端必须先拿到 line。"""
    gate = asyncio.Event()

    async def primary(_request):
        yield '{"add":"第一句","emo":"普通"}\n'
        await asyncio.wait_for(gate.wait(), timeout=2)
        yield '{"add":"第二句","emo":"普通"}\n'

    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()),
        session_id="session-incremental",
        stream_factory=primary,
    )
    stream = orchestrator.run()
    first = await asyncio.wait_for(anext(stream), timeout=2)
    assert first.event == "status"
    line_event = await asyncio.wait_for(anext(stream), timeout=2)
    assert line_event.event == "line"
    assert line_event.data["text"] == "第一句"

    gate.set()
    rest = [event async for event in stream]
    assert [event.event for event in rest] == ["line", "done"]
    assert rest[0].data["seq"] == 1


@pytest.mark.asyncio
async def test_invalid_lines_are_skipped_but_valid_ones_stream():
    chunks = [
        "模型的前言\n",
        '{"add":"（叹气）好吧。","emo":"普通"}\n',
        "```\n",
        '{"add":"（沉默）","emo":"普通"}\n',
        '{"add":"跟我来。","emo":"狰狞"}\n',
    ]
    events = await collect_events(make_orchestrator(fx.daily_request(), chunks=chunks))
    lines = [event.data for event in events if event.event == "line"]
    assert [line["text"] for line in lines] == ["好吧。", "跟我来。"]
    assert [line["emotion"] for line in lines] == ["普通", "普通"]
    assert _names(events)[-1] == "done"


@pytest.mark.asyncio
async def test_error_valid_lines_counts_model_rows_not_split_segments():
    """valid_lines 按模型输出行计、拆分前（协议 5.2 / 实施指引 3.3）。"""
    orchestrator = make_orchestrator(
        fx.daily_request(),
        chunks=[fx.F02_JSONL[0]],  # 一条超 165 的 add → 拆成 2 个 line 事件
        error=RuntimeError("stream broke after long line"),
    )
    events = await collect_events(orchestrator)
    lines = [event for event in events if event.event == "line"]
    assert len(lines) == 2
    assert [line.data["split"]["index"] for line in lines] == [0, 1]
    error = next(event for event in events if event.event == "error")
    assert error.data["valid_lines"] == 1


@pytest.mark.asyncio
async def test_error_valid_lines_does_not_count_mute():
    orchestrator = make_orchestrator(
        fx.task_request(),
        chunks=['{"mute":true}\n', '{"add":"一句","emo":"普通"}\n'],
        error=RuntimeError("stream broke"),
    )
    events = await collect_events(orchestrator)
    assert _names(events) == ["status", "mute", "line", "error"]
    assert events[-1].data["valid_lines"] == 1  # mute 不计，add 计 1


@pytest.mark.asyncio
async def test_turn_with_only_mute_is_not_treated_as_zero_valid():
    events = await collect_events(
        make_orchestrator(fx.task_request(), chunks=['{"mute":true}\n'])
    )
    assert _names(events) == ["status", "mute", "plan", "done"]
    assert events[-2].data == {"mute": True, "end": None, "edits": []}
