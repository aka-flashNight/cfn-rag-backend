"""B4：任务上下文、任务提示词、任务真模型流与 plan 语义（全部离线）。"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.dialogue_interject import api as interject_api
from services.dialogue_interject.context import (
    DailyBlocks,
    assemble_task_blocks,
    build_user_context,
)
from services.dialogue_interject.llm import stream_task_llm
from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.parser import JSONLStreamParser, ParseOptions
from services.dialogue_interject.plan import normalize_task_plan
from services.dialogue_interject.prompts import (
    TASK_SYSTEM_PROMPT,
    build_task_messages,
    build_task_system_prompt,
    tool_matrix_for,
)
from services.dialogue_interject.schemas import RemainingLine
from services.dialogue_interject.tools import DailyToolRuntime, ToolExecutionResult
from services.llm import LLMConfig, StreamEvent
from services.npc.manager import NPCState
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import collect_events, make_orchestrator, make_request, parse_sse


STATE = NPCState(
    favorability=5,
    relationship_level="熟悉",
    sex="男",
    emotions=["普通", "认真"],
    faction="A兵团",
    titles=["军需官"],
)


class _StubNpcManager:
    async def get(self, _name: str) -> NPCState:
        return STATE

    async def all_states(self) -> dict[str, NPCState]:
        return {fx.NPC: STATE}


class _FakeGameData:
    map_registry = None


def _config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key", api_base="http://unit.test/v1", model_name="test-model",
    )


def _task_post_payload() -> dict:
    payload = fx.task_request()
    payload["dialogue"]["kind"] = "task_post"
    payload["task_context"] = {
        "completed": {
            "task_id": 200124,
            "title": "北边仓库失窃",
            "npc": fx.NPC,
            "objectives": ["找回失窃货物", "返回交付"],
            "rewards": ["金币 2400", "声望 10"],
            "outcome": "玩家已找回全部失窃货物",
        },
        "upcoming": {
            "task_id": 200125,
            "title": "东边水塔接应",
            "npc": "老赵",
            "objectives": ["在水塔接应玩家"],
            "rewards": ["情报"],
        },
    }
    return payload


def _task_plan(chunks: list[str], count: int = 4):
    remaining = {
        row["index"]: RemainingLine.model_validate(row) for row in fx.TASK_REMAINING
    }
    parser = JSONLStreamParser(
        ParseOptions(mode="task", npc=fx.NPC, remaining=remaining)
    )
    entries = []
    for chunk in chunks:
        entries.extend(parser.feed(chunk))
    entries.extend(parser.finish())
    return normalize_task_plan(entries, count), parser


# ---------------------------------------------------------------------------
# 任务上下文与提示词
# ---------------------------------------------------------------------------


def test_task_pre_context_renders_completed_and_upcoming():
    text = build_user_context(make_request(fx.task_request()))
    assert "【任务上下文（任务前）】" in text
    assert "已完成任务：南边哨站补给" in text
    assert "玩家完成了什么：玩家已送回 3 箱弹药" in text
    assert "将完成任务：北边仓库失窃" in text
    assert "目标：找回失窃货物；返回交付" in text
    assert "奖励：金币 2400；声望 10" in text


def test_task_post_context_renders_completed_and_next():
    text = build_user_context(make_request(_task_post_payload()))
    assert "【任务上下文（任务后）】" in text
    assert "已完成任务：北边仓库失窃" in text
    assert "玩家完成了什么：玩家已找回全部失窃货物" in text
    assert "将完成任务：东边水塔接应" in text
    assert "发布 NPC：老赵" in text


def test_task_context_both_empty_says_no_summary():
    payload = fx.task_request()
    payload["task_context"] = {"completed": None, "upcoming": None}
    text = build_user_context(make_request(payload))
    assert "本次没有可用的已完成/将完成任务摘要" in text


def test_task_prompt_has_discipline_output_and_only_query_tools():
    system = build_task_system_prompt(npc_name=fx.NPC, state=STATE)
    assert "【任务对话纪律】" in system
    assert "必须回到当前任务对话的原有脉络" in system
    assert "后续内容能直接接回原文时必须用 continue 提前结束" in system
    assert "continue 填的编号以及后续台词均保持不变" in system
    assert "如果没有合适的 edit，且下一条无法直接接上玩家的对话时，建议先 add 一条来回应玩家" in system
    assert "只有原剩余已全部逐条处理完" in system
    assert "改写时不得自称可修改原句里的任务信息" in system
    assert '"type":"add","name":' in system
    assert '"type":"add","no":0' not in system
    assert '"type":"continue","no":4,"name":' in system
    assert "name：add、edit、keep、remove、continue 都必填" in system
    # 兜底 system 仍保留完整 schema 行（主 system 的示例在输出规则末尾）。
    assert '{"type":"mute"}' in TASK_SYSTEM_PROMPT
    assert '{"type":"end"}' in TASK_SYSTEM_PROMPT
    assert '"type":"add","no":0' not in TASK_SYSTEM_PROMPT
    for old_format in ("nochange", '"n":', '"line":', '{"end":4}', '{"mute":true}'):
        assert old_format not in system
    assert "route_search" in system
    assert "update_long_term_memory" in system
    for forbidden in ("route_task_draft", "route_task_confirm", "delegate_dialogue_rewrite",
                      "【工具（本轮可用）】\n- route_task", "草案"):
        assert forbidden not in system
    assert "后台" not in system and "Host" not in system
    assert tool_matrix_for("task") == ("route_search", "update_long_term_memory")


async def test_assemble_task_blocks_reuses_stable_blocks_without_draft(memory_store, monkeypatch):
    from services.integration.state import get_integration_state

    get_integration_state().unbind()
    monkeypatch.setattr(
        "services.dialogue.delegation.delegate_tool_or_none",
        lambda **_kwargs: {"type": "function", "function": {"name": "delegate_dialogue_rewrite"}},
    )
    await memory_store.upsert_draft(
        fx.TASK_DIALOGUE_ID,
        {"draft_id": "d1", "title": "不该出现的草案"},
        bargain_count=0,
    )
    blocks = await assemble_task_blocks(
        make_request(fx.task_request()),
        npc_manager=_StubNpcManager(),
        game_data=_FakeGameData(),
        memory=memory_store,
        rag_enabled=False,
        session_id=fx.TASK_DIALOGUE_ID,
    )
    assert blocks.task_publish_disabled is True
    assert blocks.delegate_enabled is False
    assert blocks.pending_draft_row is None
    assert "【任务对话纪律】" in blocks.system_prompt
    assert "【世界观背景概要】" in blocks.system_prompt
    assert "不该出现的草案" not in blocks.system_prompt
    assert "delegate_dialogue_rewrite" not in blocks.system_prompt

    messages = build_task_messages(make_request(fx.task_request()), blocks)
    assert [item["role"] for item in messages] == ["system", "user"]
    assert "【任务上下文（任务前）】" in messages[1]["content"]
    assert "草案" not in messages[1]["content"]


# ---------------------------------------------------------------------------
# fixture F-05 / F-05b / F-06 / F-07 / F-08 / F-09
# ---------------------------------------------------------------------------


async def test_f05_continue_reconnects_and_steps_keep_order():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F05_JSONL))
    assert [event.event for event in events] == ["status", "line", "plan", "done"]
    assert events[1].data["text"] == "你倒是爽快。"
    plan = events[2].data
    assert plan["resume"] == 3
    assert [(step["n"], step["kind"]) for step in plan["steps"]] == [
        (1, "keep"), (2, "edit"),
    ]
    assert plan["steps"][1]["lines"][0]["text"] == "北边仓库昨晚丢了一批货。"


async def test_f05b_long_edit_is_split_in_plan():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F05B_JSONL))
    plan = next(event.data for event in events if event.event == "plan")
    assert plan["resume"] == 4
    edit = plan["steps"][2]
    assert edit["n"] == 3
    parts = edit["lines"]
    assert len(parts) == 2
    assert [part["split"]["index"] for part in parts] == [0, 1]
    assert all(part["split"]["count"] == 2 for part in parts)
    assert "".join(part["text"] for part in parts) == fx.F05B_LONG_EDIT


async def test_f06_remove_reaches_plan():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F06_JSONL))
    plan = next(event.data for event in events if event.event == "plan")
    assert plan["resume"] == 3
    assert plan["steps"] == [
        {"n": 1, "kind": "keep"},
        {"n": 2, "kind": "remove"},
    ]


async def test_f07_explicit_keep_and_edit_are_ordered():
    plan, _ = _task_plan(fx.F07_JSONL)
    assert plan.resume == 4
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "keep"), (2, "keep"), (3, "edit"),
    ]


async def test_f08_end_does_not_resume_original_tail():
    plan, _ = _task_plan(fx.F08_JSONL)
    assert plan.resume == 5
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "edit"), (2, "remove"), (3, "remove"), (4, "remove"),
    ]


async def test_f09_mute_precedes_line_and_plan_keeps_mute():
    events = await collect_events(make_orchestrator(fx.task_request(), chunks=fx.F09_JSONL))
    assert [event.event for event in events] == ["status", "mute", "line", "plan", "done"]
    assert events[1].data == {"mute": True}
    assert events[2].data["text"] == "行了，这事不用你操心。"
    assert events[3].data == {"mute": True, "resume": 1, "steps": []}


# ---------------------------------------------------------------------------
# plan 边界
# ---------------------------------------------------------------------------


def test_missing_terminator_preserves_unprocessed_tail():
    plan, _ = _task_plan([
        '{"type":"keep","no":1}\n',
        '{"type":"edit","no":2,"text":"留下改写","emo":"普通"}\n',
    ])
    assert plan.resume == 3
    assert plan.fallback_continue is True
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "edit")]


def test_continue_with_gap_backfills_remove():
    plan, _ = _task_plan([
        '{"type":"keep","no":1}\n',
        '{"type":"continue","no":3}\n',
    ])
    assert plan.resume == 3
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "remove")]


def test_same_no_consecutive_edits_merge_into_one_step():
    plan, _ = _task_plan([
        '{"type":"edit","no":1,"text":"第一段","emo":"普通"}\n',
        '{"type":"edit","no":1,"text":"第二段","emo":"普通"}\n',
        '{"type":"continue","no":2}\n',
    ])
    assert plan.resume == 2
    assert len(plan.steps) == 1
    assert [line.text for line in plan.steps[0].lines] == ["第一段", "第二段"]


def test_add_after_original_op_is_not_streamed_and_enters_steps():
    plan, _ = _task_plan([
        '{"type":"add","no":0,"text":"前缀","emo":"普通"}\n',
        '{"type":"edit","no":1,"text":"改写","emo":"普通"}\n',
        '{"type":"add","no":0,"text":"插入","emo":"普通"}\n',
        '{"type":"continue","no":2}\n',
    ])
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "edit"), (0, "add")]
    assert plan.steps[1].lines[0].text == "插入"


# ---------------------------------------------------------------------------
# 失败 / 零有效 / 部分失败
# ---------------------------------------------------------------------------


async def test_task_zero_valid_retries_then_error():
    calls = []

    async def primary(_request):
        calls.append("primary")
        for chunk in fx.F10_TEXT:
            yield chunk

    async def repair(_request):
        calls.append("repair")
        for chunk in fx.F10_TEXT:
            yield chunk

    orchestrator = InterjectOrchestrator(
        make_request(fx.task_request()),
        session_id="session-task-f10",
        stream_factory=primary,
        repair_factory=repair,
    )
    events = await collect_events(orchestrator)
    assert calls == ["primary", "repair"]
    assert [event.event for event in events] == ["status", "error"]
    assert events[-1].data["code"] == "no_valid_lines"
    assert events[-1].data["valid_lines"] == 0


async def test_task_partial_failure_keeps_line_but_never_plans():
    orchestrator = make_orchestrator(
        fx.task_request(),
        chunks=fx.F05_JSONL[:2],
        error=RuntimeError("connection reset"),
    )
    events = await collect_events(orchestrator)
    assert [event.event for event in events] == ["status", "line", "error"]
    assert events[-1].data["code"] == "stream_failed"
    assert "plan" not in [event.event for event in events]


async def test_task_valid_lines_does_not_count_control_rows_or_mute():
    """协议 5.2：valid_lines 只计有效正文行（add/edit）；其他操作不计。"""
    events = await collect_events(
        make_orchestrator(
            fx.task_request(),
            chunks=[
                '{"type":"mute"}\n',
                '{"type":"keep","no":1}\n',
                '{"type":"remove","no":2}\n',
                '{"type":"continue","no":3}\n',
            ],
            error=RuntimeError("stream broke"),
        )
    )
    assert events[-1].data["valid_lines"] == 0


async def test_task_valid_lines_counts_add_and_edit_before_split():
    """正文行按模型输出行计、拆分前：一条超长 add + 一条 edit = 2。"""
    events = await collect_events(
        make_orchestrator(
            fx.task_request(),
            chunks=[
                '{"type":"add","no":0,"text":"' + fx.F05B_LONG_EDIT + '","emo":"普通"}\n',
                '{"type":"edit","no":1,"text":"北边仓库昨晚丢了一批货。","emo":"普通"}\n',
            ],
            error=RuntimeError("stream broke"),
        )
    )
    # 超长 add 被拆成多个 line 事件，但仍只算 1 条模型输出行
    assert len([event for event in events if event.event == "line"]) >= 2
    assert events[-1].data["valid_lines"] == 2


# ---------------------------------------------------------------------------
# 工具轮与工具矩阵
# ---------------------------------------------------------------------------


class _FakeRegistry:
    def get_openai_tools(self, names=None):
        return [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
            for name in (names or [])
        ]


def test_task_runtime_registers_only_search_and_memory(monkeypatch):
    monkeypatch.setattr(
        "services.dialogue.delegation.delegate_tool_or_none",
        lambda **_kwargs: {"type": "function", "function": {"name": "delegate_dialogue_rewrite"}},
    )
    blocks = DailyBlocks(
        system_prompt="SYS",
        registry=_FakeRegistry(),
        long_term=SimpleNamespace(base=None),
        task_publish_disabled=True,
        delegate_enabled=False,
    )
    runtime = DailyToolRuntime(make_request(fx.task_request()), blocks)
    names = [item["function"]["name"] for item in (runtime.tools() or [])]
    assert names == ["route_search", "update_long_term_memory"]
    assert not any(name.startswith("route_task_") for name in names)
    assert "delegate_dialogue_rewrite" not in names


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


class _FakeExecutor:
    def __init__(self, tools, result: ToolExecutionResult):
        self._tools = tools
        self.result = result
        self.calls = []

    def tools(self):
        return self._tools

    async def execute(self, tool_calls, *, round_text=""):
        self.calls.append((tool_calls, round_text))
        return self.result


SEARCH_CALL = {
    "type": "function",
    "id": "call_search_task",
    "function": {"name": "route_search", "arguments": '{"keywords":"仓库暗号"}'},
}
SEARCH_SCHEMA = [{
    "type": "function",
    "function": {"name": "route_search", "description": "", "parameters": {}},
}]


async def test_task_tool_round_emits_searching_then_final_without_tools():
    client = _ScriptedLLM([
        [(None, [SEARCH_CALL])],
        [(
            '{"type":"add","no":0,"text":"暗号是北风。","emo":"普通"}\n'
            '{"type":"keep","no":1}\n'
            '{"type":"edit","no":2,"text":"北边仓库昨晚丢了一批货。","emo":"普通"}\n'
            '{"type":"continue","no":3}\n',
            None,
        )],
    ])
    executor = _FakeExecutor(
        SEARCH_SCHEMA,
        ToolExecutionResult(
            requires_followup=True,
            messages=[],
            result_block="【办理结果】找到资料",
            instruction="用 JSONL 回答",
            route_kind="search",
        ),
    )
    factory = partial(
        stream_task_llm,
        client=client,
        blocks=DailyBlocks(
            system_prompt="SYS",
            task_publish_disabled=True,
            delegate_enabled=False,
        ),
        tool_executor=executor,
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.task_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)
    assert [event.event for event in events] == ["status", "status", "line", "plan", "done"]
    assert [event.data["stage"] for event in events[:2]] == ["thinking", "searching"]
    assert all(event.data.get("stage") != "drafting" for event in events[:2])
    assert events[2].data["text"] == "暗号是北风。"
    plan = events[3].data
    assert plan["resume"] == 3
    assert [(step["n"], step["kind"]) for step in plan["steps"]] == [(1, "keep"), (2, "edit")]

    assert client.requests[0].tools == SEARCH_SCHEMA
    assert client.requests[1].tools is None  # 工具结果后不再给工具
    text = "".join(
        event.data["text"] for event in events if event.event == "line"
    )
    assert "仓库暗号" not in text  # 工具参数不进台词


async def test_task_stream_never_emits_drafting_status_even_for_task_tool():
    client = _ScriptedLLM([
        [(None, [{
            "type": "function",
            "id": "call_bad",
            "function": {"name": "route_task_draft", "arguments": "{}"},
        }])],
        [('{"type":"add","no":0,"text":"回到正题。","emo":"普通"}\n{"type":"continue","no":1}\n', None)],
    ])
    executor = _FakeExecutor(
        SEARCH_SCHEMA,
        ToolExecutionResult(requires_followup=False),
    )
    factory = partial(
        stream_task_llm,
        client=client,
        blocks=DailyBlocks(system_prompt="SYS", task_publish_disabled=True),
        tool_executor=executor,
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.task_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)
    stages = [event.data.get("stage") for event in events if event.event == "status"]
    assert stages == ["thinking"]  # 任务模式不产生 drafting


# ---------------------------------------------------------------------------
# API 接线（假流，不调真实模型）
# ---------------------------------------------------------------------------


def _client() -> TestClient:
    interject_api.set_stream_factories(None, None)
    app = FastAPI()
    app.include_router(interject_api.router, prefix="/game")
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_factories():
    yield
    interject_api.set_stream_factories(None, None)


def test_task_api_wires_task_blocks_and_stream(memory_store, monkeypatch):
    captured: dict = {}
    blocks = DailyBlocks(system_prompt="SYS", npc_emotions=("普通",))

    async def _fake_assemble(payload, *, prompt_max=135, session_id=None):
        captured["assemble"] = (payload.mode, prompt_max, session_id)
        return blocks

    async def _fake_stream(request, *, blocks=None, llm_config=None):
        captured["stream"] = (request.mode, blocks)
        for chunk in fx.F05_JSONL:
            yield chunk

    monkeypatch.setattr(interject_api, "llm_is_configured", lambda *a, **k: True)
    monkeypatch.setattr(interject_api, "assemble_task_blocks", _fake_assemble)
    monkeypatch.setattr(interject_api, "stream_task_llm", _fake_stream)

    response = _client().post("/game/dialogue-interject", json=fx.task_request())
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "line", "plan", "done"]
    assert events[2][1]["resume"] == 3
    assert captured["assemble"][0] == "task"
    assert captured["stream"][0] == "task"
    assert captured["stream"][1] is blocks
    assert memory_store.dialogue_session_get_sync(fx.TASK_DIALOGUE_ID) is not None
