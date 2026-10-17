"""confirm 阶段的回访配对：归一化、说话人校验、一次修正重试与降级。

这里是两段式最容易被模型说错的环节——完成 NPC 与接取 NPC 不同时，模型经常把
完成对话说成接取人。覆盖 sanitizer 的场次占位、$PC 收敛、重试只加一次、以及
两条降级路径（保留原对话退回单任务 / 整份换模板）。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from services.memory.store import TaskDraftRow
from services.orchestrator.turn import TurnOrchestrator
from tests.orchestrator.conftest import NPC_NAME

NPC_A = NPC_NAME       # 接取 NPC（铁匠）
NPC_B = "药师"          # 完成 NPC

DRAFT = {
    "draft_id": "d1",
    "task_type": "传话",
    "title": "带话给药师",
    "npc_name": NPC_A,
    "get_npc": NPC_A,
    "finish_npc": NPC_B,
    "rewards": [{"item_name": "金币", "count": 20000}],
}


def _orch(env) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id,
        npc_name=NPC_A,
        query="接！",
        deps=env.deps,
    )


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(rag_context_text=None, npc_states=None)


def _draft_row() -> TaskDraftRow:
    return TaskDraftRow(
        session_id="s1", draft_id="d1", npc_name=NPC_A, draft=dict(DRAFT),
    )


def _valid_args() -> dict:
    return {
        "title": "带话给药师",
        "description": "把话带给药师。",
        "get_dialogue": [{"name": NPC_A, "emotion": "", "text": "替我把话带到。"}],
        "finish_dialogue": [{"name": NPC_B, "emotion": "", "text": "话我收到了。"}],
        "return_for_reward": True,
        "return_description": "回到铁匠处复命，领取这份委托的报酬。",
        "return_get_dialogue": [{"name": NPC_B, "emotion": "", "text": "回去找铁匠领赏。"}],
        "return_finish_dialogue": [{"name": NPC_A, "emotion": "", "text": "报酬拿好。"}],
    }


def _bad_speaker_args() -> dict:
    """完成对话说成接取人：四组里只有 finish_dialogue 非法。"""
    args = _valid_args()
    args["finish_dialogue"] = [{"name": NPC_A, "emotion": "", "text": "话我收到了。"}]
    return args


def _bad_return_args() -> dict:
    """原两组合规，只有回访接取对话说错人。"""
    args = _valid_args()
    args["return_get_dialogue"] = [{"name": NPC_A, "emotion": "", "text": "回去找药师领赏。"}]
    return args


async def _run(env, responses: list[dict | str]):
    orch = _orch(env)
    for item in responses:
        env.fake.add_chat(
            content=item if isinstance(item, str) else json.dumps(item, ensure_ascii=False),
        )
    return await orch._generate_confirm_args(
        env.fake, _ctx(), _draft_row(), env.game_data, base_messages=[],
    )


# ---------------------------------------------------------------------------
# sanitizer
# ---------------------------------------------------------------------------

def test_sanitizer_clears_return_fields_when_flag_false(env):
    orch = _orch(env)
    args = orch._sanitize_confirm_args({
        "title": "带话给药师",
        "get_dialogue": [{"name": NPC_A, "text": "替我把话带到。"}],
        "finish_dialogue": [{"name": NPC_B, "text": "话我收到了。"}],
        "return_for_reward": False,
        "return_description": "残留描述",
        "return_get_dialogue": [{"name": NPC_B, "text": "残留"}],
        "return_finish_dialogue": [{"name": NPC_A, "text": "残留"}],
    }, get_npc=NPC_A, finish_npc=NPC_B)

    assert args["return_for_reward"] is False
    assert args["return_description"] == ""
    assert args["return_get_dialogue"] == []
    assert args["return_finish_dialogue"] == []


def test_sanitizer_fills_speaker_by_scene(env):
    orch = _orch(env)
    args = orch._sanitize_confirm_args({
        "get_dialogue": [{"text": "替我把话带到。"}],
        "finish_dialogue": [{"text": "话我收到了。"}],
    }, get_npc=NPC_A, finish_npc=NPC_B)

    assert args["get_dialogue"][0]["name"] == NPC_A
    assert args["finish_dialogue"][0]["name"] == NPC_B


def test_sanitizer_placeholder_uses_finish_npc(env):
    orch = _orch(env)
    args = orch._sanitize_confirm_args(
        {"finish_dialogue": "不是数组"}, get_npc=NPC_A, finish_npc=NPC_B,
    )
    assert args["finish_dialogue"][0]["name"] == NPC_B


def test_sanitizer_keeps_return_groups_when_flag_true(env):
    orch = _orch(env)
    args = orch._sanitize_confirm_args({
        "return_for_reward": "true",
        "return_description": "  回到铁匠处复命。  ",
        "return_get_dialogue": [{"text": "回去领赏吧。"}],
        "return_finish_dialogue": [{"text": "报酬拿好。"}],
    }, get_npc=NPC_A, finish_npc=NPC_B)

    assert args["return_for_reward"] is True
    assert args["return_description"] == "回到铁匠处复命。"
    assert args["return_get_dialogue"][0]["name"] == NPC_B
    assert args["return_finish_dialogue"][0]["name"] == NPC_A


def test_sanitizer_collapses_player_aliases(env):
    orch = _orch(env)
    args = orch._sanitize_confirm_args({
        "get_dialogue": [
            {"name": "玩家", "text": "好。"},
            {"name": "$PC_TITLE", "text": "行。"},
            {"name": "（$PC）", "text": "嗯。"},
            {"name": "$pc", "text": "走。"},
        ],
        "return_for_reward": True,
        "return_get_dialogue": [{"name": "$PC_CHAR", "text": "我回去了。"}],
    }, get_npc=NPC_A, finish_npc=NPC_B)

    assert [r["name"] for r in args["get_dialogue"]] == ["$PC"] * 4
    assert args["return_get_dialogue"][0]["name"] == "$PC"


# ---------------------------------------------------------------------------
# 一次修正重试
# ---------------------------------------------------------------------------

async def test_retry_once_then_accept_pairing(env):
    args, warnings = await _run(env, [_bad_speaker_args(), _valid_args()])

    assert warnings == []
    assert args["return_for_reward"] is True
    assert args["finish_dialogue"][0]["name"] == NPC_B
    assert len(env.fake.chat_requests) == 2
    # 第二次请求带上了第一次原文 + 逐条修正说明
    retry_messages = env.fake.chat_requests[1].messages
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-1]["role"] == "user"
    assert "上一次输出未通过校验" in retry_messages[-1]["content"]
    assert "finish_dialogue" in retry_messages[-1]["content"]
    assert "$PC" in retry_messages[-1]["content"]


async def test_retry_usage_accumulates_twice(env):
    orch = _orch(env)
    env.fake.add_chat(
        content=json.dumps(_bad_speaker_args(), ensure_ascii=False),
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    env.fake.add_chat(
        content=json.dumps(_valid_args(), ensure_ascii=False),
        usage={"prompt_tokens": 20, "completion_tokens": 5},
    )

    await orch._generate_confirm_args(
        env.fake, _ctx(), _draft_row(), env.game_data, base_messages=[],
    )

    assert orch._usage["prompt_tokens"] == 30
    assert orch._usage["completion_tokens"] == 10
    assert len(orch._usage["calls"]) == 2


async def test_second_attempt_still_bad_keeps_base_dialogue_and_drops_pairing(env):
    args, warnings = await _run(env, [_bad_return_args(), _bad_return_args()])

    # 原两组合规 → 保留模型产出，只关掉配对
    assert args["return_for_reward"] is False
    assert args["return_get_dialogue"] == []
    assert args["return_finish_dialogue"] == []
    assert args["return_description"] == ""
    assert args["get_dialogue"][0]["text"] == "替我把话带到。"
    assert warnings
    assert len(env.fake.chat_requests) == 2


async def test_second_attempt_still_bad_with_broken_base_uses_fallback(env):
    args, warnings = await _run(env, [_bad_speaker_args(), _bad_speaker_args()])

    # 原两组也不合规 → 整份换模板（说话人已按场次修正）
    assert args["return_for_reward"] is False
    assert args["title"] == DRAFT["title"]
    assert args["finish_dialogue"][0]["name"] == NPC_B
    assert args["finish_dialogue"][0]["text"] == "干得漂亮，说好的报酬一分不少。"
    assert args["get_dialogue"][0]["name"] == NPC_A
    assert warnings


async def test_call_failure_falls_back_without_retry(env):
    class _BoomLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, req):
            self.calls += 1
            raise RuntimeError("上游炸了")

    llm = _BoomLLM()
    orch = _orch(env)
    args, _warnings = await orch._generate_confirm_args(
        llm, _ctx(), _draft_row(), env.game_data, base_messages=[],
    )

    assert llm.calls == 1
    assert args["title"] == DRAFT["title"]
    assert args["return_for_reward"] is False


async def test_parse_failure_falls_back_without_retry(env):
    args, _warnings = await _run(env, ["这不是 JSON"])

    assert len(env.fake.chat_requests) == 1
    assert args["title"] == DRAFT["title"]


async def test_cancelled_error_propagates(env):
    class _CancelLLM:
        async def chat(self, req):
            raise asyncio.CancelledError()

    orch = _orch(env)
    with pytest.raises(asyncio.CancelledError):
        await orch._generate_confirm_args(
            _CancelLLM(), _ctx(), _draft_row(), env.game_data, base_messages=[],
        )
