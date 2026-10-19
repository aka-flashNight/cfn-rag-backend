"""context / 工具策略接口的 P0 骨架测试（不做提示词全量断言）。"""

from __future__ import annotations

from services.dialogue_interject.context import (
    OMISSION_MARK,
    build_user_context,
    render_played,
    render_reference,
    render_remaining,
)
from services.dialogue_interject.prompts import tool_matrix_for
from services.dialogue_interject.schemas import PlayedLine, ReferenceLine, RemainingLine
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import make_request


def test_render_played_role_mapping_and_omission():
    lines = [
        PlayedLine(seq=0, role="npc", name=fx.NPC, text="原作"),
        PlayedLine(seq=1, role="player", name="$PC", text="玩家话"),
        PlayedLine(seq=2, role="other_npc", name="老赵", text="别挡道。"),
        PlayedLine(seq=3, role="ai", name=fx.NPC, text="AI 话"),
    ]
    assert render_played(lines, npc=fx.NPC) == [
        f"{fx.NPC}：原作",
        "玩家：玩家话",
        "【老赵：别挡道。】",
        f"{fx.NPC}：AI 话",
    ]
    trimmed = render_played(lines, npc=fx.NPC, max_lines=3)
    assert OMISSION_MARK in trimmed


def test_render_remaining_keeps_original_index():
    lines = [RemainingLine.model_validate(row) for row in fx.TASK_REMAINING]
    assert render_remaining(lines)[1] == f"[2] {fx.NPC}：北边仓库昨晚丢了东西。"


def test_render_reference_kinds():
    lines = [
        ReferenceLine(kind="add", name=fx.NPC, text="未播新增"),
        ReferenceLine(kind="closing", name=fx.NPC, text="未播结束语"),
        ReferenceLine(kind="edit", index=5, name=fx.NPC, text="未播改写"),
        ReferenceLine(kind="remove", index=6),
    ]
    assert render_reference(lines) == [
        "上一轮未播新增：未播新增",
        "上一轮未播结束语：未播结束语",
        "上一轮原第 5 条改写：未播改写",
        "上一轮原第 6 条移除",
    ]


def test_task_user_context_contains_task_summary_and_player_input():
    text = build_user_context(make_request(fx.task_request()))
    assert "【任务上下文（任务前）】" in text
    assert "北边仓库失窃" in text
    assert "找回失窃货物" in text
    assert "金币 2400" in text
    assert "【玩家刚刚说的话】" in text
    assert fx.task_request()["player_input"] in text


def test_tool_matrix_b2():
    """B2 日常接工具；B3 委派按注册闸可选；任务模式留 P4。"""
    assert tool_matrix_for("task") == ()
    daily = tool_matrix_for("daily")
    assert "route_task_draft" in daily
    assert "route_search" in daily
    assert "update_long_term_memory" in daily
    assert "route_task_confirm" in tool_matrix_for("daily", has_draft=True)
    assert "delegate_dialogue_rewrite" not in daily
    assert "delegate_dialogue_rewrite" in tool_matrix_for("daily", delegate_enabled=True)
