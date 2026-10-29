"""context / 工具策略接口的 P0 骨架测试（不做提示词全量断言）。"""

from __future__ import annotations

import json

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
        "$PC：玩家话",
        "【老赵：别挡道。】",
        f"{fx.NPC}：AI 话",
    ]
    trimmed = render_played(lines, npc=fx.NPC, max_lines=3)
    assert OMISSION_MARK in trimmed


def test_render_remaining_keeps_original_index_as_json():
    lines = [RemainingLine.model_validate(row) for row in fx.TASK_REMAINING]
    assert json.loads(render_remaining(lines)[1]) == {
        "no": 2,
        "name": fx.NPC,
        "text": "北边仓库昨晚丢了东西。",
    }


def test_render_remaining_counts_player_rows():
    lines = [RemainingLine.model_validate(row) for row in fx.TASK_REMAINING_MIXED]
    rendered = [json.loads(line) for line in render_remaining(lines)]
    assert [line["no"] for line in rendered] == [1, 2, 3, 4]
    assert [line["name"] for line in rendered] == [fx.NPC, "$PC", fx.NPC, "$PC"]
    assert all(set(line) == {"no", "name", "text"} for line in rendered)


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


def _mixed_task_request() -> dict:
    payload = fx.task_request()
    payload["dialogue"]["played"] = [
        {"seq": 0, "role": "player", "name": "$PC", "text": "先说正事。"},
        {"seq": 1, "role": "npc", "name": fx.NPC, "text": "你来了。", "emotion": "普通"},
    ]
    payload["dialogue"]["remaining"] = [dict(row) for row in fx.TASK_REMAINING_MIXED]
    payload["dialogue"]["reference_continuation"] = [
        {"kind": "add", "name": fx.NPC, "text": "上一轮没播完的一句。", "emotion": "普通"},
    ]
    return payload


def test_task_user_context_block_order_and_player_note():
    text = build_user_context(make_request(_mixed_task_request()))
    order = [
        "【本组已播历史】",
        "【任务上下文（任务前）】",
        "【原始剩余台词（按编号no）】",
        "【参考续写",
        "【编号与操作说明】",
        "【玩家刚刚说的话】",
    ]
    positions = [text.index(token) for token in order]
    assert positions == sorted(positions)
    # 最早出现 $PC 的是已播历史，说明必须插在它之前。
    assert text.index("【关于 $PC】") < positions[0]
    assert '"$PC"是玩家姓名的占位符' in text
    assert "name需要填玩家时" in text
    # 剩余台词里的玩家行用 $PC，编号包含玩家行。
    assert '{"no":2,"name":"$PC","text":"货是在哪丢的？"}' in text
    assert "包含所有未播台词（玩家 $PC 行也算一条）" in text


def test_daily_player_note_omits_name_rule():
    text = build_user_context(make_request(fx.daily_turn_b_request()))
    assert "【关于 $PC】" in text
    assert '"$PC"是玩家姓名的占位符' in text
    assert "name需要填玩家时" not in text


def test_tool_matrix_b2():
    """B2 日常接工具；B3 委派按注册闸可选；任务模式留 P4。"""
    assert tool_matrix_for("task") == ("route_search", "update_long_term_memory")
    daily = tool_matrix_for("daily")
    assert "route_task_draft" in daily
    assert "route_search" in daily
    assert "update_long_term_memory" in daily
    assert "route_task_confirm" in tool_matrix_for("daily", has_draft=True)
    assert "delegate_dialogue_rewrite" not in daily
    assert "delegate_dialogue_rewrite" in tool_matrix_for("daily", delegate_enabled=True)
