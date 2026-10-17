"""草案摘要标出「这是本 NPC 自己卖的东西」。

口径（用户 2026-10-18）：候选里 source=本NPC商店 的物品被选进草案后，
摘要与物品详情行要标注来源，让聊天 Agent 知道哪些东西是这位 NPC 在售的，避免信息遗漏。
"""

from __future__ import annotations

from services.agent_tools.draft_formatting import (
    _detailed_draft_summary,
    _npc_shop_marker,
)

MARKER = "本NPC商店在售"


def _draft(npc: str, rewards=None, submit=None, contain=None) -> dict:
    return {
        "draft_id": "d1",
        "npc_name": npc,
        "task_type": "通关",
        "title": "清掉废城的僵尸",
        "rewards": rewards or [],
        "finish_submit_items": submit or [],
        "finish_contain_items": contain or [],
    }


def _reward_line(game_data, npc: str, item: str) -> str:
    summary = _detailed_draft_summary(
        _draft(npc, rewards=[{"item_name": item, "count": 1}]), game_data,
    )
    return next(line for line in summary.split("\n") if line.startswith("奖励:"))


def test_shop_item_is_marked(game_data):
    """King 自己卖的马格南手枪弹药要标出来。"""
    line = _reward_line(game_data, "King", "马格南手枪弹药")
    assert MARKER in line


def test_non_shop_item_is_not_marked(game_data):
    """不在他店里的物品不标。"""
    assert "马格南手枪弹药" in set(game_data.shops.get_npc_shop("King"))
    line = _reward_line(game_data, "King", "镜之虎彻制作图纸")
    assert MARKER not in line


def test_npc_without_shop_never_marks(game_data):
    """无商店 NPC（铁匠）一律不标。"""
    assert not game_data.shops.has_shop("铁匠")
    line = _reward_line(game_data, "铁匠", "马格南手枪弹药")
    assert MARKER not in line


def test_marker_helper_edges(game_data):
    assert _npc_shop_marker(game_data, "", "马格南手枪弹药") == ""
    assert _npc_shop_marker(game_data, "King", "") == ""
    assert _npc_shop_marker(None, "King", "马格南手枪弹药") == ""


def test_submit_and_contain_lines_also_marked(game_data):
    summary = _detailed_draft_summary(
        _draft(
            "King",
            submit=[{"item_name": "手枪通用弹药", "count": 2}],
            contain=[{"item_name": "马格南手枪弹药", "count": 1}],
        ),
        game_data,
    )
    submit_line = next(l for l in summary.split("\n") if l.startswith("提交物品:"))
    contain_line = next(l for l in summary.split("\n") if l.startswith("持有物品:"))
    assert MARKER in submit_line
    assert MARKER in contain_line


def test_entity_line_marks_shop_source(game_data):
    """[物品] 详情行同样带标注（模型看物品详细信息时也能看到）。"""
    summary = _detailed_draft_summary(
        _draft("King", rewards=[{"item_name": "马格南手枪弹药", "count": 10}]),
        game_data,
    )
    item_line = next(l for l in summary.split("\n") if l.startswith("[物品] 名称：马格南手枪弹药"))
    assert MARKER in item_line
