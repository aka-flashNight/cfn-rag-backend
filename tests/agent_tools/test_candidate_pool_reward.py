"""奖励池分层抽选测试（用户 2026-10-18 定稿口径）。

1) 单例钉选必留（勾了就有）；
2) 关键词命中全保留，允许超过上限；
3) 高优先＝装备等级窗口 [L-5, L+3] ＋ 所有非装备，按 use 轮转填到 10；
4) 任务池非单例条目上限 = max(5, 勾选的非单例类型数)；
5) 窗口外且未命中关键词的装备只做补位，最多 3 条；
6) 没有玩家等级时窗口退化为阶段等级区间。
"""

from __future__ import annotations

import json

from services.agent_tools.context_builder import (
    EQUIP_LEVEL_DOWN,
    EQUIP_LEVEL_UP,
    LOW_PRIORITY_EQUIP_CAP,
    REWARD_POOL_CAP,
    TASK_POOL_DEFAULT_CAP,
    _is_reward_equipment,
    _reward_equip_window,
    prepare_task_context,
)

RT_FULL = {
    "regular": ["金币", "经验值"],
    "optional": ["K点", "技能点", "强化石", "药剂", "弹夹", "材料", "食品"],
}
SINGLETONS = {"金币", "经验值", "K点", "技能点", "强化石"}


def _rewards(game_data, npc: str, *, level=None, reward_types=None, keywords=None, stage=4) -> list[dict]:
    raw = prepare_task_context(
        task_type="通关",
        reward_types=reward_types or RT_FULL,
        npc_name=npc,
        player_progress=stage,
        player_level=level,
        npc_states={},
        reward_keywords=keywords,
        game_data=game_data,
    )
    return json.loads(raw)["reward_item_candidates"]


def _level(game_data, name: str) -> int:
    item = game_data.items.get_by_name(name)
    return int(getattr(item, "level", 0) or 0)


def test_each_side_has_its_own_budget(game_data):
    """单例≤5 ＋ 商店≤10 ＋ 任务池非单例≤max(5, 类型数)；不是全局 10 条。"""
    for npc in ("King", "铁匠", "冷兵器商人"):
        rows = _rewards(game_data, npc, level=25)
        shop = [r for r in rows if r.get("source") == "本NPC商店"]
        task_non_sing = [
            r for r in rows
            if r.get("source") != "本NPC商店" and r["name"] not in SINGLETONS
        ]
        pinned = [r for r in rows if r["name"] in SINGLETONS]
        assert len(pinned) <= 5
        assert len(shop) <= 10
        assert len(task_non_sing) <= max(5, 3)
        assert len(rows) == len(pinned) + len(shop) + len(task_non_sing)
        assert len(rows) <= 20


def test_selected_singletons_always_kept(game_data):
    rows = _rewards(game_data, "铁匠", level=25)
    names = {r["name"] for r in rows}
    assert {"金币", "经验值", "K点", "技能点", "强化石"} <= names


def test_keyword_hits_may_exceed_cap(game_data):
    rows = _rewards(game_data, "King", level=25, keywords=["药剂"])
    # 命中数（8 条）+ 单例钉选（5 条）→ 13，超过上限而不截断
    assert len(rows) > REWARD_POOL_CAP
    hit = [r for r in rows if r["name"] not in SINGLETONS]
    assert len(hit) >= 6


def test_task_pool_cap_counts_types(game_data):
    """任务池非单例条目不超过 max(5, 勾选的非单例类型数)。"""
    for npc in ("铁匠", "King", "冷兵器商人"):
        rows = _rewards(game_data, npc, level=25)
        task_non_sing = [
            r for r in rows
            if r.get("source") != "本NPC商店" and r["name"] not in SINGLETONS
        ]
        assert len(task_non_sing) <= TASK_POOL_DEFAULT_CAP


def test_equipment_window_math():
    assert _reward_equip_window(player_level=25, stage_level_range=(20, 31)) == (
        25 - EQUIP_LEVEL_DOWN, 25 + EQUIP_LEVEL_UP,
    )
    # 没有玩家等级 → 退化用阶段区间
    assert _reward_equip_window(player_level=None, stage_level_range=(20, 31)) == (20, 31)


def test_neck_equipment_is_not_equipment(game_data):
    necklace = next(i for i in game_data.items.items if i.use == "颈部装备")
    assert _is_reward_equipment(necklace) is False
    weapon = next(i for i in game_data.items.items if i.type == "武器")
    assert _is_reward_equipment(weapon) is True


def test_high_priority_equipment_inside_window(game_data):
    """有等级时，进池的装备要么在窗口内，要么是低优先补位（≤3 件）。"""
    level = 25
    lo, hi = level - EQUIP_LEVEL_DOWN, level + EQUIP_LEVEL_UP
    rows = _rewards(game_data, "冷兵器商人", level=level,
                    reward_types={"regular": ["金币"], "optional": ["武器"]})
    equip = [r for r in rows if _is_reward_equipment(game_data.items.get_by_name(r["name"]))]
    outside = [r for r in equip if not (lo <= _level(game_data, r["name"]) <= hi)]
    assert len(outside) <= LOW_PRIORITY_EQUIP_CAP


def test_no_level_falls_back_to_stage_range(game_data):
    """没有玩家等级时窗口退化为阶段区间（阶段 4＝20-31），装备仍能进池。"""
    rows = _rewards(game_data, "冷兵器商人", level=None,
                    reward_types={"regular": ["金币"], "optional": ["武器"]}, stage=4)
    equip = [r for r in rows if _is_reward_equipment(game_data.items.get_by_name(r["name"]))]
    assert equip, "退化到阶段区间后仍应选出装备"
    assert all(
        20 <= _level(game_data, r["name"]) <= 31 for r in equip
    ), [(r["name"], _level(game_data, r["name"])) for r in equip]


def test_uses_are_spread(game_data):
    """高优先层按 use 轮转：同一 use 不会把 10 个名额占满。"""
    rows = [
        r for r in _rewards(game_data, "铁匠", level=25)
        if r["name"] not in SINGLETONS and r.get("source") != "本NPC商店"
    ]
    uses = []
    for r in rows:
        item = game_data.items.get_by_name(r["name"])
        uses.append(getattr(item, "use", None) or getattr(item, "type", None))
    from collections import Counter

    top = Counter(uses).most_common(1)[0][1]
    assert top <= 3, Counter(uses)
