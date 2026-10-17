"""关卡候选池收窄（关键词池＋随机池）测试。

口径（用户 2026-10-17）：
- 每个池子最多 10 条，不足则全放；
- 关键词命中的行全部进池；超额时按相似度留 5 + 其余在命中里随机；来源标关键词；
- 不足 10 条时按 8:2 随机补足，标「随机候选」；
- 中立外交地图（type=外交地图）不是可攻打关卡，必须排除；
- 军阀这类「本方区域全是自家关」的势力不注册「周边地区」，也不能被当标签命中。
"""

from __future__ import annotations

import json

from services.agent_tools import pool_tags as pt
from services.agent_tools.context_builder import (
    STAGE_POOL_CAP,
    _split_stage_keywords,
    prepare_task_context,
)

DIPLOMATIC_STAGES = {
    "幸存者营地", "同盟卸货站", "外交-军阀", "外交-联合大学", "外交-堕落城商业街",
    "外交-堕落城酒吧", "外交-摇滚公园", "外交-第一防线防区", "外交-隧道据点",
    "外交-黑铁会修炼场", "外交-黑铁阁",
}

RT = {"regular": ["金币", "经验值"], "optional": ["材料"]}


def _rows(game_data, *, task_type="通关", keywords=None, faction="", stage=1) -> list[dict]:
    raw = prepare_task_context(
        task_type=task_type,
        reward_types=RT,
        npc_name="测试NPC",
        npc_faction=faction,
        player_progress=stage,
        npc_states={},
        requirement_keywords=keywords,
        game_data=game_data,
    )
    ctx = json.loads(raw)
    key = "stage_list" if task_type != "通关并收集" else "stage_loot_list"
    return [(b["area"], s) for b in ctx.get(key, []) for s in b["stages"]]


def test_diplomatic_maps_never_enter_pool(game_data):
    for stage in (1, 4, 7):
        for task_type in ("通关", "通关并收集"):
            for _area, row in _rows(game_data, task_type=task_type, stage=stage):
                assert row["name"] not in DIPLOMATIC_STAGES, row["name"]


def test_pool_cap_is_ten(game_data):
    for stage in (1, 4, 7):
        assert len(_rows(game_data, stage=stage)) <= STAGE_POOL_CAP


def test_every_row_carries_matched_by(game_data):
    rows = _rows(game_data, keywords=["势均力敌"], stage=4)
    assert rows
    for _area, row in rows:
        assert row["matched_by"] == "势均力敌" or row["matched_by"] == pt.RANDOM_PICK_LABEL


def test_enemies_field_is_at_most_two_and_known(game_data):
    for _area, row in _rows(game_data, stage=4):
        enemies = row.get(pt.ENEMIES_FIELD) or []
        assert len(enemies) <= 2
        assert set(enemies) <= set(pt.ENEMY_LABEL_VALUES), enemies


def test_warlord_has_no_home_area_tag(game_data):
    tags, words, _all = _split_stage_keywords(
        ["周边地区", "敌对区域"], publisher_faction="军阀", game_data=game_data,
    )
    assert "周边地区" not in tags          # 军阀不注册
    assert "周边地区" in words             # 当自由词处理，不会静默变成标签
    assert "敌对区域" in tags


def test_blackiron_has_home_area_tag(game_data):
    tags, _words, _all = _split_stage_keywords(
        ["周边地区", "敌对区域"], publisher_faction="黑铁会", game_data=game_data,
    )
    assert "周边地区" in tags and "敌对区域" in tags


def test_merchant_has_neither_tag(game_data):
    tags, words, _all = _split_stage_keywords(
        ["周边地区", "敌对区域"], publisher_faction="商人", game_data=game_data,
    )
    assert tags == []
    assert set(words) == {"周边地区", "敌对区域"}


def test_stage_tags_are_mutually_complementary(game_data):
    in_range = _rows(game_data, keywords=["势均力敌"], stage=4)
    below = _rows(game_data, keywords=["轻松应对"], stage=4)
    assert all(not row.get("below_progress") for _a, row in in_range)
    assert all(row.get("below_progress") for _a, row in below)


def test_exact_name_hit_ranks_first(game_data):
    target = "地铁站"
    rows = _rows(game_data, keywords=[target], stage=1)
    assert rows, "地铁站应可命中"
    assert rows[0][1]["name"] == target


def test_overflow_hits_are_all_keyword_sourced(game_data):
    """命中数超过上限时，10 条全部来自关键词，不含随机候选。"""
    rows = _rows(game_data, keywords=["势均力敌"], stage=4)
    assert len(rows) == STAGE_POOL_CAP
    assert all(row["matched_by"] != pt.RANDOM_PICK_LABEL for _a, row in rows)


def test_shortfall_is_filled_with_random(game_data):
    """命中不足上限时，其余名额随机补足并标「随机候选」。"""
    rare = "新手练习场"
    rows = _rows(game_data, keywords=[rare], stage=1)
    assert len(rows) == STAGE_POOL_CAP
    assert any(row["matched_by"] == pt.RANDOM_PICK_LABEL for _a, row in rows)


def test_no_keywords_pool_is_all_random(game_data):
    rows = _rows(game_data, stage=4)
    assert len(rows) == STAGE_POOL_CAP
    assert all(row["matched_by"] == pt.RANDOM_PICK_LABEL for _a, row in rows)
