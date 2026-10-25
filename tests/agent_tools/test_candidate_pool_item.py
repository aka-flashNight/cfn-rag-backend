"""物品候选池收窄（关键词池＋随机池）测试。

口径（用户 2026-10-17/18）：
- 每池上限 10 条，命中的全进、超额留 5 + 其余随机、不足随机补足；
- 来源标 `matched_by`（标签原文 / 随机候选）；
- 酒水＝「调酒」配方产物 ∪ 商店在售且名字含「酒」；药剂不含酒水；礼包不注册。
"""

from __future__ import annotations

import json

from services.agent_tools import pool_tags as pt
from services.agent_tools.context_builder import ITEM_POOL_CAP, prepare_task_context

RT = {"regular": ["金币", "经验值"], "optional": ["药剂", "材料"]}

KEY_BY_TYPE = {
    "资源收集": "collectable_items",
    "装备缴纳": "equipment_items",
    "特殊物品获取": "special_items",
    "物品持有": "holdable_items",
}


def _rows(game_data, task_type: str, *, keywords=None, stage: int = 4) -> list[dict]:
    raw = prepare_task_context(
        task_type=task_type,
        reward_types=RT,
        npc_name="冷兵器商人",
        npc_faction="商人",
        player_progress=stage,
        npc_states={},
        requirement_keywords=keywords,
        game_data=game_data,
    )
    return json.loads(raw).get(KEY_BY_TYPE[task_type]) or []


def test_item_pool_cap(game_data):
    for task_type in KEY_BY_TYPE:
        for stage in (1, 4):
            assert len(_rows(game_data, task_type, stage=stage)) <= ITEM_POOL_CAP


def test_every_row_has_matched_by(game_data):
    for task_type in KEY_BY_TYPE:
        rows = _rows(game_data, task_type)
        assert rows
        for r in rows:
            assert r.get("matched_by")


def test_without_keywords_all_rows_are_random(game_data):
    for task_type in KEY_BY_TYPE:
        rows = _rows(game_data, task_type)
        assert all(r["matched_by"] == pt.RANDOM_PICK_LABEL for r in rows)


def test_liquor_tag_only_returns_liquor(game_data):
    liquor = pt.liquor_names(game_data)
    rows = _rows(game_data, "特殊物品获取", keywords=["酒水"])
    assert rows
    for r in rows:
        assert r["name"] in liquor, r["name"]
        assert r["matched_by"] == "酒水"


def test_medicine_tag_excludes_liquor(game_data):
    liquor = pt.liquor_names(game_data)
    rows = _rows(game_data, "特殊物品获取", keywords=["药剂"])
    assert rows
    assert all(r["name"] not in liquor for r in rows), "药剂标签不应命中酒水"


def test_equipment_tag_matches_type(game_data):
    rows = _rows(game_data, "装备缴纳", keywords=["防具"])
    assert rows
    for r in rows:
        item = game_data.items.get_by_name(r["name"])
        assert item is not None and item.type == "防具"


def test_armor_slot_tag(game_data):
    rows = _rows(game_data, "装备缴纳", keywords=["头部装备"])
    assert rows
    for r in rows:
        item = game_data.items.get_by_name(r["name"])
        assert item is not None and item.use == "头部装备"


def test_unknown_tag_falls_back_to_freeword(game_data):
    """没注册的标签（如资源收集的「食材」、任何类型的「礼包」）不会静默当标签。"""
    tags, words, _all = pt.ITEM_TAGS_PER_TASK, [], []
    assert "食材" not in tags["资源收集"]
    for task_tags in tags.values():
        assert "礼包" not in task_tags

    # 用真实请求验证：食材 会被当自由词，池里没命中就退化为全随机
    rows = _rows(game_data, "资源收集", keywords=["食材"])
    assert len(rows) <= ITEM_POOL_CAP


def test_overflow_hits_are_all_keyword_sourced(game_data):
    rows = _rows(game_data, "装备缴纳", keywords=["防具"], stage=7)
    if len(rows) == ITEM_POOL_CAP:
        assert all(r["matched_by"] != pt.RANDOM_PICK_LABEL for r in rows)


def test_exact_name_hit_ranks_first(game_data):
    target = "扎啤"   # 调酒产物，在特殊物品池里
    rows = _rows(game_data, "特殊物品获取", keywords=[target], stage=7)
    assert rows and rows[0]["name"] == target


def test_vip_price_threshold(game_data):
    rows = _rows(game_data, "特殊物品获取", keywords=["贵重"])
    assert rows
    for r in rows:
        assert int(r.get("price") or 0) >= pt.VALUABLE_PRICE_THRESHOLD
