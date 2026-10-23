from __future__ import annotations

import random
from types import SimpleNamespace

from services.dialogue_interject.item_pool import (
    build_gift_pool,
    build_take_pool,
    category_counts,
    item_category,
    remaining_pool_counts,
    sample_names,
    search_names,
)
from services.dialogue_interject.item_value import (
    enhancement_description,
    format_gift_failure,
    format_item_info,
    format_take_completion,
    gift_value_cap,
    reduce_gift_quantity,
)
from services.game_data.models import Item


class FakeItems:
    def __init__(self, items):
        self._by_name = {item.name: item for item in items}

    def get_by_name(self, name):
        return self._by_name.get(name)

    def get_price(self, name):
        item = self._by_name.get(name)
        return int(item.price or 0) if item else 0

    def search(self, keyword, *, limit=50):
        return [item for item in self._by_name.values() if keyword in item.name][:limit]


class FakeTasks:
    def __init__(self, rewards=(), submits=()):
        self._rewards = set(rewards)
        self._submits = set(submits)

    def list_core_reward_item_names(self):
        return set(self._rewards)

    def list_core_submit_items(self):
        return set(self._submits)


def make_game_data(items, *, rewards=(), submits=()):
    return SimpleNamespace(
        items=FakeItems(items),
        tasks=FakeTasks(rewards, submits),
        equipment_mods=None,
        crafting=SimpleNamespace(_recipes=[]),
        shops=SimpleNamespace(_shops={}),
        kshop=SimpleNamespace(list_items=lambda: []),
        stages=SimpleNamespace(_stage_infos={}),
    )


def _items():
    return [
        Item(name="普通hp药剂", type="消耗品", use="药剂", price=250),
        Item(name="普通mp药剂", type="消耗品", use="药剂", price=200),
        Item(name="M4A1", type="武器", use="步枪", price=500),
        Item(name="铁甲", type="防具", use="身体装备", price=800),
        Item(name="铁矿石", type="收集品", use="材料", price=30),
        Item(name="战场口粮", type="消耗品", use="战场补给", price=100),
        Item(name="杂交测试种", type="消耗品", use="药剂", price=100),
        Item(name="任务信物", type="收集品", use="材料", price=100),
    ]


def test_gift_pool_filters_and_keeps_singleton_currency():
    gd = make_game_data(
        _items(),
        rewards=[
            "普通hp药剂", "金币", "战场口粮", "杂交测试种", "任务信物",
            "有声望", "零价物品",
        ],
        submits=["任务信物"],
    )
    pool = build_gift_pool(gd)
    assert "普通hp药剂" in pool
    assert "金币" in pool
    assert "战场口粮" not in pool
    assert "杂交测试种" not in pool
    assert "任务信物" not in pool
    assert "有声望" not in pool


def test_take_pool_gold_first_and_filters_currencies():
    gd = make_game_data(_items())
    pool = build_take_pool(
        gd,
        ["普通hp药剂", "M4A1", "经验值", "K点", "技能点", "杂交测试种"],
    )
    assert pool[0] == "金币"
    assert "M4A1" in pool
    assert "普通hp药剂" in pool
    assert all(name not in pool for name in ("经验值", "K点", "技能点", "杂交测试种"))


def test_search_and_sample():
    gd = make_game_data(_items())
    pool = build_take_pool(gd, ["普通hp药剂", "普通mp药剂", "M4A1"])
    hits = search_names(gd, pool, "药剂")
    assert hits == ["普通hp药剂", "普通mp药剂"]
    assert search_names(gd, pool, "不存在") == []

    picked = sample_names(pool, count=2, gold_first=True, rng=random.Random(1))
    assert picked[0] == "金币"
    assert len(picked) == 2
    assert len(set(picked)) == 2


def test_category_counts_and_remaining():
    gd = make_game_data(_items())
    pool = ["金币", "M4A1", "铁甲", "普通hp药剂", "铁矿石"]
    assert category_counts(gd, pool) == {
        "武器": 1, "防具": 1, "消耗品": 1, "材料": 1, "其他": 0, "金币": 1,
    }
    counts, total = remaining_pool_counts(gd, pool, ["M4A1"])
    assert total == 5
    assert counts["武器"] == 0
    assert counts["金币"] == 1


def test_item_category():
    gd = make_game_data(_items())
    assert item_category(gd, "金币") == "金币"
    assert item_category(gd, "M4A1") == "武器"
    assert item_category(gd, "铁甲") == "防具"
    assert item_category(gd, "普通hp药剂") == "消耗品"
    assert item_category(gd, "铁矿石") == "材料"
    assert item_category(gd, "未知物品") == "其他"


def test_gift_cap_and_reduce():
    gd = make_game_data(_items())
    cap = gift_value_cap(stage=1, favorability=40)
    assert cap == 30000
    assert reduce_gift_quantity(gd, "普通hp药剂", 200, cap=cap) == (120, None)
    assert reduce_gift_quantity(gd, "金币", 50000, cap=cap) == (30000, None)
    assert reduce_gift_quantity(gd, "M4A1", 1, cap=100) == (0, "value_exceeded")
    assert reduce_gift_quantity(gd, "未知物品", 1, cap=cap) == (0, "no_price")
    assert format_gift_failure("M4A1") == "你选择了赠送M4A1，因价值过高，赠送失败。"


def test_enhancement_description_boundaries():
    assert enhancement_description(1) == ""
    assert enhancement_description(2) == "经过了一定幅度的强化"
    assert enhancement_description(4) == "经过了一定幅度的强化"
    assert enhancement_description(5) == "经过了较高成本的强化"
    assert enhancement_description(8) == "经过了较高成本的强化"
    assert enhancement_description(9) == "经过了极高成本的强化"


def test_take_completion_text():
    assert format_take_completion("普通hp药剂", 2) == "玩家把 2 个 普通hp药剂 交给了你。"
    assert (
        format_take_completion("M4A1", 1, item_level=2)
        == "玩家把 1 个 M4A1（强化度2，经过了一定幅度的强化）交给了你。"
    )


def test_item_info_block_for_registry_and_pseudo():
    gd = make_game_data(_items())
    text = format_item_info(gd, "普通hp药剂")
    assert text.startswith("名称：普通hp药剂；")
    assert "单价：250" in text
    gold = format_item_info(gd, "金币")
    assert gold.startswith("名称：金币；")
    assert "单价：1" in gold
