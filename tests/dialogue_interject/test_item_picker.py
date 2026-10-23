from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.dialogue_interject.item_picker import ItemPicker
from services.game_data.models import Item


class FakeItems:
    def __init__(self, items):
        self._by_name = {item.name: item for item in items}

    def get_by_name(self, name):
        return self._by_name.get(name)

    def get_price(self, name):
        item = self._by_name.get(name)
        return int(item.price or 0) if item else 0


def _game_data():
    items = [
        Item(name="普通hp药剂", type="消耗品", use="药剂", price=250),
        Item(name="高级治疗药剂", type="消耗品", use="药剂", price=800),
        Item(name="M4A1", type="武器", use="步枪", price=500),
    ]
    return SimpleNamespace(
        items=FakeItems(items), tasks=None, equipment_mods=None,
        crafting=SimpleNamespace(_recipes=[]),
        shops=SimpleNamespace(_shops={}),
        kshop=SimpleNamespace(list_items=lambda: []),
        stages=SimpleNamespace(_stage_infos={}),
    )


@pytest.mark.asyncio
async def test_exact_name_skips_picker():
    called = False

    async def caller(_messages):
        nonlocal called
        called = True
        return '{"action":"none"}'

    picker = ItemPicker(caller=caller)
    name = await picker.pick(
        direction="take", name="M4A1", keyword="", instruction="",
        pool=["金币", "M4A1"], game_data=_game_data(),
    )
    assert name == "M4A1"
    assert called is False


@pytest.mark.asyncio
async def test_keyword_pick_and_prompt_shape():
    seen = []

    async def caller(messages):
        seen.append(messages)
        return '{"action":"pick","name":"普通hp药剂"}'

    picker = ItemPicker(caller=caller)
    name = await picker.pick(
        direction="take", name="", keyword="药剂", instruction="能回血的",
        pool=["金币", "普通hp药剂", "高级治疗药剂"], game_data=_game_data(),
    )
    assert name == "普通hp药剂"
    user = seen[0][1]["content"]
    assert "候选列表" in user
    assert "剩余" in user
    assert "普通hp药剂" in user


@pytest.mark.asyncio
async def test_search_then_pick():
    calls = []

    async def caller(messages):
        calls.append(messages)
        if len(calls) == 1:
            return '{"action":"search","keyword":"治疗"}'
        return '{"action":"pick","name":"高级治疗药剂"}'

    picker = ItemPicker(caller=caller)
    name = await picker.pick(
        direction="take", name="", keyword="回血", instruction="",
        pool=["金币", "普通hp药剂", "高级治疗药剂"], game_data=_game_data(),
    )
    assert name == "高级治疗药剂"
    assert len(calls) == 2
    assert "第二轮" in calls[1][1]["content"]


@pytest.mark.asyncio
async def test_invalid_pick_and_none():
    async def bad(messages):
        return '{"action":"pick","name":"不存在的物品"}'

    picker = ItemPicker(caller=bad)
    assert await picker.pick(
        direction="give", name="", keyword="药剂", instruction="",
        pool=["普通hp药剂"], game_data=_game_data(),
    ) is None

    async def none(messages):
        return '{"action":"none"}'

    picker = ItemPicker(caller=none)
    assert await picker.pick(
        direction="give", name="", keyword="", instruction="随便",
        pool=["普通hp药剂"], game_data=_game_data(),
    ) is None


@pytest.mark.asyncio
async def test_second_round_search_is_rejected():
    calls = []

    async def caller(messages):
        calls.append(messages)
        if len(calls) == 1:
            return '{"action":"search","keyword":"药剂"}'
        return '{"action":"search","keyword":"治疗"}'

    picker = ItemPicker(caller=caller)
    assert await picker.pick(
        direction="take", name="", keyword="回血", instruction="",
        pool=["金币", "普通hp药剂", "高级治疗药剂"], game_data=_game_data(),
    ) is None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_empty_pool_returns_none():
    async def caller(_messages):  # pragma: no cover - 不应被调用
        raise AssertionError("空池不应调用模型")

    picker = ItemPicker(caller=caller)
    assert await picker.pick(
        direction="give", name="", keyword="", instruction="", pool=[], game_data=_game_data(),
    ) is None


@pytest.mark.asyncio
async def test_exact_name_missing_fails_without_random_fallback():
    """玩家指定的物品不在池内：判失败，不调用模型、不改选随机物品。"""
    called = False

    async def caller(_messages):
        nonlocal called
        called = True
        return '{"action":"pick","name":"普通hp药剂"}'

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="牛肉罐头", keyword="", instruction="",
        pool=["普通hp药剂"], game_data=_game_data(),
    )
    assert result.name is None
    assert result.reason == "not_found"
    assert result.target == "牛肉罐头"
    assert called is False


@pytest.mark.asyncio
async def test_exact_name_over_cap_fails_before_picker():
    async def caller(_messages):  # pragma: no cover - 不应被调用
        raise AssertionError("超上限的精确物品不应进入选择模型")

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="高级治疗药剂", keyword="", instruction="",
        pool=["普通hp药剂", "高级治疗药剂"], game_data=_game_data(), cap=500,
    )
    assert result.name is None
    assert result.reason == "value_exceeded"
    assert result.target == "高级治疗药剂"


@pytest.mark.asyncio
async def test_keyword_hit_over_cap_is_annotated_and_can_fail():
    seen = []

    async def caller(messages):
        seen.append(messages)
        return '{"action":"fail","reason":"value_exceeded"}'

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="", keyword="高级", instruction="",
        pool=["普通hp药剂", "高级治疗药剂"], random_pool=["普通hp药剂"],
        game_data=_game_data(), cap=500,
        player_text="给我高级治疗药剂", recent_dialogue="玩家：给我高级治疗药剂",
    )

    user = seen[0][1]["content"]
    assert "高级治疗药剂" in user
    assert "该物品价值超出可赠送范围，不可赠送" in user
    assert "玩家原话：给我高级治疗药剂" in user
    assert "玩家近期对话" in user
    assert result.name is None
    assert result.reason == "value_exceeded"


@pytest.mark.asyncio
async def test_pick_over_cap_candidate_is_rejected():
    async def caller(_messages):
        return '{"action":"pick","name":"高级治疗药剂"}'

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="", keyword="高级", instruction="",
        pool=["普通hp药剂", "高级治疗药剂"], random_pool=["普通hp药剂"],
        game_data=_game_data(), cap=500,
    )
    assert result.name is None
    assert result.reason == "value_exceeded"


@pytest.mark.asyncio
async def test_random_pool_excludes_over_cap_items():
    seen = []

    async def caller(messages):
        seen.append(messages)
        return '{"action":"pick","name":"普通hp药剂"}'

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="", keyword="", instruction="",
        pool=["普通hp药剂", "高级治疗药剂"], random_pool=["普通hp药剂"],
        game_data=_game_data(), cap=500,
    )
    assert result.name == "普通hp药剂"
    assert "高级治疗药剂" not in seen[0][1]["content"]
