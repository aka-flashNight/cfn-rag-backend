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


def _gift_game_data():
    """赠送方向用的 fake：索引里含池外的三个罐头。"""
    from services.game_data.models import Item as _Item

    items = [
        _Item(name="普通hp药剂", type="消耗品", use="药剂", price=250),
        _Item(name="牛肉罐头", type="消耗品", use="食品", price=300),
        _Item(name="菠萝罐头", type="消耗品", use="食品", price=4000),
        _Item(name="白米饭罐头", type="消耗品", use="食品", price=1000),
    ]
    gd = SimpleNamespace(
        items=FakeItems(items),
        tasks=SimpleNamespace(
            list_core_reward_item_names=lambda: {"普通hp药剂"},
            list_core_submit_items=lambda: set(),
        ),
        equipment_mods=None,
        crafting=SimpleNamespace(_recipes=[]),
        shops=SimpleNamespace(_shops={}),
        kshop=SimpleNamespace(list_items=lambda: []),
        stages=SimpleNamespace(_stage_infos={}),
    )
    gd._item_source_index_cache = {
        name: {} for name in (
            "普通hp药剂", "牛肉罐头", "菠萝罐头", "白米饭罐头",
        )
    }
    return gd


_OUT_OF_POOL_NOTE = (
    "注释：查询到以下可能相关的物品，但不在你拥有的物品池范围中，不可选择："
    "牛肉罐头,菠萝罐头,白米饭罐头。如果这些物品不相关，无视即可。"
    "如果确认是这些不可选择的物品，且可选候选池内无相关物品，"
    "请返回fail并给出out_of_pool这个reason。"
)


@pytest.mark.asyncio
async def test_exact_name_out_of_pool_returns_known_names():
    called = False

    async def caller(_messages):
        nonlocal called
        called = True
        return '{"action":"none"}'

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="牛肉罐头", keyword="", instruction="",
        pool=["普通hp药剂"], game_data=_gift_game_data(),
    )
    assert called is False
    assert result.name is None
    assert result.reason == "not_found"
    assert result.known_names == ("牛肉罐头",)


@pytest.mark.asyncio
async def test_exact_name_truly_missing_has_no_known_names():
    async def caller(_messages):
        return '{"action":"none"}'

    result = await ItemPicker(caller=caller).pick_result(
        direction="give", name="南瓜罐头", keyword="", instruction="",
        pool=["普通hp药剂"], game_data=_gift_game_data(),
    )
    assert result.known_names == ()


@pytest.mark.asyncio
async def test_keyword_out_of_pool_note_placed_below_candidates():
    seen = []

    async def caller(messages):
        seen.append(messages)
        return '{"action":"fail","reason":"out_of_pool"}'

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="", keyword="罐头",
        instruction="给玩家一点吃的", pool=["普通hp药剂"],
        game_data=_gift_game_data(), player_text="能不能给我个罐头",
    )
    assert result.reason == "out_of_pool"
    assert result.known_names == ("牛肉罐头", "菠萝罐头", "白米饭罐头")
    assert len(seen) == 1
    user = seen[0][1]["content"]
    assert _OUT_OF_POOL_NOTE in user
    assert user.index("候选列表（") < user.index(_OUT_OF_POOL_NOTE) < user.index("玩家原话：")
    assert "本轮指示：给玩家一点吃的" in user
    assert "玩家指示：" not in user
    assert "请返回 pick / fail / search。" in user


@pytest.mark.asyncio
async def test_mixed_pool_hits_still_appends_note():
    """池内命中 + 池外也有：仍进子 Agent，注释照样拼。"""
    seen = []

    async def caller(messages):
        seen.append(messages)
        return '{"action":"pick","name":"普通hp药剂"}'

    picker = ItemPicker(caller=caller)
    gd = _gift_game_data()
    gd._item_source_index_cache["大HP药剂"] = {}
    gd.items._by_name["大HP药剂"] = Item(
        name="大HP药剂", type="消耗品", use="药剂", price=800,
    )
    result = await picker.pick_result(
        direction="give", name="", keyword="药剂",
        instruction="给玩家一点药剂", pool=["普通hp药剂"], game_data=gd,
    )
    assert result.name == "普通hp药剂"
    assert seen and "注释：" in seen[0][1]["content"]
    assert "池内匹配 1 件" in seen[0][1]["content"]


@pytest.mark.asyncio
async def test_second_round_search_rechecks_pool_outside_with_new_keyword():
    """二轮 search 无池内结果：用新关键词重查池外，名字仍带回。"""
    calls = []

    async def caller(messages):
        calls.append(messages)
        return '{"action":"search","keyword":"罐头"}'

    picker = ItemPicker(caller=caller)
    result = await picker.pick_result(
        direction="give", name="", keyword="回血", instruction="",
        pool=["普通hp药剂"], game_data=_gift_game_data(),
        player_text="能不能给我个罐头",
    )
    assert len(calls) == 1  # 二轮无池内命中，不再调子 Agent
    assert result.reason == "none"
    assert result.known_names == ("牛肉罐头", "菠萝罐头", "白米饭罐头")


@pytest.mark.asyncio
async def test_second_round_messages_also_carry_note():
    """二轮 messages 同样拼注释；且用二轮关键词查池外。"""
    calls = []

    async def caller(messages):
        calls.append(messages)
        if len(calls) == 1:
            return '{"action":"search","keyword":"药剂"}'
        return '{"action":"fail","reason":"out_of_pool"}'

    picker = ItemPicker(caller=caller)
    gd = _gift_game_data()
    # 二轮关键词「药剂」在池外也有合法物品，才会拼注释。
    gd._item_source_index_cache["大HP药剂"] = {}
    gd.items._by_name["大HP药剂"] = Item(
        name="大HP药剂", type="消耗品", use="药剂", price=800,
    )
    result = await picker.pick_result(
        direction="give", name="", keyword="回血", instruction="",
        pool=["普通hp药剂"], game_data=gd,
    )
    assert len(calls) == 2
    second = calls[1][1]["content"]
    assert "注释：" in second and "大HP药剂" in second
    assert "这是第二轮，只能返回 pick 或 fail，不能再 search。" in second
    assert result.reason == "out_of_pool"
