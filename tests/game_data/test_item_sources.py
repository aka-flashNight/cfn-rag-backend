"""物品获取途径反查（物品图鉴注释口径）的测试。

口径：合成 / NPC 商店 / K点商店 / 关卡掉落 / 核心任务池（提交·奖励）；
查不到任何来源＝空串（不写「未知」、不留占位）。
"""

from __future__ import annotations

from services.game_data.item_sources import format_item_sources
from services.game_entity_prompts import format_item_prompt_line


def test_crafting_item_reports_craft_source(game_data):
    # 抗生素：合成（化学生产）+ 商店 + K点 + 掉落 + 任务，五个来源都能反查到
    text = format_item_sources(game_data, "抗生素")
    assert text.startswith("合成（")
    assert "商店（" in text
    assert "K点商店" in text
    assert "关卡掉落（" in text
    assert "任务" in text


def test_shop_only_item(game_data):
    text = format_item_sources(game_data, "啤酒")
    assert text.startswith("商店（")
    assert "酒保" in text


def test_stage_drop_item(game_data):
    text = format_item_sources(game_data, "牛肉罐头")
    assert "关卡掉落（" in text


def test_unknown_item_or_missing_registry_yields_empty(game_data):
    assert format_item_sources(game_data, "库里没有的东西") == ""
    assert format_item_sources(None, "牛肉罐头") == ""
    assert format_item_sources(game_data, "") == ""


def test_all_shops_listed_by_default(game_data):
    # 数据侧每件物品最多 4 家在售，默认全列（不再折叠成「等N家」）
    text = format_item_sources(game_data, "杰森面具")
    assert text.count("、") >= 3
    assert "等" not in text


def test_shop_limit_still_supported(game_data):
    text = format_item_sources(game_data, "抗生素", max_shop=1)
    assert "等2家商店" in text


def test_prompt_line_appends_sources_last(game_data):
    item = game_data.items.get_by_name("抗生素")
    line = format_item_prompt_line(
        item, reward_tags=[], price=item.price,
        sources=format_item_sources(game_data, "抗生素"),
    )
    assert line.startswith("名称：抗生素；")
    # 获取途径是最后一段（内部可能自带「；」分隔的多来源）
    assert "；获取途径：" in line
    assert line.endswith("任务提交或奖励")


def test_prompt_line_omits_sources_when_empty(game_data):
    item = game_data.items.get_by_name("抗生素")
    line = format_item_prompt_line(item, reward_tags=[], price=item.price, sources="")
    assert "获取途径" not in line


def test_index_is_cached_on_registry(game_data):
    format_item_sources(game_data, "抗生素")
    first = getattr(game_data, "_item_source_index_cache", None)
    assert isinstance(first, dict) and first
    format_item_sources(game_data, "啤酒")
    assert getattr(game_data, "_item_source_index_cache", None) is first
