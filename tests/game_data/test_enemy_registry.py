"""P2 关卡→怪物注册表单测（覆盖 §8.2 实测缺口：无模板 spritename 跳过 / 数量并列 / 无系数不参与强度比较）。"""

from __future__ import annotations

from services.game_entity_prompts import format_stage_monster_block


def test_unit_type_maps_to_spritename(game_data):
    assert game_data.enemy_registry.spritename_for_unit_type("兵种61") == "敌人-小日本小兵"
    assert game_data.enemy_registry.spritename_for_unit_type("兵种999") is None


def test_stage_spawns_skip_blank_type(game_data):
    spawns = game_data.stages.get_enemy_spawns("基地门口", "废墟武器店")
    assert spawns
    assert all(s.unit_type for s in spawns)


def test_stage_aggregation_sums_quantity_and_strength(game_data):
    """一级防线：同兵种多次出现按 displayname 聚合、Quantity 求和；强度＝阶段×档次系数。"""
    spawns = game_data.stages.get_enemy_spawns("亡灵沙漠", "一级防线")
    monsters = game_data.enemy_registry.monsters_for_spawns(spawns)
    assert [(m.displayname, m.quantity, m.strength) for m in monsters] == [
        ("小日本杂兵", 42, 0.3),
        ("小日本特种兵", 26, 0.3),
        ("小日本少佐", 34, 0.4),
    ]


def test_stage_without_xml_returns_empty(game_data):
    assert game_data.stages.get_enemy_spawns("基地门口", "不存在的关卡") == []


def test_no_template_spritename_skipped(game_data):
    """兵种293（敌人-柜员女僵尸）被关卡引用但无模板（无 displayname）→ 不进名单。"""
    spawns = game_data.stages.get_enemy_spawns("基地门口", "地铁站")
    assert any(s.unit_type == "兵种293" for s in spawns)
    monsters = game_data.enemy_registry.monsters_for_spawns(spawns)
    names = [m.displayname for m in monsters]
    assert "柜员女僵尸" not in names
    assert names[0] == "黄衣僵尸"


def test_block_verbatim_text_for_stage(game_data):
    spawns = game_data.stages.get_enemy_spawns("基地门口", "地铁站")
    block = format_stage_monster_block(game_data.enemy_registry.monsters_for_spawns(spawns))
    assert block == (
        "该关卡有以下怪物：黄衣僵尸，金发女僵尸，红衣僵尸，小儿麻痹僵尸，胖僵尸，"
        "眼镜僵尸，光头军人僵尸，狂暴僵尸，防暴僵尸，特战僵尸，狂躁僵尸，"
        "裸体兽化僵尸，兽化僵尸，巨型僵尸。"
        "其中数量最多的是兽化僵尸，大致数量约16个。强度最高的是巨型僵尸。"
        "根据你的身份，你可能知道也可能不知道这个关卡的详情，但绝对不要编造信息，"
        "要么表现的不清楚，要么根据实际信息去说话，不能编造未提供的信息。"
    )


def test_block_lists_quantity_ties_with_approx_count(game_data):
    """数量并列全列，并附「大致数量约X个」（X＝求和值）。"""
    spawns = game_data.stages.get_enemy_spawns("基地车库", "堕落城深处")
    block = format_stage_monster_block(game_data.enemy_registry.monsters_for_spawns(spawns))
    assert "其中数量最多的是跳蚤、铁拳，大致数量约8个。" in block


def test_block_keeps_ungraded_monsters_but_excludes_from_strength(game_data):
    """无阶段×档次系数的怪物（魔女）仍进名单，只是不参与强度比较。"""
    spawns = game_data.stages.get_enemy_spawns("基地车库", "堕落城保卫战")
    monsters = game_data.enemy_registry.monsters_for_spawns(spawns)
    assert any(m.displayname == "魔女" and m.strength is None for m in monsters)
    block = format_stage_monster_block(monsters)
    assert "魔女" in block
    assert "强度最高的是魔女" not in block


def test_block_empty_when_no_monsters():
    assert format_stage_monster_block([]) == ""
