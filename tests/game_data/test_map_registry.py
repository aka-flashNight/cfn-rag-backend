"""P1 NPC→场景注册表单测（覆盖 §8.2 实测缺口：多驻地 / 大小写对 / 别名 / 无驻地 / 范围外 NPC）。"""

from __future__ import annotations

from services.game_data.models import MapScene


def _pairs(scenes: list[MapScene]) -> list[tuple[str, str]]:
    return [(s.label, s.linkage) for s in scenes]


def test_single_location_scene(game_data):
    scenes = game_data.map_registry.scenes_for_npc("冷兵器商人")
    assert _pairs(scenes) == [("基地车库", "基地场景-基地车库")]


def test_multi_location_npcs_report_all_scenes(game_data):
    assert _pairs(game_data.map_registry.scenes_for_npc("前治安官")) == [
        ("基地大厅", "基地场景-基地1层"),
        ("第一防线", "地图-第一防线防区"),
    ]
    assert _pairs(game_data.map_registry.scenes_for_npc("武器大师")) == [
        ("健身房", "基地场景-健身房"),
        ("第一防线", "地图-第一防线防区"),
    ]
    assert _pairs(game_data.map_registry.scenes_for_npc("人修罗")) == [
        ("幸存者营地", "地图-幸存者营地"),
        ("彩蛋地图", "地图-彩蛋地图"),
    ]


def test_db_only_npcs_have_no_location(game_data):
    """17 名无驻地 NPC 不拼场景块：注册表查无驻地返回空列表。"""
    for name in ["lochan", "暮光闪闪", "歌留多", "BLACK-L", "枫", "王道", "LSY大人",
                 "PD421", "和平", "乔恩", "劳埃德", "清水结衣", "千夏沐晓", "云吞魔芋",
                 "PROPHET2", "短波无线电", "$PC_CHAR"]:
        assert game_data.map_registry.scenes_for_npc(name) == [], name


def test_case_insensitive_name_pairs(game_data):
    for lower, upper in [("guitar", "Guitar"), ("singer", "Singer"),
                         ("keyboard", "Keyboard"), ("itinerant", "Itinerant")]:
        assert (
            _pairs(game_data.map_registry.scenes_for_npc(lower))
            == _pairs(game_data.map_registry.scenes_for_npc(upper))
            != []
        ), lower


def test_alias_resolves_to_same_scene(game_data):
    direct = _pairs(game_data.map_registry.scenes_for_npc("杀马特"))
    via_alias = _pairs(game_data.map_registry.scenes_for_npc("∞天ㄙ★使的剪∞"))
    assert direct == via_alias == [("基地大厅", "基地场景-基地1层")]


def test_scene_list_includes_npcs_outside_state_db(game_data):
    """名单来源＝地图文件：含 npc_state_db 范围外的 NPC，按 placements 文件顺序。"""
    labels = [n.label for n in game_data.map_registry.npcs_in_scene("base_garage")]
    assert labels == ["Pig", "Boy", "King", "冷兵器商人"]
    # 武器大师无 db 记录但要在名单里
    gym = [n.label for n in game_data.map_registry.npcs_in_scene("gym")]
    assert gym == ["武器大师"]
