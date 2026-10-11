"""场景块与场景补充描述单测（B8-1 文本逐字；覆盖单/多驻地、无同场景、无驻地、补充去重）。"""

from __future__ import annotations

from services.game_data.models import MapScene
from services.npc.manager import NPCState
from services.npc.scene_context import (
    build_scene_block,
    build_scene_supplement,
    npc_scene_text,
    scene_display,
)


def _states(**items) -> dict[str, NPCState]:
    return {name: state for name, state in items.items()}


def test_scene_display_falls_back_when_linkage_missing():
    assert scene_display(MapScene(location_id="x", label="食堂", linkage="地图-食堂")) == "食堂（地图-食堂）"
    assert scene_display(MapScene(location_id="x", label="食堂", linkage="")) == "食堂"


def test_single_location_block_verbatim(game_data):
    states = _states(
        Pig=NPCState(sex="男", faction="A兵团元老", titles=["野猪"]),
        Boy=NPCState(sex="男", faction="A兵团元老"),
        King=NPCState(sex="女", faction="A兵团元老"),
    )
    block = build_scene_block(
        map_reg=game_data.map_registry, npc_states=states, npc_name="冷兵器商人",
    )
    assert block == (
        "你所在场景：基地车库（基地场景-基地车库）。\n"
        "同场景npc（均在你视线范围内）：Pig，Boy，King。"
    )


def test_single_location_without_others_uses_fixed_line(game_data):
    block = build_scene_block(
        map_reg=game_data.map_registry, npc_states={}, npc_name="厨师",
    )
    assert block == "你所在场景：食堂（地图-食堂）。\n同场景无其他npc。"


def test_multi_location_block_verbatim(game_data):
    block = build_scene_block(
        map_reg=game_data.map_registry, npc_states={}, npc_name="前治安官",
    )
    assert block == (
        "你所在场景（以下多个场景为你常去的场景，你可能在任意其一）：\n"
        "基地大厅（基地场景-基地1层）：同场景npc（均在你视线范围内）："
        "杀马特，宝石线人，黑铁会外交部长，学生妹，幸存老兵，迷之盔甲君，爱国青年。\n"
        "第一防线（地图-第一防线防区）：同场景npc（均在你视线范围内）：artist，soldier，武器大师。"
    )


def test_multi_location_empty_scene_still_annotated(game_data):
    block = build_scene_block(
        map_reg=game_data.map_registry, npc_states={}, npc_name="武器大师",
    )
    assert block == (
        "你所在场景（以下多个场景为你常去的场景，你可能在任意其一）：\n"
        "健身房（基地场景-健身房）：同场景无其他npc。\n"
        "第一防线（地图-第一防线防区）：同场景npc（均在你视线范围内）：artist，soldier，前治安官。"
    )


def test_no_location_npc_gets_no_block_or_supplement(game_data):
    assert build_scene_block(map_reg=game_data.map_registry, npc_states={}, npc_name="lochan") == ""
    assert build_scene_supplement(
        map_reg=game_data.map_registry, npc_states={}, npc_name="lochan", exclude_names=set(),
    ) == ""


def test_db_name_preferred_over_map_label(game_data):
    """跨库名字对齐：db 记的是别名时，场景名单用 db 名（同阵营块同款名字）。"""
    states = _states(**{"∞天ㄙ★使的剪∞": NPCState(sex="男", faction="闲杂人等")})
    block = build_scene_block(
        map_reg=game_data.map_registry, npc_states=states, npc_name="宝石线人",
    )
    assert "∞天ㄙ★使的剪∞" in block
    assert "杀马特" not in block


def test_supplement_lines_and_exclusion(game_data):
    states = _states(
        Pig=NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-右前锋", "野猪"]),
        Boy=NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-队长"]),
        King=NPCState(sex="女", faction="A兵团元老", titles=["BPK战队-左前锋"]),
    )
    text = build_scene_supplement(
        map_reg=game_data.map_registry, npc_states=states,
        npc_name="冷兵器商人", exclude_names=set(),
    )
    assert text == (
        "「Pig」（性别：男）（阵营：A兵团元老）（身份或称呼：BPK战队-右前锋、野猪）\n"
        "「Boy」（性别：男）（阵营：A兵团元老）（身份或称呼：BPK战队-队长）\n"
        "「King」（性别：女）（阵营：A兵团元老）（身份或称呼：BPK战队-左前锋）"
    )
    # 已列过者不重复
    text2 = build_scene_supplement(
        map_reg=game_data.map_registry, npc_states=states,
        npc_name="冷兵器商人", exclude_names={"Pig", "Boy"},
    )
    assert text2 == "「King」（性别：女）（阵营：A兵团元老）（身份或称呼：BPK战队-左前锋）"


def test_supplement_skips_self_and_map_only_npcs(game_data):
    """自己不出现在补充里；地图有、db 无记录的 NPC 不补（无法给出描述）。"""
    states = _states(**{
        "冷兵器商人": NPCState(faction="商人"),
        "Pig": NPCState(sex="男"),
    })
    text = build_scene_supplement(
        map_reg=game_data.map_registry, npc_states=states,
        npc_name="冷兵器商人", exclude_names=set(),
    )
    assert text == "「Pig」（性别：男）"

    # publisher=artist：第一防线同场景有 前治安官（db 有记录）与 武器大师（地图有、db 无记录）
    states2 = _states(**{"前治安官": NPCState(sex="男", faction="闲杂人等")})
    text2 = build_scene_supplement(
        map_reg=game_data.map_registry, npc_states=states2,
        npc_name="artist", exclude_names=set(),
    )
    assert "「前治安官」（性别：男）（阵营：闲杂人等）" in text2
    assert "武器大师" not in text2


def test_npc_scene_text_joins_multi_location(game_data):
    assert npc_scene_text(game_data.map_registry, "冷兵器商人") == "基地车库（基地场景-基地车库）"
    assert npc_scene_text(game_data.map_registry, "前治安官") == (
        "基地大厅（基地场景-基地1层）、第一防线（地图-第一防线防区）"
    )
    assert npc_scene_text(game_data.map_registry, "lochan") == ""
