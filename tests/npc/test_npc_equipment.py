"""NPC 固定层装备块（units.json ＋ items 解析 ＋ 共用段落标记）。

口径：
- 只列真实存在且能解析的装备；``gender``/``face``/``hairstyle`` 不是装备；
- 名字对不齐（``Andy Law`` ↔ ``Andy``、``Shop Girl`` ↔ ``ShopGirl``、``小F`` ↔ ``BYD 小F``）
  先归一化精确匹配、再退唯一包含匹配；歧义载荷不同＝fail-closed 整块不输出；
- 大段重复介绍（≥20 字符）框成共用段落，纯 difflib、不调模型。

全部夹具都在临时 data_root 里造，不读游戏仓数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.game_data.item_registry import ItemRegistry
from services.npc.npc_equipment import (
    EQUIPMENT_HEADER,
    build_npc_equipment_block,
    mark_shared_paragraphs,
    normalize_name,
)

ITEM_XML = """<items>
  <item name="Andy领子-NPC">
    <displayname>Andy领子-NPC</displayname>
    <description>&lt;FONT COLOR='#FFFFFF'&gt;An dy 的定制领子。&lt;/FONT&gt;</description>
    <data><weight>0</weight></data>
  </item>
  <item name="Andy上装-NPC">
    <displayname>Andy上装-NPC</displayname>
    <data><weight>0.5</weight></data>
  </item>
  <item name="东区最强战士项链">
    <displayname>东区最强战士项链</displayname>
    <description>东区最强战士的荣誉象征。</description>
    <data>
      <weight>0</weight>
      <title>&lt;FONT COLOR='#C80000'&gt;完美士兵&lt;/FONT&gt;</title>
    </data>
  </item>
  <item name="K5项链">
    <displayname>K5项链</displayname>
    <data><weight>abc</weight><title>永恒强者</title></data>
  </item>
  <item name="MK23手枪">
    <displayname>MK23手枪</displayname>
    <data><weight>1</weight></data>
  </item>
  <item name="小F头部">
    <displayname>小F头部</displayname>
    <data><weight>0</weight></data>
  </item>
</items>
"""


def _item(name: str, **kwargs) -> dict:
    xmlns = {
        "name": name,
        "displayname": name,
        "description": kwargs.get("description"),
        "data": {},
    }
    return xmlns


def _units(records: list[dict]) -> list[dict]:
    return records


def _unit(name: str, data: dict, *, npc: bool = True) -> dict:
    return {"name": name, "NPC": npc, "data": data}


@pytest.fixture
def registry(tmp_path: Path) -> ItemRegistry:
    root = tmp_path / "data"
    (root / "items").mkdir(parents=True)
    (root / "items" / "gear.xml").write_text(ITEM_XML, encoding="utf-8")
    reg = ItemRegistry(data_root=root)
    reg.load()
    return reg


class _FakeGameData:
    """只给装备块需要的两样：data_root 与 items。"""

    def __init__(self, root: Path, items: ItemRegistry) -> None:
        self.data_root = root
        self.items = items


def _game_data(registry: ItemRegistry, units: list[dict]) -> _FakeGameData:
    root = registry.data_root
    (root / "units").mkdir(parents=True, exist_ok=True)
    (root / "units" / "units.json").write_text(
        json.dumps(_units(units), ensure_ascii=False), encoding="utf-8",
    )
    return _FakeGameData(root, registry)


def _block(registry: ItemRegistry, units: list[dict], npc_name: str = "Andy") -> str:
    return build_npc_equipment_block(
        npc_name=npc_name, game_data=_game_data(registry, units),
    )


# ---------------------------------------------------------------------------
# items 解析：title
# ---------------------------------------------------------------------------

def test_item_title_is_stripped_of_markup(registry):
    neck = registry.get_by_name("东区最强战士项链")
    assert neck.title == "完美士兵"
    assert registry.get_by_name("K5项链").title == "永恒强者"
    # 没有 title 的物品＝空串（不是 None，也不是名字）
    assert registry.get_by_name("MK23手枪").title == ""


def test_item_description_still_stripped(registry):
    assert registry.get_by_name("Andy领子-NPC").description == "An dy 的定制领子。"


# ---------------------------------------------------------------------------
# 名称匹配
# ---------------------------------------------------------------------------

def test_exact_match_and_case_and_space(registry):
    units = [
        _unit("ShopGirl", {"secondary": "MK23手枪", "neck": "K5项链"}),
        _unit("Andy", {"head": "Andy领子-NPC"}),
    ]
    assert _block(registry, units, "ShopGirl").startswith(EQUIPMENT_HEADER)
    # 大小写 / 空格差异都要能对上（归一化后相等）
    assert _block(registry, units, "shop girl").startswith(EQUIPMENT_HEADER)
    assert _block(registry, units, " ANDY ").startswith(EQUIPMENT_HEADER)


def test_contains_match_for_unaligned_names(registry):
    units = [
        _unit("Andy", {"head": "Andy领子-NPC"}),
        _unit("BYD 小F", {"hand": "小F头部"}),
    ]
    assert "Andy领子-NPC" in _block(registry, units, "Andy Law")
    assert "小F头部" in _block(registry, units, "小F")


def test_unknown_npc_returns_empty(registry):
    units = [_unit("Andy", {"head": "Andy领子-NPC"})]
    assert _block(registry, units, "路人甲") == ""
    # 空名字
    assert _block(registry, units, "") == ""


def test_non_npc_units_are_ignored(registry):
    units = [_unit("Andy", {"head": "Andy领子-NPC"}, npc=False)]
    assert _block(registry, units, "Andy") == ""


def test_only_non_equipment_fields_returns_empty(registry):
    units = [_unit("Andy", {"gender": "男", "face": "f", "hairstyle": "h"})]
    assert _block(registry, units, "Andy") == ""
    # data 整个是空对象（吉他/键盘等 11 个 NPC 的现状）
    assert _block(registry, [_unit("吉他", {})], "吉他") == ""


def test_duplicate_records_with_same_payload_hit(registry):
    """Pig/Boy/King 各重复两条：载荷相同就取任意一条。"""
    units = [
        _unit("Pig", {"head": "Andy领子-NPC"}),
        _unit("Pig", {"head": "Andy领子-NPC"}),
    ]
    assert "Andy领子-NPC" in _block(registry, units, "Pig")


def test_duplicate_records_with_different_payload_fail_closed(registry):
    """载荷不同＝无法确定唯一候选，整块不输出（宁缺勿错）。"""
    units = [
        _unit("King", {"head": "Andy领子-NPC"}),
        _unit("King", {"neck": "K5项链"}),
    ]
    assert _block(registry, units, "King") == ""


# ---------------------------------------------------------------------------
# 字段映射与展示格式
# ---------------------------------------------------------------------------

def test_field_mapping_and_order(registry):
    units = [_unit("Andy", {
        "primary": "MK23手枪", "secondary": "MK23手枪", "secondary2": "MK23手枪",
        "melee": "MK23手枪", "grenade": "MK23手枪",
        "head": "Andy领子-NPC", "body": "Andy上装-NPC",
        "leg": "Andy上装-NPC", "hand": "Andy领子-NPC", "foot": "Andy上装-NPC",
        "neck": "东区最强战士项链",
    })]
    block = _block(registry, units, "Andy")
    labels = [line.split("：", 1)[0] for line in block.splitlines()[1:]]
    assert labels == [
        "头部装备", "上装装备", "下装装备", "手部装备", "脚部装备", "颈部装备",
        "长枪", "手枪", "副手枪", "刀", "手雷",
    ]


def test_weight_and_title_rendering(registry):
    units = [_unit("Andy", {
        "head": "Andy领子-NPC",      # weight 0 → 0kg
        "body": "Andy上装-NPC",      # weight 0.5 → 0.5kg
        "neck": "K5项链",            # weight abc → 省略重量；title=永恒强者
    })]
    block = _block(registry, units, "Andy")
    assert "头部装备：Andy领子-NPC，重量0kg，介绍：An dy 的定制领子；" in block
    assert "上装装备：Andy上装-NPC，重量0.5kg；" in block
    assert "颈部装备：K5项链；称号：永恒强者；" in block


def test_neck_title_is_rendered(registry):
    units = [_unit("Andy", {"neck": "东区最强战士项链"})]
    assert "颈部装备：东区最强战士项链，重量0kg，介绍：东区最强战士的荣誉象征；称号：完美士兵；" in _block(
        registry, units, "Andy",
    )


def test_partial_null_fields_only_list_existing(registry):
    units = [_unit("Andy", {
        "primary": "黄金QBZ", "secondary": None, "melee": None,
        "head": "Andy领子-NPC", "neck": None,
    })]
    block = _block(registry, units, "Andy")
    # 解析不到的物品（黄金QBZ 不在物品表里）与 null 字段都跳过，只留真实存在的那件
    assert block == "你身上的装备是：\n头部装备：Andy领子-NPC，重量0kg，介绍：An dy 的定制领子；"


def test_missing_items_skipped_and_all_missing_empty(registry):
    units = [_unit("Andy", {"head": "不存在的物品", "body": "Andy上装-NPC"})]
    block = _block(registry, units, "Andy")
    assert "Andy上装-NPC" in block
    assert "不存在的物品" not in block

    units = [_unit("Andy", {"head": "不存在的物品", "body": "也不存在"})]
    assert _block(registry, units, "Andy") == ""


def test_missing_units_file_or_game_data_returns_empty(registry, tmp_path):
    class _NoUnits:
        data_root = tmp_path / "nowhere"
        items = registry

    assert build_npc_equipment_block(npc_name="Andy", game_data=_NoUnits()) == ""

    class _NoDataRoot:
        items = registry

    assert build_npc_equipment_block(npc_name="Andy", game_data=_NoDataRoot()) == ""
    assert build_npc_equipment_block(npc_name="Andy", game_data=None) == ""


# ---------------------------------------------------------------------------
# 共用段落
# ---------------------------------------------------------------------------

_long = "这套装备是东区军械库为最强战士专门打造的，材质与工艺都极其罕见。"


def test_identical_descriptions_get_shared_paragraph():
    a = _long
    assert len(a) >= 20
    marked = mark_shared_paragraphs([a, a])
    assert marked[0] == f"<共用段落1开始>{a}<共用段落1结束>"
    assert marked[1] == "<此处与共用段落1相同>"


def test_three_items_repeat_same_paragraph_number():
    first = f"甲{_long}甲"
    second = f"乙{_long}乙"
    third = f"丙{_long}丙"
    marked = mark_shared_paragraphs([first, second, third])
    assert f"<共用段落1开始>{_long}<共用段落1结束>" in marked[0]
    assert marked[1].count("<此处与共用段落1相同>") == 1
    assert marked[2].count("<此处与共用段落1相同>") == 1
    # 不该冒出第二个编号
    assert "共用段落2" not in "".join(marked)


def test_short_repeat_is_not_marked():
    short = "东区军械库出品"
    assert len(short) < 20
    assert mark_shared_paragraphs([f"甲{short}", f"乙{short}"]) == [f"甲{short}", f"乙{short}"]


def test_same_description_internal_repeat_untouched():
    inside = f"{_long}中间{_long}"
    marked = mark_shared_paragraphs([inside, f"别的东西{_long}"])
    # 第一条内部同内容出现两次：不替换（只在后续物品里替换）
    assert marked[0].startswith(f"<共用段落1开始>{_long}<共用段落1结束>")
    assert marked[1].count("<此处与共用段落1相同>") == 1


def test_trailing_punctuation_normalized_before_semicolon(registry):
    units = [_unit("Andy", {"head": "Andy领子-NPC"})]
    block = _block(registry, units, "Andy")
    # 介绍正文自带的句号被去掉后统一补「；」，不会出现「。。；」或「。；」
    assert "。。；" not in block
    assert "。；" not in block
    assert "定制领子；" in block


# ---------------------------------------------------------------------------
# 装配位置（system 层2：appearance 之后、任务发布硬约束之前）
# ---------------------------------------------------------------------------

def test_block_sits_after_appearance_before_publish_constraint():
    from services.npc.manager import NPCState
    from services.orchestrator.prompts import build_static_system

    state = NPCState(
        favorability=10, relationship_level="陌生",
        emotions=["普通"], faction="A兵团", titles=["铁匠"],
        appearance="一个满脸胡茬的壮汉。",
    )
    system = build_static_system(
        npc_name="铁匠", state=state, equipment_block="你身上的装备是：\n头部装备：斗笠，重量0kg；",
    )
    assert "【你的形象】" in system
    assert EQUIPMENT_HEADER in system
    assert system.index("【你的形象】") < system.index(EQUIPMENT_HEADER)
    assert system.index(EQUIPMENT_HEADER) < system.index("【任务发布硬约束")


def test_empty_equipment_block_leaves_prompt_unchanged():
    from services.npc.manager import NPCState
    from services.orchestrator.prompts import build_static_system

    state = NPCState(favorability=10, emotions=["普通"], faction="A兵团", titles=["铁匠"])
    without = build_static_system(npc_name="铁匠", state=state)
    with_empty = build_static_system(npc_name="铁匠", state=state, equipment_block="   ")
    assert without == with_empty
    assert EQUIPMENT_HEADER not in without


def test_normalize_name_drops_spaces_and_case():
    assert normalize_name("  Shop\u3000Girl ") == "shopgirl"
    assert normalize_name("BYD 小F") == "byd小f"
