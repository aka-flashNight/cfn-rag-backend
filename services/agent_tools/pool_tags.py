"""候选池标签：关卡/物品/NPC 三类池子的关键词表、势力归属与敌对关系。

设计口径（用户 2026-10-17 定稿）：
- **关键词是闭集标签**，每个标签一句说明，随提示词一起给模型；
- 每个阵营**只注册它真有的标签**（军阀没有「周边地区」、没有本方区域的势力两个都不注册），
  这样「军阀让玩家打军阀」在结构上不可能发生；
- 势力归属判定**优先看关卡名**（摇滚/黑铁/下水道/盗贼），判不出来才用**区域表**（轻打表）；
- 敌对关系是「势力 × 势力」三态：敌对 / 可打但名义不敌对 / 友军。

本模块只放静态表与纯函数，不碰注册表、不产生 IO。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# 阵营
# ---------------------------------------------------------------------------

FACTION_A_ARMY = "A兵团"
FACTION_A_VETERAN = "A兵团元老"
FACTION_MEMBERS = "成员"
FACTION_WARLORD = "军阀"
FACTION_ZOMBIE = "僵尸"
FACTION_BANDIT = "盗贼"
FACTION_BLACKIRON = "黑铁会"
FACTION_UNIVERSITY = "联合大学"
FACTION_ROCK = "摇滚公园"
FACTION_NOAH = "诺亚"
FACTION_SKYNET = "天网"
NO_FACTION = "无主"

#: NPC 状态库里的阵营名（不含彩蛋——彩蛋单位不参与任务候选）
NPC_FACTIONS: tuple[str, ...] = (
    FACTION_A_ARMY,
    FACTION_A_VETERAN,
    FACTION_MEMBERS,
    FACTION_WARLORD,
    FACTION_BLACKIRON,
    FACTION_UNIVERSITY,
    FACTION_ROCK,
    "禁区人员",
    "闲杂人等",
    "商人",
    "探索者",
    "通缉犯",
)

#: 友军簇（簇内互不攻击）
FRIENDLY_CLUSTERS: tuple[frozenset[str], ...] = (
    frozenset({FACTION_A_ARMY, FACTION_A_VETERAN, FACTION_MEMBERS}),
)

#: 全部势力（不含无主）
ALL_FACTIONS: frozenset[str] = frozenset({
    FACTION_A_ARMY, FACTION_A_VETERAN, FACTION_MEMBERS, FACTION_WARLORD,
    FACTION_ZOMBIE, FACTION_BANDIT, FACTION_BLACKIRON, FACTION_UNIVERSITY,
    FACTION_ROCK, FACTION_NOAH, FACTION_SKYNET,
})

#: 「可打但名义不敌对」的单向关系：发布者 -> 目标
TARGETABLE: dict[str, frozenset[str]] = {
    FACTION_BLACKIRON: frozenset({FACTION_BANDIT, FACTION_ROCK}),
}

#: 显式敌对关系（不对称也可以写，查询时取并集的对称闭包）。
#: 默认**不敌对**——没写进来的势力之间不会互相派单，避免凭空冒出「打自己人」的委托。
HOSTILE: dict[str, frozenset[str]] = {
    # 与所有人敌对
    FACTION_WARLORD: ALL_FACTIONS - {FACTION_WARLORD},
    FACTION_ZOMBIE: ALL_FACTIONS - {FACTION_ZOMBIE},
    FACTION_SKYNET: ALL_FACTIONS - {FACTION_SKYNET},
    FACTION_NOAH: ALL_FACTIONS - {FACTION_NOAH},
    # 盗贼：除黑铁会外都敌对
    FACTION_BANDIT: ALL_FACTIONS - {FACTION_BANDIT, FACTION_BLACKIRON},
    # 堕落城内部互相斗争
    FACTION_ROCK: frozenset({FACTION_UNIVERSITY, FACTION_BANDIT, FACTION_WARLORD,
                             FACTION_ZOMBIE, FACTION_NOAH, FACTION_SKYNET}),
    FACTION_UNIVERSITY: frozenset({FACTION_ROCK, FACTION_BANDIT, FACTION_WARLORD,
                                   FACTION_ZOMBIE, FACTION_NOAH, FACTION_SKYNET}),
    # 黑铁会：打军阀/僵尸/诺亚/天网；盗贼与摇滚走 TARGETABLE（名义不敌对）
    FACTION_BLACKIRON: frozenset({FACTION_WARLORD, FACTION_ZOMBIE, FACTION_NOAH, FACTION_SKYNET}),
    # A 兵团系
    FACTION_A_ARMY: frozenset({FACTION_WARLORD, FACTION_ZOMBIE, FACTION_BANDIT,
                               FACTION_NOAH, FACTION_SKYNET}),
    FACTION_A_VETERAN: frozenset({FACTION_WARLORD, FACTION_ZOMBIE, FACTION_BANDIT,
                                  FACTION_NOAH, FACTION_SKYNET}),
    FACTION_MEMBERS: frozenset({FACTION_WARLORD, FACTION_ZOMBIE, FACTION_BANDIT,
                                FACTION_NOAH, FACTION_SKYNET}),
}


def _in_same_cluster(a: str, b: str) -> bool:
    return any(a in c and b in c for c in FRIENDLY_CLUSTERS)


def is_friendly(a: str, b: str) -> bool:
    """a 与 b 是否友军（同势力也算）。"""
    if not a or not b:
        return False
    if a == b:
        return True
    return _in_same_cluster(a, b)


def is_hostile(attacker: str, target: str) -> bool:
    """attacker 是否敌视 target（可派任务去打），敌对关系取对称闭包。"""
    if not attacker or not target or attacker == target:
        return False
    if target == NO_FACTION:
        return False
    if is_friendly(attacker, target):
        return False
    return target in HOSTILE.get(attacker, frozenset()) or attacker in HOSTILE.get(target, frozenset())


def can_target(attacker: str, target: str) -> bool:
    """attacker 能不能打 target：敌对 或 名义不敌对的追责对象。"""
    if is_hostile(attacker, target):
        return True
    return target in TARGETABLE.get(attacker, frozenset())


# ---------------------------------------------------------------------------
# 区域 -> 势力（轻打表；关卡名优先，判不出来才回落这里）
# ---------------------------------------------------------------------------

AREA_DEFAULT_FACTION: dict[str, str] = {
    "基地门口": FACTION_ZOMBIE,
    "基地车库": FACTION_BANDIT,          # 混战区，名字规则优先，兜底算盗贼
    "基地房顶": FACTION_WARLORD,
    "黑铁会总部": FACTION_BLACKIRON,
    "地下2层": FACTION_SKYNET,
    "诺亚前线基地深处": FACTION_NOAH,
    "雪山": FACTION_NOAH,
    "试炼场深处": NO_FACTION,
    "副本任务": NO_FACTION,              # 跨区，不标势力
}

#: 混战区第二势力（给候选行的 enemies 补第二个标签）
AREA_SECOND_FACTION: dict[str, str] = {
    "基地车库": FACTION_ROCK,
    "地下2层": FACTION_ZOMBIE,
}

#: 关卡名关键词 -> 势力（按顺序匹配，先中先得）
STAGE_NAME_FACTION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("摇滚",), FACTION_ROCK),
    (("黑铁",), FACTION_BLACKIRON),
    (("下水道",), FACTION_ZOMBIE),
    (("盗", "贼", "寇"), FACTION_BANDIT),
    (("军阀",), FACTION_WARLORD),
    (("诺亚",), FACTION_NOAH),
    (("僵尸",), FACTION_ZOMBIE),
    (("终结者",), FACTION_SKYNET),
    (("大学",), FACTION_UNIVERSITY),
)

#: 势力 -> 本方区域（决定「周边地区」标签是否注册）
FACTION_HOME_AREA: dict[str, str] = {
    FACTION_A_ARMY: "基地门口",
    FACTION_A_VETERAN: "基地门口",
    FACTION_MEMBERS: "基地门口",
    FACTION_BLACKIRON: "基地车库",       # 自家总堂在黑铁会总部，不在车库
    FACTION_UNIVERSITY: "基地车库",
    FACTION_ROCK: "基地车库",
    FACTION_BANDIT: "基地车库",
    FACTION_WARLORD: "基地房顶",
    "禁区人员": "地下2层",
}

#: 区域别名（自由词兜底：玩家说的地名不是关卡区域名时也能命中该区关卡）
AREA_ALIASES: dict[str, str] = {
    "堕落城": "基地车库",
    "大学城": "基地车库",
    "废城": "基地门口",
    "荒漠": "基地房顶",
    "禁区": "地下2层",
    "诺亚外围": "地下2层",
    "诺亚前线": "诺亚前线基地深处",
}

#: 外交地图（中立区域，不是可攻打关卡）：StageInfo.type == 该值
DIPLOMATIC_STAGE_TYPE = "外交地图"


def stage_faction(area: str, stage_name: str) -> str:
    """关卡的势力归属：先看关卡名，再看区域默认。"""
    name = stage_name or ""
    for words, faction in STAGE_NAME_FACTION_RULES:
        if any(w in name for w in words):
            return faction
    return AREA_DEFAULT_FACTION.get(area or "", NO_FACTION)


def stage_enemy_tags(area: str, stage_name: str, *, limit: int = 2) -> list[str]:
    """候选行的敌人标签（≤limit 个）：主势力 + 混战区的第二势力。"""
    primary = stage_faction(area, stage_name)
    out: list[str] = [] if primary == NO_FACTION else [primary]
    second = AREA_SECOND_FACTION.get(area or "")
    if second and second not in out and len(out) < limit:
        out.append(second)
    return out[:limit]


# ---------------------------------------------------------------------------
# 关键词标签表（每个标签一句说明；顺序即提示词里的展示顺序）
# ---------------------------------------------------------------------------

STAGE_TAGS: tuple[tuple[str, str], ...] = (
    ("势均力敌", "和玩家当前进度相当的关卡，打起来有来有回。"),
    ("轻松应对", "不属于势均力敌的关卡都能算轻松应对，玩家应付得来。"),
    ("周边地区", "发布者势力附近、可以派玩家去打的关卡，不含发布者自己的本部。"),
    ("敌对区域", "跟发布者敌对的势力盘踞的关卡区域。"),
    ("副本", "副本任务里的关卡，多数只开「简单」，少数多一档。"),
    ("有情报产出", "箱子里会掉情报类物品的关卡。"),
    ("掉落丰厚", "箱子产出价值偏高的关卡。"),
    ("僵尸", "主要敌人是僵尸的关卡。"),
    ("盗贼", "主要敌人是盗贼的关卡。"),
    ("诺亚", "主要敌人是诺亚的关卡。"),
    ("军阀", "主要敌人是军阀的关卡。"),
    ("天网", "主要敌人是天网机械的关卡。"),
)

#: 势力标签（可用作关键词，也是候选行的 enemies 取值）
FACTION_TAGS: tuple[str, ...] = (
    FACTION_ZOMBIE, FACTION_BANDIT, FACTION_NOAH, FACTION_WARLORD, FACTION_SKYNET,
)

#: 互斥标签对（同组内只能选一个；模型仍同时填时后端取并集，不报错）
STAGE_TAG_EXCLUSIVE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("势均力敌", "轻松应对"),
)

ITEM_TAGS: tuple[tuple[str, str], ...] = (
    ("武器", "刀枪一类的武器。"),
    ("防具", "头/上装/下装/脚/颈/手六部位的总称。"),
    ("头部装备", "头部部位的装备。"),
    ("上装装备", "上装部位的装备。"),
    ("下装装备", "下装部位的装备。"),
    ("脚部装备", "脚部部位的装备。"),
    ("颈部装备", "颈部部位的装备。"),
    ("手部装备", "手部部位的装备。"),
    ("刀", "刀类武器。"),
    ("长枪", "长枪类武器。"),
    ("手枪", "手枪类武器。"),
    ("大威力手枪", "大威力手枪。"),
    ("突击步枪", "突击步枪。"),
    ("冲锋枪", "冲锋枪。"),
    ("霰弹枪", "霰弹枪。"),
    ("狙击步枪", "狙击步枪。"),
    ("战斗步枪", "战斗步枪。"),
    ("机枪", "机枪。"),
    ("发射器", "发射器。"),
    ("反器材武器", "反器材武器。"),
    ("特殊", "特殊类武器。"),
    ("药剂", "恢复或增益类药剂，不含调酒调出来的饮品。"),
    ("酒水", "调酒调出来的饮品，以及商店在卖的名字带「酒」的东西。"),
    ("食品", "能直接吃的成品。"),
    ("食材", "做菜用的原料。"),
    ("材料", "收集类材料。"),
    ("弹夹", "弹药类。"),
    ("手雷", "投掷物。"),
    ("情报", "情报、线索、线报一类。"),
    ("插件", "装备插件。"),
    ("贵重", "单价偏高的东西。"),
)

ITEM_TAG_EXCLUSIVE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("武器", "防具"),
    ("药剂", "酒水", "食品", "食材", "材料", "弹夹", "手雷", "情报"),
)

NPC_TAGS: tuple[tuple[str, str], ...] = (
    ("同阵营", "跟发布者同一阵营的人。"),
    ("其他阵营", "跟发布者不同阵营的人。"),
    ("男", "男性。"),
    ("女", "女性。"),
    ("商人", "有商店、能供货的人。"),
    ("可切磋", "配了切磋关卡、能跟玩家比划的人。"),
    ("同区域", "和发布者在同一大区域的人。"),
    ("跨区域", "和发布者不在同一大区域的人。"),
) + tuple((f, f"{f}阵营的人。") for f in NPC_FACTIONS)

NPC_TAG_EXCLUSIVE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("同阵营", "其他阵营"),
    ("男", "女"),
    ("同区域", "跨区域"),
)

#: 敌人标签在候选行上的字段名（短字段，每行最多两个）
ENEMIES_FIELD = "enemies"
#: enemies 字段的合法取值（比关键词表宽：黑铁会/摇滚/大学也可能成为某关的敌人）
ENEMY_LABEL_VALUES = ALL_FACTIONS
#: 候选来源标注字段名（不能用 source：物品行已占用）
MATCHED_BY_FIELD = "matched_by"
RANDOM_PICK_LABEL = "随机候选"


# ---------------------------------------------------------------------------
# 阵营可用标签（只注册真有的）
# ---------------------------------------------------------------------------

def stage_tags_for_faction(
    faction: str,
    *,
    home_area_factions: Iterable[str] = (),
) -> list[str]:
    """该阵营可用的关卡标签（按 STAGE_TAGS 顺序，只注册真能命中东西的）。

    - 没有本方区域的势力（商人/探索者/通缉犯/闲杂人等/彩蛋）：`周边地区` 与 `敌对区域` 都不注册；
    - 本方区域里全是自家或友军关卡（军阀的基地房顶）：不注册 `周边地区`——这正是
      「军阀不可能让玩家去打军阀」的结构性保证；
    - `敌对区域` 只要本方区域之外还有能打的势力就注册。
    """
    f = faction or ""
    home = FACTION_HOME_AREA.get(f, "")
    if not home:
        hidden = {"周边地区", "敌对区域"}
    else:
        present = {x for x in home_area_factions if x and x != NO_FACTION}
        has_target_in_home = any(
            x != f and can_target(f, x) for x in present
        )
        hidden = set()
        if not has_target_in_home:
            hidden.add("周边地区")
    return [t for t, _ in STAGE_TAGS if t not in hidden]


def tag_description(tag: str) -> str:
    for table in (STAGE_TAGS, ITEM_TAGS, NPC_TAGS):
        for name, desc in table:
            if name == tag:
                return desc
    return ""


def exclusive_groups_for(tags: Iterable[str]) -> list[tuple[str, ...]]:
    """返回与这些标签相关的互斥组（只回涉及到的组）。"""
    chosen = set(tags or ())
    out: list[tuple[str, ...]] = []
    for groups in (STAGE_TAG_EXCLUSIVE_GROUPS, ITEM_TAG_EXCLUSIVE_GROUPS, NPC_TAG_EXCLUSIVE_GROUPS):
        for g in groups:
            if len(chosen & set(g)) >= 2:
                out.append(g)
    return out


def format_tag_catalog(tags: Iterable[str]) -> str:
    """「标签：说明」逐行文本，供提示词内联。"""
    lines: list[str] = []
    for t in tags:
        desc = tag_description(t)
        lines.append(f"- {t}：{desc}" if desc else f"- {t}")
    return "\n".join(lines)


def normalize_tag(raw: str, available: Iterable[str]) -> Optional[str]:
    """把模型给的词对齐到闭集标签（原样 / 去空白 / 常见别名）；对不上返回 None。"""
    s = str(raw or "").strip()
    if not s:
        return None
    avail = list(available)
    if s in avail:
        return s
    aliases = {
        "酒": "酒水", "美酒": "酒水", "饮品": "酒水",
        "吃货": "食品", "食物": "食品",
        "装备": "防具", "护甲": "防具",
        "枪": "武器", "刀剑": "刀",
        "弹药": "弹夹",
    }
    canon = aliases.get(s)
    if canon and canon in avail:
        return canon
    return None


def is_diplomatic_stage(stage_info: Any) -> bool:
    """中立外交地图不是可攻打关卡。"""
    return str(getattr(stage_info, "type", "") or "") == DIPLOMATIC_STAGE_TYPE
