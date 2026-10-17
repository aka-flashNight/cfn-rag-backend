"""NPC 固定内容层的装备块（system 层2，appearance 之后）。

数据源（只读，缺任何一环都整块不输出）：
- ``<data_root>/units/units.json``：顶层数组，``"NPC": true`` 的记录里 ``data`` 段
  直接存物品的 ``<name>``（不是 displayname）；
- ``<data_root>/items/*.xml``（经 ``list.xml`` 索引，由 ``GameDataRegistry`` 加载）：
  按 name 解析出 displayname / weight / description / title 用于展示。

三条口径（照用户裁定，改动前先读）：
① ``gender`` / ``face`` / ``hairstyle`` **不是装备**，一律排除；
② 名字对不齐是常态（``Andy Law`` ↔ ``Andy``、``Shop Girl`` ↔ ``ShopGirl``、
   ``小F`` ↔ ``BYD 小F``）：先归一化精确匹配，再退到唯一包含匹配；
   命中多条且载荷不同就 fail-closed（整块不输出），绝不猜；
③ 大段重复介绍框成共用段落（纯 difflib，不调模型），规则见
   :func:`mark_shared_paragraphs`。

本模块不写任何文件、不碰存档与状态库；units.json 只有几百条，
缓存键是 ``(路径, mtime_ns, size)``——文件一变就失效，绝不按路径永久缓存。
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

#: 装备字段 → 中文标签（固定顺序＝输出顺序；来自游戏仓「兵种系统_兼容.as」）
EQUIPMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("head", "头部装备"),
    ("body", "上装装备"),
    ("leg", "下装装备"),
    ("hand", "手部装备"),
    ("foot", "脚部装备"),
    ("neck", "颈部装备"),
    ("primary", "长枪"),
    ("secondary", "手枪"),
    ("secondary2", "副手枪"),
    ("melee", "刀"),
    ("grenade", "手雷"),
)

#: data 段里这些键不是装备（外观/生理属性），整块不参与解析
NON_EQUIPMENT_FIELDS: frozenset[str] = frozenset({"gender", "face", "hairstyle"})

#: 共用段落的最小长度（Python len，不是字节）；短重复不动
SHARED_MIN_CHUNK = 20

#: units.json 缓存：键＝(路径, mtime_ns, size)，只留最新一份
_UNITS_CACHE: dict[tuple[str, int, int], list[dict[str, Any]]] = {}

#: 内部装备行（展示名已按 displayname 回退 name 解析好）
@dataclass(frozen=True)
class EquipmentEntry:
    label: str
    displayname: str
    weight: str = ""
    description: str = ""
    title: str = ""


# ---------------------------------------------------------------------------
# units.json 读取与名称匹配
# ---------------------------------------------------------------------------

def normalize_name(text: Any) -> str:
    """名称归一化：casefold ＋ 去掉所有空白（含全角空格），保留中文与符号。"""
    return "".join(ch for ch in str(text or "").casefold() if not ch.isspace())


def _load_units(path: Path) -> list[dict[str, Any]]:
    """读 units.json（mtime+size 失效的缓存）；读不到/坏 JSON＝空列表。"""
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _UNITS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        doc = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 %s 失败，装备块跳过: %s", path, exc)
        return []
    units = [u for u in doc if isinstance(u, dict)] if isinstance(doc, list) else []
    _UNITS_CACHE.clear()  # 路径固定、只留最新状态，避免历史键堆积
    _UNITS_CACHE[key] = units
    return units


def _equipment_payload(unit: dict[str, Any]) -> dict[str, str]:
    """该单位的装备载荷 {字段: 物品 name}；空值与非装备字段都不进。"""
    data = unit.get("data")
    if not isinstance(data, dict):
        return {}
    payload: dict[str, str] = {}
    for field, _label in EQUIPMENT_FIELDS:
        if field in NON_EQUIPMENT_FIELDS:  # 防御：映射表与排除表永远不交叉
            continue
        name = str(data.get(field) or "").strip()
        if name:
            payload[field] = name
    return payload


def _matches(unit: dict[str, Any], target: str) -> bool:
    """精确或唯一包含匹配的候选判定（包含是双向的：一方含另一方）。"""
    name = normalize_name(unit.get("name"))
    if not name or not target:
        return False
    if name == target:
        return False  # 精确匹配由调用方单独处理，这里只算「非精确」的包含
    return target in name or name in target


def resolve_unit(units: Iterable[dict[str, Any]], npc_name: str) -> Optional[dict[str, Any]]:
    """按 NPC 名找 units 记录；歧义一律 fail-closed 返回 None。

    只看 ``"NPC": true`` 的记录。先归一化精确匹配，再退到包含匹配；
    命中多条时，装备载荷完全相同才算命中（Pig/Boy/King 这类同载荷重复记录），
    载荷不同就放弃——宁可没有装备块，也不给模型一个错的装备表。
    """
    target = normalize_name(npc_name)
    if not target:
        return None
    npc_units = [u for u in units if u.get("NPC") is True]
    if not npc_units:
        return None
    exact = [u for u in npc_units if normalize_name(u.get("name")) == target]
    if exact:
        candidates = exact
    else:
        candidates = [u for u in npc_units if _matches(u, target)]
        if len(candidates) > 1:
            logger.debug("NPC「%s」包含匹配到 %d 条 units 记录", npc_name, len(candidates))
    if not candidates:
        return None
    payloads = {
        tuple(sorted(_equipment_payload(u).items())) for u in candidates
    }
    if len(payloads) > 1:
        logger.warning(
            "NPC「%s」在 units.json 命中 %d 条装备载荷不同的记录，装备块 fail-closed 不输出",
            npc_name, len(candidates),
        )
        return None
    return candidates[0]


# ---------------------------------------------------------------------------
# 展示字段格式化
# ---------------------------------------------------------------------------

def _format_weight(raw: Any) -> str:
    """重量短句：能转数字就格式化（0→0kg、0.5→0.5kg）；缺失/非数字＝空串。"""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return ""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        return ""
    if value == int(value):
        return f"{int(value)}kg"
    return f"{value:g}kg"


def _clean_clause(text: Any) -> str:
    """介绍/称号：去首尾空白与句末标点，交给调用方统一补「；」。"""
    return str(text or "").strip().rstrip("。；;，,.、 ").strip()


# ---------------------------------------------------------------------------
# 共用段落（大段重复介绍的标记化，纯函数）
# ---------------------------------------------------------------------------

def _overlaps(marks: list[tuple[int, int, str, Optional[str], int]], start: int, end: int) -> bool:
    return any(not (end <= mark[0] or start >= mark[1]) for mark in marks)


def _registered_id(registered: list[tuple[int, str]], text: str) -> Optional[int]:
    for sid, chunk in registered:
        if chunk == text:
            return sid
    return None


def _render_marks(description: str, marks: list[tuple[int, int, str, Optional[str], int]]) -> str:
    if not marks:
        return description
    out: list[str] = []
    cursor = 0
    for start, end, prefix, suffix, _sid in sorted(marks, key=lambda m: m[0]):
        if start > cursor:
            out.append(description[cursor:start])
        if suffix is None:
            out.append(prefix)
        else:
            out.append(f"{prefix}{description[start:end]}{suffix}")
        cursor = end
    out.append(description[cursor:])
    return "".join(out)


def mark_shared_paragraphs(descriptions: list[str]) -> list[str]:
    """把大段重复介绍框成共用段落（确定性，不调 LLM）。

    口径：
    - 只处理 ≥ ``SHARED_MIN_CHUNK`` 字符的匹配块（``difflib.SequenceMatcher``，
      ``autojunk=False``），短重复原样保留；
    - 共享段按**首次出现顺序**编号 1、2、3…：首次出现那条用
      ``<共用段落N开始> … <共用段落N结束>`` 包住，后续物品同一段替换为
      ``<此处与共用段落N相同>``；3 条以上重复同一段时都指向同一个编号；
    - 与已标记区域重叠的新共享段跳过（不嵌套）；
    - 完全相同的两条介绍：第一条整段标记，第二条整条变成「此处与共用段落N相同」；
    - 只替换后续物品里的重复，第一条内部同内容出现两次保持原样；
    - 标记逐字使用中文尖括号，编号从 1 开始、不加空格；其余文本逐字保留。
    """
    count = len(descriptions)
    marks: list[list[tuple[int, int, str, Optional[str], int]]] = [[] for _ in range(count)]
    registered: list[tuple[int, str]] = []
    next_id = 1
    for i in range(count):
        for j in range(i + 1, count):
            left, right = descriptions[i], descriptions[j]
            if not left or not right:
                continue
            matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
            for block in matcher.get_matching_blocks():
                size = block.size
                if size < SHARED_MIN_CHUNK:
                    continue
                # 后续物品这一侧不能被已标记区域覆盖（不嵌套、也不重复贴同一段）
                if _overlaps(marks[j], block.b, block.b + size):
                    continue
                chunk = left[block.a:block.a + size]
                shared_id = _registered_id(registered, chunk)
                if shared_id is None:
                    # 新共享段：首次出现的那条要被包起来，所以它这一侧也不能已标记
                    if _overlaps(marks[i], block.a, block.a + size):
                        continue
                    shared_id = next_id
                    next_id += 1
                    registered.append((shared_id, chunk))
                    marks[i].append((
                        block.a, block.a + size,
                        f"<共用段落{shared_id}开始>", f"<共用段落{shared_id}结束>", shared_id,
                    ))
                # 已登记的共享段：首次出现那条早已包好，这里只补后续物品这一侧
                marks[j].append((
                    block.b, block.b + size,
                    f"<此处与共用段落{shared_id}相同>", None, shared_id,
                ))
    return [_render_marks(descriptions[i], marks[i]) for i in range(count)]


# ---------------------------------------------------------------------------
# 块文本
# ---------------------------------------------------------------------------

EQUIPMENT_HEADER = "你身上的装备是："


def format_equipment_block(entries: list[EquipmentEntry]) -> str:
    """装备块文本（固定顺序、只输出实际存在的项）；无条目＝空串。"""
    if not entries:
        return ""
    marked = mark_shared_paragraphs([e.description for e in entries])
    lines: list[str] = []
    for entry, description in zip(entries, marked):
        clauses = [f"{entry.label}：{entry.displayname}"]
        if entry.weight:
            clauses.append(f"重量{entry.weight}")
        if description:
            clauses.append(f"介绍：{description}")
        line = "，".join(clauses)
        if entry.title:
            line += f"；称号：{entry.title}"
        lines.append(f"{line}；")
    return f"{EQUIPMENT_HEADER}\n" + "\n".join(lines)


def build_npc_equipment_block(*, npc_name: str, game_data: Any) -> str:
    """按 NPC 名拼「你身上的装备是：…」块；任何一环缺失都返回空串（不占位、不编造）。

    调用方（``orchestrator/turn.py``）对异常另有一层兜底，故这里只保证**不抛**。
    """
    try:
        data_root = getattr(game_data, "data_root", None)
        if data_root is None:
            return ""
        units = _load_units(Path(data_root) / "units" / "units.json")
        if not units:
            return ""
        unit = resolve_unit(units, npc_name)
        if unit is None:
            return ""
        payload = _equipment_payload(unit)
        if not payload:
            return ""
        items = getattr(game_data, "items", None)
        if items is None:
            return ""
        entries: list[EquipmentEntry] = []
        for field, label in EQUIPMENT_FIELDS:
            item_name = payload.get(field)
            if not item_name:
                continue
            try:
                item = items.get_by_name(item_name)
            except Exception:  # pragma: no cover - 注册表异常按解析不到处理
                item = None
            if item is None:
                continue
            entries.append(EquipmentEntry(
                label=label,
                displayname=str(item.displayname or item.name),
                weight=_format_weight(item.weight),
                description=_clean_clause(item.description),
                title=_clean_clause(getattr(item, "title", "")),
            ))
        if not entries:
            return ""
        return format_equipment_block(entries)
    except Exception as exc:
        logger.warning("NPC 装备块装配失败，降级为空: %s", exc)
        return ""
