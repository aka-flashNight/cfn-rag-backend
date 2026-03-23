from __future__ import annotations

import re
from typing import Any, Optional

from services.game_progress import stage_root_region_hint

from services.game_data.models import Item, StageInfo

# RAG 中与草案去重共用的块标题（与 game_rag_service 输出一致）
_RE_ITEM_SECTION_HDR = re.compile(r"【玩家可能提到的物品类型[^】]*】\s*", re.MULTILINE)
_RE_STAGE_SECTION_HDR = re.compile(r"【玩家可能提到的关卡】\s*", re.MULTILINE)
_RE_NEXT_SECTION = re.compile(r"\n\s*【", re.MULTILINE)


def parse_rag_game_entity_mentions(text: str | None) -> tuple[set[str], set[str]]:
    """
    从本轮检索上下文（含【玩家可能提到的物品类型】【玩家可能提到的关卡】）解析已出现的实体名，供草案摘要去重。

    约定：每条物品行以「名称：」开头、每条关卡行以「关卡名称：」开头（取第一个全角分号「；」前的片段为名称）。
    """
    if not text or not str(text).strip():
        return set(), set()
    raw = str(text)
    items: set[str] = set()
    stages: set[str] = set()

    def _first_field_value(line: str, prefix: str) -> str:
        s = line.strip()
        if not s.startswith(prefix):
            return ""
        rest = s[len(prefix) :]
        i = rest.find("；")
        return (rest[:i] if i >= 0 else rest).strip()

    m_item = _RE_ITEM_SECTION_HDR.search(raw)
    if m_item:
        tail = raw[m_item.end() :]
        cut = _RE_NEXT_SECTION.search(tail)
        block = tail[: cut.start()] if cut else tail
        for line in block.splitlines():
            v = _first_field_value(line, "名称：")
            if v:
                items.add(v)

    m_st = _RE_STAGE_SECTION_HDR.search(raw)
    if m_st:
        tail = raw[m_st.end() :]
        cut = _RE_NEXT_SECTION.search(tail)
        block = tail[: cut.start()] if cut else tail
        for line in block.splitlines():
            v = _first_field_value(line, "关卡名称：")
            if v:
                stages.add(v)

    return items, stages


def compute_reward_tags(item: Item, equipment_mods: Any) -> list[str]:
    """与任务奖励类型标注一致：药剂/弹夹/材料/食品/武器/防具/插件等（食材按食品标注）。"""
    allowed_use = {"药剂", "弹夹", "材料", "食品"}
    allowed_type = {"武器", "防具"}
    name = (item.name or "").strip()
    it_use = (item.use or "").strip()
    it_type = (item.type or "").strip()
    type_tags: list[str] = []
    use_tags: list[str] = []
    if it_type in allowed_type:
        type_tags.append(it_type)
        if it_use:
            use_tags.append(it_use)
    elif it_use == "食材" or it_type == "食材":
        use_tags.append("食品")
    elif it_use in allowed_use:
        use_tags.append(it_use)
    plugin = False
    try:
        plugin = bool(equipment_mods and equipment_mods.is_plugin(name))
    except Exception:
        plugin = False
    if not type_tags and not use_tags and not plugin:
        return []
    ordered = type_tags + use_tags + (["插件"] if plugin else [])
    dedup: list[str] = []
    for t in ordered:
        if t not in dedup:
            dedup.append(t)
    return dedup


def format_item_prompt_line(
    item: Item,
    *,
    reward_tags: list[str],
    price: Optional[int],
) -> str:
    """
    对话/RAG 用单行：名称、称呼(若)、任务类型标注、单价 靠前；其余字段随后。
    reward_tags 为空时，用分类/用途作简短替代（语义检索命中的物品等）。

    约定（勿改）：每条必须以「名称：{name}；」开头，供 parse_rag_game_entity_mentions 与草案摘要去重。
    """
    parts: list[str] = [f"名称：{item.name}"]
    ed = item.effective_displayname
    if ed:
        parts.append(f"称呼：{ed}")
    if reward_tags:
        parts.append(f"任务类型标注：{'、'.join(reward_tags)}")
    else:
        bits: list[str] = []
        if item.type:
            bits.append(item.type)
        if item.use:
            bits.append(item.use)
        if bits:
            parts.append(f"分类用途：{'/'.join(bits)}")
    if price is not None:
        parts.append(f"单价：{price}")
    if item.actiontype:
        parts.append(f"动作类型：{item.actiontype}")
    if item.weapontype:
        parts.append(f"枪械类型：{item.weapontype}")
    if item.description:
        parts.append(f"描述：{item.description}")
    if item.weight:
        parts.append(f"重量：{item.weight}")
    if item.clipname:
        parts.append(f"弹夹名：{item.clipname}")
    return "；".join(parts)



def format_item_embedding_text(item: Item) -> str:
    """向量片段：多行键值，与关键词展示信息等价。"""
    lines: list[str] = [f"名称：{item.name}"]
    ed = item.effective_displayname
    if ed:
        lines.append(f"称呼：{ed}")
    if item.type:
        lines.append(f"类型：{item.type}")
    if item.use:
        lines.append(f"用途：{item.use}")
    if item.actiontype:
        lines.append(f"动作类型：{item.actiontype}")
    if item.weapontype:
        lines.append(f"枪械类型：{item.weapontype}")
    if item.price is not None:
        lines.append(f"价格：{item.price}")
    if item.description:
        lines.append(f"描述：{item.description}")
    if item.weight:
        lines.append(f"重量：{item.weight}")
    if item.clipname:
        lines.append(f"弹夹名：{item.clipname}")
    return "\n".join(lines)


def format_stage_detail_line(si: StageInfo) -> str:
    """
    约定（勿改）：合并后首段为「关卡名称：{name}；」，供 parse_rag_game_entity_mentions 与草案摘要去重。
    """
    parts: list[str] = [
        f"关卡名称：{si.name}",
        f"所在区域：{si.area}",
        f"区域说明：{stage_root_region_hint(si.area)}",
    ]
    if si.description:
        parts.insert(1, f"关卡描述：{si.description}")
    else:
        parts.insert(1, "关卡描述：（无）")
    if si.unlock_condition is not None:
        parts.append(f"解锁主线ID：{si.unlock_condition}")
    return "；".join(parts)


def format_stage_embedding_text(si: StageInfo) -> str:
    lines = [
        f"关卡名称：{si.name}",
        f"关卡描述：{si.description or '（无）'}",
        f"所在区域：{si.area}",
        f"区域说明：{stage_root_region_hint(si.area)}",
    ]
    if si.unlock_condition is not None:
        lines.append(f"解锁主线ID：{si.unlock_condition}")
    return "\n".join(lines)


def pick_stage_area_for_name(stage_registry: Any, stage_name: str) -> Optional[str]:
    """与 context_builder._pick_stage_root_for_stage_name 一致，避免循环 import。"""
    areas = {a for (a, n), _ in stage_registry._stage_infos.items() if n == stage_name}
    if not areas:
        return None
    if "副本任务" in areas:
        return "副本任务"
    return min(areas)


def get_stage_info_for_name(stage_registry: Any, stage_name: str) -> Optional[StageInfo]:
    area = pick_stage_area_for_name(stage_registry, stage_name)
    if area is None:
        return None
    return stage_registry._stage_infos.get((area, stage_name))


def iter_all_stage_infos(stage_registry: Any) -> list[StageInfo]:
    return [si for si in stage_registry._stage_infos.values()]

