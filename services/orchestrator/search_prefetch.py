"""route_search 的编排层预取（检索链结构优化）。

设计口径（作者定稿 2026-10-17）：route_search 直接承担检索子 Agent 的**第一步**。
聊天主 Agent 在路由参数里给出 keywords 与 sources，编排层用同一组关键词在所选
范围内检索并返回并集；默认不启动子 Agent，结果直接拼进下一轮聊天调用。

- knowledge：沿用 Tier-1 的检索管线（池阈值 + top_k），按融合分排序，再用
  dense 相对下限筛掉明显低相关项；与聊天 Agent 已有的 Tier-1 结果按节点 id 去重。
- items / stages：与 search_items / search_stages 工具同语义（名称/区域子串），
  按匹配程度排序；不使用嵌入分，避免自由文本查询把弱相关实体带进来。
- 截断：knowledge 3600 字、items 1200 字、stages 1200 字，合计约 6000 字；
  每个类型内按排名依次填充，装不下时截断当前项并停止（不再跳过去找更短的）。

旧行为（本次替换）：search_knowledge 工具结果 `[:2000]` 硬截断；search_items
最多 10 条、search_stages 最多 8 条且按数据顺序取；子 Agent 最终结论
`_FINDINGS_MAX_CHARS=400` 硬截断（提示词也写 400，模型多写即被砍）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

SOURCE_KNOWLEDGE = "knowledge"
SOURCE_ITEMS = "items"
SOURCE_STAGES = "stages"
ALL_SOURCES: tuple[str, ...] = (SOURCE_KNOWLEDGE, SOURCE_ITEMS, SOURCE_STAGES)

# 每个类型的字数上限（合计约 6000；类型内不足不向其他类型借额度）
KNOWLEDGE_MAX_CHARS = 3600
ITEMS_MAX_CHARS = 1200
STAGES_MAX_CHARS = 1200
TOTAL_MAX_CHARS = KNOWLEDGE_MAX_CHARS + ITEMS_MAX_CHARS + STAGES_MAX_CHARS

# 单条资料的展示上限：避免一条超长世界观把整段预算吃光
NODE_MAX_CHARS = 800
_ITEM_LINE_MAX_CHARS = 320
_STAGE_LINE_MAX_CHARS = 320

# dense 相对下限：低于本类型最高分 35% 的视为明显低相关（池的绝对阈值仍先生效）
KNOWLEDGE_RELATIVE_FLOOR = 0.35

# 剩余预算不足这个数时不再截半句塞进去
_MIN_TAIL_CHARS = 120

_TERM_SPLIT_RE = re.compile(r"[\s,，、;；/|]+")

_KNOWLEDGE_POOLS: tuple[str, ...] = (
    "npc_dialogue",
    "world_lore",
    "loading",
    "npc_task",
    "supp_intel",
    "other_npc",
)

_KNOWLEDGE_LABELS: dict[str, str] = {
    "npc_dialogue": "你的过往台词",
    "world_lore": "世界观",
    "loading": "提示",
    "npc_task": "你的参考任务对话",
    "other_npc": "其他角色（仅参考设定）",
    "supp_intel": "补充设定/情报",
}


@dataclass
class SearchPrefetch:
    """一次关键词预取的结果（text 为空表示没有查到任何新资料）。"""

    text: str = ""
    sources: tuple[str, ...] = ()
    knowledge_count: int = 0
    item_count: int = 0
    stage_count: int = 0
    node_ids: frozenset[str] = field(default_factory=frozenset)
    item_names: frozenset[str] = field(default_factory=frozenset)
    stage_names: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def normalize_sources(raw: Any) -> tuple[str, ...]:
    """sources 归一化：不填/all → 全部；未知值忽略；空结果回落全部。"""
    if raw is None:
        return ALL_SOURCES
    if isinstance(raw, str):
        items: list[Any] = [raw]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = list(raw)
    else:
        return ALL_SOURCES

    picked: list[str] = []
    for item in items:
        value = str(item or "").strip().lower()
        if not value:
            continue
        if value == "all":
            return ALL_SOURCES
        if value in ALL_SOURCES and value not in picked:
            picked.append(value)
    return tuple(picked) or ALL_SOURCES


def run_search_prefetch(
    *,
    engine: Any,
    game_data: Any,
    keywords: str,
    sources: Any = None,
    npc_name: str = "",
    npc_titles: Sequence[str] | None = None,
    npc_faction: str | None = None,
    forbidden: set[str] | None = None,
    existing_bundle: Any = None,
) -> SearchPrefetch:
    """同步执行一次关键词预取（调用方用 to_thread 包装）。

    任何单侧失败只降级该侧为空，不影响其它来源；整体异常由调用方兜底。
    """
    query = (keywords or "").strip()
    picked_sources = normalize_sources(sources)
    forbidden = forbidden or set()
    if not query:
        return SearchPrefetch(sources=picked_sources)

    existing_ids, existing_item_names, existing_stage_names = _existing_keys(existing_bundle)

    knowledge_text = ""
    knowledge_count = 0
    knowledge_ids: set[str] = set()
    if SOURCE_KNOWLEDGE in picked_sources:
        try:
            knowledge_text, knowledge_ids = _prefetch_knowledge(
                engine=engine,
                keywords=query,
                npc_name=npc_name,
                npc_titles=npc_titles or [],
                npc_faction=npc_faction,
                forbidden=forbidden,
                existing_ids=existing_ids,
            )
            knowledge_count = len(knowledge_ids)
        except Exception as exc:
            logger.warning("预取 knowledge 失败，降级为空: %s", exc)

    items_text = ""
    item_names: set[str] = set()
    if SOURCE_ITEMS in picked_sources:
        try:
            items_text, item_names = _prefetch_items(
                game_data=game_data,
                keywords=query,
                existing_names=existing_item_names,
            )
        except Exception as exc:
            logger.warning("预取 items 失败，降级为空: %s", exc)

    stages_text = ""
    stage_names: set[str] = set()
    if SOURCE_STAGES in picked_sources:
        try:
            stages_text, stage_names = _prefetch_stages(
                game_data=game_data,
                keywords=query,
                existing_names=existing_stage_names,
            )
        except Exception as exc:
            logger.warning("预取 stages 失败，降级为空: %s", exc)

    sections = [block for block in (knowledge_text, items_text, stages_text) if block.strip()]
    text = "\n\n".join(sections)
    result = SearchPrefetch(
        text=text,
        sources=picked_sources,
        knowledge_count=knowledge_count,
        item_count=len(item_names),
        stage_count=len(stage_names),
        node_ids=frozenset(knowledge_ids),
        item_names=frozenset(item_names),
        stage_names=frozenset(stage_names),
    )
    logger.info(
        "预取检索完成：keywords=%.40s sources=%s knowledge=%d items=%d stages=%d chars=%d",
        query, "/".join(picked_sources), result.knowledge_count,
        result.item_count, result.stage_count, len(text),
    )
    return result


# ---------------------------------------------------------------------------
# knowledge
# ---------------------------------------------------------------------------

def _prefetch_knowledge(
    *,
    engine: Any,
    keywords: str,
    npc_name: str,
    npc_titles: list[str],
    npc_faction: str | None,
    forbidden: set[str],
    existing_ids: set[str],
) -> tuple[str, set[str]]:
    from services.orchestrator.context import retrieve_tier1

    bundle = retrieve_tier1(
        engine=engine,
        user_query=keywords,
        npc_name=npc_name,
        npc_titles=npc_titles,
        npc_faction=npc_faction,
        npc_last_message=None,
        forbidden=forbidden,
    )
    if bundle is None:
        return "", set()

    entries: list[tuple[str, Any]] = []
    top_dense = 0.0
    for pool_name in _KNOWLEDGE_POOLS:
        for scored in getattr(bundle, pool_name, None) or []:
            node = scored.node
            if not (node.text or "").strip():
                continue
            if node.id in existing_ids:
                continue
            top_dense = max(top_dense, float(scored.dense_score))
            entries.append((pool_name, scored))
    if not entries:
        return "", set()

    floor = top_dense * KNOWLEDGE_RELATIVE_FLOOR
    entries = [
        (pool_name, scored)
        for pool_name, scored in entries
        if float(scored.dense_score) >= floor
    ]
    entries.sort(key=lambda item: (-float(item[1].fused_score), -float(item[1].dense_score)))

    seen_lines: set[str] = set()
    parts: list[str] = []
    budget = KNOWLEDGE_MAX_CHARS
    used_ids: set[str] = set()
    for pool_name, scored in entries:
        label = _knowledge_label(pool_name, scored, forbidden)
        body = _node_body(scored.node.text, seen_lines)
        if not body:
            continue
        entry = f"[{label}] {body}"
        if len(entry) <= budget:
            parts.append(entry)
            budget -= len(entry) + 1
            used_ids.add(scored.node.id)
            continue
        if budget >= _MIN_TAIL_CHARS:
            parts.append(entry[: budget - 1].rstrip() + "…")
            used_ids.add(scored.node.id)
        break
    return "\n".join(parts), used_ids


def _knowledge_label(pool_name: str, scored: Any, forbidden: set[str]) -> str:
    base = _KNOWLEDGE_LABELS.get(pool_name, "资料")
    character = (scored.node.character or "").strip()
    if pool_name in ("npc_dialogue", "npc_task") and character:
        return f"{base}·{character}"
    if pool_name == "other_npc":
        if character and character.lower() in forbidden:
            return f"{base}·{character}（非正式角色）"
        return f"{base}·{character}" if character else base
    return base


def _node_body(text: str, seen_lines: set[str]) -> str:
    """单条正文：去空行、跨条去重、限长。"""
    body = (text or "").strip()
    if len(body) > NODE_MAX_CHARS:
        body = body[:NODE_MAX_CHARS].rstrip() + "…"
    lines: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line in seen_lines:
            continue
        seen_lines.add(line)
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# items / stages（与 search_items / search_stages 工具同语义）
# ---------------------------------------------------------------------------

def _search_terms(keywords: str) -> list[str]:
    terms: list[str] = []
    for term in [keywords, *_TERM_SPLIT_RE.split(keywords)]:
        value = (term or "").strip()
        if value and value not in terms:
            terms.append(value)
    return terms


def _match_rank(haystack: str, term: str) -> int:
    """0 完全相等 / 1 前缀 / 2 子串 / 99 不匹配。"""
    hay = (haystack or "").strip().lower()
    needle = (term or "").strip().lower()
    if not hay or not needle:
        return 99
    if hay == needle:
        return 0
    if hay.startswith(needle):
        return 1
    if needle in hay:
        return 2
    return 99


def _prefetch_items(
    *, game_data: Any, keywords: str, existing_names: set[str],
) -> tuple[str, set[str]]:
    from services.game_entity_prompts import compute_reward_tags, format_item_prompt_line

    items_registry = getattr(game_data, "items", None)
    if items_registry is None:
        return "", set()

    terms = _search_terms(keywords)
    picked: dict[str, Any] = {}
    ranks: dict[str, int] = {}
    for term in terms:
        # registry 的 limit 是数据序提前截断，不参与相关度；这里先取全量匹配再统一排序
        for item in items_registry.search(term, limit=10_000) or []:
            name = (item.name or "").strip()
            if not name or name.lower() in existing_names:
                continue
            hay = " ".join(part for part in (name, item.displayname or "") if part)
            rank = min((_match_rank(hay, t) for t in terms), default=99)
            if rank >= 99:
                continue
            old_rank = ranks.get(name)
            if old_rank is None or rank < old_rank:
                picked[name] = item
                ranks[name] = rank

    ordered = sorted(
        picked.values(),
        key=lambda item: (ranks.get(item.name, 99), len(item.name or ""), item.name or ""),
    )
    budget = ITEMS_MAX_CHARS
    header = "【物品资料】"
    lines = [header]
    budget -= len(header) + 1
    used: set[str] = set()
    for item in ordered:
        tags = compute_reward_tags(item, getattr(game_data, "equipment_mods", None))
        line = format_item_prompt_line(item, reward_tags=tags, price=item.price)
        if len(line) > _ITEM_LINE_MAX_CHARS:
            line = line[:_ITEM_LINE_MAX_CHARS].rstrip() + "…"
        if len(line) + 1 <= budget:
            lines.append(line)
            budget -= len(line) + 1
            used.add(item.name)
            continue
        if budget >= _MIN_TAIL_CHARS:
            lines.append(line[: budget - 1].rstrip() + "…")
            used.add(item.name)
        break
    if len(lines) == 1:
        return "", set()
    return "\n".join(lines), used


def _prefetch_stages(
    *, game_data: Any, keywords: str, existing_names: set[str],
) -> tuple[str, set[str]]:
    from services.game_entity_prompts import format_stage_detail_line

    stage_infos = getattr(getattr(game_data, "stages", None), "_stage_infos", None)
    if not stage_infos:
        return "", set()

    terms = _search_terms(keywords)
    picked: dict[tuple[str, str], tuple[int, Any]] = {}
    for (area, name), info in stage_infos.items():
        area_value = (area or "").strip()
        name_value = (name or "").strip()
        if not name_value:
            continue
        if name_value.lower() in existing_names or f"{area_value}::{name_value}".lower() in existing_names:
            continue
        rank = 99
        for term in terms:
            name_rank = _match_rank(name_value, term)
            if name_rank < 99:
                rank = min(rank, name_rank)
                continue
            area_rank = _match_rank(area_value, term)
            if area_rank < 99:
                rank = min(rank, 3 + area_rank)
        if rank >= 99:
            continue
        key = (area_value, name_value)
        old = picked.get(key)
        if old is None or rank < old[0]:
            picked[key] = (rank, info)

    ordered = sorted(
        picked.values(),
        key=lambda pair: (pair[0], len(pair[1].name or ""), pair[1].name or "", pair[1].area or ""),
    )
    budget = STAGES_MAX_CHARS
    header = "【关卡资料】"
    lines = [header]
    budget -= len(header) + 1
    used: set[str] = set()
    for _rank, info in ordered:
        line = format_stage_detail_line(info)
        if len(line) > _STAGE_LINE_MAX_CHARS:
            line = line[:_STAGE_LINE_MAX_CHARS].rstrip() + "…"
        if len(line) + 1 <= budget:
            lines.append(line)
            budget -= len(line) + 1
            used.add((info.name or "").strip().lower())
            continue
        if budget >= _MIN_TAIL_CHARS:
            lines.append(line[: budget - 1].rstrip() + "…")
            used.add((info.name or "").strip().lower())
        break
    if len(lines) == 1:
        return "", set()
    return "\n".join(lines), used


# ---------------------------------------------------------------------------
# 与聊天 Agent 已有 Tier-1 结果的去重键
# ---------------------------------------------------------------------------

_ALL_BUNDLE_FIELDS: tuple[str, ...] = (
    "npc_dialogue", "world_lore", "loading", "npc_task",
    "supp_intel", "other_npc", "entity_items", "entity_stages",
)


def _existing_keys(bundle: Any) -> tuple[set[str], set[str], set[str]]:
    """返回 (节点 id, 物品名, 关卡名/区域键)，全部小写。"""
    if bundle is None:
        return set(), set(), set()
    ids: set[str] = set()
    item_names: set[str] = set()
    stage_names: set[str] = set()
    for field_name in _ALL_BUNDLE_FIELDS:
        for scored in getattr(bundle, field_name, None) or []:
            node = scored.node
            if node.id:
                ids.add(node.id)
            if node.item_name:
                item_names.add(node.item_name.strip().lower())
            if node.stage_name:
                stage_names.add(node.stage_name.strip().lower())
                if node.stage_area:
                    stage_names.add(f"{node.stage_area.strip()}::{node.stage_name.strip()}".lower())
    return ids, item_names, stage_names


def build_prefetch_result_block(text: str) -> str:
    """拼进聊天 Agent 下一轮的结果块（与 MergeCoordinator 的搜索块同标题）。"""
    body = (text or "").strip()
    return "【后台结果：检索到的资料】\n" + (body or "（本次没有检索到新的相关资料）")
