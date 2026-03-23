"""
prepare_task_context 工具执行器。

根据 task_type 和 reward_types 一次性返回该类型所需的全部筛选后数据与规则说明，
减少 LLM 的决策负担（文档 6.3.2）。
"""

from __future__ import annotations

import json
import random
from typing import Any, Optional

from services.game_data.registry import GameDataRegistry, get_game_data_registry
from services.game_data.reward_utils import (
    REWARD_STAGE_BASE_MAX,
    REWARD_STAGE_BASE_MIN,
    parse_name_count,
)
from services.game_progress import (
    _STAGE_ROOT_REGION_HINT_EXTRA,
    get_progress_stage_config,
    get_progress_stage_level_range,
    get_progress_stage_main_task_range,
    PROGRESS_STAGE_CONFIG,
    stage_root_region_hint,
)

# 当 challenge 无 llm_hint 时使用的默认提醒（仅在选择该难度时随 challenge_modes 返回，按需占用 token）
DEFAULT_CHALLENGE_LLM_HINT = "若选择此难度，请在任务介绍中明确说明当前难度，并在对话中提醒具有挑战性。"


# ---------------------------------------------------------------------------
# 关键词筛选（prepare_task_context 可选参数）
# ---------------------------------------------------------------------------

def _normalize_kw_list(keywords: Optional[list[str]]) -> list[str]:
    if not keywords:
        return []
    out: list[str] = []
    for k in keywords:
        if not isinstance(k, str):
            continue
        s = k.strip()
        if s:
            out.append(s)
    return out


def _any_keyword_in_text(text: str, keywords: list[str]) -> bool:
    if not text or not keywords:
        return False
    return any(kw in text for kw in keywords)


def _item_keyword_search_text(item: Any, equipment_mods: Any) -> str:
    """用于子串匹配：name / displayname / type / use，并补充 插件、食材、食品 等语义标签。"""
    parts: list[str] = [
        getattr(item, "name", "") or "",
        getattr(item, "displayname", "") or "",
        getattr(item, "type", "") or "",
        getattr(item, "use", "") or "",
    ]
    name = getattr(item, "name", "") or ""
    if name and equipment_mods.is_plugin(name):
        parts.append("插件")
    use = getattr(item, "use", "") or ""
    typ = getattr(item, "type", "") or ""
    if use == "食材" or typ == "食材":
        parts.append("食材")
    if "食品" in typ or use == "食品" or "菜" in typ:
        parts.append("食品")
    return "".join(parts)


def _entry_matches_item_keywords(
    entry: dict[str, Any],
    item: Optional[Any],
    equipment_mods: Any,
    keywords: list[str],
) -> bool:
    if not keywords:
        return False
    blob = "".join([
        str(entry.get("name") or ""),
        str(entry.get("type") or ""),
        str(entry.get("source") or ""),
    ])
    if _any_keyword_in_text(blob, keywords):
        return True
    if item is not None:
        if _any_keyword_in_text(_item_keyword_search_text(item, equipment_mods), keywords):
            return True
    return False


def _allocate_remainder_quota_pools(
    remainder: int,
    pools: list[list[dict[str, Any]]],
    mode: str,
) -> list[dict[str, Any]]:
    """
    关键词等优先档占满后，对「剩余名额」按档位比例分配：
    - mode '82'：两档 pools[0] 高 : pools[1] 低 = 8 : 2（低档为 round(remainder×20%)，四舍五入）；
    - mode '55'：两档 pools[0] : pools[1] = 5 : 5（先 round(remainder×50%)，余数给另一档）；
    - mode '532'：三档 = 5 : 3 : 2（先 round 前两档，余数给第三档）。

    pools 顺序为高优先级 → 低优先级；各池内部会先 shuffle。
    某档数量不足时，先从高档池依次补齐，再从中档、低档耗尽为止。
    """
    if remainder <= 0 or not pools:
        return []
    n = len(pools)
    if mode == "82":
        if n != 2:
            raise ValueError("mode 82 requires exactly 2 pools")
        q_low = int(round(remainder * 2 / 10))
        q_high = remainder - q_low
        quotas = [q_high, q_low]
    elif mode == "55":
        if n != 2:
            raise ValueError("mode 55 requires exactly 2 pools")
        q0 = int(round(remainder * 5 / 10))
        q1 = remainder - q0
        quotas = [q0, q1]
    elif mode == "532":
        if n != 3:
            raise ValueError("mode 532 requires exactly 3 pools")
        q0 = int(round(remainder * 5 / 10))
        q1 = int(round(remainder * 3 / 10))
        q2 = remainder - q0 - q1
        quotas = [q0, q1, q2]
    else:
        raise ValueError(f"unknown remainder quota mode: {mode!r}")

    work = [list(p) for p in pools]
    for w in work:
        random.shuffle(w)
    out: list[dict[str, Any]] = []
    for i, q in enumerate(quotas):
        take = min(q, len(work[i]))
        out.extend(work[i][:take])
        work[i] = work[i][take:]
    while len(out) < remainder:
        progressed = False
        for i in range(len(work)):
            if work[i]:
                out.append(work[i].pop(0))
                progressed = True
                if len(out) >= remainder:
                    break
        if not progressed:
            break
    return out[:remainder]


def _shuffle_each_bucket_concat(buckets: list[list[Any]]) -> list[Any]:
    """保持桶之间的优先级顺序；同一优先级桶内随机打乱，避免长尾条目永远进不了截断列表。"""
    out: list[Any] = []
    for b in buckets:
        bb = list(b)
        random.shuffle(bb)
        out.extend(bb)
    return out


def _item_level_matches_stage(item: Optional[Any], min_level: int, max_level: int) -> bool:
    """等级为 0 或落在玩家阶段等级区间内视为「符合当前进度」。"""
    if item is None:
        return True
    lv = int(getattr(item, "level", 0) or 0)
    if lv == 0:
        return True
    return min_level <= lv <= max_level


def _matched_items_ordered_level_then_source(
    matched: list[dict[str, Any]],
    *,
    item_registry: Any,
    min_level: int,
    max_level: int,
) -> list[dict[str, Any]]:
    """装备缴纳：关键词命中项先「等级在阶段内」再「等级外」；各档内再按 合成 > 非本阵营商店 > K点商店；桶内随机。"""
    mk_ok_synth: list[dict[str, Any]] = []
    mk_ok_shop: list[dict[str, Any]] = []
    mk_ok_k: list[dict[str, Any]] = []
    mk_bad_synth: list[dict[str, Any]] = []
    mk_bad_shop: list[dict[str, Any]] = []
    mk_bad_k: list[dict[str, Any]] = []
    for e in matched:
        it = item_registry.get_by_name(e.get("name") or "")
        ok = _item_level_matches_stage(it, min_level, max_level)
        src = e.get("source")
        if ok:
            if src == "合成":
                mk_ok_synth.append(e)
            elif src == "非本阵营商店":
                mk_ok_shop.append(e)
            elif src == "K点商店":
                mk_ok_k.append(e)
            else:
                mk_ok_synth.append(e)
        else:
            if src == "合成":
                mk_bad_synth.append(e)
            elif src == "非本阵营商店":
                mk_bad_shop.append(e)
            elif src == "K点商店":
                mk_bad_k.append(e)
            else:
                mk_bad_synth.append(e)
    return _shuffle_each_bucket_concat(
        [
            mk_ok_synth,
            mk_ok_shop,
            mk_ok_k,
            mk_bad_synth,
            mk_bad_shop,
            mk_bad_k,
        ]
    )


# ---------------------------------------------------------------------------
# 任务类型规则说明模板
# ---------------------------------------------------------------------------

_TASK_RULES: dict[str, str] = {
    "问候": (
        "问候/闲聊类任务：最简单的任务类型，玩家只需与指定NPC对话即可完成。"
        "可选当前NPC自己或其他NPC为完成NPC。奖励较少。"
        "完成任务对话内容中是完成NPC和玩家进行对话，奖励也由完成NPC提供。"
        "适合低好感度或初次对话时使用。"
    ),
    "传话": (
        "传话类任务：要求玩家去找另一个NPC对话。"
        "在有理由的情况下，可选择任意NPC作为完成NPC，包括同阵营传话、不同阵营的外交等。"
        "注意，传话类任务中，完成NPC和发布NPC不同。完成NPC一定要是需要传话的NPC对象，完成任务时的对话内容也是该完成NPC和玩家对话，奖励也由完成NPC提供。"
        "奖励较少。适合推动NPC间剧情互动。"
    ),
    "通关": (
        "通关类任务：要求玩家通关指定关卡。"
        "优先选择在玩家当前进度范围内的关卡，其次可选小于玩家进度的。（解锁ID ≤ 当前主线ID）"
        "area为“副本任务”的关卡通常仅可选择“简单”。只有当关卡配置了额外的难度候选项时（副本类只有最多两个选项），才允许选择该额外难度，选择时需明确提醒难度要求；否则只能选择“简单”。"
        "area为其他区域的地图关卡可选任意难度。"
        "基础奖励上下限更高。"
    ),
    "清理": (
        "清理类任务：要求玩家通关指定关卡，但叙事上侧重'清除威胁/清理区域'。"
        "优先选择在玩家当前进度范围内的关卡，其次可选小于玩家进度的。（解锁ID ≤ 当前主线ID）"
        "area为“副本任务”的关卡通常仅可选择“简单”。只有当关卡配置了额外的难度候选项时（副本类只有最多两个选项），才允许选择该额外难度，选择时需明确提醒难度要求；否则只能选择“简单”。"
        "area为其他区域的地图关卡可选任意难度。"
        "基础奖励上下限更高。"
    ),
    "挑战": (
        "挑战类任务：高难度通关，建议选择修罗或地狱难度。"
        "area为“副本任务”的关卡，只有当关卡配置了额外的难度候选项时（副本类只有最多两个选项），才允许选择该额外难度，选择时需明确提醒难度要求。"
        "奖励中经验值占比应 ≥ 50%，金币占比可降低。可适当给一些技能点。"
        "基础奖励上下限更高。"
    ),
    "切磋": (
        "切磋类任务：要求玩家通关当前NPC配置的专属切磋关卡。"
        "只能使用当前NPC的challenge属性对应的关卡。"
        "基础奖励上下限更高。"
    ),
    "资源收集": (
        "资源收集类任务：要求玩家收集并提交指定数量的食材/药剂/材料/弹夹。"
        "提交物品必须在现有任务的提交物品+奖励物品池中。"
        "提交品总价值不超过基础奖励的200%。"
        "奖励在原有区间要求上，额外增加提交品总价值的1~2倍（下限+1×、上限+2×）。"
        "收集的资源是玩家提交给完成NPC的。完成NPC可以选自己，也可以选他人，但需要任务文本的配合。"
        "如果玩家是在索要某种资源，且未主动提出交换资源，立刻选择其他任务类型重新调用prepare_task_context，并在奖励中选择对应资源！绝对禁止选择本任务类型！绝对禁止将玩家需求的物品放入finish_submit_items中！"
    ),
    "装备缴纳": (
        "装备缴纳类任务：要求玩家获取并提交一件装备（武器/防具）。"
        "来源：合成配方、K点商店、其他阵营的NPC商店。"
        "提交品总价值不超过基础奖励的300%。"
        "奖励在原有区间要求上，额外增加提交品价值的1~2倍（下限+1×、上限+2×）。"
        "缴纳的装备是玩家提交给完成NPC的。完成NPC可以选自己，也可以选他人，但需要任务文本的配合。"
        "如果玩家是在索要某种装备，且未主动提出交换资源，立刻选择其他任务类型重新调用prepare_task_context，并在奖励中选择对应装备！绝对禁止选择本任务类型！绝对禁止将玩家需求的物品放入finish_submit_items中！"
    ),
    "特殊物品获取": (
        "特殊物品获取类任务：要求玩家获取并提交一个特殊物品。"
        "通常只需要1个物品。来源：合成配方、K点商店、其他阵营的NPC商店。"
        "包括插件、药剂、菜品、贵重消耗品等非装备物品。"
        "获取的特殊物品是玩家提交给完成NPC的。完成NPC可以选自己，也可以选他人，但需要任务文本的配合。"
    ),
    "物品持有": (
        "物品持有类任务：要求玩家持有（不提交）指定物品。"
        "来源：情报类物品或合成配方产出。"
        "奖励额外增加持有品价值的0.5倍，上限不超过基础奖励的50%。"
        "持有品本身总价值上限为基础奖励的200%。"
        "持有类可以是让玩家获取情报，也可以是指引玩家去制作物品并检验成果，或者其他需要获取物品但无需提交的情况。"
    ),
    "通关并收集": (
        "通关并收集类任务：组合通关+收集要求。"
        "收集物品必须是该关卡箱子的产出物品。"
        "收集数量建议使用箱子的最小产出数量。"
        "基础奖励上下限更高，并需要再叠加收集品加成（提交品价值计入奖励区间：下限+1×、上限+2×）。"
        "收集品是玩家提交给完成NPC的。完成NPC可以选自己，也可以选他人，但需要任务文本的配合。"
        "如果当前关卡的area是“副本任务”，且只有两个难度选择，而你在 finish_requirements 里选择了非“简单”的难度（例如“地狱”），请务必在任务说明中明显提醒玩家正在选择挑战模式，并在接取/完成台词里明确提到要挑战该高难度模式。"
    ),
    "通关并持有": (
        "通关并持有类任务：组合通关+持有要求。"
        "持有物品必须是该关卡箱子的产出物品。"
        "基础奖励上下限更高，并需要再叠加持有品加成（持有品按 0.5× 计入奖励区间，上限不超过基础奖励的50%。）。"
        "持有品本身总价值上限为基础奖励的200%。"
        "持有品不要求玩家提交。可以是需要情报物品，也可以是检验玩家的搜索成果等情况。"
        "如果当前关卡的area是“副本任务”，且只有两个难度选择，而你 finish_requirements 里选择了非“简单”的难度（例如“地狱”），请务必在任务说明中明显提醒玩家正在选择挑战模式，并在接取/完成台词里明确提到要挑战该高难度模式。"
    ),
}


# ---------------------------------------------------------------------------
# 奖励预算计算
# ---------------------------------------------------------------------------

_COMBAT_TYPES = frozenset({
    "通关", "清理", "挑战", "切磋", "通关并收集", "通关并持有",
})


def _compute_reward_budget(
    *,
    stage: int,
    task_type: str,
    affinity: int,
) -> dict[str, Any]:
    base_min = stage * REWARD_STAGE_BASE_MIN
    base_max = stage * REWARD_STAGE_BASE_MAX
    multiplier = 2 if task_type in _COMBAT_TYPES else 1

    if affinity >= 80:
        aff_mod = 1.20
    elif affinity >= 50:
        aff_mod = 1.10
    elif affinity >= 20:
        aff_mod = 1.00
    else:
        aff_mod = 0.90

    final_min = int(base_min * multiplier * aff_mod)
    final_max = int(base_max * multiplier * aff_mod)

    # 只返回模型定价用的上下限；基础值/倍率由后端内部使用，校验 V7 亦不从本字段读取。
    return {
        "final_min": final_min,
        "final_max": final_max,
    }


# ---------------------------------------------------------------------------
# 奖励物品候选筛选
# ---------------------------------------------------------------------------

def _matches_reward_type(
    item: Any,
    reward_type: str,
    equipment_mods: Any,
) -> bool:
    """
    判断一个物品是否匹配指定的奖励类型。
    规则：name 或 type 或 use 任意一项与 reward_type 相等即可。
    特殊：\"插件\" 通过 equipment_mods_registry 判断。
    材料：reward_type「材料」同时匹配 item.use==材料 或 item.type==收集品（数据里材料多为 type 收集品 + use 材料）。
    食品：Parser 将 消耗品_材料_食材 的 use 标为「食材」；奖励枚举仍只有「食品」，此处将食材视为食品以便入选奖励池。
    """
    if reward_type == "插件":
        return equipment_mods.is_plugin(item.name)

    if reward_type == "武器":
        if item.type == "武器":
            return True
        if item.use == "手雷":
            return True

    if item.name == reward_type:
        return True
    if item.type and item.type == reward_type:
        return True
    if item.use and item.use == reward_type:
        return True
    # 材料：数据中常为 type=收集品、use=材料；若 use 未标或归一化，用 type 收集品 兜底
    if reward_type == "材料" and item.type and item.type == "收集品":
        return True
    if reward_type == "食品":
        if item.use == "食材":
            return True
        if item.type == "食材":
            return True
    return False


def _ordered_reward_item_names_from_tasks(
    *,
    task_registry: Any,
    mercenary_registry: Any,
    main_task_min_id: int,
    main_task_max_id: int,
    level_min: int,
    level_max: int,
) -> list[str]:
    """
    按“当前区间优先”顺序产出任务奖励池中的物品名，用于 reward_item_candidates 排序。
    顺序：主线 id 在区间内 → 前置含区间内主线 id 的非主线 → mercenary 推荐等级与当前等级区间有交集 → 其他。
    """
    all_tasks = task_registry.list_all_tasks()
    main_range_ids = {t.id for t in all_tasks if main_task_min_id <= t.id <= main_task_max_id}
    # 前置任务包含当前区间内任一主线 id 的任务（且自身不在区间内，避免重复）
    precondition_in_range_ids = {
        t.id for t in all_tasks
        if t.id not in main_range_ids
        and (t.get_requirements or [])
        and any(rid in main_range_ids for rid in (t.get_requirements or []))
    }
    # mercenary_tasks.json 中存在且推荐等级与当前等级区间有交集的任务 id（无推荐等级的副本视为最低优先级，不放入 tier3）
    merc_level_overlap_ids: set[int] = set()
    for m in (mercenary_registry.list_all() if hasattr(mercenary_registry, "list_all") else []):
        rmin = getattr(m, "recommended_min_level", None)
        rmax = getattr(m, "recommended_max_level", None)
        if rmin is None and rmax is None:
            continue  # 无推荐等级 -> 不视为“符合”，归入 tier4
        rmin = rmin if rmin is not None else 0
        rmax = rmax if rmax is not None else 999
        if rmax < level_min or rmin > level_max:
            continue
        merc_level_overlap_ids.add(m.id)
    # 分档：tier1 主线区间内, tier2 前置在区间内, tier3 mercenary 等级有交集, tier4 其余
    tier1: list[int] = [t.id for t in all_tasks if t.id in main_range_ids]
    tier2: list[int] = [t.id for t in all_tasks if t.id in precondition_in_range_ids]
    tier3: list[int] = [t.id for t in all_tasks if t.id in merc_level_overlap_ids and t.id not in main_range_ids and t.id not in precondition_in_range_ids]
    tier4: list[int] = [t.id for t in all_tasks if t.id not in main_range_ids and t.id not in precondition_in_range_ids and t.id not in merc_level_overlap_ids]
    ordered_ids = tier1 + tier2 + tier3 + tier4
    # 按任务顺序收集奖励物品名（保持顺序、可重复，后续用 seen 去重）
    out: list[str] = []
    for tid in ordered_ids:
        t = task_registry.get_by_id(tid)
        if not t or not t.rewards:
            continue
        for r in t.rewards:
            name, _ = parse_name_count(r)
            if name:
                out.append(name)
    return out


def _reward_item_name_progress_tier_map(
    *,
    task_registry: Any,
    mercenary_registry: Any,
    main_task_min_id: int,
    main_task_max_id: int,
    level_min: int,
    level_max: int,
) -> dict[str, int]:
    """
    任务奖励物品名 → 进度档 1~4（与 _ordered_reward_item_names_from_tasks 分档一致），
    未出现在任务奖励池中的名不会出现在 map 中。
    """
    all_tasks = task_registry.list_all_tasks()
    main_range_ids = {t.id for t in all_tasks if main_task_min_id <= t.id <= main_task_max_id}
    precondition_in_range_ids = {
        t.id for t in all_tasks
        if t.id not in main_range_ids
        and (t.get_requirements or [])
        and any(rid in main_range_ids for rid in (t.get_requirements or []))
    }
    merc_level_overlap_ids: set[int] = set()
    for m in (mercenary_registry.list_all() if hasattr(mercenary_registry, "list_all") else []):
        rmin = getattr(m, "recommended_min_level", None)
        rmax = getattr(m, "recommended_max_level", None)
        if rmin is None and rmax is None:
            continue
        rmin = rmin if rmin is not None else 0
        rmax = rmax if rmax is not None else 999
        if rmax < level_min or rmin > level_max:
            continue
        merc_level_overlap_ids.add(m.id)
    tier1: list[int] = [t.id for t in all_tasks if t.id in main_range_ids]
    tier2: list[int] = [t.id for t in all_tasks if t.id in precondition_in_range_ids]
    tier3: list[int] = [
        t.id for t in all_tasks
        if t.id in merc_level_overlap_ids
        and t.id not in main_range_ids
        and t.id not in precondition_in_range_ids
    ]
    tier4: list[int] = [
        t.id for t in all_tasks
        if t.id not in main_range_ids
        and t.id not in precondition_in_range_ids
        and t.id not in merc_level_overlap_ids
    ]
    name_tier: dict[str, int] = {}
    for tier_num, tier_ids in enumerate([tier1, tier2, tier3, tier4], start=1):
        for tid in tier_ids:
            t = task_registry.get_by_id(tid)
            if not t or not t.rewards:
                continue
            for r in t.rewards:
                n, _ = parse_name_count(r)
                if n and n not in name_tier:
                    name_tier[n] = tier_num
    return name_tier


def _ordered_collectable_item_names_from_tasks(
    *,
    task_registry: Any,
    mercenary_registry: Any,
    main_task_min_id: int,
    main_task_max_id: int,
    level_min: int,
    level_max: int,
) -> list[str]:
    """
    按“当前区间优先”顺序产出资源收集相关物品池中的物品名（可能重复），
    顺序：主线 id 在区间内 → 前置含区间主线 → mercenary 等级重合 → 其他。

    同一任务内：先其 `finish_submit_items`，再其 `rewards`。
    """
    all_tasks = task_registry.list_all_tasks()
    main_range_ids = {t.id for t in all_tasks if main_task_min_id <= t.id <= main_task_max_id}
    precondition_in_range_ids = {
        t.id
        for t in all_tasks
        if t.id not in main_range_ids
        and (t.get_requirements or [])
        and any(rid in main_range_ids for rid in (t.get_requirements or []))
    }

    merc_level_overlap_ids: set[int] = set()
    for m in (mercenary_registry.list_all() if hasattr(mercenary_registry, "list_all") else []):
        rmin = getattr(m, "recommended_min_level", None)
        rmax = getattr(m, "recommended_max_level", None)
        if rmin is None and rmax is None:
            continue
        rmin = rmin if rmin is not None else 0
        rmax = rmax if rmax is not None else 999
        if rmax < level_min or rmin > level_max:
            continue
        merc_level_overlap_ids.add(m.id)

    tier1: list[int] = [t.id for t in all_tasks if t.id in main_range_ids]
    tier2: list[int] = [t.id for t in all_tasks if t.id in precondition_in_range_ids]
    tier3: list[int] = [
        t.id
        for t in all_tasks
        if t.id in merc_level_overlap_ids
        and t.id not in main_range_ids
        and t.id not in precondition_in_range_ids
    ]
    tier4: list[int] = [
        t.id
        for t in all_tasks
        if t.id not in main_range_ids
        and t.id not in precondition_in_range_ids
        and t.id not in merc_level_overlap_ids
    ]
    ordered_ids = tier1 + tier2 + tier3 + tier4

    out: list[str] = []
    for tid in ordered_ids:
        t = task_registry.get_by_id(tid)
        if not t:
            continue
        for expr in (t.finish_submit_items or []):
            name, _ = parse_name_count(expr)
            if name:
                out.append(name)
        for r in (t.rewards or []):
            name, _ = parse_name_count(r)
            if name:
                out.append(name)
    return out


def _build_reward_item_candidates(
    *,
    reward_types: dict[str, list[str]],
    game_data: GameDataRegistry,
    npc_name: str,
    stage: int,
    min_level: int,
    max_level: int,
    main_task_range: tuple[int, int],
    reward_keywords: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    组装奖励物品候选列表（文档 6.3.2 通用字段 reward_item_candidates）。

    数据来源与顺序：NPC 商店优先 → 当前区间任务（主线/前置/mercenary 等级重合）→ 其他任务。
    """
    item_registry = game_data.items
    task_registry = game_data.tasks
    shop_registry = game_data.shops
    equipment_mods = game_data.equipment_mods
    mercenary_registry = game_data.mercenary_tasks

    all_types = list(reward_types.get("regular", [])) + list(reward_types.get("optional", []))

    if stage < 4 and "K点" in all_types:
        all_types.remove("K点")

    npc_shop_items = set(shop_registry.get_npc_shop(npc_name))
    has_shop = shop_registry.has_shop(npc_name)
    if not has_shop:
        for restricted in ("武器", "防具", "插件"):
            if restricted in all_types:
                all_types.remove(restricted)

    if not all_types:
        return []

    reward_stats = task_registry.get_reward_stats()
    main_min, main_max = main_task_range
    ordered_task_item_names = _ordered_reward_item_names_from_tasks(
        task_registry=task_registry,
        mercenary_registry=mercenary_registry,
        main_task_min_id=main_min,
        main_task_max_id=main_max,
        level_min=min_level,
        level_max=max_level,
    )
    reward_task_tier_by_name = _reward_item_name_progress_tier_map(
        task_registry=task_registry,
        mercenary_registry=mercenary_registry,
        main_task_min_id=main_min,
        main_task_max_id=main_max,
        level_min=min_level,
        level_max=max_level,
    )

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    # 当 NPC 商店的武器/防具候选在等级区间内不足 3 件时：
    # 1) 先仅返回区间内（min_level~max_level）的商店装备；
    # 2) 不足后再从「低于 min_level 但不超过 max_level」的商店装备中补足；
    # 3) 补足按等级从高到低进行（尽量贴近 min_level）。
    _SHOP_EQUIPMENT_SUPPLEMENT_TYPES: set[str] = {"武器", "防具"}
    _shop_inrange_equipment_by_type: dict[str, list[dict[str, Any]]] = {
        rt: [] for rt in _SHOP_EQUIPMENT_SUPPLEMENT_TYPES
    }
    _shop_below_equipment_by_type: dict[str, list[dict[str, Any]]] = {
        rt: [] for rt in _SHOP_EQUIPMENT_SUPPLEMENT_TYPES
    }

    # 来源 2 优先：当前NPC商店物品
    for shop_item_name in npc_shop_items:
        if shop_item_name in seen:
            continue
        item = item_registry.get_by_name(shop_item_name)
        if item is None:
            continue
        is_weapon_or_armor = item.type in ("武器", "防具")
        is_grenade = bool(item.use) and item.use == "手雷"

        # 所有商店物品都需匹配 reward_types（含「材料」匹配 item.use/type）
        matched_types: list[str] = []
        for rt in all_types:
            if _matches_reward_type(item, rt, equipment_mods):
                matched_types.append(rt)
        if not matched_types:
            continue

        # entry 字段统一（后续补足逻辑复用）
        entry = {
            "name": item.name,
            "type": item.type,
            "price": item.price or 0,
            "source": "本NPC商店",
        }
        if item.level > 0:
            entry["level"] = item.level
        if equipment_mods.is_plugin(item.name):
            tier = equipment_mods.get_plugin_tier(item.name)
            if tier:
                entry["plugin_tier"] = tier

        # 等级筛选只针对「武器/防具/手雷」这一类装备输入项
        if is_weapon_or_armor or is_grenade:
            if item.level > max_level:
                continue
            if item.level < min_level:
                for rt in matched_types:
                    if rt in _SHOP_EQUIPMENT_SUPPLEMENT_TYPES:
                        _shop_below_equipment_by_type[rt].append(entry)
                continue

            # 区间内：立即加入候选，并在后续做“不足 3 才补”的计数
            candidates.append(entry)
            seen.add(shop_item_name)
            for rt in matched_types:
                if rt in _SHOP_EQUIPMENT_SUPPLEMENT_TYPES:
                    _shop_inrange_equipment_by_type[rt].append(entry)
            continue

        # 非装备类：直接加入候选
        candidates.append(entry)
        seen.add(shop_item_name)

    # 不足 3 件时再补：从高到低补「低于 min_level」但不超过 max_level 的商店装备
    for rt in _SHOP_EQUIPMENT_SUPPLEMENT_TYPES:
        if rt not in all_types:
            continue
        need = 3 - len(_shop_inrange_equipment_by_type.get(rt) or [])
        if need <= 0:
            continue

        below_items = _shop_below_equipment_by_type.get(rt) or []
        below_items.sort(key=lambda e: e.get("level") or 0, reverse=True)
        for e in below_items:
            nm = e.get("name") or ""
            if not nm or nm in seen:
                continue
            candidates.append(e)
            seen.add(nm)
            _shop_inrange_equipment_by_type.setdefault(rt, []).append(e)
            need -= 1
            if need <= 0:
                break

    # 来源 1：任务奖励池（按当前区间优先顺序：主线区间 → 前置在区间 → mercenary 等级重合 → 其他）
    for item_name in ordered_task_item_names:
        if item_name in seen:
            continue
        item = item_registry.get_by_name(item_name)
        if item is None:
            continue
        if item.level > max_level:
            continue
        if item.type in ("武器", "防具") and getattr(item, "use", None) != "手雷":
            continue
        for rt in all_types:
            if _matches_reward_type(item, rt, equipment_mods):
                entry = {
                    "name": item.name,
                    "type": item.type,
                    "price": item.price or 0,
                    "source": "任务奖励可选",
                }
                if item.level > 0:
                    entry["level"] = item.level
                stats = reward_stats.get(item_name)
                if stats:
                    entry["min_qty"] = stats[0]
                    entry["max_qty"] = stats[1]
                if equipment_mods.is_plugin(item.name):
                    tier = equipment_mods.get_plugin_tier(item.name)
                    if tier:
                        entry["plugin_tier"] = tier
                candidates.append(entry)
                seen.add(item_name)
                break

    return _finalize_reward_item_candidates(
        candidates,
        item_registry=item_registry,
        equipment_mods=equipment_mods,
        reward_keywords=reward_keywords,
        reward_task_tier_by_name=reward_task_tier_by_name,
        max_n=20,
        selected_reward_types=all_types,
    )


# ---------------------------------------------------------------------------
# 类型专属字段构建
# ---------------------------------------------------------------------------

def _pick_stage_root_for_stage_name(stage_registry: Any, stage_name: str) -> Optional[str]:
    """根据关卡 XML 名解析其所在的 stages 子目录（大区）；副本优先归为「副本任务」。"""
    areas = {a for (a, n), _ in stage_registry._stage_infos.items() if n == stage_name}
    if not areas:
        return None
    if "副本任务" in areas:
        return "副本任务"
    return min(areas)


def _reorder_stage_list_by_keywords(
    stage_list: list[dict[str, Any]],
    requirement_keywords: Optional[list[str]],
) -> None:
    kw = _normalize_kw_list(requirement_keywords)
    for block in stage_list:
        stages = block.get("stages") or []
        if not stages:
            continue
        area = block.get("area") or ""
        hint_p = stage_root_region_hint(area)
        extra = _STAGE_ROOT_REGION_HINT_EXTRA.get(area, "")

        def _stage_kw_hit(s: dict[str, Any]) -> bool:
            if not kw:
                return False
            blob = "".join([
                area,
                hint_p,
                extra,
                str(s.get("name") or ""),
                str(s.get("area_region_hint") or ""),
            ])
            return _any_keyword_in_text(blob, kw)

        if not kw:
            in_prog = [s for s in stages if not s.get("below_progress")]
            below = [s for s in stages if s.get("below_progress")]
            block["stages"] = _shuffle_each_bucket_concat([in_prog, below])
            continue

        b0: list[dict[str, Any]] = []  # 关键词 + 当前进度内
        b1: list[dict[str, Any]] = []  # 关键词 + 低于进度
        b2: list[dict[str, Any]] = []  # 无关键词 + 当前进度内
        b3: list[dict[str, Any]] = []  # 无关键词 + 低于进度
        for s in stages:
            kh = _stage_kw_hit(s)
            bp = bool(s.get("below_progress"))
            if kh and not bp:
                b0.append(s)
            elif kh and bp:
                b1.append(s)
            elif not kh and not bp:
                b2.append(s)
            else:
                b3.append(s)
        block["stages"] = _shuffle_each_bucket_concat([b0, b1, b2, b3])


def _stage_loot_row_matches_keywords(
    entry: dict[str, Any],
    keywords: list[str],
    item_registry: Any,
    equipment_mods: Any,
) -> bool:
    blob = "".join([
        str(entry.get("area") or ""),
        str(entry.get("area_region_hint") or ""),
        str(entry.get("stage_name") or ""),
    ])
    if _any_keyword_in_text(blob, keywords):
        return True
    for li in entry.get("loot_items") or []:
        name = li.get("item_name")
        if not name:
            continue
        item = item_registry.get_by_name(name)
        ed: dict[str, Any] = {"name": name}
        if _entry_matches_item_keywords(ed, item, equipment_mods, keywords):
            return True
    return False


def _get_all_stages_for_progress(
    game_data: GameDataRegistry,
    stage: int,
    requirement_keywords: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    获取所有有效关卡的二级结构列表（通关/清理/挑战类用）。
    筛选：unlock_condition ≤ main_task_max_id，副本按推荐等级筛选。
    """
    stage_registry = game_data.stages
    mercenary_registry = game_data.mercenary_tasks

    cfg = get_progress_stage_config(stage)
    if cfg is None:
        return []

    main_task_max_id = cfg.main_task_max_id or 0
    main_task_min_id = cfg.main_task_min_id or 0
    max_level = cfg.max_level or 50

    # 副本任务（副本关卡）按 stage_name 聚合
    mercs_by_stage: dict[str, list[Any]] = {}
    for mt in mercenary_registry.list_all():
        if not mt.stage_name:
            continue
        mercs_by_stage.setdefault(mt.stage_name, []).append(mt)

    area_map: dict[str, list[dict[str, Any]]] = {}

    for (area, name), si in stage_registry._stage_infos.items():
        if si.unlock_condition is None:
            continue

        is_dungeon = area == "副本任务"

        if is_dungeon:
            merc_tasks = mercs_by_stage.get(name) or []
            # 文档：没有标注推荐等级的副本一律剔除（沿用旧逻辑）
            # 只有当至少存在一个 merc task 的 recommended_min_level <= max_level 时才放行
            allowed_by_root = [
                mt for mt in merc_tasks
                if mt.recommended_min_level is not None and mt.recommended_min_level <= max_level
            ]
            if not allowed_by_root:
                continue

            allowed_difficulties: set[str] = {"简单"}
            challenge_modes_map: dict[str, str] = {}
            challenge_hint_map: dict[str, str] = {}

            # 额外难度：需要满足 challenge.recommended_level 的下限
            for mt in merc_tasks:
                if not mt.challenge_difficulty or mt.challenge_difficulty == "简单":
                    continue
                cmin = mt.challenge_recommended_min_level
                if cmin is None:
                    continue
                if cmin <= max_level:
                    allowed_difficulties.add(mt.challenge_difficulty)
                    if mt.challenge_description and mt.challenge_difficulty not in challenge_modes_map:
                        challenge_modes_map[mt.challenge_difficulty] = mt.challenge_description
                    if mt.challenge_difficulty not in challenge_hint_map:
                        challenge_hint_map[mt.challenge_difficulty] = (
                            mt.challenge_llm_hint or DEFAULT_CHALLENGE_LLM_HINT
                        )

            # 保持稳定顺序：按照 difficulty 枚举顺序输出
            difficulties = [d for d in ("简单", "冒险", "修罗", "地狱") if d in allowed_difficulties]

            # 返回给 LLM 的挑战模式说明（仅当包含非简单时）
            if len(difficulties) > 1 and challenge_modes_map:
                entry_challenges = [
                    {
                        "difficulty": diff,
                        "description": desc,
                        "hint": challenge_hint_map.get(diff, DEFAULT_CHALLENGE_LLM_HINT),
                    }
                    for diff, desc in challenge_modes_map.items()
                    if diff in difficulties and diff != "简单"
                ]
            else:
                entry_challenges = []
        else:
            if si.unlock_condition > main_task_max_id:
                continue
            difficulties = ["简单", "冒险", "修罗", "地狱"]

        below_progress = (
            not is_dungeon
            and si.unlock_condition < main_task_min_id
        )

        entry: dict[str, Any] = {
            "name": name,
            "unlock_id": si.unlock_condition,
            "difficulties": difficulties,
            "is_dungeon": is_dungeon,
            "area_region_hint": stage_root_region_hint(area),
        }
        if is_dungeon:
            # 根 recommended_level 用于“副本可选性”提示（这里取放行集合的最小下限）
            entry["recommended_level"] = min(
                mt.recommended_min_level
                for mt in merc_tasks
                if mt.recommended_min_level is not None
            )
            if entry_challenges:
                entry["challenge_modes"] = entry_challenges
        if below_progress:
            entry["below_progress"] = True
        # difficulties/challenge_modes/recommended_level 已在 is_dungeon 分支里处理

        area_map.setdefault(area, []).append(entry)

    # 构建二级结构
    result: list[dict[str, Any]] = []
    for stage_num in sorted(PROGRESS_STAGE_CONFIG.keys()):
        sc = PROGRESS_STAGE_CONFIG[stage_num]
        area_name = sc.stage_name
        if area_name and area_name in area_map:
            lr = get_progress_stage_level_range(stage_num)
            result.append({
                "area": area_name,
                "area_level_range": list(lr) if lr else None,
                "stages": area_map[area_name],
            })
    # 追加跨进度区域
    for cross_area in ("地下2层", "副本任务", "试炼场深处"):
        if cross_area in area_map:
            result.append({
                "area": cross_area,
                "area_level_range": None,
                "stages": area_map[cross_area],
            })

    _reorder_stage_list_by_keywords(result, requirement_keywords)
    return result


def _build_npc_list(
    game_data: GameDataRegistry,
    npc_states: dict[str, Any],
    current_npc: str,
    requirement_keywords: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """问候/传话类：所有NPC列表。"""
    # 文档约束：如果当前NPC不是“成员/彩蛋”阵营，则候选列表里排除这些类型NPC，
    # 避免 LLM 在对话中把非正式角色当作可选完成NPC。
    current_state = npc_states.get(current_npc)
    current_faction = getattr(current_state, "faction", None) or ""
    allow_special_npcs = current_faction in {"成员", "彩蛋"}

    banned_special_factions = {"成员", "彩蛋"}
    result: list[dict[str, Any]] = []
    for name, state in npc_states.items():
        if name == "$PC_CHAR":
            continue
        faction = getattr(state, "faction", None) or ""
        if (
            not allow_special_npcs
            and faction in banned_special_factions
            and name != current_npc
        ):
            continue
        entry: dict[str, Any] = {
            "name": name,
            "faction": getattr(state, "faction", None) or "",
        }
        titles = getattr(state, "titles", None)
        if titles:
            entry["title"] = titles[0] if titles else ""
            entry["titles"] = list(titles) if isinstance(titles, list) else titles
        else:
            entry["titles"] = []
        emotions = getattr(state, "emotions", None)
        if isinstance(emotions, list) and emotions:
            entry["emotions"] = list(emotions)
        else:
            entry["emotions"] = ["普通"]
        if name == current_npc:
            entry["is_current"] = True
        result.append(entry)

    kw = _normalize_kw_list(requirement_keywords)

    def _npc_kw_hit(entry: dict[str, Any]) -> bool:
        if not kw:
            return False
        titles = entry.get("titles") or []
        title_parts = list(titles) if isinstance(titles, list) else [str(titles)]
        blob = "".join([
            str(entry.get("name") or ""),
            str(entry.get("faction") or ""),
            str(entry.get("title") or ""),
            *title_parts,
        ])
        return _any_keyword_in_text(blob, kw)

    if not kw:
        cur = [e for e in result if e.get("is_current")]
        rest = [e for e in result if not e.get("is_current")]
        ordered = _shuffle_each_bucket_concat([cur, rest])
        return ordered[:30]

    b0: list[dict[str, Any]] = []  # 关键词 + 当前 NPC
    b1: list[dict[str, Any]] = []  # 关键词 + 其他
    b2: list[dict[str, Any]] = []  # 无关键词 + 当前 NPC
    b3: list[dict[str, Any]] = []  # 无关键词 + 其他
    for entry in result:
        kh = _npc_kw_hit(entry)
        cur = bool(entry.get("is_current"))
        if kh and cur:
            b0.append(entry)
        elif kh and not cur:
            b1.append(entry)
        elif not kh and cur:
            b2.append(entry)
        else:
            b3.append(entry)
    ordered = _shuffle_each_bucket_concat([b0, b1, b2, b3])
    return ordered[:30]


def _build_challenge_targets(
    game_data: GameDataRegistry,
    npc_name: str,
    npc_challenge: Optional[str],
    stage: int,
) -> list[dict[str, Any]] | dict[str, str]:
    """切磋类：当前NPC的切磋关卡列表。"""
    if not npc_challenge:
        return {"error": "当前NPC无可用的切磋目标，请选择其他任务类型"}

    mercenary_registry = game_data.mercenary_tasks
    cfg = get_progress_stage_config(stage)
    max_level = cfg.max_level if cfg else 50
    main_task_range = get_progress_stage_main_task_range(stage) or (0, 77)
    main_task_max_id = int(main_task_range[1]) if main_task_range and len(main_task_range) > 1 else 77

    # 同名关卡可能出现在多个 area；不能只用“副本优先”的单一归属判断。
    # 规则：无推荐等级时，若存在任一非副本 area 且其解锁id满足进度，则按关卡可用。
    areas = {
        area
        for (area, name), _si in game_data.stages._stage_infos.items()
        if name == npc_challenge
    }
    has_dungeon_area = "副本任务" in areas
    non_dungeon_areas = [a for a in areas if a != "副本任务"]

    # 根据 mercenary_tasks.json 的 recommended_level 过滤：不展示推荐等级高于当前阶段上限的关卡
    matched_merc_tasks = [m for m in mercenary_registry.list_all() if m.stage_name == npc_challenge]
    if matched_merc_tasks:
        eligible: list[Any] = []
        unlock_ok = False
        for area in non_dungeon_areas:
            unlock_id = game_data.stages.get_unlock_condition(area, npc_challenge)
            if unlock_id > 0 and unlock_id <= int(main_task_max_id):
                unlock_ok = True
                break

        for m in matched_merc_tasks:
            rec = getattr(m, "recommended_min_level", None)
            if rec is not None:
                if int(rec) <= int(max_level):
                    eligible.append(m)
                continue

            # 没有 recommended_min_level 的条目：
            # - 存在可解锁的非副本 area：按关卡可用
            if unlock_ok:
                eligible.append(m)
                continue

            # - 否则若仅副本 area（或关卡 unlock 不满足）：排除
            if has_dungeon_area:
                continue

        if not eligible:
            return {"error": "玩家当前阶段实力/进度不满足该NPC切磋关卡条件，请选择其他任务类型"}

        # 后续难度/提示等只使用可判定的有效条目
        matched_merc_tasks = eligible
    else:
        # 无 mercenary 条目时：按关卡解锁条件兜底判定
        # - 存在可解锁的非副本 area：可用
        # - 否则不可用（例如仅副本但无推荐等级数据）
        unlock_ok = False
        for area in non_dungeon_areas:
            unlock_id = game_data.stages.get_unlock_condition(area, npc_challenge)
            if unlock_id > 0 and unlock_id <= int(main_task_max_id):
                unlock_ok = True
                break
        if not unlock_ok:
            return {"error": "玩家当前阶段实力/进度不满足该NPC切磋关卡条件，请选择其他任务类型"}

    # 基础：至少包含“简单”
    allowed_difficulties: set[str] = {"简单"}
    challenge_modes_map: dict[str, str] = {}
    challenge_hint_map: dict[str, str] = {}
    for m in matched_merc_tasks:
        if not m.challenge_difficulty or m.challenge_difficulty == "简单":
            continue
        cmin = m.challenge_recommended_min_level
        if cmin is None:
            continue
        if int(cmin) <= int(max_level):
            allowed_difficulties.add(m.challenge_difficulty)
            if m.challenge_description and m.challenge_difficulty not in challenge_modes_map:
                challenge_modes_map[m.challenge_difficulty] = m.challenge_description
            if m.challenge_difficulty not in challenge_hint_map:
                challenge_hint_map[m.challenge_difficulty] = m.challenge_llm_hint or DEFAULT_CHALLENGE_LLM_HINT

    difficulties = [d for d in ("简单", "冒险", "修罗", "地狱") if d in allowed_difficulties]

    entry: dict[str, Any] = {
        "dungeon_name": npc_challenge,
        "target_npc": npc_name,
        "difficulties": difficulties,
    }
    ch_area = _pick_stage_root_for_stage_name(game_data.stages, npc_challenge)
    if ch_area:
        entry["stage_area"] = ch_area
        entry["area_region_hint"] = stage_root_region_hint(ch_area)

    if len(difficulties) > 1:
        extra_modes = [
            {
                "difficulty": d,
                "description": desc,
                "hint": challenge_hint_map.get(d, DEFAULT_CHALLENGE_LLM_HINT),
            }
            for d, desc in challenge_modes_map.items()
            if d in difficulties and d != "简单"
        ]
        if extra_modes:
            entry["challenge_modes"] = extra_modes

    return [entry]


_COLLECTABLE_MAX_ITEMS = 20


def _collectable_entry_for_name(
    item_name: str,
    *,
    item_registry: Any,
    submit_stats: Any,
    reward_stats: Any,
    base_max: int,
    max_level: int,
) -> Optional[dict[str, Any]]:
    item = item_registry.get_by_name(item_name)
    if item is None or item.level > max_level:
        return None
    price = item.price or 0
    entry: dict[str, Any] = {
        "name": item.name,
        "type": item.type,
        "price": price,
        "min_qty": 1,
    }
    stats = submit_stats.get(item_name)
    submit_max = int(stats[1] or 0) if stats and int(stats[1] or 0) > 0 else 0
    reward_max = int(reward_stats.get(item_name, (None, 0))[1] or 0)
    effective_max = max(submit_max, reward_max)
    if effective_max > 0:
        entry["max_qty"] = effective_max * 2
    else:
        unit_price = price or 1
        entry["max_qty"] = max(1, int(base_max / unit_price))
    if item.level > 0:
        entry["level"] = item.level
    return entry


def _build_collectable_items(
    game_data: GameDataRegistry,
    stage: int,
    max_level: int,
    base_max: int,
    requirement_keywords: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """资源收集类：食材/药剂/材料/弹夹，且在现有任务物品池中。"""
    item_registry = game_data.items
    task_registry = game_data.tasks
    mercenary_registry = game_data.mercenary_tasks
    equipment_mods = game_data.equipment_mods
    keywords = _normalize_kw_list(requirement_keywords)

    allowed_uses = {"食材", "药剂", "材料", "弹夹"}
    pool = task_registry.list_submit_items() | task_registry.list_reward_item_names()
    price_cap = base_max * 2

    main_task_range = get_progress_stage_main_task_range(stage) or (0, 77)
    level_range = get_progress_stage_level_range(stage) or (1, 50)
    ordered_item_names = _ordered_collectable_item_names_from_tasks(
        task_registry=task_registry,
        mercenary_registry=mercenary_registry,
        main_task_min_id=main_task_range[0],
        main_task_max_id=main_task_range[1],
        level_min=level_range[0],
        level_max=max_level,
    )
    seen: set[str] = set()
    submit_stats = task_registry.get_submit_stats()
    reward_stats = task_registry.get_reward_stats()

    def _can_use_item(it: Any) -> bool:
        return (it.use in allowed_uses) or (it.type in allowed_uses) or (it.name in allowed_uses)

    candidates: list[dict[str, Any]] = []

    for item_name in ordered_item_names:
        if item_name in seen or item_name not in pool:
            continue
        item = item_registry.get_by_name(item_name)
        if item is None or not _can_use_item(item):
            continue
        entry = _collectable_entry_for_name(
            item_name,
            item_registry=item_registry,
            submit_stats=submit_stats,
            reward_stats=reward_stats,
            base_max=base_max,
            max_level=max_level,
        )
        if entry:
            entry["_from_ordered"] = True
            candidates.append(entry)
            seen.add(item_name)

    for item_name in sorted(pool - seen):
        item = item_registry.get_by_name(item_name)
        if item is None or not _can_use_item(item):
            continue
        entry = _collectable_entry_for_name(
            item_name,
            item_registry=item_registry,
            submit_stats=submit_stats,
            reward_stats=reward_stats,
            base_max=base_max,
            max_level=max_level,
        )
        if entry:
            entry["_from_ordered"] = False
            candidates.append(entry)
            seen.add(item_name)

    if keywords:
        b0, b1, b2, b3 = [], [], [], []
        for e in candidates:
            it = item_registry.get_by_name(e.get("name") or "")
            km = _entry_matches_item_keywords(e, it, equipment_mods, keywords)
            fo = bool(e.get("_from_ordered"))
            if km and fo:
                b0.append(e)
            elif km and not fo:
                b1.append(e)
            elif not km and fo:
                b2.append(e)
            else:
                b3.append(e)
        ordered = _shuffle_each_bucket_concat([b0, b1, b2, b3])
    else:
        fo_yes = [e for e in candidates if e.get("_from_ordered")]
        fo_no = [e for e in candidates if not e.get("_from_ordered")]
        ordered = _shuffle_each_bucket_concat([fo_yes, fo_no])

    result: list[dict[str, Any]] = []
    for e in ordered:
        e.pop("_from_ordered", None)
        if len(result) >= _COLLECTABLE_MAX_ITEMS:
            break
        p = e.get("price") or 0
        # 候选项池只按「单件」对照任务规则（提交品单价不超过基础奖励 200%），不累加列表总价。
        if p > price_cap:
            continue
        result.append(e)
    return result


_EQUIPMENT_ITEMS_MAX_COUNT = 20


def _npc_faction_from_states(
    npc_states: Optional[dict[str, Any]], npc_name: str
) -> str:
    if not npc_states:
        return ""
    st = npc_states.get(npc_name)
    if st is None:
        return ""
    if isinstance(st, dict):
        return str(st.get("faction") or "").strip()
    fac = getattr(st, "faction", None)
    return (str(fac).strip() if fac is not None else "") or ""


def _skip_easter_egg_shop_for_cross_faction_pool(
    other_npc: str, npc_states: Optional[dict[str, Any]]
) -> bool:
    """「非本阵营商店」候选不收录彩蛋阵营 NPC 的商店。"""
    return _npc_faction_from_states(npc_states, other_npc) == "彩蛋"


def _build_equipment_items(
    game_data: GameDataRegistry,
    npc_name: str,
    npc_faction: str,
    stage: int,
    min_level: int,
    max_level: int,
    requirement_keywords: Optional[list[str]] = None,
    npc_states: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """装备缴纳类：合成 + 非本阵营商店 + K 点（同名保留合成优先）。"""
    item_registry = game_data.items
    shop_registry = game_data.shops
    crafting_registry = game_data.crafting
    kshop_registry = game_data.kshop
    equipment_mods = game_data.equipment_mods
    keywords = _normalize_kw_list(requirement_keywords)

    equipment_types = {"武器", "防具"}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 1. 合成配方产出的装备（去重优先级最高）
    for recipe in crafting_registry._recipes:
        if recipe.name in seen:
            continue
        item = item_registry.get_by_name(recipe.name)
        if item is None or item.type not in equipment_types:
            continue
        if item.level > max_level:
            continue
        seen.add(recipe.name)
        result.append({
            "name": item.name,
            "type": item.type,
            "price": item.price or 0,
            "level": item.level,
            "source": "合成",
        })

    # 2. 非本阵营 NPC 商店的装备（排除彩蛋阵营商人的商店）
    for other_npc in shop_registry._shops:
        if other_npc == npc_name:
            continue
        if _skip_easter_egg_shop_for_cross_faction_pool(other_npc, npc_states):
            continue
        for item_name in shop_registry.get_npc_shop(other_npc):
            if item_name in seen:
                continue
            item = item_registry.get_by_name(item_name)
            if item is None or item.type not in equipment_types:
                continue
            if item.level > max_level:
                continue
            seen.add(item_name)
            result.append({
                "name": item.name,
                "type": item.type,
                "price": item.price or 0,
                "level": item.level,
                "source": "非本阵营商店",
            })

    # 3. K点商店装备
    kprice_cap = stage * 100
    for kitem in kshop_registry.list_items():
        if kitem.item in seen:
            continue
        item = item_registry.get_by_name(kitem.item)
        if item is None or item.type not in equipment_types:
            continue
        if item.level > max_level:
            continue
        if kitem.price > kprice_cap:
            continue
        seen.add(kitem.item)
        result.append({
            "name": item.name,
            "type": item.type,
            "price": item.price or 0,
            "level": item.level,
            "source": "K点商店",
        })

    max_total = _EQUIPMENT_ITEMS_MAX_COUNT
    matched: list[dict[str, Any]] = []
    if keywords:
        for e in result:
            it = item_registry.get_by_name(e.get("name") or "")
            if _entry_matches_item_keywords(e, it, equipment_mods, keywords):
                matched.append(e)

    matched_names = {e.get("name") for e in matched}
    unmatched = [e for e in result if e.get("name") not in matched_names]

    synth_u, shop_u, k_u = _partition_special_by_source(unmatched)
    random.shuffle(synth_u)
    random.shuffle(shop_u)
    random.shuffle(k_u)

    picked: list[dict[str, Any]] = []
    if keywords:
        mk = _matched_items_ordered_level_then_source(
            matched,
            item_registry=item_registry,
            min_level=min_level,
            max_level=max_level,
        )
        take_m = min(len(mk), max_total)
        picked.extend(mk[:take_m])
        rem = max_total - len(picked)
    else:
        rem = max_total

    if rem > 0:
        picked.extend(
            _allocate_remainder_quota_pools(rem, [synth_u, shop_u, k_u], "532")
        )

    return picked[:max_total]


# 合成产出（如菜品）单价常低于「奖励下限×10%」，此处单独用固定门槛，避免被筛掉
_SPECIAL_ITEM_SYNTHESIS_MIN_PRICE = 1000
_SPECIAL_ITEMS_MAX_COUNT = 25
# 有关键词时略抬高上限，便于列出匹配项后再按比例补足
_SPECIAL_ITEMS_MAX_COUNT_WITH_KEYWORDS = 40


def _partition_special_by_source(entries: list[dict[str, Any]]) -> tuple[list, list, list]:
    synth: list[dict[str, Any]] = []
    shop: list[dict[str, Any]] = []
    ksh: list[dict[str, Any]] = []
    for e in entries:
        src = e.get("source")
        if src == "合成":
            synth.append(e)
        elif src == "非本阵营商店":
            shop.append(e)
        elif src == "K点商店":
            ksh.append(e)
    return synth, shop, ksh


def _matched_special_items_ordered_by_source(
    matched: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """特殊物品获取：关键词命中项不按等级分档，仅按 合成 > 非本阵营商店 > K点商店；桶内随机。"""
    s, sh, k = _partition_special_by_source(matched)
    return _shuffle_each_bucket_concat([s, sh, k])


def _build_special_items(
    game_data: GameDataRegistry,
    npc_name: str,
    stage: int,
    min_level: int,
    max_level: int,
    reward_final_min: Optional[int] = None,
    requirement_keywords: Optional[list[str]] = None,
    npc_states: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """特殊物品获取类：非装备的特殊物品（合成 / 非本阵营商店 / K 点）。"""
    item_registry = game_data.items
    shop_registry = game_data.shops
    crafting_registry = game_data.crafting
    kshop_registry = game_data.kshop
    equipment_mods = game_data.equipment_mods
    keywords = _normalize_kw_list(requirement_keywords)

    equipment_types = {"武器", "防具"}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 按来源优先级依次收录，同名物品保留优先级更高来源（合成 > 非本阵营商店 > K 点）
    # 1. 合成配方产出的非装备物品
    for recipe in crafting_registry._recipes:
        if recipe.name in seen:
            continue
        item = item_registry.get_by_name(recipe.name)
        if item is None or item.type in equipment_types:
            continue
        if item.level > max_level:
            continue
        seen.add(recipe.name)
        entry: dict[str, Any] = {
            "name": item.name if item else recipe.name,
            "type": item.type if item else None,
            "price": (item.price if item else recipe.price) or 0,
            "source": "合成",
        }
        if equipment_mods.is_plugin(recipe.name):
            tier = equipment_mods.get_plugin_tier(recipe.name)
            if tier:
                entry["plugin_tier"] = tier
        result.append(entry)

    # 2. 非本阵营商店的非装备物品（排除彩蛋阵营商人的商店）
    for other_npc in shop_registry._shops:
        if other_npc == npc_name:
            continue
        if _skip_easter_egg_shop_for_cross_faction_pool(other_npc, npc_states):
            continue
        for item_name in shop_registry.get_npc_shop(other_npc):
            if item_name in seen:
                continue
            item = item_registry.get_by_name(item_name)
            if item is None or item.type in equipment_types:
                continue
            if item.level > max_level:
                continue
            seen.add(item_name)
            entry = {
                "name": item.name,
                "type": item.type,
                "price": item.price or 0,
                "source": "非本阵营商店",
            }
            if equipment_mods.is_plugin(item.name):
                tier = equipment_mods.get_plugin_tier(item.name)
                if tier:
                    entry["plugin_tier"] = tier
            result.append(entry)

    # 3. K点商店非装备物品
    kprice_cap = stage * 100
    for kitem in kshop_registry.list_items():
        if kitem.item in seen:
            continue
        item = item_registry.get_by_name(kitem.item)
        if item is not None and item.type in equipment_types:
            continue
        if item is not None and item.level > max_level:
            continue
        if kitem.price > kprice_cap:
            continue
        seen.add(kitem.item)
        result.append({
            "name": kitem.item,
            "type": item.type if item else None,
            "price": (item.price if item else 0) or 0,
            "source": "K点商店",
        })

    min_shop_k_price = 0
    if reward_final_min is not None and reward_final_min > 0:
        min_shop_k_price = int(reward_final_min * 0.1)

    def _passes_non_keyword_price(e: dict[str, Any]) -> bool:
        p = e.get("price") or 0
        if e.get("source") == "合成":
            return p >= _SPECIAL_ITEM_SYNTHESIS_MIN_PRICE
        return p >= min_shop_k_price

    max_total = (
        _SPECIAL_ITEMS_MAX_COUNT_WITH_KEYWORDS if keywords else _SPECIAL_ITEMS_MAX_COUNT
    )

    matched: list[dict[str, Any]] = []
    if keywords:
        for e in result:
            it = item_registry.get_by_name(e.get("name") or "")
            if _entry_matches_item_keywords(e, it, equipment_mods, keywords):
                matched.append(e)

    matched_names = {e.get("name") for e in matched}
    unmatched_price_ok: list[dict[str, Any]] = []
    for e in result:
        if e.get("name") in matched_names:
            continue
        if _passes_non_keyword_price(e):
            unmatched_price_ok.append(e)

    synth_u, shop_u, k_u = _partition_special_by_source(unmatched_price_ok)
    random.shuffle(synth_u)
    random.shuffle(shop_u)
    random.shuffle(k_u)

    picked: list[dict[str, Any]] = []
    if keywords:
        mk = _matched_special_items_ordered_by_source(matched)
        take_m = min(len(mk), max_total)
        picked.extend(mk[:take_m])
        rem = max_total - len(picked)
    else:
        rem = max_total

    if rem > 0:
        picked.extend(
            _allocate_remainder_quota_pools(rem, [synth_u, shop_u, k_u], "532")
        )

    return picked[:max_total]


_HOLDABLE_MAX_ITEMS = 20


def _build_holdable_items(
    game_data: GameDataRegistry,
    stage: int,
    max_level: int,
    base_max: int,
    requirement_keywords: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """物品持有类：情报类物品 + 合成配方产出。"""
    item_registry = game_data.items
    crafting_registry = game_data.crafting
    equipment_mods = game_data.equipment_mods
    keywords = _normalize_kw_list(requirement_keywords)
    price_cap = base_max * 2

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in item_registry.find(use="情报"):
        if item.name in seen:
            continue
        if item.level > max_level:
            continue
        seen.add(item.name)
        candidates.append({
            "name": item.name,
            "type": item.type,
            "price": item.price or 0,
            "source": "情报",
        })

    for recipe in crafting_registry._recipes:
        if recipe.name in seen:
            continue
        item = item_registry.get_by_name(recipe.name)
        if item and item.level > max_level:
            continue
        price = (item.price if item else recipe.price) or 0
        seen.add(recipe.name)
        entry: dict[str, Any] = {
            "name": recipe.name,
            "type": item.type if item else None,
            "price": price,
            "source": "合成",
        }
        if item and item.level > 0:
            entry["level"] = item.level
        if equipment_mods.is_plugin(recipe.name):
            tier = equipment_mods.get_plugin_tier(recipe.name)
            if tier:
                entry["plugin_tier"] = tier
        candidates.append(entry)

    if keywords:
        b0, b1, b2, b3 = [], [], [], []
        for e in candidates:
            it = item_registry.get_by_name(e.get("name") or "")
            km = _entry_matches_item_keywords(e, it, equipment_mods, keywords)
            intel = e.get("source") == "情报"
            if km and intel:
                b0.append(e)
            elif km and not intel:
                b1.append(e)
            elif not km and intel:
                b2.append(e)
            else:
                b3.append(e)
        ordered = _shuffle_each_bucket_concat([b0, b1, b2, b3])
    else:
        intel_e = [e for e in candidates if e.get("source") == "情报"]
        craft_e = [e for e in candidates if e.get("source") != "情报"]
        intel_ok = [e for e in intel_e if (e.get("price") or 0) <= price_cap]
        craft_ok = [e for e in craft_e if (e.get("price") or 0) <= price_cap]
        return _allocate_remainder_quota_pools(
            _HOLDABLE_MAX_ITEMS, [intel_ok, craft_ok], "55"
        )

    result: list[dict[str, Any]] = []
    for e in ordered:
        if len(result) >= _HOLDABLE_MAX_ITEMS:
            break
        p = e.get("price") or 0
        if p > price_cap:
            continue
        result.append(e)
    return result


def _build_stage_loot_list(
    game_data: GameDataRegistry,
    stage: int,
    requirement_keywords: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """通关并收集/通关并持有：仅有箱子掉落的关卡。"""
    stage_registry = game_data.stages
    item_registry = game_data.items
    mercenary_registry = game_data.mercenary_tasks
    equipment_mods = game_data.equipment_mods

    cfg = get_progress_stage_config(stage)
    if cfg is None:
        return []

    main_task_max_id = cfg.main_task_max_id or 0
    main_task_min_id = cfg.main_task_min_id or 0
    max_level = cfg.max_level or 50

    mercs_by_stage: dict[str, list[Any]] = {}
    for mt in mercenary_registry.list_all():
        if not mt.stage_name:
            continue
        mercs_by_stage.setdefault(mt.stage_name, []).append(mt)

    result: list[dict[str, Any]] = []

    for (area, name), si in stage_registry._stage_infos.items():
        if si.unlock_condition is None:
            continue

        is_dungeon = area == "副本任务"
        if is_dungeon:
            merc_tasks = mercs_by_stage.get(name) or []
            allowed_by_root = [
                mt for mt in merc_tasks
                if mt.recommended_min_level is not None and mt.recommended_min_level <= max_level
            ]
            if not allowed_by_root:
                continue

            allowed_difficulties: set[str] = {"简单"}
            challenge_modes_map: dict[str, str] = {}
            challenge_hint_map: dict[str, str] = {}
            for mt in merc_tasks:
                if not mt.challenge_difficulty or mt.challenge_difficulty == "简单":
                    continue
                cmin = mt.challenge_recommended_min_level
                if cmin is None:
                    continue
                if cmin <= max_level:
                    allowed_difficulties.add(mt.challenge_difficulty)
                    if mt.challenge_description and mt.challenge_difficulty not in challenge_modes_map:
                        challenge_modes_map[mt.challenge_difficulty] = mt.challenge_description
                    if mt.challenge_difficulty not in challenge_hint_map:
                        challenge_hint_map[mt.challenge_difficulty] = (
                            mt.challenge_llm_hint or DEFAULT_CHALLENGE_LLM_HINT
                        )

            difficulties = [d for d in ("简单", "冒险", "修罗", "地狱") if d in allowed_difficulties]
            entry_challenges = [
                {
                    "difficulty": diff,
                    "description": desc,
                    "hint": challenge_hint_map.get(diff, DEFAULT_CHALLENGE_LLM_HINT),
                }
                for diff, desc in challenge_modes_map.items()
                if diff in difficulties and diff != "简单"
            ]
        else:
            if si.unlock_condition > main_task_max_id:
                continue
            difficulties = ["简单", "冒险", "修罗", "地狱"]

        crates = stage_registry.get_stage_loot(area, name)
        if not crates:
            continue

        # 汇总同一关卡下的重复掉落物，减少 prompt 长度：
        # 相同 item_name：min/max 分别求和，仅保留一条记录。
        loot_items: list[dict[str, Any]] = []
        loot_map: dict[str, dict[str, Any]] = {}
        total_loot_value = 0
        for crate in crates:
            for drop in crate.drops:
                unit_price = item_registry.get_price(drop.name) or 0
                prev = loot_map.get(drop.name)
                if prev is None:
                    entry = {
                        "item_name": drop.name,
                        "min_qty": drop.min_count,
                        "max_qty": drop.max_count,
                        "unit_price": unit_price,
                    }
                    loot_map[drop.name] = entry
                    loot_items.append(entry)  # 保持首次出现顺序
                else:
                    # unit_price 按 item_name 取值应当恒定；这里只累加数量
                    prev["min_qty"] += drop.min_count
                    prev["max_qty"] += drop.max_count
                total_loot_value += unit_price * drop.min_count

        if not loot_items:
            continue

        below_progress = (
            not is_dungeon
            and si.unlock_condition < main_task_min_id
        )

        lr = get_progress_stage_level_range(stage)
        entry: dict[str, Any] = {
            "area": area,
            "area_region_hint": stage_root_region_hint(area),
            "area_level_range": list(lr) if lr else None,
            "stage_name": name,
            "unlock_id": si.unlock_condition,
            "is_dungeon": is_dungeon,
            "difficulties": difficulties,
            "loot_items": loot_items,
            "total_loot_value": total_loot_value,
        }
        if below_progress:
            entry["below_progress"] = True
        if is_dungeon:
            entry["recommended_level"] = min(
                mt.recommended_min_level
                for mt in merc_tasks
                if mt.recommended_min_level is not None
            )
            if entry_challenges:
                entry["challenge_modes"] = entry_challenges

        result.append(entry)

    kw = _normalize_kw_list(requirement_keywords)
    if not kw:
        in_prog = [r for r in result if not r.get("below_progress")]
        below = [r for r in result if r.get("below_progress")]
        result = _shuffle_each_bucket_concat([in_prog, below])
    else:
        b0, b1, b2, b3 = [], [], [], []
        for row in result:
            kh = _stage_loot_row_matches_keywords(row, kw, item_registry, equipment_mods)
            bp = bool(row.get("below_progress"))
            if kh and not bp:
                b0.append(row)
            elif kh and bp:
                b1.append(row)
            elif not kh and not bp:
                b2.append(row)
            else:
                b3.append(row)
        result = _shuffle_each_bucket_concat([b0, b1, b2, b3])

    return result[:20]


# 每种在数据侧通常唯一对应条目；若玩家勾选了该奖励类型，截断列表中优先保证出现。
SINGLETON_REWARD_LABELS: frozenset[str] = frozenset({
    "金币", "经验值", "K点", "技能点", "强化石",
})


def _sort_candidates_by_task_tier_shuffle(
    items: list[dict[str, Any]],
    *,
    reward_task_tier_by_name: dict[str, int],
) -> list[dict[str, Any]]:
    """任务奖励池侧：tier 1→4→其余；同 tier 内随机（商店物品多为 tier 5，落在最后一桶）。"""
    t1: list[dict[str, Any]] = []
    t2: list[dict[str, Any]] = []
    t3: list[dict[str, Any]] = []
    t4: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for c in items:
        tr = reward_task_tier_by_name.get(c.get("name") or "", 5)
        if tr <= 1:
            t1.append(c)
        elif tr == 2:
            t2.append(c)
        elif tr == 3:
            t3.append(c)
        elif tr == 4:
            t4.append(c)
        else:
            rest.append(c)
    return _shuffle_each_bucket_concat([t1, t2, t3, t4, rest])


def _pin_requested_singleton_candidates(
    candidates: list[dict[str, Any]],
    selected_reward_types: list[str],
    item_registry: Any,
    equipment_mods: Any,
) -> tuple[list[dict[str, Any]], set[str]]:
    """按勾选顺序为每种「单例」奖励类型至多保留 1 条；优先 NPC 商店来源。"""
    pinned: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for t in selected_reward_types:
        if t not in SINGLETON_REWARD_LABELS:
            continue
        matches: list[dict[str, Any]] = []
        for c in candidates:
            name = c.get("name") or ""
            if not name or name in seen_names:
                continue
            it = item_registry.get_by_name(name)
            if it is None:
                continue
            if _matches_reward_type(it, t, equipment_mods):
                matches.append(c)
        if not matches:
            continue
        matches.sort(key=lambda x: 0 if x.get("source") == "本NPC商店" else 1)
        best = matches[0]
        nm = best.get("name") or ""
        if nm:
            seen_names.add(nm)
        pinned.append(best)
    return pinned, seen_names


def _proportional_pick_one_source(
    pool: list[dict[str, Any]],
    non_sing_types: list[str],
    n: int,
    item_registry: Any,
    equipment_mods: Any,
    reward_task_tier_by_name: dict[str, int],
) -> list[dict[str, Any]]:
    """
    单侧来源（仅商店或仅任务池）：在非单例类型间轮询均分名额；
    每类型内部按任务 tier 分桶后桶内随机；无法归入类型的条目最后补齐。
    """
    if n <= 0 or not pool:
        return []
    if not non_sing_types:
        ordered = _sort_candidates_by_task_tier_shuffle(
            pool, reward_task_tier_by_name=reward_task_tier_by_name,
        )
        return ordered[:n]

    groups: dict[str, list[dict[str, Any]]] = {t: [] for t in non_sing_types}
    orphans: list[dict[str, Any]] = []
    for c in pool:
        it = item_registry.get_by_name(c.get("name") or "")
        placed = False
        if it is not None:
            for t in non_sing_types:
                if _matches_reward_type(it, t, equipment_mods):
                    groups[t].append(c)
                    placed = True
                    break
        if not placed:
            orphans.append(c)

    for t in non_sing_types:
        groups[t] = _sort_candidates_by_task_tier_shuffle(
            groups[t], reward_task_tier_by_name=reward_task_tier_by_name,
        )
    orphans = _sort_candidates_by_task_tier_shuffle(
        orphans, reward_task_tier_by_name=reward_task_tier_by_name,
    )
    non_empty = [t for t in non_sing_types if groups[t]]
    picked: list[dict[str, Any]] = []
    if non_empty:
        idxs = {t: 0 for t in non_empty}
        while len(picked) < n:
            progressed = False
            for t in non_empty:
                if len(picked) >= n:
                    break
                arr = groups[t]
                i = idxs[t]
                if i < len(arr):
                    picked.append(arr[i])
                    idxs[t] = i + 1
                    progressed = True
            if not progressed:
                break
    if len(picked) < n and orphans:
        picked.extend(orphans[: n - len(picked)])
    return picked[:n]


def _pick_layer_shop_task_82_proportional(
    items: list[dict[str, Any]],
    n: int,
    non_sing_types: list[str],
    item_registry: Any,
    equipment_mods: Any,
    reward_task_tier_by_name: dict[str, int],
) -> list[dict[str, Any]]:
    """
    某一档内（全量池 / 关键词命中池 / 关键词未命中池）：商店 : 任务池 = 8 : 2
   （与 _allocate_remainder_quota_pools mode「82」同一口径：低档 round(n×20%)，余数给商店侧）；
    各侧再按类型轮询 + tier 内随机；名额不足时从另一侧与剩余项按 tier 顺序补齐。
    """
    if n <= 0 or not items:
        return []
    shop = [c for c in items if c.get("source") == "本NPC商店"]
    task = [c for c in items if c.get("source") != "本NPC商店"]
    q_task = int(round(n * 2 / 10))
    q_shop = n - q_task
    ps = _proportional_pick_one_source(
        shop, non_sing_types, q_shop,
        item_registry, equipment_mods, reward_task_tier_by_name,
    )
    pt = _proportional_pick_one_source(
        task, non_sing_types, q_task,
        item_registry, equipment_mods, reward_task_tier_by_name,
    )
    picked = ps + pt
    seen: set[str] = {x.get("name") or "" for x in picked if x.get("name")}
    if len(picked) < n:
        rest = [
            c for c in items
            if (c.get("name") or "") not in seen
        ]
        rest = _sort_candidates_by_task_tier_shuffle(
            rest, reward_task_tier_by_name=reward_task_tier_by_name,
        )
        for c in rest:
            if len(picked) >= n:
                break
            nm = c.get("name") or ""
            if nm and nm in seen:
                continue
            picked.append(c)
            if nm:
                seen.add(nm)
    return picked[:n]


def _finalize_reward_item_candidates(
    candidates: list[dict[str, Any]],
    *,
    item_registry: Any,
    equipment_mods: Any,
    reward_keywords: Optional[list[str]],
    reward_task_tier_by_name: dict[str, int],
    max_n: int = 20,
    selected_reward_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    截断优先级（从高到低）：
    1) 勾选的单例类型（金币/经验值/K点/技能点/强化石）各至多 1 条且优先商店——排在一切关键词之前，
       因勾选即表示玩家/NPC 明确需要；
    2) 剩余名额：有关键词时先填「关键词命中」再填「未命中」；
    3) 上述每一层内部：商店 : 任务池 = 8 : 2，再按非单例奖励类型轮询均分，
       类型内按任务进度 tier 分桶、桶内随机；不足则跨源补齐。
    """
    kw = _normalize_kw_list(reward_keywords)
    sel = list(selected_reward_types or [])
    pinned, pinned_names = _pin_requested_singleton_candidates(
        candidates, sel, item_registry, equipment_mods,
    )
    pool = [c for c in candidates if (c.get("name") or "") not in pinned_names]
    budget = max(0, max_n - len(pinned))
    non_sing = [t for t in sel if t not in SINGLETON_REWARD_LABELS]

    if budget <= 0:
        return pinned[:max_n]

    if not kw:
        rest = _pick_layer_shop_task_82_proportional(
            pool, budget, non_sing,
            item_registry, equipment_mods, reward_task_tier_by_name,
        )
        return (pinned + rest)[:max_n]

    # 关键词分层在单例占位之后；命中池、未命中池各自内部仍为 8:2 商店优先 + 类型轮询。
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for c in pool:
        it = item_registry.get_by_name(c.get("name") or "")
        if _entry_matches_item_keywords(c, it, equipment_mods, kw):
            matched.append(c)
        else:
            unmatched.append(c)

    take_kw = min(budget, len(matched))
    p_kw = _pick_layer_shop_task_82_proportional(
        matched, take_kw, non_sing,
        item_registry, equipment_mods, reward_task_tier_by_name,
    )
    budget -= len(p_kw)
    p_nkw: list[dict[str, Any]] = []
    if budget > 0 and unmatched:
        p_nkw = _pick_layer_shop_task_82_proportional(
            unmatched, budget, non_sing,
            item_registry, equipment_mods, reward_task_tier_by_name,
        )
    return (pinned + p_kw + p_nkw)[:max_n]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def prepare_task_context(
    *,
    task_type: str,
    reward_types: dict[str, list[str]],
    npc_name: str,
    npc_faction: str = "",
    npc_challenge: Optional[str] = None,
    player_progress: int = 1,
    npc_affinity: int = 0,
    npc_states: Optional[dict[str, Any]] = None,
    requirement_keywords: Optional[list[str]] = None,
    reward_keywords: Optional[list[str]] = None,
    game_data: Optional[GameDataRegistry] = None,
) -> str:
    """
    prepare_task_context 工具执行器。

    返回 JSON 字符串，包含该任务类型所需的全部筛选后数据与规则说明。
    """
    if game_data is None:
        game_data = get_game_data_registry()

    stage = max(1, min(7, player_progress))
    cfg = get_progress_stage_config(stage)

    level_range = get_progress_stage_level_range(stage) or (1, 50)
    main_task_range = get_progress_stage_main_task_range(stage) or (0, 77)
    max_level = level_range[1]

    # 通用字段
    reward_budget = _compute_reward_budget(
        stage=stage, task_type=task_type, affinity=npc_affinity,
    )

    existing_tasks: list[dict[str, Any]] = []
    # 注释掉已存在任务
    # for t in game_data.tasks.list_by_npc(npc_name):
    #     existing_tasks.append({
    #         "id": t.id,
    #         "title": t.title,
    #         "type": t.chain or "",
    #     })

    reward_item_candidates = _build_reward_item_candidates(
        reward_types=reward_types,
        game_data=game_data,
        npc_name=npc_name,
        stage=stage,
        min_level=level_range[0],
        max_level=max_level,
        main_task_range=(main_task_range[0], main_task_range[1]),
        reward_keywords=reward_keywords,
    )

    task_rules = _TASK_RULES.get(task_type, "")

    context: dict[str, Any] = {
        "level_range": list(level_range),
        "main_task_range": list(main_task_range),
        "reward_budget": reward_budget,
        "existing_tasks": existing_tasks[:15],
        "reward_item_candidates": reward_item_candidates,
        "task_rules": task_rules,
    }

    if len(existing_tasks) > 15:
        context["truncated"] = True

    # 类型专属字段
    if task_type in ("问候", "传话"):
        context["npc_list"] = _build_npc_list(
            game_data,
            npc_states or {},
            npc_name,
            requirement_keywords=requirement_keywords,
        )

    elif task_type in ("通关", "清理", "挑战"):
        context["stage_list"] = _get_all_stages_for_progress(
            game_data, stage, requirement_keywords=requirement_keywords,
        )

    elif task_type == "切磋":
        context["challenge_targets"] = _build_challenge_targets(
            game_data, npc_name, npc_challenge, stage,
        )

    elif task_type == "资源收集":
        base_max = stage * REWARD_STAGE_BASE_MAX
        context["collectable_items"] = _build_collectable_items(
            game_data,
            stage,
            max_level,
            base_max,
            requirement_keywords=requirement_keywords,
        )

    elif task_type == "装备缴纳":
        context["equipment_items"] = _build_equipment_items(
            game_data,
            npc_name,
            npc_faction,
            stage,
            level_range[0],
            max_level,
            requirement_keywords=requirement_keywords,
            npc_states=npc_states,
        )

    elif task_type == "特殊物品获取":
        context["special_items"] = _build_special_items(
            game_data,
            npc_name,
            stage,
            level_range[0],
            max_level,
            reward_final_min=reward_budget.get("final_min"),
            requirement_keywords=requirement_keywords,
            npc_states=npc_states,
        )

    elif task_type == "物品持有":
        base_max = stage * REWARD_STAGE_BASE_MAX
        context["holdable_items"] = _build_holdable_items(
            game_data,
            stage,
            max_level,
            base_max,
            requirement_keywords=requirement_keywords,
        )

    elif task_type in ("通关并收集", "通关并持有"):
        context["stage_loot_list"] = _build_stage_loot_list(
            game_data, stage, requirement_keywords=requirement_keywords,
        )

    if task_type in ("资源收集", "装备缴纳", "特殊物品获取", "通关并收集"):
        context["submit_vs_reward_hint"] = (
            "拟定草案时 `finish_submit_items` 与 `rewards` 不得出现相同物品；玩家需要的物品要放到rewards里。"
        )

    return json.dumps(context, ensure_ascii=False)
