"""
prepare_task_context 工具执行器。

根据 task_type 和 reward_types 一次性返回该类型所需的全部筛选后数据与规则说明，
减少 LLM 的决策负担（文档 6.3.2）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from services.game_data.registry import GameDataRegistry, get_game_data_registry
from services.game_data.reward_utils import parse_name_count
from services.game_progress import (
    get_progress_stage_config,
    get_progress_stage_level_range,
    get_progress_stage_main_task_range,
    PROGRESS_STAGE_CONFIG,
)

# 当 challenge 无 llm_hint 时使用的默认提醒（仅在选择该难度时随 challenge_modes 返回，按需占用 token）
DEFAULT_CHALLENGE_LLM_HINT = "若选择此难度，请在任务介绍中明确说明当前难度，并在对话中提醒具有挑战性。"


# ---------------------------------------------------------------------------
# 任务类型规则说明模板
# ---------------------------------------------------------------------------

_TASK_RULES: dict[str, str] = {
    "问候": (
        "问候/闲聊类任务：最简单的任务类型，玩家只需与指定NPC对话即可完成。"
        "可选当前NPC自己或其他NPC为完成NPC。奖励较少，以金币为主。"
        "适合低好感度或初次对话时使用。"
    ),
    "传话": (
        "传话类任务：要求玩家去找另一个NPC对话。"
        "在有理由的情况下，可选择任意NPC作为完成NPC，包括同阵营传话、不同阵营的外交等。"
        "奖励较少，以金币为主。适合推动NPC间剧情互动。"
    ),
    "通关": (
        "通关类任务：要求玩家通关指定关卡。"
        "关卡必须在玩家当前进度范围内（解锁ID ≤ 当前主线ID）。"
        "副本关卡通常仅可选择“简单”。只有当副本配置了 challenge 额外难度（且满足玩家等级校验）时，才允许选择该额外难度（特征是仅有两个难度候选项），选择时需明确提醒难度要求；否则只能选择“简单”。"
        "地图关卡可选任意难度。"
        "基础奖励 ×2。"
    ),
    "清理": (
        "清理类任务：与通关类似，但叙事上侧重'清除威胁/清理区域'。"
        "规则与通关类相同。基础奖励 ×2。"
    ),
    "挑战": (
        "挑战类任务：高难度通关，建议选择修罗或地狱难度。"
        "奖励中经验占比应 ≥ 50%，金币占比可降低。基础奖励 ×2。"
        "适合高进度、高好感度的玩家。"
    ),
    "切磋": (
        "切磋类任务：要求玩家通关当前NPC配置的专属切磋关卡。"
        "必须使用当前NPC的challenge属性对应的关卡。"
        "基础奖励 ×2。"
    ),
    "资源收集": (
        "资源收集类任务：要求玩家收集并提交指定数量的食材/药剂/材料/弹夹。"
        "提交物品必须在现有任务的提交物品+奖励物品池中。"
        "提交品总价值不超过基础奖励的200%。"
        "奖励额外增加提交品价值的1.5~2倍。"
    ),
    "装备缴纳": (
        "装备缴纳类任务：要求玩家获取并提交一件装备（武器/防具）。"
        "来源：非本阵营NPC商店、合成配方、K点商店。"
        "提交品总价值不超过基础奖励的300%。"
        "奖励额外增加提交品价值的1.5~2倍。"
    ),
    "特殊物品获取": (
        "特殊物品获取类任务：要求玩家获取并提交一个特殊物品。"
        "通常只需要1个物品。来源同装备缴纳类。"
        "包括插件、药剂、菜品、贵重消耗品等非装备物品。"
    ),
    "物品持有": (
        "物品持有类任务：要求玩家持有（不提交）指定物品。"
        "来源：情报类物品或合成配方产出。"
        "奖励额外增加持有品价值的0.5倍，上限不超过基础奖励的50%。"
        "持有品本身总价值上限为基础奖励的200%。"
    ),
    "通关并收集": (
        "通关并收集类任务：组合通关+收集要求。"
        "收集物品必须是该关卡箱子的产出物品。"
        "收集数量建议使用箱子的最小产出数量。"
        "基础奖励 ×2，再叠加收集品加成。"
        "如果你在 finish_requirements 里选择了非“简单”的副本难度（例如“地狱”），请务必在任务说明（title/description）中明显提醒玩家正在选择挑战模式，并在接取/完成台词里明确提到要挑战该高难度模式。"
    ),
    "通关并持有": (
        "通关并持有类任务：组合通关+持有要求。"
        "持有物品必须是该关卡箱子的产出物品。"
        "基础奖励 ×2，再叠加持有品加成。"
        "如果你在 finish_requirements 里选择了非“简单”的副本难度（例如“地狱”），请务必在任务说明（title/description）中明显提醒玩家正在选择挑战模式，并在接取/完成台词里明确提到要挑战该高难度模式。"
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
    base_min = stage * 10000
    base_max = stage * 20000
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

    return {
        "base_min": base_min,
        "base_max": base_max,
        "multiplier": multiplier,
        "affinity_modifier": aff_mod,
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
    """
    if reward_type == "插件":
        return equipment_mods.is_plugin(item.name)

    if item.name == reward_type:
        return True
    if item.type and item.type == reward_type:
        return True
    if item.use and item.use == reward_type:
        return True
    # 材料：数据中常为 type=收集品、use=材料；若 use 未标或归一化，用 type 收集品 兜底
    if reward_type == "材料" and item.type and item.type == "收集品":
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

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    # 来源 2 优先：当前NPC商店物品
    for shop_item_name in npc_shop_items:
        if shop_item_name in seen:
            continue
        item = item_registry.get_by_name(shop_item_name)
        if item is None:
            continue
        is_weapon_or_armor = item.type in ("武器", "防具")
        is_grenade = bool(item.use) and item.use == "手雷"
        if is_weapon_or_armor or is_grenade:
            if item.level <= min_level or item.level > max_level:
                continue
        # 所有商店物品都需匹配 reward_types（含「材料」匹配 item.use/type）
        matched = False
        for rt in all_types:
            if _matches_reward_type(item, rt, equipment_mods):
                matched = True
                break
        if not matched:
            continue
        entry = {
            "name": item.name,
            "type": item.type,
            "price": item.price or 0,
            "source": "NPC商店",
        }
        if item.level > 0:
            entry["level"] = item.level
        if equipment_mods.is_plugin(item.name):
            tier = equipment_mods.get_plugin_tier(item.name)
            if tier:
                entry["plugin_tier"] = tier
        candidates.append(entry)
        seen.add(shop_item_name)

    # 来源 1：任务奖励池（按当前区间优先顺序：主线区间 → 前置在区间 → mercenary 等级重合 → 其他）
    for item_name in ordered_task_item_names:
        if item_name in seen:
            continue
        item = item_registry.get_by_name(item_name)
        if item is None:
            continue
        if item.level > max_level:
            continue
        if item.type in ("武器", "防具"):
            continue
        for rt in all_types:
            if _matches_reward_type(item, rt, equipment_mods):
                entry = {
                    "name": item.name,
                    "type": item.type,
                    "price": item.price or 0,
                    "source": "任务奖励常见",
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

    return candidates[:20]


# ---------------------------------------------------------------------------
# 类型专属字段构建
# ---------------------------------------------------------------------------

def _get_all_stages_for_progress(
    game_data: GameDataRegistry,
    stage: int,
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

    return result


def _build_npc_list(
    game_data: GameDataRegistry,
    npc_states: dict[str, Any],
    current_npc: str,
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
    return result[:30]


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

    # 根据 mercenary_tasks.json 的 recommended_level 过滤：不展示推荐等级高于当前阶段上限的关卡
    matched_merc_tasks = [m for m in mercenary_registry.list_all() if m.stage_name == npc_challenge]
    if matched_merc_tasks:
        # 规则：只要存在一个条目满足 rec_min <= player_max_level（或该条目无推荐等级），则认为可用
        ok = any(
            (m.recommended_min_level is None) or (m.recommended_min_level <= max_level)
            for m in matched_merc_tasks
        )
        if not ok:
            return {"error": "当前阶段等级不满足该NPC切磋关卡的推荐等级，请选择其他任务类型"}

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


def _build_collectable_items(
    game_data: GameDataRegistry,
    stage: int,
    max_level: int,
    base_max: int,
) -> list[dict[str, Any]]:
    """资源收集类：食材/药剂/材料/弹夹，且在现有任务物品池中。"""
    item_registry = game_data.items
    task_registry = game_data.tasks
    mercenary_registry = game_data.mercenary_tasks

    allowed_uses = {"食材", "药剂", "材料", "弹夹"}
    pool = task_registry.list_submit_items() | task_registry.list_reward_item_names()
    price_cap = base_max * 2

    # 资源池按“当前区间任务优先级”排序（提交物品 + 奖励物品都参与）
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

    result: list[dict[str, Any]] = []
    total = 0

    def _can_use_item(it: Any) -> bool:
        return (it.use in allowed_uses) or (it.type in allowed_uses) or (it.name in allowed_uses)

    for item_name in ordered_item_names:
        if item_name in seen or item_name not in pool:
            continue
        item = item_registry.get_by_name(item_name)
        if item is None:
            continue
        if item.level > max_level:
            continue
        if not _can_use_item(item):
            continue

        price = item.price or 0
        if total + price > price_cap:
            continue
        total += price

        entry: dict[str, Any] = {
            "name": item.name,
            "type": item.type,
            "price": price,
            "min_qty": 1,
        }

        stats = submit_stats.get(item_name)
        submit_max = int(stats[1] or 0) if stats and int(stats[1] or 0) > 0 else 0
        reward_max = int(reward_stats.get(item_name, (None, 0))[1] or 0)
        # 并集原则：提交历史 max_qty 与奖励历史 max_qty 取并集上限
        effective_max = max(submit_max, reward_max)
        if effective_max > 0:
            # 统一沿用 V2 的 allowed_max = effective_max * 2
            entry["max_qty"] = effective_max * 2
        else:
            # 历史统计缺失：用当前阶段奖励上限/单价估算一个“不会轻易超限”的数量上限
            unit_price = price or 1
            entry["max_qty"] = max(1, int(base_max / unit_price))

        if item.level > 0:
            entry["level"] = item.level

        result.append(entry)
        seen.add(item_name)
        if len(result) >= 20:
            break

    # 若区间优先池不足，再补齐剩余物品（按名称稳定排序）
    if len(result) < 20:
        remaining = sorted(pool - seen)
        for item_name in remaining:
            if len(result) >= 20:
                break
            item = item_registry.get_by_name(item_name)
            if item is None or item.level > max_level:
                continue
            if not _can_use_item(item):
                continue
            price = item.price or 0
            if total + price > price_cap:
                continue

            stats = submit_stats.get(item_name)
            submit_max = int(stats[1] or 0) if stats and int(stats[1] or 0) > 0 else 0
            reward_max = int(reward_stats.get(item_name, (None, 0))[1] or 0)
            effective_max = max(submit_max, reward_max)
            max_qty = effective_max * 2 if effective_max > 0 else max(1, int(base_max / (price or 1)))

            result.append({
                "name": item.name,
                "type": item.type,
                "price": price,
                "min_qty": 1,
                "max_qty": max_qty,
                **({"level": item.level} if item.level > 0 else {}),
            })
            total += price
    return result


def _build_equipment_items(
    game_data: GameDataRegistry,
    npc_name: str,
    npc_faction: str,
    stage: int,
    max_level: int,
) -> list[dict[str, Any]]:
    """装备缴纳类：非本阵营商店+合成+K点商店的装备。"""
    item_registry = game_data.items
    shop_registry = game_data.shops
    crafting_registry = game_data.crafting
    kshop_registry = game_data.kshop

    equipment_types = {"武器", "防具"}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 1. 非本阵营NPC商店的装备
    # (简化：遍历所有商店，排除当前NPC)
    for other_npc in shop_registry._shops:
        if other_npc == npc_name:
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

    # 2. 合成配方产出的装备
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

    # 3. K点商店装备（筛选价格合理的）
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

    return result[:20]


def _build_special_items(
    game_data: GameDataRegistry,
    npc_name: str,
    stage: int,
    max_level: int,
) -> list[dict[str, Any]]:
    """特殊物品获取类：非装备的特殊物品。"""
    item_registry = game_data.items
    shop_registry = game_data.shops
    crafting_registry = game_data.crafting
    kshop_registry = game_data.kshop
    equipment_mods = game_data.equipment_mods

    equipment_types = {"武器", "防具"}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 1. 非本阵营商店的非装备物品
    for other_npc in shop_registry._shops:
        if other_npc == npc_name:
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
            entry: dict[str, Any] = {
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

    # 2. 合成配方产出的非装备物品
    for recipe in crafting_registry._recipes:
        if recipe.name in seen:
            continue
        item = item_registry.get_by_name(recipe.name)
        if item is None or item.type in equipment_types:
            continue
        if item.level > max_level:
            continue
        seen.add(recipe.name)
        entry = {
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

    return result[:20]


def _build_holdable_items(
    game_data: GameDataRegistry,
    stage: int,
    max_level: int,
    base_max: int,
) -> list[dict[str, Any]]:
    """物品持有类：情报类物品 + 合成配方产出。"""
    item_registry = game_data.items
    crafting_registry = game_data.crafting
    equipment_mods = game_data.equipment_mods
    price_cap = base_max * 2

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0

    # 1. 情报类物品
    for item in item_registry.find(use="情报"):
        if item.name in seen:
            continue
        if item.level > max_level:
            continue
        price = item.price or 0
        if total + price > price_cap:
            continue
        total += price
        seen.add(item.name)
        result.append({
            "name": item.name,
            "type": item.type,
            "price": price,
            "source": "情报",
        })

    # 2. 合成配方产出
    for recipe in crafting_registry._recipes:
        if recipe.name in seen:
            continue
        item = item_registry.get_by_name(recipe.name)
        if item and item.level > max_level:
            continue
        price = (item.price if item else recipe.price) or 0
        if total + price > price_cap:
            continue
        total += price
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
        result.append(entry)
        if len(result) >= 20:
            break

    return result


def _build_stage_loot_list(
    game_data: GameDataRegistry,
    stage: int,
) -> list[dict[str, Any]]:
    """通关并收集/通关并持有：仅有箱子掉落的关卡。"""
    stage_registry = game_data.stages
    item_registry = game_data.items
    mercenary_registry = game_data.mercenary_tasks

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

        loot_items: list[dict[str, Any]] = []
        total_loot_value = 0
        for crate in crates:
            for drop in crate.drops:
                unit_price = item_registry.get_price(drop.name)
                loot_items.append({
                    "item_name": drop.name,
                    "min_qty": drop.min_count,
                    "max_qty": drop.max_count,
                    "unit_price": unit_price,
                })
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

    return result[:20]


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
    game_data: Optional[GameDataRegistry] = None,
) -> str:
    """
    prepare_task_context 工具执行器。

    返回 JSON 字符串，包含该任务类型所需的全部筛选后数据与规则说明。
    """
    if game_data is None:
        game_data = get_game_data_registry()

    stage = max(1, min(6, player_progress))
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
            game_data, npc_states or {}, npc_name,
        )

    elif task_type in ("通关", "清理", "挑战"):
        context["stage_list"] = _get_all_stages_for_progress(game_data, stage)

    elif task_type == "切磋":
        context["challenge_targets"] = _build_challenge_targets(
            game_data, npc_name, npc_challenge, stage,
        )

    elif task_type == "资源收集":
        base_max = stage * 20000
        context["collectable_items"] = _build_collectable_items(
            game_data, stage, max_level, base_max,
        )

    elif task_type == "装备缴纳":
        context["equipment_items"] = _build_equipment_items(
            game_data, npc_name, npc_faction, stage, max_level,
        )

    elif task_type == "特殊物品获取":
        context["special_items"] = _build_special_items(
            game_data, npc_name, stage, max_level,
        )

    elif task_type == "物品持有":
        base_max = stage * 20000
        context["holdable_items"] = _build_holdable_items(
            game_data, stage, max_level, base_max,
        )

    elif task_type in ("通关并收集", "通关并持有"):
        context["stage_loot_list"] = _build_stage_loot_list(game_data, stage)

    return json.dumps(context, ensure_ascii=False)
