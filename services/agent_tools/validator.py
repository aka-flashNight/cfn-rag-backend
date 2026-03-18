from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING

from services.game_progress import VALID_STAGE_ROOTS

if TYPE_CHECKING:
    from services.game_data.registry import GameDataRegistry


@dataclass(frozen=True)
class DraftValidationContext:
    """
    校验所需的"玩家当前进度信息"。

    - V4 依赖 main_task_max_id
    - V7 依赖 stage / affinity
    - V8 依赖 npc_name（用于检查 NPC 商店物品）
    - V10 依赖 max_level
    """

    main_task_max_id: int
    max_level: int
    stage: int = 1
    affinity: int = 0
    npc_name: Optional[str] = None
    # 讨价还价阶段上限放大倍数（Phase 4 传入），默认为 1.0 表示不放大。
    # 文档 V7 中默认不含此项；此项为兼容后续讨价还价扩展。
    bargain_rate: float = 1.0


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

COMBAT_TASK_TYPES: frozenset[str] = frozenset({
    "通关", "清理", "挑战", "切磋", "通关并收集", "通关并持有",
})

EQUIPMENT_TYPES: frozenset[str] = frozenset({"武器", "防具"})


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def _as_list(v: Any) -> list[Any]:
    if isinstance(v, list):
        return v
    return []


def _reward_item_iter(draft: Mapping[str, Any], key: str) -> Iterable[Mapping[str, Any]]:
    for it in _as_list(draft.get(key)):
        if isinstance(it, dict):
            yield it


def _stage_requirement_iter(draft: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for sr in _as_list(draft.get("finish_requirements")):
        if isinstance(sr, dict):
            yield sr


def _compute_items_value(
    items: Iterable[Mapping[str, Any]], item_registry: Any,
) -> int:
    """计算结构化物品列表 [{item_name, count}, ...] 的总价值。"""
    total = 0
    for it in items:
        name = it.get("item_name")
        count = it.get("count", 0)
        if isinstance(name, str) and name.strip():
            try:
                n = int(count)
            except (TypeError, ValueError):
                n = 0
            price = item_registry.get_price(name)
            total += price * max(n, 0)
    return total


def _compute_reward_value_range(
    *,
    stage: int,
    task_type: str,
    submit_value: int,
    contain_value: int,
    affinity: int,
    bargain_rate: float = 1.0,
) -> tuple[int, int]:
    """
    按文档 6.4.2 V7 公式计算奖励总价值允许区间 (final_min, final_max)。

    bargain_rate：讨价还价上限放大倍数（只影响 final_max）。
    """
    base_min = stage * 10000
    base_max = stage * 20000

    type_mult = 2 if task_type in COMBAT_TASK_TYPES else 1

    mult_min = base_min * type_mult
    mult_max = base_max * type_mult

    # ---- 提交品加成 ----
    # 提交品总价值上限 = 基础奖励 × 200%
    submit_cap_lo = mult_min * 2.0
    submit_cap_hi = mult_max * 2.0
    capped_submit_lo = min(submit_value, submit_cap_lo)
    capped_submit_hi = min(submit_value, submit_cap_hi)
    # 额外 += capped × 1.5（下限）~ 2.0（上限）
    submit_bonus_lo = capped_submit_lo * 1.5
    submit_bonus_hi = capped_submit_hi * 2.0

    # ---- 持有品加成 ----
    # 持有品总价值上限 = 基础奖励 × 200%
    contain_val_cap_lo = mult_min * 2.0
    contain_val_cap_hi = mult_max * 2.0
    capped_contain_lo = min(contain_value, contain_val_cap_lo)
    capped_contain_hi = min(contain_value, contain_val_cap_hi)
    # bonus = capped × 0.5, bonus 上限 = 基础奖励 × 50%
    contain_bonus_cap_lo = mult_min * 0.5
    contain_bonus_cap_hi = mult_max * 0.5
    contain_bonus_lo = min(capped_contain_lo * 0.5, contain_bonus_cap_lo)
    contain_bonus_hi = min(capped_contain_hi * 0.5, contain_bonus_cap_hi)

    # ---- 好感度修正 ----
    if affinity >= 80:
        aff = 1.20
    elif affinity >= 50:
        aff = 1.10
    elif affinity >= 20:
        aff = 1.00
    else:
        aff = 0.90

    final_min = int((mult_min + submit_bonus_lo + contain_bonus_lo) * aff)
    final_max = int((mult_max + submit_bonus_hi + contain_bonus_hi) * aff)

    # 讨价还价修正：只放大上限，避免放大导致下限被随意改变。
    try:
        br = float(bargain_rate)
    except Exception:
        br = 1.0
    # 文档：±0%~+50% -> 仅允许在 [1.0, 1.5] 范围内扩展
    br = max(1.0, min(br, 1.5))
    final_max = int(final_max * br)

    return final_min, final_max


# =========================================================================
# V1: 物品存在性
# =========================================================================

def _validate_v1_item_existence(
    *,
    draft: Mapping[str, Any],
    item_registry: Any,
    keys: tuple[str, ...] = ("rewards", "finish_submit_items", "finish_contain_items"),
) -> Optional[dict[str, Any]]:
    missing: set[str] = set()

    for k in keys:
        for it in _reward_item_iter(draft, k):
            item_name = it.get("item_name")
            if not isinstance(item_name, str) or not item_name.strip():
                continue
            if item_registry.get_by_name(item_name) is None:
                missing.add(item_name)

    if missing:
        return {
            "step": "V1",
            "error": "物品不存在",
            "missing_item_names": sorted(missing),
        }
    return None


# =========================================================================
# V2: 物品数量合理性
# =========================================================================

def _validate_v2_item_quantity_reasonableness(
    *,
    draft: Mapping[str, Any],
    task_registry: Any,
    keys: tuple[str, ...] = ("rewards", "finish_submit_items", "finish_contain_items"),
) -> Optional[dict[str, Any]]:
    reward_stats: dict[str, tuple[int, int]] = {}
    submit_stats: dict[str, tuple[int, int]] = {}
    contain_stats: dict[str, tuple[int, int]] = {}
    try:
        reward_stats = task_registry.get_reward_stats()
    except Exception:
        reward_stats = {}

    try:
        submit_stats = task_registry.get_submit_stats()
    except Exception:
        submit_stats = {}

    try:
        contain_stats = task_registry.get_contain_stats()
    except Exception:
        contain_stats = {}

    over: list[dict[str, Any]] = []

    for k in keys:
        for it in _reward_item_iter(draft, k):
            item_name = it.get("item_name")
            count = it.get("count")
            if not isinstance(item_name, str) or not item_name.strip():
                continue
            try:
                n = int(count)
            except Exception:
                n = -999999

            # 物品数量基准来源不同：
            # - rewards：用奖励历史数量统计
            # - finish_submit_items：用提交历史数量统计
            # - finish_contain_items：用持有历史数量统计
            if k == "rewards":
                stats = reward_stats
            elif k == "finish_submit_items":
                stats = submit_stats
            elif k == "finish_contain_items":
                stats = contain_stats
            else:
                stats = reward_stats

            _, max_qty = stats.get(item_name, (None, 0))  # type: ignore[assignment]
            # 若 submit/hold 的统计不存在（max_qty=0），但该物品在奖励历史中存在，
            # 则回退到 rewards 的统计，避免出现 allowed_range=[1,0] 这种“天然不可能”区间。
            if int(max_qty or 0) == 0:
                _, reward_max_qty = reward_stats.get(item_name, (None, 0))  # type: ignore[assignment]
                if int(reward_max_qty or 0) > 0:
                    max_qty = reward_max_qty
            allowed_max = int(max_qty) * 2
            allowed_min = 1

            if n < allowed_min or n > allowed_max:
                over.append(
                    {
                        "item_name": item_name,
                        "count": n,
                        "allowed_range": [allowed_min, allowed_max],
                    }
                )

    if over:
        return {
            "step": "V2",
            "error": "物品数量不合理",
            "quantity_issues": over,
        }
    return None


# =========================================================================
# V3: 关卡存在性与解锁
# =========================================================================

def _validate_v3_stage_existence_and_area(
    *,
    draft: Mapping[str, Any],
    stage_registry: Any,
) -> Optional[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for sr in _stage_requirement_iter(draft):
        stage_area = sr.get("stage_area")
        stage_name = sr.get("stage_name")
        if not isinstance(stage_area, str) or not stage_area.strip():
            continue
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue

        if stage_area not in VALID_STAGE_ROOTS:
            invalid.append({"stage_area": stage_area, "stage_name": stage_name, "reason": "无效大区"})
            continue

        unlock = stage_registry.get_unlock_condition(stage_area, stage_name)
        if not isinstance(unlock, int) or unlock <= 0:
            invalid.append({"stage_area": stage_area, "stage_name": stage_name, "reason": "关卡无效或缺少解锁条件"})

    if invalid:
        return {
            "step": "V3",
            "error": "关卡存在性与解锁",
            "invalid_stages": invalid,
        }
    return None


# =========================================================================
# V4: 关卡解锁条件匹配
# =========================================================================

def _validate_v4_stage_unlock_condition(
    *,
    draft: Mapping[str, Any],
    stage_registry: Any,
    main_task_max_id: int,
) -> Optional[dict[str, Any]]:
    over: list[dict[str, Any]] = []
    for sr in _stage_requirement_iter(draft):
        stage_area = sr.get("stage_area")
        stage_name = sr.get("stage_name")
        if not isinstance(stage_area, str) or not stage_area.strip():
            continue
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue

        unlock_id = stage_registry.get_unlock_condition(stage_area, stage_name)
        if int(unlock_id) > int(main_task_max_id):
            over.append(
                {
                    "stage_area": stage_area,
                    "stage_name": stage_name,
                    "unlock_id": int(unlock_id),
                    "main_task_max_id": int(main_task_max_id),
                }
            )

    if over:
        return {
            "step": "V4",
            "error": "关卡解锁条件匹配失败（超进度）",
            "over_progress_stages": over,
        }
    return None


# =========================================================================
# V5: 副本关卡难度
# =========================================================================

def _validate_v5_replica_stage_difficulty(
    *,
    draft: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for sr in _stage_requirement_iter(draft):
        stage_area = sr.get("stage_area")
        stage_name = sr.get("stage_name")
        difficulty = sr.get("difficulty")
        if not isinstance(stage_area, str) or not stage_area.strip():
            continue
        if stage_area != "副本任务":
            continue
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue
        if not isinstance(difficulty, str) or not difficulty.strip():
            continue
        if difficulty != "普通":
            invalid.append(
                {
                    "stage_area": stage_area,
                    "stage_name": stage_name,
                    "difficulty": difficulty,
                    "expected": "普通",
                }
            )

    if invalid:
        return {"step": "V5", "error": "副本关卡难度违规", "invalid_replica_stages": invalid}
    return None


# =========================================================================
# V6: 前置任务合法性
# =========================================================================

def _validate_v6_precondition_tasks(
    *,
    draft: Mapping[str, Any],
    task_registry: Any,
) -> Optional[dict[str, Any]]:
    ids = _as_list(draft.get("get_requirements"))
    invalid: list[int] = []
    for x in ids:
        try:
            tid = int(x)
        except Exception:
            continue
        if tid == -1:
            invalid.append(tid)
            continue
        if task_registry.get_by_id(tid) is None:
            invalid.append(tid)

    if invalid:
        return {"step": "V6", "error": "前置任务合法性失败", "invalid_precondition_ids": sorted(set(invalid))}
    return None


# =========================================================================
# V7: 奖励总价值
# =========================================================================

def _validate_v7_reward_total_value(
    *,
    draft: Mapping[str, Any],
    item_registry: Any,
    stage: int,
    task_type: str,
    affinity: int,
    bargain_rate: float = 1.0,
) -> Optional[dict[str, Any]]:
    reward_items = list(_reward_item_iter(draft, "rewards"))
    rewards_value = _compute_items_value(reward_items, item_registry)
    submit_value = _compute_items_value(
        _reward_item_iter(draft, "finish_submit_items"), item_registry,
    )
    contain_value = _compute_items_value(
        _reward_item_iter(draft, "finish_contain_items"), item_registry,
    )

    range_min, range_max = _compute_reward_value_range(
        stage=stage,
        task_type=task_type,
        submit_value=submit_value,
        contain_value=contain_value,
        affinity=affinity,
        bargain_rate=bargain_rate,
    )

    if range_min <= rewards_value <= range_max:
        return None

    item_prices: list[dict[str, Any]] = []
    for it in reward_items:
        name = it.get("item_name", "")
        count = it.get("count", 0)
        try:
            n = int(count)
        except (TypeError, ValueError):
            n = 0
        price = item_registry.get_price(name) if isinstance(name, str) else 0
        item_prices.append({
            "item_name": name,
            "count": n,
            "unit_price": price,
            "subtotal": price * max(n, 0),
        })

    return {
        "step": "V7",
        "error": "奖励总价值超出允许区间",
        "item_prices": item_prices,
        "total_value": rewards_value,
        "allowed_range": [range_min, range_max],
    }


# =========================================================================
# V8: 奖励类型合规
# =========================================================================

def _validate_v8_reward_type_compliance(
    *,
    draft: Mapping[str, Any],
    item_registry: Any,
    task_registry: Any,
    shop_registry: Any,
    npc_name: Optional[str],
) -> Optional[dict[str, Any]]:
    # 从已有任务奖励物品名集合，推导出合法的物品 *类型* 集合
    existing_reward_names = task_registry.list_reward_types()
    valid_types: set[str] = set()
    for name in existing_reward_names:
        item = item_registry.get_by_name(name)
        if item and item.type:
            valid_types.add(item.type)

    # NPC 商店物品名集合
    npc_shop_items: set[str] = set()
    if npc_name:
        npc_shop_items = set(shop_registry.get_npc_shop(npc_name))

    non_compliant: list[dict[str, Any]] = []
    for it in _reward_item_iter(draft, "rewards"):
        name = it.get("item_name", "")
        if not isinstance(name, str) or not name.strip():
            continue

        if name in npc_shop_items:
            continue

        item = item_registry.get_by_name(name)
        if item is None:
            continue  # V1 已处理存在性
        if not item.type:
            continue  # 无类型信息则跳过
        if item.type in valid_types:
            continue

        non_compliant.append({
            "item_name": name,
            "item_type": item.type,
            "reason": f"类型 '{item.type}' 未在已有任务奖励中出现，且不属于当前NPC商店物品",
        })

    if non_compliant:
        return {
            "step": "V8",
            "error": "奖励类型不合规",
            "non_compliant_items": non_compliant,
        }
    return None


# =========================================================================
# V9: 任务不完全重复
# =========================================================================

def _validate_v9_task_uniqueness(
    *,
    draft: Mapping[str, Any],
    task_registry: Any,
    npc_name: Optional[str],
) -> Optional[dict[str, Any]]:
    # 为 draft 构建"结构指纹"
    draft_reqs: set[str] = set()
    for sr in _stage_requirement_iter(draft):
        sn = sr.get("stage_name", "")
        diff = sr.get("difficulty", "")
        if sn and diff:
            draft_reqs.add(f"{sn}#{diff}")

    draft_submit: set[str] = set()
    for it in _reward_item_iter(draft, "finish_submit_items"):
        n = it.get("item_name", "")
        c = it.get("count", 0)
        if n:
            draft_submit.add(f"{n}#{c}")

    draft_contain: set[str] = set()
    for it in _reward_item_iter(draft, "finish_contain_items"):
        n = it.get("item_name", "")
        c = it.get("count", 0)
        if n:
            draft_contain.add(f"{n}#{c}")

    # 若草案完全没有结构化内容（如纯问候任务），跳过重复检测
    if not (draft_reqs or draft_submit or draft_contain):
        return None

    agent_tasks = task_registry.list_agent_tasks()

    similar_ids: list[int] = []
    for t in agent_tasks:
        # 仅比较同一 NPC 发布的任务
        if npc_name and t.get_npc != npc_name:
            continue

        existing_reqs = set(t.finish_requirements or [])
        existing_submit = set(t.finish_submit_items or [])
        existing_contain = set(t.finish_contain_items or [])

        if (draft_reqs == existing_reqs
                and draft_submit == existing_submit
                and draft_contain == existing_contain):
            similar_ids.append(t.id)

    if similar_ids:
        return {
            "step": "V9",
            "error": "任务与已有agent任务高度雷同",
            "similar_task_ids": similar_ids,
        }
    return None


# =========================================================================
# V10: 装备等级匹配
# =========================================================================

def _validate_v10_equipment_level_match(
    *,
    draft: Mapping[str, Any],
    item_registry: Any,
    max_level: int,
    keys: tuple[str, ...] = ("rewards", "finish_submit_items", "finish_contain_items"),
) -> Optional[dict[str, Any]]:
    over_level: list[dict[str, Any]] = []

    for k in keys:
        for it in _reward_item_iter(draft, k):
            name = it.get("item_name", "")
            if not isinstance(name, str) or not name.strip():
                continue
            item = item_registry.get_by_name(name)
            if item is None:
                continue  # V1 已处理
            if item.type not in EQUIPMENT_TYPES:
                continue
            if item.level > max_level:
                over_level.append({
                    "item_name": name,
                    "item_type": item.type,
                    "item_level": item.level,
                    "max_level": max_level,
                })

    if over_level:
        return {
            "step": "V10",
            "error": "装备等级超出当前阶段上限",
            "over_level_items": over_level,
        }
    return None


# =========================================================================
# 校验结果
# =========================================================================

@dataclass(frozen=True)
class DraftValidationResult:
    success: bool
    validation_errors: list[dict[str, Any]]


# =========================================================================
# 完整校验管线 V1-V10
# =========================================================================

def validate_task_draft(
    draft: Mapping[str, Any],
    *,
    context: DraftValidationContext,
    changed_fields: Optional[set[str]] = None,
    game_data: Optional["GameDataRegistry"] = None,
) -> DraftValidationResult:
    """
    完整校验管线（V1-V10）。

    - draft_agent_task：全量校验（changed_fields=None）
    - update_task_draft：增量校验（changed_fields 为仅变更字段的名称集合）
    """

    if game_data is None:
        from services.game_data.registry import get_game_data_registry
        game_data = get_game_data_registry()

    item_registry = game_data.items
    stage_registry = game_data.stages
    task_registry = game_data.tasks
    shop_registry = game_data.shops

    full_mode = changed_fields is None
    changed = changed_fields or set()

    # -- 字段分组 --
    rewards_keys = {"rewards", "finish_submit_items", "finish_contain_items"}
    stage_keys = {"finish_requirements"}
    precondition_keys = {"get_requirements"}
    text_keys = {"title", "description", "get_conversation_text", "finish_conversation_text"}

    reward_keys_to_validate = rewards_keys if full_mode else (changed & rewards_keys)

    run_rewards = full_mode or bool(changed & rewards_keys)
    run_stages = full_mode or bool(changed & stage_keys)
    run_preconditions = full_mode or bool(changed & precondition_keys)
    run_v7 = full_mode or bool(changed & rewards_keys)
    run_v8 = full_mode or bool(changed & {"rewards"})
    run_v9 = full_mode or bool(changed & (rewards_keys | stage_keys | text_keys))
    run_v10 = full_mode or bool(changed & rewards_keys)

    # ---- V1: 物品存在性 ----
    if run_rewards:
        e = _validate_v1_item_existence(
            draft=draft,
            item_registry=item_registry,
            keys=tuple(sorted(reward_keys_to_validate)),
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V2: 物品数量合理性 ----
    if run_rewards:
        e = _validate_v2_item_quantity_reasonableness(
            draft=draft,
            task_registry=task_registry,
            keys=tuple(sorted(reward_keys_to_validate)),
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V3: 关卡存在性与解锁 ----
    if run_stages:
        e = _validate_v3_stage_existence_and_area(
            draft=draft, stage_registry=stage_registry,
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V4: 关卡解锁条件匹配 ----
    if run_stages:
        e = _validate_v4_stage_unlock_condition(
            draft=draft,
            stage_registry=stage_registry,
            main_task_max_id=context.main_task_max_id,
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V5: 副本关卡难度 ----
    if run_stages:
        e = _validate_v5_replica_stage_difficulty(draft=draft)
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V6: 前置任务合法性 ----
    if run_preconditions:
        e = _validate_v6_precondition_tasks(
            draft=draft, task_registry=task_registry,
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V7: 奖励总价值 ----
    if run_v7:
        task_type = draft.get("task_type", "")
        if isinstance(task_type, str) and task_type:
            e = _validate_v7_reward_total_value(
                draft=draft,
                item_registry=item_registry,
                stage=context.stage,
                task_type=task_type,
                affinity=context.affinity,
                bargain_rate=context.bargain_rate,
            )
            if e:
                return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V8: 奖励类型合规 ----
    if run_v8:
        e = _validate_v8_reward_type_compliance(
            draft=draft,
            item_registry=item_registry,
            task_registry=task_registry,
            shop_registry=shop_registry,
            npc_name=context.npc_name,
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V9: 任务不完全重复 ----
    if run_v9:
        e = _validate_v9_task_uniqueness(
            draft=draft,
            task_registry=task_registry,
            npc_name=context.npc_name,
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    # ---- V10: 装备等级匹配 ----
    if run_v10:
        keys_for_v10 = tuple(sorted(reward_keys_to_validate))
        e = _validate_v10_equipment_level_match(
            draft=draft,
            item_registry=item_registry,
            max_level=context.max_level,
            keys=keys_for_v10,
        )
        if e:
            return DraftValidationResult(success=False, validation_errors=[e])

    return DraftValidationResult(success=True, validation_errors=[])


# =========================================================================
# 向后兼容：仅 V1-V6 校验（旧接口）
# =========================================================================

def validate_task_draft_v1_v6(
    draft: Mapping[str, Any],
    *,
    context: DraftValidationContext,
    changed_fields: Optional[set[str]] = None,
    game_data: Optional["GameDataRegistry"] = None,
) -> DraftValidationResult:
    """向后兼容入口，仅执行 V1-V6。新代码推荐使用 ``validate_task_draft``。"""

    if game_data is None:
        from services.game_data.registry import get_game_data_registry
        game_data = get_game_data_registry()

    item_registry = game_data.items
    stage_registry = game_data.stages
    task_registry = game_data.tasks

    full_mode = changed_fields is None
    changed_fields_set = changed_fields or set()

    rewards_keys = {"rewards", "finish_submit_items", "finish_contain_items"}
    stage_keys = {"finish_requirements"}
    precondition_keys = {"get_requirements"}

    reward_keys_to_validate = rewards_keys if full_mode else (changed_fields_set & rewards_keys)

    run_rewards_steps = full_mode or bool(changed_fields_set & rewards_keys)
    run_stage_steps = full_mode or bool(changed_fields_set & stage_keys)
    run_precondition_steps = full_mode or bool(changed_fields_set & precondition_keys)

    if run_rewards_steps:
        e1 = _validate_v1_item_existence(
            draft=draft,
            item_registry=item_registry,
            keys=tuple(sorted(reward_keys_to_validate)),
        )
        if e1:
            return DraftValidationResult(success=False, validation_errors=[e1])

        e2 = _validate_v2_item_quantity_reasonableness(
            draft=draft,
            task_registry=task_registry,
            keys=tuple(sorted(reward_keys_to_validate)),
        )
        if e2:
            return DraftValidationResult(success=False, validation_errors=[e2])

    if run_stage_steps:
        e3 = _validate_v3_stage_existence_and_area(draft=draft, stage_registry=stage_registry)
        if e3:
            return DraftValidationResult(success=False, validation_errors=[e3])

        e4 = _validate_v4_stage_unlock_condition(
            draft=draft,
            stage_registry=stage_registry,
            main_task_max_id=context.main_task_max_id,
        )
        if e4:
            return DraftValidationResult(success=False, validation_errors=[e4])

        e5 = _validate_v5_replica_stage_difficulty(draft=draft)
        if e5:
            return DraftValidationResult(success=False, validation_errors=[e5])

    if run_precondition_steps:
        e6 = _validate_v6_precondition_tasks(draft=draft, task_registry=task_registry)
        if e6:
            return DraftValidationResult(success=False, validation_errors=[e6])

    return DraftValidationResult(success=True, validation_errors=[])
