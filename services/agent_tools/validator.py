from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING

from services.game_progress import VALID_STAGE_ROOTS

if TYPE_CHECKING:
    from services.game_data.registry import GameDataRegistry


@dataclass(frozen=True)
class DraftValidationContext:
    """
    校验所需的“玩家当前进度信息”，Phase 2 仅实现 V1-V6：
    - V4 依赖 main_task_max_id
    - V10 以后会依赖 max_level（此处保留字段以免后续改签名）
    """

    main_task_max_id: int
    max_level: int


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


def _validate_v2_item_quantity_reasonableness(
    *,
    draft: Mapping[str, Any],
    task_registry: Any,
    keys: tuple[str, ...] = ("rewards", "finish_submit_items", "finish_contain_items"),
) -> Optional[dict[str, Any]]:
    reward_stats: dict[str, tuple[int, int]] = {}
    try:
        reward_stats = task_registry.get_reward_stats()
    except Exception:
        # 退化：若无法拿到 reward_stats，则按 max=0 处理以确保保守校验
        reward_stats = {}

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
                # 非法数量直接作为超限处理：最低也应为 1
                n = -999999

            # “已有任务奖励中出现过的最大数量的 2 倍”
            _, max_qty = reward_stats.get(item_name, (None, 0))  # type: ignore[assignment]
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

        # StageRegistry.get_unlock_condition 在找不到/UnlockCondition 不存在时会返回 0
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


@dataclass(frozen=True)
class DraftValidationResult:
    success: bool
    validation_errors: list[dict[str, Any]]


def validate_task_draft_v1_v6(
    draft: Mapping[str, Any],
    *,
    context: DraftValidationContext,
    changed_fields: Optional[set[str]] = None,
    game_data: Optional["GameDataRegistry"] = None,
) -> DraftValidationResult:
    """
    只实现文档 Phase 2 2.4 的 V1-V6：
    - draft_agent_task：full 校验（changed_fields=None）
    - update_task_draft：增量校验（changed_fields 为仅变更字段集合）
    """

    if game_data is None:
        # 延迟导入，避免在缺少 pydantic 等依赖时导致模块导入失败
        from services.game_data.registry import get_game_data_registry

        game_data = get_game_data_registry()

    item_registry = game_data.items
    stage_registry = game_data.stages
    task_registry = game_data.tasks

    full_mode = changed_fields is None
    changed_fields_set = changed_fields or set()

    # 触发条件：只校验与变化字段相关的步骤（文档要求“仅校验变更字段”）
    rewards_keys = {"rewards", "finish_submit_items", "finish_contain_items"}
    stage_keys = {"finish_requirements"}
    precondition_keys = {"get_requirements"}

    reward_keys_to_validate = rewards_keys if full_mode else (changed_fields_set & rewards_keys)

    run_rewards_steps = full_mode or bool(changed_fields_set & rewards_keys)
    run_stage_steps = full_mode or bool(changed_fields_set & stage_keys)
    run_precondition_steps = full_mode or bool(changed_fields_set & precondition_keys)

    # 按文档顺序逐项校验：V1 -> V2 -> V3 -> V4 -> V5 -> V6
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

