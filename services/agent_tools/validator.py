"""任务草案校验管线 V1~V11（聚合模式 + 增强反馈，对应 docs/v3-developer/05 §2）。

与旧实现的差异（修 D1/D3）：
- **聚合模式**：一次跑完所有（或增量关联的）规则，全量收集 ``ValidationIssue``，
  不再串行短路「一次只报一个错」；
- **增强反馈**：每条 issue 携带 root_cause（根因）/ fix_hint（具体修正动作，含数字）/
  candidates（可选项清单），模型一轮即可修正；
- V9（雷同）维持 warning 不阻塞。

规则本身（V1~V11 的业务语义、常量体系）为保留资产，与旧版一致。
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING

from services.game_progress import VALID_STAGE_ROOTS
from services.game_data.reward_utils import (
    REWARD_STAGE_BASE_MAX,
    REWARD_STAGE_BASE_MIN,
    parse_name_count,
)

if TYPE_CHECKING:
    from services.game_data.registry import GameDataRegistry


# ---------------------------------------------------------------------------
# 反馈结构（05 §2.1）
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """单条校验问题：规则、字段、人话描述、根因定位、修正指引、候选清单。"""

    rule: str                    # "V1".."V11"
    field: str                   # 出错字段（如 rewards / finish_requirements / get_requirements）
    message: str                 # 人话描述
    root_cause: str = ""         # 后端定位的根因
    fix_hint: str = ""           # 具体修正动作
    candidates: list[str] = field(default_factory=list)   # 可选项（≤10 条）
    auto_repairable: bool = False
    # 修复层数据（allowed_range / item_prices 等）；仅 repair.py 使用，不进模型可见 JSON
    detail: dict[str, Any] = field(default_factory=dict)

    def to_model_json(self) -> dict[str, Any]:
        """模型可见形态（不含 detail 内部数据）。"""
        out: dict[str, Any] = {
            "rule": self.rule,
            "field": self.field,
            "message": self.message,
        }
        if self.root_cause:
            out["root_cause"] = self.root_cause
        if self.fix_hint:
            out["fix_hint"] = self.fix_hint
        if self.candidates:
            out["candidates"] = self.candidates
        return out


@dataclass
class ValidationReport:
    """聚合校验结果：一次跑完全部规则后统一返回（修 D1）。"""

    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)  # V9 等，不阻塞

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_model_json(self) -> dict[str, Any]:
        """模型可见 JSON：issues 数组 + 逐条 root_cause/fix_hint/candidates。"""
        return {
            "status": "validation_failed",
            "issue_count": len(self.issues),
            "issues": [i.to_model_json() for i in self.issues],
            "warnings": self.warnings or None,
        }


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
    # 讨价还价阶段上限放大倍数，默认为 1.0 表示不放大。
    bargain_rate: float = 1.0


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

COMBAT_TASK_TYPES: frozenset[str] = frozenset({
    "通关", "清理", "挑战", "切磋", "通关并收集", "通关并持有",
})

EQUIPMENT_TYPES: frozenset[str] = frozenset({"武器", "防具"})

# 反馈清单上限
_MAX_CANDIDATES = 10
_MAX_FUZZY_CANDIDATES = 5


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def _as_list(v: Any) -> list[Any]:
    if isinstance(v, list):
        return v
    return []


def _reward_item_iter(draft: Mapping[str, Any], key: str) -> Iterable[Mapping[str, Any]]:
    """遍历奖励/提交/持有列表，支持 dict {item_name, count} 与字符串 \"物品名#数量\" 两种格式。"""
    for it in _as_list(draft.get(key)):
        if isinstance(it, dict):
            yield it
        elif isinstance(it, str) and it.strip():
            name, count = parse_name_count(it.strip())
            if name:
                yield {"item_name": name, "count": count}


def _stage_requirement_iter(draft: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for sr in _as_list(draft.get("finish_requirements")):
        if isinstance(sr, dict):
            yield sr


def _get_stage_infos_by_name(*, stage_registry: Any, stage_name: str) -> list[tuple[str, Any]]:
    """
    通过 stage_name 在所有大区中查找 stage 信息。
    注意：stage_area 不参与 LLM 输入；由后端在筛选候选关卡时使用，本校验只做 stage_name 级校验。
    """
    if not isinstance(stage_name, str) or not stage_name.strip():
        return []
    stage_infos_raw = getattr(stage_registry, "_stage_infos", None)
    if not isinstance(stage_infos_raw, dict):
        return []

    out: list[tuple[str, Any]] = []
    for (area, name), si in stage_infos_raw.items():
        if name == stage_name:
            out.append((str(area), si))
    return out


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
    base_min = stage * REWARD_STAGE_BASE_MIN
    base_max = stage * REWARD_STAGE_BASE_MAX

    type_mult = 2 if task_type in COMBAT_TASK_TYPES else 1

    mult_min = base_min * type_mult
    mult_max = base_max * type_mult

    # ---- 提交品加成 ----
    # 提交品总价值上限 = 基础奖励 × 200%
    submit_cap_lo = mult_min * 2.0
    submit_cap_hi = mult_max * 2.0
    capped_submit_lo = min(submit_value, submit_cap_lo)
    capped_submit_hi = min(submit_value, submit_cap_hi)
    # 额外 += capped × 1.0（下限）~ 2.0（上限）：下限与提交品等价，上限为 2 倍，降低「略增奖励仍低于区间」的概率
    submit_bonus_lo = capped_submit_lo * 1.0
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


def _fuzzy_item_candidates(
    name: str, item_registry: Any, limit: int = _MAX_FUZZY_CANDIDATES,
) -> list[str]:
    """V1 增强：按编辑距离 + 子串双重策略给出真实物品名候选（05 §2.2）。"""
    all_names: list[str] = []
    try:
        all_names = [it.name for it in item_registry.items if it.name]
    except Exception:
        return []
    if not all_names:
        return []

    query = name.strip()
    lower = query.lower()
    scored: dict[str, float] = {}

    # 子串命中（物品名包含查询词或反之）优先
    for n in all_names:
        nl = n.lower()
        if lower and (lower in nl or nl in lower):
            scored[n] = 0.0  # 排在最前

    close = difflib.get_close_matches(query, all_names, n=limit * 2, cutoff=0.4)
    for rank, n in enumerate(close):
        if n not in scored:
            scored[n] = 1.0 + rank  # 编辑距离命中次之

    return sorted(scored, key=lambda n: (scored[n], n))[:limit]


def _available_stage_candidates(
    *,
    stage_registry: Any,
    main_task_max_id: int,
    limit: int = _MAX_CANDIDATES,
) -> list[str]:
    """V3/V4/V5 增强：当前进度下可用的关卡候选（unlock_condition ≤ 进度上限），去重后截断。"""
    infos = getattr(stage_registry, "_stage_infos", None)
    if not isinstance(infos, dict):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for (_area, name), si in infos.items():
        if name in seen:
            continue
        unlock = getattr(si, "unlock_condition", None)
        # 无解锁条件（副本等）视为可用；有解锁条件则要求 ≤ 当前进度
        if unlock is not None and isinstance(unlock, int) and unlock > main_task_max_id:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= limit:
            break
    return out


# =========================================================================
# V1: 物品存在性
# =========================================================================

def _validate_v1_item_existence(
    *,
    draft: Mapping[str, Any],
    item_registry: Any,
    keys: tuple[str, ...] = ("rewards", "finish_submit_items", "finish_contain_items"),
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    # 同名物品只在第一个出现的字段报一次，避免重复刷屏
    reported: set[str] = set()

    for k in keys:
        for it in _reward_item_iter(draft, k):
            item_name = it.get("item_name")
            if not isinstance(item_name, str) or not item_name.strip():
                continue
            if item_name in reported:
                continue
            if item_registry.get_by_name(item_name) is not None:
                continue
            reported.add(item_name)
            candidates = _fuzzy_item_candidates(item_name, item_registry)
            root = f"物品「{item_name}」不在游戏物品数据中"
            if candidates:
                root += f"，是否想表达：{'、'.join(candidates)}？"
            issues.append(ValidationIssue(
                rule="V1",
                field=k,
                message=f"物品「{item_name}」不存在，无法用于 {k}。",
                root_cause=root,
                fix_hint=(
                    f"把 {k} 中的「{item_name}」改为真实存在的物品名"
                    + (f"（候选：{'、'.join(candidates)}）" if candidates else "")
                    + "；候选不足时重新调用 prepare_task_context 查看可选物品。"
                ),
                candidates=candidates,
            ))
    return issues


# =========================================================================
# V2: 物品数量合理性
# =========================================================================

def _validate_v2_item_quantity_reasonableness(
    *,
    draft: Mapping[str, Any],
    task_registry: Any,
    item_registry: Any,
    context: DraftValidationContext,
    keys: tuple[str, ...] = ("rewards", "finish_submit_items", "finish_contain_items"),
) -> list[ValidationIssue]:
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

    issues: list[ValidationIssue] = []
    stage = int(getattr(context, "stage", 1) or 1)
    base_max = stage * REWARD_STAGE_BASE_MAX
    task_type = draft.get("task_type", "") if isinstance(draft, dict) else ""
    type_mult = 2 if isinstance(task_type, str) and task_type in COMBAT_TASK_TYPES else 1

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
            reward_min_qty, reward_max_qty = reward_stats.get(item_name, (None, 0))  # type: ignore[assignment]

            # 并集原则：提交/持有的数量上限同时参考奖励历史（同一物品在 rewards 里出现很多次时，
            # 仅看 finish_submit_items/finish_contain_items 的统计会过小，导致 V2 误拒绝）。
            if k in ("finish_submit_items", "finish_contain_items") and int(reward_max_qty or 0) > 0:
                max_qty = max(int(max_qty or 0), int(reward_max_qty or 0))

            # 若 submit/hold 的统计仍然不存在（max_qty=0），且为装备类：
            # 回退到 rewards 统计，避免出现 allowed_range=[1,0] 的“天然不可能”区间。
            if int(max_qty or 0) == 0 and k in ("finish_submit_items", "finish_contain_items"):
                item = item_registry.get_by_name(item_name) if item_registry else None
                item_type = getattr(item, "type", None) if item else None
                if item_type in EQUIPMENT_TYPES:
                    if int(reward_max_qty or 0) > 0:
                        max_qty = reward_max_qty
            allowed_min = 1
            effective_max = int(max_qty or 0)
            if effective_max <= 0:
                # 无历史统计：用“阶段奖励上限/单价”估算一个数量上限，
                # 比固定 [1,2] 更贴近实际预算，避免生成直接触发 V2。
                if k == "rewards":
                    allowed_max = 2
                else:
                    item = item_registry.get_by_name(item_name) if item_registry else None
                    unit_price = int(getattr(item, "price", 0) or 0) if item else 0
                    if unit_price <= 0:
                        allowed_max = 2
                    else:
                        # 武器/防具单价高，允许更宽松的“数量”上限
                        item_type = getattr(item, "type", None) if item else None
                        qty_multiplier = 3 if item_type in ("武器", "防具") else 1
                        allowed_max = int((base_max * type_mult * qty_multiplier) / unit_price)
                        allowed_max = max(1, allowed_max)
            else:
                allowed_max = effective_max * 2

            if n < allowed_min or n > allowed_max:
                suggest = min(max(n, allowed_min), allowed_max)
                issues.append(ValidationIssue(
                    rule="V2",
                    field=k,
                    message=f"物品「{item_name}」数量 {n} 不在合理范围 [{allowed_min}, {allowed_max}]。",
                    root_cause=(
                        f"「{item_name}」在历史任务中的数量统计上限为 {effective_max if effective_max > 0 else '无（按预算估算）'}，"
                        f"当前填写的 {n} 超出可接受区间。"
                    ),
                    fix_hint=f"把 {k} 中「{item_name}」的数量从 {n} 调整到 {suggest}（合理范围 [{allowed_min}, {allowed_max}]）。",
                    auto_repairable=True,
                    detail={
                        "key": k,
                        "item_name": item_name,
                        "current": n,
                        "allowed_range": [allowed_min, allowed_max],
                        "suggest": suggest,
                    },
                ))
    return issues


# =========================================================================
# V3: 关卡存在性与解锁
# =========================================================================

def _validate_v3_stage_existence_and_area(
    *,
    draft: Mapping[str, Any],
    stage_registry: Any,
    context: DraftValidationContext,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    candidates = _available_stage_candidates(
        stage_registry=stage_registry, main_task_max_id=context.main_task_max_id,
    )
    for sr in _stage_requirement_iter(draft):
        stage_name = sr.get("stage_name")
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue
        stage_infos = _get_stage_infos_by_name(stage_registry=stage_registry, stage_name=stage_name)
        if not stage_infos:
            issues.append(ValidationIssue(
                rule="V3",
                field="finish_requirements",
                message=f"关卡「{stage_name}」不存在。",
                root_cause=f"游戏关卡数据中没有名为「{stage_name}」的关卡。",
                fix_hint=(
                    f"把 finish_requirements 中的关卡改为可用关卡"
                    + (f"（候选：{'、'.join(candidates)}）" if candidates else "；候选不足时重新调用 prepare_task_context")
                    + "。"
                ),
                candidates=candidates,
            ))
            continue

        # stage_area 由筛选阶段负责；这里只确认：至少存在一个匹配关卡，且有 unlock_condition
        has_valid_unlock = False
        for _area, si in stage_infos:
            unlock = getattr(si, "unlock_condition", None)
            if isinstance(unlock, int) and unlock > 0:
                has_valid_unlock = True
                break

        if not has_valid_unlock:
            issues.append(ValidationIssue(
                rule="V3",
                field="finish_requirements",
                message=f"关卡「{stage_name}」无效或缺少解锁条件。",
                root_cause=f"关卡「{stage_name}」在数据中缺少有效的解锁条件，不能用于任务。",
                fix_hint="改用其他有有效解锁条件的关卡。",
                candidates=candidates,
            ))
    return issues


def _validate_v3_dungeon_recommended_level(
    *,
    draft: Mapping[str, Any],
    context: DraftValidationContext,
    game_data: "GameDataRegistry",
) -> list[ValidationIssue]:
    """
    对副本/切磋类的关卡按 mercenary_tasks.json 的 recommended_level 做强校验：
    - 如果某关卡在 mercenary_tasks 中存在推荐下限且推荐下限 > 玩家当前阶段上限，则拒绝。
    - 如果该关卡没有推荐等级（或 recommended_min_level 为 None），则不做推荐筛选。
    """
    mercenary_registry = getattr(game_data, "mercenary_tasks", None)
    if mercenary_registry is None:
        return []

    max_level = int(getattr(context, "max_level", 50) or 50)
    stage_registry = game_data.stages
    candidates = _available_stage_candidates(
        stage_registry=stage_registry, main_task_max_id=context.main_task_max_id,
    )

    issues: list[ValidationIssue] = []
    for sr in _stage_requirement_iter(draft):
        stage_name = sr.get("stage_name")
        difficulty = sr.get("difficulty")
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue

        matched = [m for m in mercenary_registry.list_all() if m.stage_name == stage_name]
        if not matched:
            continue

        # 只要有一个匹配项满足推荐条件，则认为该关卡可用
        ok = False
        for m in matched:
            if m.recommended_min_level is None:
                ok = True
                break
            if int(m.recommended_min_level or 0) <= max_level:
                ok = True
                break
        if not ok:
            rec = max(m.recommended_min_level for m in matched if m.recommended_min_level is not None)
            issues.append(ValidationIssue(
                rule="V3R",
                field="finish_requirements",
                message=f"关卡「{stage_name}」推荐等级 {rec} 超出玩家当前等级上限 {max_level}。",
                root_cause=f"「{stage_name}」为副本/切磋关卡，推荐最低等级 {rec}，玩家当前阶段等级上限为 {max_level}。",
                fix_hint=(
                    "改用推荐等级更低的副本关卡，或把任务类型改为普通通关类。"
                    + (f"可用关卡候选：{'、'.join(candidates)}" if candidates else "")
                ),
                candidates=candidates,
            ))
    return issues


# =========================================================================
# V4: 关卡解锁条件匹配
# =========================================================================

def _validate_v4_stage_unlock_condition(
    *,
    draft: Mapping[str, Any],
    stage_registry: Any,
    main_task_max_id: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    candidates = _available_stage_candidates(
        stage_registry=stage_registry, main_task_max_id=main_task_max_id,
    )
    for sr in _stage_requirement_iter(draft):
        stage_name = sr.get("stage_name")
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue
        stage_infos = _get_stage_infos_by_name(stage_registry=stage_registry, stage_name=stage_name)
        if not stage_infos:
            continue  # V3 会处理

        unlock_ids: list[int] = []
        for _area, si in stage_infos:
            unlock = getattr(si, "unlock_condition", None)
            if isinstance(unlock, int) and unlock > 0:
                unlock_ids.append(int(unlock))

        if not unlock_ids:
            continue  # V3 会处理

        # 只要存在一种“该 stage_name”对应关卡在当前进度可解锁即可
        # 否则表示所有匹配关卡都超进度。
        min_unlock = min(unlock_ids)
        if min_unlock > int(main_task_max_id):
            issues.append(ValidationIssue(
                rule="V4",
                field="finish_requirements",
                message=f"关卡「{stage_name}」解锁进度不足（需主线 ID ≥ {min_unlock}，当前进度上限 {main_task_max_id}）。",
                root_cause=f"「{stage_name}」最早解锁需要完成主线任务 {min_unlock}，玩家当前主线进度只到 {main_task_max_id}。",
                fix_hint=(
                    "改用玩家当前进度已解锁的关卡。"
                    + (f"可用关卡候选：{'、'.join(candidates)}" if candidates else "")
                ),
                candidates=candidates,
            ))
    return issues


# =========================================================================
# V5: 副本关卡难度
# =========================================================================

def _validate_v5_replica_stage_difficulty(
    *,
    draft: Mapping[str, Any],
    context: DraftValidationContext,
    game_data: "GameDataRegistry",
) -> list[ValidationIssue]:
    """
    mercenary_tasks.json 绑定的关卡难度校验：
    - 永远允许 "简单"
    - 非简单难度仅当该 stage_name 在 mercenary_tasks.json 的对应任务配置了 challenge 额外难度，且玩家满足其推荐等级下限时才允许
    """
    issues: list[ValidationIssue] = []
    mercenary_registry = getattr(game_data, "mercenary_tasks", None)
    max_level = int(getattr(context, "max_level", 50) or 50)
    for sr in _stage_requirement_iter(draft):
        stage_name = sr.get("stage_name")
        difficulty = sr.get("difficulty")
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue
        if not isinstance(difficulty, str) or not difficulty.strip():
            continue
        if mercenary_registry is None:
            continue

        # 判定“副本类/委托类”：以 mercenary_tasks.json 绑定的 stage_name 为准
        matched = [m for m in mercenary_registry.list_all() if m.stage_name == stage_name]
        if not matched:
            continue

        allowed_difficulties: set[str] = {"简单"}
        for m in matched:
            if not m.challenge_difficulty or m.challenge_difficulty == "简单":
                continue
            cmin = m.challenge_recommended_min_level
            if cmin is not None and int(cmin) <= max_level:
                allowed_difficulties.add(m.challenge_difficulty)

        if difficulty not in allowed_difficulties:
            issues.append(ValidationIssue(
                rule="V5",
                field="finish_requirements",
                message=f"关卡「{stage_name}」不允许难度「{difficulty}」（当前可用：{'、'.join(sorted(allowed_difficulties))}）。",
                root_cause=(
                    f"「{stage_name}」的额外难度仅当玩家等级满足其推荐下限时开放，"
                    f"玩家当前等级上限为 {max_level}。"
                ),
                fix_hint=f"把「{stage_name}」的难度改为「{'、'.join(sorted(allowed_difficulties))}」之一。",
                candidates=sorted(allowed_difficulties),
            ))
    return issues


# =========================================================================
# V6: 前置任务合法性
# =========================================================================

def _validate_v6_precondition_tasks(
    *,
    draft: Mapping[str, Any],
    task_registry: Any,
    context: DraftValidationContext,
) -> list[ValidationIssue]:
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

    if not invalid:
        return []
    return [ValidationIssue(
        rule="V6",
        field="get_requirements",
        message=f"前置任务 ID 不合法：{sorted(set(invalid))}（-1 禁止使用）。",
        root_cause="get_requirements 只能填写真实存在的主线任务 ID，且不允许 -1。",
        fix_hint=(
            "移除这些无效 ID，或改为 ≤ 当前进度上限 "
            f"{context.main_task_max_id} 的合法主线任务 ID。"
        ),
    )]


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
) -> list[ValidationIssue]:
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
        return []

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

    if rewards_value < range_min:
        delta = range_min - rewards_value
        direction = "低于"
        need = f"增加 ≥{delta}"
        suggest_target = range_min
    else:
        delta = rewards_value - range_max
        direction = "高于"
        need = f"减少 ≥{delta}"
        suggest_target = range_max

    # 给出按单价可换算的具体建议（调整最大占比奖励项的数量）
    best = max(item_prices, key=lambda x: x["subtotal"], default=None)
    adjust_hint = ""
    if best and best["unit_price"] > 0:
        best_name = best["item_name"]
        step = max(1, delta // best["unit_price"] + (1 if delta % best["unit_price"] else 0))
        if rewards_value < range_min:
            new_count = best["count"] + step
        else:
            new_count = max(0, best["count"] - step)
        adjust_hint = (
            f"；可把「{best_name}」（单价 {best['unit_price']}）数量从 {best['count']} 调整到 {new_count}"
        )

    # 总值偏差一律可由后端按取整阶梯微调（数值类问题不进打回流程，05 §4）
    auto_repairable = True

    return [ValidationIssue(
        rule="V7",
        field="rewards",
        message=(
            f"奖励总价值 {rewards_value} {direction}允许区间 [{range_min}, {range_max}]，需{need}。"
        ),
        root_cause=(
            f"按玩家进度（阶段 {stage}）、任务类型「{task_type}」与好感度 {affinity} 计算，"
            f"奖励总值允许区间为 [{range_min}, {range_max}]，当前总值 {rewards_value}。"
        ),
        fix_hint=(
            f"当前总值 {rewards_value}，{direction}允许区间 [{range_min}, {range_max}]，需{need}{adjust_hint}。"
        ),
        auto_repairable=auto_repairable,
        detail={
            "total_value": rewards_value,
            "allowed_range": [range_min, range_max],
            "item_prices": item_prices,
        },
    )]


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
) -> list[ValidationIssue]:
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
        # 商店物品的类型也是合规奖励类型（05 §2.2：商店 ∪ 历史奖励类型）
        for shop_name in npc_shop_items:
            item = item_registry.get_by_name(shop_name)
            if item and item.type:
                valid_types.add(item.type)

    compliant_types = sorted(valid_types)
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
        })

    issues: list[ValidationIssue] = []
    for nc in non_compliant:
        name = nc["item_name"]
        itype = nc["item_type"]
        issues.append(ValidationIssue(
            rule="V8",
            field="rewards",
            message=f"奖励物品「{name}」类型「{itype}」不合规。",
            root_cause=f"类型「{itype}」未在已有任务奖励或当前 NPC 商店中出现，「{name}」不能作为该 NPC 的奖励。",
            fix_hint=(
                f"把「{name}」替换为合规类型的奖励物品"
                + (f"（合规类型：{'、'.join(compliant_types)}）" if compliant_types else "")
                + "。"
            ),
            candidates=compliant_types[:_MAX_CANDIDATES],
        ))
    return issues


# =========================================================================
# V9: 任务不完全重复（仅 warning）
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
        n = len(similar_ids)
        return {
            "step": "V9",
            "warning": f"此前已发布{n}个高度雷同的任务，请谨慎发布，可视情况取消任务/变更任务或继续委派任务。",
            "similar_task_count": n,
            "similar_task_ids": similar_ids,
        }
    return None


# =========================================================================
# V10: 装备等级匹配
# =========================================================================

def _low_level_replacements(
    *,
    item_registry: Any,
    item_type: str,
    max_level: int,
    limit: int = _MAX_FUZZY_CANDIDATES,
) -> list[str]:
    """V10 增强：同类型、等级 ≤ max_level 的替代装备候选（按等级从高到低）。"""
    try:
        same_type = item_registry.list_by_type(item_type)
    except Exception:
        return []
    usable = [it for it in same_type if (it.level or 0) <= max_level]
    usable.sort(key=lambda it: -(it.level or 0))
    return [it.name for it in usable[:limit]]


def _validate_v10_equipment_level_match(
    *,
    draft: Mapping[str, Any],
    item_registry: Any,
    max_level: int,
    keys: tuple[str, ...] = ("rewards", "finish_submit_items", "finish_contain_items"),
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

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
            if item.level <= max_level:
                continue
            replacements = _low_level_replacements(
                item_registry=item_registry, item_type=item.type, max_level=max_level,
            )
            issues.append(ValidationIssue(
                rule="V10",
                field=k,
                message=f"装备「{name}」等级 {item.level} 超出当前阶段上限 {max_level}。",
                root_cause=f"「{name}」（{item.type}，等级 {item.level}）超过玩家当前阶段允许的装备等级上限 {max_level}。",
                fix_hint=(
                    f"把 {k} 中的「{name}」替换为低级版本"
                    + (f"（替代候选：{'、'.join(replacements)}）" if replacements else "，或直接移除该项")
                    + "。"
                ),
                candidates=replacements,
                auto_repairable=True,
                detail={
                    "key": k,
                    "item_name": name,
                    "item_level": item.level,
                    "max_level": max_level,
                    "replacement": replacements[0] if replacements else None,
                },
            ))
    return issues


# =========================================================================
# V11: 提交品与奖励物品不得重名
# =========================================================================

def _collect_reward_item_names(draft: Mapping[str, Any], key: str) -> set[str]:
    """物品名字符串集合（strip 后），用于跨字段重复检测。"""
    names: set[str] = set()
    for it in _reward_item_iter(draft, key):
        name = it.get("item_name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _validate_v11_submit_reward_no_overlap(
    *,
    draft: Mapping[str, Any],
) -> list[ValidationIssue]:
    """
    finish_submit_items 与 rewards 不得出现相同物品名（易把「玩家想要的报酬」误填进提交要求）。
    """
    submit_names = _collect_reward_item_names(draft, "finish_submit_items")
    reward_names = _collect_reward_item_names(draft, "rewards")
    overlap = submit_names & reward_names
    if not overlap:
        return []
    overlap_sorted = sorted(overlap)
    return [ValidationIssue(
        rule="V11",
        field="finish_submit_items",
        message=(
            "`finish_submit_items` 与 `rewards` 不能包含相同物品："
            f"{overlap_sorted}。"
        ),
        root_cause=(
            "玩家需要的物品只能写在 `rewards`；需要玩家提交给你的物品只能写在 "
            "`finish_submit_items`。同名物品同时出现在两侧时任务语义冲突。"
        ),
        fix_hint=(
            f"从 `finish_submit_items` 中移除与 `rewards` 重名的物品 {overlap_sorted}"
            "（保留先出现侧，自动修复时默认保留 rewards 侧）。"
        ),
        auto_repairable=True,
        detail={"overlap": overlap_sorted},
    )]


# =========================================================================
# 完整校验管线 V1-V11（聚合模式）
# =========================================================================

def validate_task_draft(
    draft: Mapping[str, Any],
    *,
    context: DraftValidationContext,
    changed_fields: Optional[set[str]] = None,
    game_data: Optional["GameDataRegistry"] = None,
) -> ValidationReport:
    """
    完整校验管线（V1-V11，聚合模式）。

    - draft_agent_task：全量校验（changed_fields=None）
    - update_task_draft：增量校验（changed_fields 为仅变更字段的名称集合）
    一次跑完所有（或增量关联的）规则，全量收集 issue 后统一返回（修 D1）。
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

    reward_keys_to_validate = rewards_keys if full_mode else (changed & rewards_keys)

    run_rewards = full_mode or bool(changed & rewards_keys)
    run_stages = full_mode or bool(changed & stage_keys)
    run_preconditions = full_mode or bool(changed & precondition_keys)
    run_v7 = full_mode or bool(changed & rewards_keys)
    run_v8 = full_mode or bool(changed & {"rewards"})
    # V9 的指纹只依赖：关卡要求 + 提交/持有物品，不应因为“对话/标题描述”变更而触发
    run_v9 = full_mode or bool(changed & (rewards_keys | stage_keys))
    run_v10 = full_mode or bool(changed & rewards_keys)
    run_v11 = full_mode or bool(changed & {"rewards", "finish_submit_items"})

    issues: list[ValidationIssue] = []
    warnings: list[dict[str, Any]] = []

    # ---- V1: 物品存在性 ----
    if run_rewards:
        issues.extend(_validate_v1_item_existence(
            draft=draft,
            item_registry=item_registry,
            keys=tuple(sorted(reward_keys_to_validate)),
        ))

    # ---- V2: 物品数量合理性 ----
    if run_rewards:
        issues.extend(_validate_v2_item_quantity_reasonableness(
            draft=draft,
            task_registry=task_registry,
            item_registry=item_registry,
            context=context,
            keys=tuple(sorted(reward_keys_to_validate)),
        ))

    # ---- V3 / V3R / V4 / V5: 关卡族 ----
    if run_stages:
        issues.extend(_validate_v3_stage_existence_and_area(
            draft=draft, stage_registry=stage_registry, context=context,
        ))
        issues.extend(_validate_v3_dungeon_recommended_level(
            draft=draft, context=context, game_data=game_data,
        ))
        issues.extend(_validate_v4_stage_unlock_condition(
            draft=draft,
            stage_registry=stage_registry,
            main_task_max_id=context.main_task_max_id,
        ))
        issues.extend(_validate_v5_replica_stage_difficulty(
            draft=draft, context=context, game_data=game_data,
        ))

    # ---- V6: 前置任务合法性 ----
    if run_preconditions:
        issues.extend(_validate_v6_precondition_tasks(
            draft=draft, task_registry=task_registry, context=context,
        ))

    # ---- V7: 奖励总价值 ----
    if run_v7:
        task_type = draft.get("task_type", "")
        if isinstance(task_type, str) and task_type:
            issues.extend(_validate_v7_reward_total_value(
                draft=draft,
                item_registry=item_registry,
                stage=context.stage,
                task_type=task_type,
                affinity=context.affinity,
                bargain_rate=context.bargain_rate,
            ))

    # ---- V8: 奖励类型合规 ----
    if run_v8:
        issues.extend(_validate_v8_reward_type_compliance(
            draft=draft,
            item_registry=item_registry,
            task_registry=task_registry,
            shop_registry=shop_registry,
            npc_name=context.npc_name,
        ))

    # ---- V9: 任务高度雷同（仅警告，不阻止发布） ----
    if run_v9:
        w = _validate_v9_task_uniqueness(
            draft=draft,
            task_registry=task_registry,
            npc_name=context.npc_name,
        )
        if w:
            warnings.append(w)

    # ---- V10: 装备等级匹配 ----
    if run_v10:
        keys_for_v10 = tuple(sorted(reward_keys_to_validate))
        issues.extend(_validate_v10_equipment_level_match(
            draft=draft,
            item_registry=item_registry,
            max_level=context.max_level,
            keys=keys_for_v10,
        ))

    # ---- V11: 提交品与奖励物品不得重名 ----
    if run_v11:
        issues.extend(_validate_v11_submit_reward_no_overlap(draft=draft))

    return ValidationReport(issues=issues, warnings=warnings)


def report_to_validation_errors(report: ValidationReport) -> list[dict[str, Any]]:
    """兼容旧字段名的序列化（tools 结果 JSON 中 errors 数组的形态）。"""
    return [i.to_model_json() for i in report.issues]
