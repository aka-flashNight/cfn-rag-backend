"""草案自动修复层（对应 docs/v3-developer/05 §3）。

在 ``validate`` 之后、草案落库之前执行。原则：**后端只动数值，不替模型做选择**——
存在性/合规性问题（V1/V3/V4/V5/V6/V8）一律打回模型重新选择，不做自动修复。

可自动修复的场景：
- V7 总值偏差 ≤ ±10%：按单价明细调整一个奖励项数量到区间内最近值；
- V2 数量越界：clamp 到 allowed_range 最近端点；
- V10 装备超等级：降级到阶段上限内的同类型装备，无则移除该项；
- V11 提交/奖励重名：保留 rewards 侧，从 finish_submit_items 去重。

返回 (修复后草案, 修复说明列表, 复审报告)；修复说明随工具结果回传模型并记入
草案 ``repaired_notes``，confirm 展示时可一并呈现。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, TYPE_CHECKING

from services.agent_tools.validator import (
    DraftValidationContext,
    ValidationReport,
    _reward_item_iter,
    validate_task_draft,
)

if TYPE_CHECKING:
    from services.game_data.registry import GameDataRegistry


def _items_list(draft: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """把 rewards/finish_submit_items/finish_contain_items 归一为 [{item_name, count}] 列表。"""
    out: list[dict[str, Any]] = []
    for it in _reward_item_iter(draft, key):
        out.append({"item_name": str(it.get("item_name")), "count": int(it.get("count") or 0)})
    return out


def _items_total(items: list[dict[str, Any]], item_registry: Any) -> int:
    total = 0
    for it in items:
        total += item_registry.get_price(it["item_name"]) * max(it["count"], 0)
    return total


# ---------------------------------------------------------------------------
# V2：数量 clamp
# ---------------------------------------------------------------------------

def _repair_v2(draft: dict[str, Any], issue) -> str | None:
    detail = issue.detail
    key = detail.get("key")
    name = detail.get("item_name")
    lo, hi = detail.get("allowed_range", [1, 1])
    if not key or not name:
        return None
    items = draft.get(key)
    if not isinstance(items, list):
        return None
    for it in items:
        target_name = it.get("item_name") if isinstance(it, dict) else None
        if isinstance(target_name, str) and target_name.strip() == name:
            try:
                cur = int(it.get("count"))
            except Exception:
                cur = lo
            fixed = min(max(cur, int(lo)), int(hi))
            if fixed == cur:
                return None
            it["count"] = fixed
            return f"已自动微调：{key} 中「{name}」数量 {cur}→{fixed}（合理范围 [{lo}, {hi}]）。"
    return None


# ---------------------------------------------------------------------------
# V7：总值微调（单币种取整阶梯，用户定稿规则）
# ---------------------------------------------------------------------------

# 取整阶梯：金币/经验值按 5000（区间上限 < 20000 的低级任务按 1000）；K点按 100；其他物品按 1
_ROUND_STEP_MAJOR = 5000
_ROUND_STEP_MINOR = 1000
_ROUND_STEP_MINOR_THRESHOLD = 20000
_ROUND_STEP_KPOINT = 100
_ROUND_STEP_ITEM = 1

# 优先调整的奖励项（按顺序取第一个存在的）
_PREFERRED_ADJUST_TARGETS = ("金币", "经验值", "K点")


def _round_step_for(item_name: str, range_max: int) -> int:
    if item_name in ("金币", "经验值"):
        return _ROUND_STEP_MAJOR if range_max >= _ROUND_STEP_MINOR_THRESHOLD else _ROUND_STEP_MINOR
    if item_name == "K点":
        return _ROUND_STEP_KPOINT
    return _ROUND_STEP_ITEM


def _repair_v7(draft: dict[str, Any], issue, item_registry: Any) -> str | None:
    """V7 总值微调：优先动金币/经验值/K点（按取整阶梯），否则动占比最大的奖励项。

    只调奖励数量，不触碰提交品/持有品/关卡（任务内容归模型）。
    """
    detail = issue.detail
    total = int(detail.get("total_value") or 0)
    lo, hi = detail.get("allowed_range", [0, 0])
    prices: list[dict[str, Any]] = detail.get("item_prices") or []
    rewards = draft.get("rewards")
    if not isinstance(rewards, list) or not prices:
        return None
    if total > hi:
        target = int(hi)
    elif total < lo:
        target = int(lo)
    else:
        return None

    # 调整项优先级：金币 → 经验值 → K点 → 单项小节最大者
    name_by_priority: str | None = None
    for preferred in _PREFERRED_ADJUST_TARGETS:
        if any(p.get("item_name") == preferred for p in prices):
            name_by_priority = preferred
            break
    if name_by_priority is None:
        best = max(prices, key=lambda p: int(p.get("subtotal") or 0), default=None)
        name_by_priority = str(best.get("item_name")) if best else None
    if not name_by_priority:
        return None

    unit_price = 0
    for p in prices:
        if p.get("item_name") == name_by_priority:
            unit_price = int(p.get("unit_price") or 0)
            break
    if unit_price <= 0:
        return None

    target_item = next(
        (it for it in rewards if isinstance(it, dict) and it.get("item_name") == name_by_priority),
        None,
    )
    if target_item is None:
        return None
    try:
        cur = int(target_item.get("count"))
    except Exception:
        return None

    step = _round_step_for(name_by_priority, hi)
    excess = total - target  # 正=需减少，负=需增加
    magnitude = abs(excess)

    if excess > 0:
        if unit_price >= step:
            # 高价物品：按单价进位减少
            new_count = cur - max(1, -(-magnitude // unit_price))
        else:
            # 总值对齐到 ≤hi 的最近阶梯值
            aligned_total = (target // step) * step
            if aligned_total < lo:
                aligned_total += step
            new_count = cur - max(1, -(-(total - aligned_total) // unit_price))
    else:
        if unit_price >= step:
            new_count = cur + max(1, -(-magnitude // unit_price))
        else:
            # 总值对齐到 ≥lo 的最近阶梯值
            aligned_total = -(-target // step) * step
            if aligned_total > hi:
                aligned_total -= step
            new_count = cur + max(1, -(-(aligned_total - total) // unit_price))
    if new_count < 1:
        if excess > 0:
            return None  # 该项已减无可减
        new_count = 1

    target_item["count"] = new_count
    new_total = _items_total(_items_list(draft, "rewards"), item_registry)
    if not (lo <= new_total <= hi):
        # 一次未落回区间：按剩余差额精确补一刀（落区间优先于取整美感）
        remain = target - new_total
        if remain != 0 and unit_price > 0:
            fix_items = remain // unit_price
            candidate = new_count + fix_items
            trial = new_total + fix_items * unit_price
            if fix_items != 0 and candidate >= 1 and lo <= trial <= hi:
                target_item["count"] = candidate
                new_total = trial
    if lo <= new_total <= hi:
        return (
            f"已自动微调：奖励「{name_by_priority}」数量 {cur}→{target_item['count']}"
            f"（总值 {total}→{new_total}，区间 [{lo}, {hi}]，按 {_round_step_for(name_by_priority, hi)} 取整）。"
        )
    target_item["count"] = cur
    return None


# ---------------------------------------------------------------------------
# V10：装备降级 / 移除
# ---------------------------------------------------------------------------

def _repair_v10(draft: dict[str, Any], issue) -> str | None:
    detail = issue.detail
    key = detail.get("key")
    name = detail.get("item_name")
    replacement = detail.get("replacement")
    if not key or not name:
        return None
    items = draft.get(key)
    if not isinstance(items, list):
        return None
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or it.get("item_name") != name:
            continue
        if replacement:
            it["item_name"] = replacement
            return (
                f"已自动微调：{key} 中装备「{name}」超过阶段等级上限，"
                f"已降级替换为「{replacement}」。"
            )
        removed = items.pop(idx)
        _ = removed
        return f"已自动微调：{key} 中装备「{name}」超过阶段等级上限且无低级替代，已移除该项。"
    return None


# ---------------------------------------------------------------------------
# V11：提交/奖励去重（保留 rewards 侧）
# ---------------------------------------------------------------------------

def _repair_v11(draft: dict[str, Any], issue) -> str | None:
    overlap = issue.detail.get("overlap") or []
    if not overlap:
        return None
    submit = draft.get("finish_submit_items")
    if not isinstance(submit, list):
        return None
    removed: list[str] = []
    kept: list[Any] = []
    for it in submit:
        name = it.get("item_name") if isinstance(it, dict) else (
            str(it).split("#", 1)[0] if isinstance(it, str) else None
        )
        if isinstance(name, str) and name.strip() in overlap:
            removed.append(name.strip())
        else:
            kept.append(it)
    if not removed:
        return None
    draft["finish_submit_items"] = kept
    return f"已自动微调：已从 finish_submit_items 移除与 rewards 重名的物品 {'、'.join(sorted(set(removed)))}（保留 rewards 侧）。"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def auto_repair(
    draft: dict[str, Any],
    report: ValidationReport,
    *,
    context: DraftValidationContext,
    game_data: "GameDataRegistry | None" = None,
) -> tuple[dict[str, Any], list[str], ValidationReport]:
    """按 05 §3 对可自动修复的 issue 逐条修复并复审。

    返回 (修复后草案, 修复说明列表, 剩余报告)。
    """
    if game_data is None:
        from services.game_data.registry import get_game_data_registry
        game_data = get_game_data_registry()

    notes: list[str] = []
    working = dict(draft)

    for issue in report.issues:
        if not issue.auto_repairable:
            continue
        try:
            if issue.rule == "V2":
                note = _repair_v2(working, issue)
            elif issue.rule == "V7":
                note = _repair_v7(working, issue, game_data.items)
            elif issue.rule == "V10":
                note = _repair_v10(working, issue)
            elif issue.rule == "V11":
                note = _repair_v11(working, issue)
            else:
                note = None
        except Exception:
            note = None
        if note:
            notes.append(note)

    if not notes:
        return working, [], report

    fresh = validate_task_draft(working, context=context, game_data=game_data)
    # 保留与修复无关的 warning（如 V9 雷同提醒）
    fresh.warnings = list(fresh.warnings) + [
        w for w in report.warnings if w not in fresh.warnings
    ]
    return working, notes, fresh
