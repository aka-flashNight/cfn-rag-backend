"""后端兜底草案（对应 docs/v3-developer/05 §5）。

TaskRunner 轮限耗尽仍拿不到合法草案时的最终保险：用 prepare_task_context 的候选，
按最保守规则构造**100% 通过全量校验**的草案，保证「拟定了方向就能拿出草案」。

- task_type 取 direction 指定类型（子串匹配），缺省「资源收集」；
- 目标/关卡取候选中等级最低、进度合法的；
- 奖励从合规候选（商店 ∪ 历史奖励）中选单价最低物品，数量对齐 V7 区间中位数附近；
- 标题/说明用模板（「委托：收集XX」），confirm 时再润色。
兜底草案照常进入 HITL（玩家可见可拒），标记 fallback=True。
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from services.agent_tools.context_builder import prepare_task_context
from services.agent_tools.schemas import normalize_reward_types_for_prepare_context
from services.agent_tools.validator import (
    DraftValidationContext,
    _compute_reward_value_range,
    validate_task_draft,
)
from services.game_progress import get_progress_stage_config

if TYPE_CHECKING:
    from services.game_data.registry import GameDataRegistry


class FallbackDraftError(RuntimeError):
    """兜底草案构造失败（候选完全不可用等极端情况）。"""


# direction 文本 → 任务类型 的子串映射（按声明顺序匹配）
_TASK_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("通关并收集", "通关并收集"),
    ("通关并持有", "通关并持有"),
    ("收集", "资源收集"),
    ("缴纳", "装备缴纳"),
    ("装备", "装备缴纳"),
    ("武器", "装备缴纳"),
    ("持有", "物品持有"),
    ("切磋", "切磋"),
    ("清理", "清理"),
    ("挑战", "挑战"),
    ("通关", "通关"),
    ("传话", "传话"),
    ("问候", "问候"),
    ("特殊物品", "特殊物品获取"),
]

_FALLBACK_STAGES = ("通关", "清理", "挑战", "通关并收集", "通关并持有", "切磋")


def detect_task_type(direction: str) -> str:
    """从任务方向文本推断任务类型（无命中回退「资源收集」）。"""
    text = (direction or "").strip()
    for keyword, task_type in _TASK_TYPE_KEYWORDS:
        if keyword in text:
            return task_type
    return "资源收集"


def _prepare_candidates(
    *,
    task_type: str,
    direction: str,
    reward_hint: str,
    npc_name: str,
    npc_faction: str,
    npc_challenge: str | None,
    player_progress: int,
    npc_affinity: int,
    npc_states: dict[str, Any] | None,
    game_data: "GameDataRegistry",
) -> dict[str, Any]:
    """调用 prepare_task_context 取候选（关键词用 direction 全文，尽量贴合方向）。"""
    keywords = [w for w in (direction or "").replace("，", " ").replace(",", " ").split() if w]
    reward_types = normalize_reward_types_for_prepare_context(None, keywords + [reward_hint])
    raw = prepare_task_context(
        task_type=task_type,
        reward_types=reward_types,
        npc_name=npc_name,
        npc_faction=npc_faction,
        npc_challenge=npc_challenge,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
        npc_states=npc_states,
        requirement_keywords=keywords or None,
        game_data=game_data,
    )
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise FallbackDraftError("prepare_task_context 返回结构异常")
    return parsed


def _pick_submit_items(
    *, task_type: str, candidates: dict[str, Any], game_data: "GameDataRegistry",
) -> list[dict[str, Any]]:
    """提交/持有物品：取候选列表中等级最低、单价最低的前 1~2 项（最保守）。"""
    for key in ("collectable_items", "holdable_items", "equipment_items", "special_items"):
        rows = candidates.get(key)
        if isinstance(rows, list) and rows:
            def _sort_key(r: dict[str, Any]) -> tuple[int, int]:
                lv = int(r.get("level") or 0)
                price = int(r.get("price") or 0)
                return (lv, price)

            ordered = sorted(rows, key=_sort_key)
            picked: list[dict[str, Any]] = []
            for r in ordered:
                name = str(r.get("name") or "").strip()
                if not name or game_data.items.get_by_name(name) is None:
                    continue
                picked.append({"item_name": name, "count": 1})
                if len(picked) >= 1:
                    break
            if picked:
                return picked
    # 通关类可无提交物
    return []


def _pick_stage_requirements(
    *, task_type: str, candidates: dict[str, Any], context: DraftValidationContext,
    game_data: "GameDataRegistry",
) -> list[dict[str, Any]]:
    """通关族任务：取进度合法、解锁最早的关卡，难度固定「简单」。"""
    if task_type not in _FALLBACK_STAGES:
        return []
    rows: list[dict[str, Any]] = []
    for key in ("stage_list", "stage_loot_list"):
        val = candidates.get(key)
        if isinstance(val, list) and val:
            rows = val
            break
    if not rows:
        raise FallbackDraftError(f"任务类型「{task_type}」没有可用关卡候选")

    # stage_list 为大区分组结构：[{area, stages: [{name, unlock_id, ...}]}]；
    # 拍平后按解锁主线 ID 升序取最保守的一个（难度固定「简单」）。
    flat: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        inner = row.get("stages")
        if isinstance(inner, list):
            flat.extend(s for s in inner if isinstance(s, dict))
        else:
            flat.append(row)

    def _sort_key(r: dict[str, Any]) -> int:
        return int(r.get("unlock_id") or r.get("unlock_condition") or 0)

    ordered = sorted(flat, key=_sort_key)
    for r in ordered:
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        req = {"stage_name": name, "difficulty": "简单"}
        # 用最小区间快速预检 V4（超进度直接跳过）
        si = None
        infos = getattr(game_data.stages, "_stage_infos", {})
        for (_area, n), info in infos.items():
            if n == name:
                si = info
                break
        unlock = getattr(si, "unlock_condition", None) if si else None
        if isinstance(unlock, int) and unlock > context.main_task_max_id:
            continue
        return [req]
    raise FallbackDraftError("没有进度合法的关卡候选")


def _build_rewards(
    *,
    draft: dict[str, Any],
    candidates: dict[str, Any],
    context: DraftValidationContext,
    game_data: "GameDataRegistry",
) -> list[dict[str, Any]]:
    """奖励：取合规候选里单价最低的物品，数量对齐 V7 区间中位数（且满足 V2 上限）。"""
    submit_value = 0
    for it in draft.get("finish_submit_items") or []:
        submit_value += game_data.items.get_price(it["item_name"]) * it["count"]
    contain_value = 0
    for it in draft.get("finish_contain_items") or []:
        contain_value += game_data.items.get_price(it["item_name"]) * it["count"]

    task_type = str(draft.get("task_type") or "")
    lo, hi = _compute_reward_value_range(
        stage=context.stage,
        task_type=task_type,
        submit_value=submit_value,
        contain_value=contain_value,
        affinity=context.affinity,
        bargain_rate=context.bargain_rate,
    )
    target = (lo + hi) // 2

    rows = candidates.get("reward_item_candidates")
    if not isinstance(rows, list) or not rows:
        raise FallbackDraftError("没有可用的奖励物品候选")

    def _sort_key(r: dict[str, Any]) -> tuple[int, int]:
        # 单价从低到高；同价取等级低者
        price = int(r.get("price") or 0)
        level = int(r.get("level") or 0)
        return (price, level)

    ordered = sorted(rows, key=_sort_key)
    for r in ordered:
        name = str(r.get("name") or "").strip()
        item = game_data.items.get_by_name(name) if name else None
        if item is None:
            continue
        price = int(item.price or 0)
        if price <= 0:
            continue
        # V2 约束：rewards 无历史统计时上限 2；有统计时按统计×2
        reward_max_qty = 2
        try:
            stats = game_data.tasks.get_reward_stats()
            _, hist_max = stats.get(name, (None, 0))
            if int(hist_max or 0) > 0:
                reward_max_qty = int(hist_max) * 2
        except Exception:
            pass
        count = max(1, min(target // price, reward_max_qty))
        total = price * count
        # 数量受 V2 封顶后总值若仍越界，换下一候选
        if not (lo <= total <= hi):
            continue
        draft.setdefault("rewards", []).append({"item_name": name, "count": count})
        return draft["rewards"]
    raise FallbackDraftError("奖励候选无法落在 V7 区间内")


def build_fallback_draft(
    *,
    direction: str,
    reward_hint: str = "",
    npc_name: str,
    npc_faction: str = "",
    npc_challenge: str | None = None,
    player_progress: int = 1,
    npc_affinity: int = 0,
    npc_states: dict[str, Any] | None = None,
    game_data: "GameDataRegistry | None" = None,
) -> dict[str, Any]:
    """构造 100% 通过全量校验的保守草案；失败抛 FallbackDraftError。

    返回的草案带 ``fallback: True`` 标记与 ``fallback_note``（供 NPC 汇合话术使用）。
    """
    if game_data is None:
        from services.game_data.registry import get_game_data_registry
        game_data = get_game_data_registry()

    task_type = detect_task_type(direction)
    candidates = _prepare_candidates(
        task_type=task_type,
        direction=direction,
        reward_hint=reward_hint,
        npc_name=npc_name,
        npc_faction=npc_faction,
        npc_challenge=npc_challenge,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
        npc_states=npc_states,
        game_data=game_data,
    )

    cfg = get_progress_stage_config(max(1, min(7, player_progress)))
    context = DraftValidationContext(
        main_task_max_id=(cfg.main_task_max_id if cfg else 0) or 0,
        max_level=(cfg.max_level if cfg else 50) or 50,
        stage=max(1, min(7, player_progress)),
        affinity=npc_affinity,
        npc_name=npc_name or None,
    )

    draft: dict[str, Any] = {
        "task_type": task_type,
        "title": f"委托：{task_type}",
        "npc_name": npc_name,
        "fallback": True,
    }
    import uuid as _uuid

    draft["draft_id"] = str(_uuid.uuid4())[:8]
    draft["finish_requirements"] = _pick_stage_requirements(
        task_type=task_type, candidates=candidates, context=context, game_data=game_data,
    )
    if task_type in ("资源收集", "装备缴纳", "物品持有", "特殊物品获取"):
        submit_key = "finish_submit_items" if task_type != "物品持有" else "finish_contain_items"
        items = _pick_submit_items(task_type=task_type, candidates=candidates, game_data=game_data)
        if items:
            draft[submit_key] = items

    _build_rewards(draft=draft, candidates=candidates, context=context, game_data=game_data)

    report = validate_task_draft(draft, context=context, game_data=game_data)
    if not report.ok:
        raise FallbackDraftError(
            "兜底草案未通过校验: " + "; ".join(i.message for i in report.issues)
        )
    draft["fallback_note"] = "该草案由后端按最保守规则生成，信息可能比常规模型草案更模糊。"
    return draft
