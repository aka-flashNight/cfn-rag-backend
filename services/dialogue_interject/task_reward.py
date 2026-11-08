"""任务前插话：临时新增任务奖励（只新增、不改原有奖励、不做减少）。

写盘权威在游戏侧：后端只发决策（SSE `action`）与收回执，不写任何游戏仓文件、
不写覆盖层。额度与「赠送价值上限」同源，见 `item_value.gift_value_cap`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from services.dialogue_interject.item_pool import item_price
from services.dialogue_interject.item_value import gift_value_cap
from services.game_data.reward_utils import REWARD_STAGE_BASE_MAX, parse_name_count

#: 工具名与硬顶：最多 2 样、总额不超过同期一个对话类任务额度
ADJUST_TOOL = "add_task_reward"
MAX_EXTRA_REWARDS = 2
#: 与 gift_value_cap 同源的任务类型；本能力不吃讨价还价放大
REWARD_TASK_TYPE = "传话"
BARGAIN_RATE = 1.0

#: 任务模式工具结果回来后的收尾要求（物品往来与临时新增奖励共用同一条主干）
TASK_NUMBERED_FOLLOWUP = (
    "\n按【原始剩余台词（按编号no）】从 1 开始逐条处理，"
    "最后用 continue 或 end 收尾；没有原剩余时用 end。"
)

_ADJUST_ARGS = {
    "type": "object",
    "properties": {
        "extra_rewards": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_EXTRA_REWARDS,
            "description": (
                "要额外新增的奖励，格式「名称#数量」，最多 2 样；数量为正整数。"
                "沿用任务原有奖励清单里的物品名。"
            ),
        },
    },
    "required": ["extra_rewards"],
    "additionalProperties": False,
}


def affinity_multiplier(favorability: int) -> float:
    """好感度系数：与 `validator._compute_reward_value_range` 同款分档。"""
    value = int(favorability or 0)
    if value >= 80:
        return 1.20
    if value >= 50:
        return 1.10
    if value >= 20:
        return 1.00
    return 0.90


def task_reward_enabled(request: Any, pending_claim: Any) -> bool:
    """注册条件：仅任务前插话、无选项动作、无待处理索取。"""
    return (
        str(getattr(request, "mode", "") or "") == "task"
        and getattr(getattr(request, "dialogue", None), "kind", "") == "task_pre"
        and getattr(request, "action", None) is None
        and pending_claim is None
    )


def reward_cap(*, stage: int, favorability: int) -> int:
    """同期一个对话类任务额度：与赠送价值上限同一公式。"""
    return int(gift_value_cap(
        stage=max(1, int(stage or 1)),
        favorability=int(favorability or 0),
    ))


def reward_basis(*, stage: int, favorability: int) -> dict[str, Any]:
    """action 载荷里的「额度依据」；字段名按两仓约定一律用中文名。"""
    return {
        "任务类型": REWARD_TASK_TYPE,
        "阶段": int(stage),
        "好感度": int(favorability),
        "好感度系数": affinity_multiplier(favorability),
        "阶段基础上限": REWARD_STAGE_BASE_MAX,
        "讨价还价放大倍数": BARGAIN_RATE,
    }


@dataclass(frozen=True)
class RewardCheck:
    """硬校验结果：不通过时 reason 非空，绝不产出超额 action。"""

    ok: bool
    reason: str
    total: int
    rewards: tuple[str, ...]


def check_extra_rewards(game_data: Any, raw: Any, *, cap: int) -> RewardCheck:
    """最多 2 样、数量正整数、可估价、总额 ≤ 上限；失败一律拒绝，不做自动降量。"""
    if not isinstance(raw, list) or not raw:
        return RewardCheck(False, "空清单", 0, ())
    if len(raw) > MAX_EXTRA_REWARDS:
        return RewardCheck(False, "超过 2 样", 0, ())
    total = 0
    normalized: list[str] = []
    for expr in raw:
        if not isinstance(expr, str):
            return RewardCheck(False, "格式不是「名称#数量」", 0, ())
        name, count = parse_name_count(expr)
        if not name or count <= 0:
            return RewardCheck(False, "数量必须是正整数", 0, ())
        price = item_price(game_data, name)
        if price <= 0:
            # 无法估价的物品按 0 计会绕开硬顶，整笔拒绝。
            return RewardCheck(False, f"{name} 无法估价", 0, ())
        total += price * count
        normalized.append(f"{name}#{count}")
    if total > int(cap):
        # 失败一律不给清单，调用方拿不到可下发的超额载荷。
        return RewardCheck(False, "总额超过上限", total, ())
    return RewardCheck(True, "", total, tuple(normalized))


def parse_adjust_arguments(call: dict[str, Any]) -> list[str]:
    """取工具调用里的 extra_rewards；非法一律回落为空清单。"""
    function = call.get("function") or {}
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except (ValueError, TypeError):
            args = {}
    if not isinstance(args, dict):
        return []
    raw = args.get("extra_rewards")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def task_reward_tool_schema() -> dict[str, Any]:
    """任务前插话专用工具说明：只写行为判据，不提 Host/写盘/系统机制。"""
    return {
        "type": "function",
        "function": {
            "name": ADJUST_TOOL,
            "description": (
                "任务前对话里，你想在原有报酬之外额外添一点报酬时调用。"
                "只新增奖励，任务原有奖励不变，也不减少任何奖励。"
                "extra_rewards 用「名称#数量」，最多 2 样，数量是正整数；"
                "名称沿用【任务上下文】里奖励清单已有的物品名。"
                "只有办理结果确认办成，才能说加了。"
                "调用这一轮不要输出台词。"
            ),
            "parameters": _ADJUST_ARGS,
        },
    }
