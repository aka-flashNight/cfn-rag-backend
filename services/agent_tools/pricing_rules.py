"""任务定价卡与字段说明生成（对应 docs/v3-developer/05 §2 的模型侧呈现）。

**与校验器同源**：区间数字一律调用 ``validator._compute_reward_value_range`` 计算，
保证"给模型的提示词"与"后端校验"口径完全一致，杜绝两套公式漂移。

每个节点只注入当前需要的块（03 §5）：
- ``build_pricing_card``：draft/update 轮注入（V7 区间、提交品/持有品加成精确公式+算例）；
- ``build_bargain_card``：仅 update（讨价还价）轮注入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from services.agent_tools.validator import (
    COMBAT_TASK_TYPES,
    _compute_reward_value_range,
)
from services.game_progress import get_progress_stage_config

if TYPE_CHECKING:
    from services.game_data.registry import GameDataRegistry

# 常用单例物品的单价提示（与 items 注册表一致；写死避免每次查表）
_SINGLETON_PRICES = {"金币": 1, "经验值": 1, "K点": 50, "技能点": 1500, "强化石": 300}

# 讨价还价上限放大倍数（与 handlers.BARGAIN 的 bargain_rate=1.5 一致）
BARGAIN_RATE = 1.5


def _basic_range_text(task_type: str, stage: int, affinity: int) -> tuple[int, int, str]:
    """基础区间（无提交品/持有品）与展开式说明。"""
    cfg = get_progress_stage_config(stage)
    aff_name = "0.9" if affinity < 20 else ("1.0" if affinity < 50 else ("1.1" if affinity < 80 else "1.2"))
    lo, hi = _compute_reward_value_range(
        stage=stage, task_type=task_type, submit_value=0, contain_value=0,
        affinity=affinity, bargain_rate=1.0,
    )
    type_mult = 2 if task_type in COMBAT_TASK_TYPES else 1
    explain = (
        f"基础区间 = 阶段 {stage} 基数 [{stage * 10000}, {stage * 30000}]"
        f" × 类型倍率 {type_mult}（{'战斗类 ×2' if type_mult == 2 else '非战斗类 ×1'}）"
        f" × 好感系数 {aff_name}（当前好感 {affinity}）"
    )
    _ = cfg
    return lo, hi, explain


def build_pricing_card(
    *,
    task_type: str,
    stage: int,
    affinity: int,
    bargain_rate: float = 1.0,
    game_data: "GameDataRegistry | None" = None,
) -> str:
    """draft 轮的定价卡：奖励总价怎么算、区间是多少、加成怎么叠加。"""
    lo, hi, explain = _basic_range_text(task_type, stage, affinity)
    lines = [
        "【奖励定价规则（与后端校验完全一致，按此计算即可通过）】",
        "1. 奖励总价值 = Σ(每个奖励的数量 × 该物品单价)。常用物品单价："
        + "、".join(f"{k}={v}" for k, v in _SINGLETON_PRICES.items())
        + "；其他物品单价以候选列表中标注的「单价/price」为准（金币单价恒为 1）。",
        f"2. 本次任务（{task_type}，玩家阶段 {stage}）的基础区间：[{lo}, {hi}]。",
        f"   计算式：{explain}。",
        "3. 若任务带提交品（finish_submit_items）或持有品（finish_contain_items），区间会**上移**：",
        "   - 提交品加成：设提交品总价 S = Σ(数量×单价)，则 下限 +min(S, 基础下限×2)，上限 +min(S, 基础上限×2)；",
        "     算例：基础区间 [9000, 27000]，提交「普通hp药剂×10」(单价250，S=2500) → 下限 +2500、上限 +5000 → 实际区间 [11500, 32000]；",
        "   - 持有品加成：设持有品总价 H，则 下限 +min(H×0.5, 基础下限×0.5)，上限同式；",
        "     算例：持有「战宠灵石×4」(单价300，H=1200) → 上下限各 +600 → [9600, 27600]。",
        f"4. 你的 rewards 总价必须落在**最终区间**（基础区间 + 加成）内，越界会被校验打回。",
    ]
    if bargain_rate > 1.0:
        lines.append(
            f"5. 当前为讨价还价轮：区间上限已放宽至 ×{bargain_rate}（即 [{lo}, {int(hi * bargain_rate)}]"
            "，下限不变）。"
        )
    lines.append(
        "5. 常见错误（都会被打回）：rewards 与 finish_submit_items 出现同名物品（报酬误填进提交）；"
        "数量超过候选列表标注的合理范围；选了候选列表之外的物品。"
    )
    return "\n".join(lines)


def build_bargain_card(*, stage: int, affinity: int, task_type: str, game_data: "GameDataRegistry | None" = None) -> str:
    """update（讨价还价）轮的规则卡：只讲调幅与边界。"""
    lo, hi, _ = _basic_range_text(task_type, stage, affinity)
    return "\n".join([
        "【讨价还价规则】",
        "1. 使用 update_task_draft(draft_id, modify_fields) 只修改要变更的字段，未提及的字段保持原值。",
        f"2. 当前区间 [{lo}, {hi}]，讨价还价时上限放宽 ×{BARGAIN_RATE}（→ [{lo}, {int(hi * BARGAIN_RATE)}]，下限不变），"
        "修改后的 rewards 总价必须落在放宽后的区间内。",
        "3. 讨价还价最多 2 次（已计入计数）；超过会返回错误，此时应向玩家说明无法再让步。",
        "4. 不要传 description / get_dialogue / finish_dialogue（只在玩家接受发布时写入）。",
    ])


def build_direction_block(
    *,
    direction: str,
    reward_hint: str = "",
    note: str = "",
    deviation_note: str = "",
) -> str:
    """任务方向继承块：交流 Agent 给的思路，任务 Agent 必须遵守。"""
    lines = ["【任务方向（继承自对话，必须严格遵守）】"]
    if direction:
        lines.append(f"方向：{direction}")
    if reward_hint:
        lines.append(f"奖励偏好：{reward_hint}")
    if note:
        lines.append(f"备注：{note}")
    if deviation_note:
        lines.append(f"此前偏离说明（供参考）：{deviation_note}")
    lines.append(
        "候选不足时才允许偏离方向，且最终必须说明偏离原因；"
        "拟定的任务内容必须与最近对话中 NPC 的发言一致，不得冲突。"
    )
    return "\n".join(lines)
