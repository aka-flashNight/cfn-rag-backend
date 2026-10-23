"""赠送价值上限、物品信息块与装备强化度描述。"""

from __future__ import annotations

from typing import Any, Optional

from services.agent_tools.validator import _compute_reward_value_range
from services.game_data.models import Item
from services.game_data.item_sources import format_item_sources
from services.game_entity_prompts import compute_reward_tags, format_item_prompt_line

from .item_pool import item_price

#: 与任务奖励校验同一公式：传话、无提交/持有加成、不吃讨价还价
GIFT_TASK_TYPE = "传话"
#: 金币/经验值的自动降量再按 500 取整
_ROUND_500 = {"金币", "经验值"}


def gift_value_cap(*, stage: int, favorability: int) -> int:
    _, cap = _compute_reward_value_range(
        stage=max(1, int(stage or 1)),
        task_type=GIFT_TASK_TYPE,
        submit_value=0,
        contain_value=0,
        affinity=int(favorability or 0),
        bargain_rate=1.0,
    )
    return int(cap)


def reduce_gift_quantity(
    game_data: Any,
    name: str,
    requested: int,
    *,
    cap: int,
) -> tuple[int, Optional[str]]:
    """返回（实际数量，失败原因）。失败原因为 None 表示可赠送。"""
    qty = int(requested or 0)
    if qty <= 0:
        return 0, "invalid_quantity"
    price = item_price(game_data, name)
    if price <= 0:
        return 0, "no_price"
    if qty * price <= cap:
        return qty, None
    qty = int(cap // price)
    if name in _ROUND_500:
        qty = (qty // 500) * 500
    if qty < 1:
        return 0, "value_exceeded"
    return qty, None


def enhancement_description(level: Optional[int]) -> str:
    """装备强化度成本描述；未强化（≤1）返回空串。"""
    value = int(level or 0)
    if value <= 1:
        return ""
    if value < 5:
        return "经过了一定幅度的强化"
    if value <= 8:
        return "经过了较高成本的强化"
    return "经过了极高成本的强化"


def format_item_info(game_data: Any, name: str) -> str:
    """物品信息块：优先走注册表；伪货币/成长项合成最小 Item。"""
    item = None
    reg = getattr(game_data, "items", None)
    getter = getattr(reg, "get_by_name", None)
    if getter is not None:
        try:
            item = getter(name)
        except Exception:
            item = None
    price = item_price(game_data, name)
    if item is None:
        if price <= 0:
            return f"名称：{name}"
        item = Item(name=name, type="货币" if name == "金币" else "成长", price=price)
    tags: list[str] = []
    try:
        tags = compute_reward_tags(item, getattr(game_data, "equipment_mods", None))
    except Exception:
        tags = []
    sources = ""
    try:
        sources = format_item_sources(game_data, name)
    except Exception:
        sources = ""
    return format_item_prompt_line(
        item,
        reward_tags=tags,
        price=price,
        sources=sources or None,
    )


def format_gift_result(
    name: str,
    *,
    requested: int,
    actual: int,
    adjusted: bool,
) -> str:
    if adjusted:
        return f"已成功赠送 {actual} 个 {name}（原请求 {requested}，因价值上限调整）"
    return f"已成功赠送 {actual} 个 {name}"


def format_gift_failure(name: str) -> str:
    return f"你选择了赠送{name}，因价值过高，赠送失败。"


def format_gift_no_price(name: str) -> str:
    return f"你选择了赠送{name}，无法估价，赠送失败。"


def format_take_completion(
    name: str,
    quantity: int,
    *,
    item_level: Optional[int] = None,
) -> str:
    desc = enhancement_description(item_level)
    if desc:
        return f"玩家把 {quantity} 个 {name}（强化度{int(item_level or 0)}，{desc}）交给了你。"
    return f"玩家把 {quantity} 个 {name} 交给了你。"


def format_recent_take(
    name: str,
    quantity: int,
    *,
    item_level: Optional[int] = None,
) -> str:
    desc = enhancement_description(item_level)
    if desc:
        return (
            f"玩家刚刚给了你 {quantity} 个 {name}"
            f"（强化度{int(item_level or 0)}，{desc}）"
        )
    return f"玩家刚刚给了你 {quantity} 个 {name}"
