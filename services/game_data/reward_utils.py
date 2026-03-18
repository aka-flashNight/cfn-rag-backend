from __future__ import annotations

from typing import Iterable

from .item_registry import ItemRegistry


def parse_name_count(expr: str) -> tuple[str, int]:
    """
    解析通用的 "物品名#数量" 表达式。
    - 数量缺失/非法：按 0 处理
    """

    if expr is None:
        return "", 0
    s = str(expr).strip()
    if not s:
        return "", 0
    if "#" not in s:
        return s, 0
    name, cnt = s.split("#", 1)
    name = name.strip()
    try:
        count = int(str(cnt).strip() or "0")
    except Exception:
        count = 0
    return name, count


def calculate_rewards_value(rewards: Iterable[str], *, items: ItemRegistry) -> int:
    """
    将 rewards（["物品名#数量", ...]）按 items.price 换算为金币等价总额。
    - 若物品不存在或无 price：该项按 0 计
    """

    total = 0
    for r in rewards or []:
        name, count = parse_name_count(r)
        if not name or count <= 0:
            continue
        price = items.get_price(name)
        total += price * count
    return int(total)

