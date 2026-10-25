"""物品赠送/索取的有效池与候选生成。

池只决定“可不可选”，不代表玩家真的持有；持有与装备实例由 Host/AS2 查询。
本模块不写盘、不改任务数据。
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

from services.agent_tools.pricing_rules import _SINGLETON_PRICES
from services.game_data.item_sources import _index_for

GOLD = "金币"
#: 索取不允许的成长/点数货币
_NON_TAKE_CURRENCIES = {"经验值", "K点", "技能点"}
#: 即时补给不进背包，不能作为赠礼/索取对象
_EXCLUDED_USES = {"战场补给"}
#: 测试/调试物品
_EXCLUDED_NAME_MARKERS = ("测试", "调试")

CATEGORIES = ("武器", "防具", "消耗品", "材料", "其他", "金币")


def _items_registry(game_data: Any) -> Any:
    return getattr(game_data, "items", None)


def _get_item(game_data: Any, name: str) -> Any:
    reg = _items_registry(game_data)
    getter = getattr(reg, "get_by_name", None)
    if getter is None:
        return None
    try:
        return getter(name)
    except Exception:
        return None


def item_price(game_data: Any, name: str) -> int:
    """单价：注册表优先，伪货币/成长项用固定单价；未知返回 0。"""
    fixed = _SINGLETON_PRICES.get(name)
    if fixed is not None:
        return int(fixed)
    reg = _items_registry(game_data)
    getter = getattr(reg, "get_price", None)
    if getter is None:
        return 0
    try:
        return int(getter(name) or 0)
    except Exception:
        return 0


def _task_submit_names(game_data: Any) -> set[str]:
    tasks = getattr(game_data, "tasks", None)
    getter = getattr(tasks, "list_core_submit_items", None)
    if getter is None:
        return set()
    try:
        return {str(name) for name in (getter() or set()) if str(name).strip()}
    except Exception:
        return set()


def _is_excluded_name(name: str) -> bool:
    return any(marker in name for marker in _EXCLUDED_NAME_MARKERS)


def is_pool_name(game_data: Any, name: str, *, direction: str) -> bool:
    """名称能否进入指定方向的有效池（不含来源过滤）。"""
    text = (name or "").strip()
    if not text or _is_excluded_name(text):
        return False
    if direction == "take" and text in _NON_TAKE_CURRENCIES:
        return False
    item = _get_item(game_data, text)
    if item is not None and str(getattr(item, "use", "") or "") in _EXCLUDED_USES:
        return False
    if text in _task_submit_names(game_data):
        return False
    return item_price(game_data, text) > 0


def build_gift_pool(game_data: Any) -> list[str]:
    """赠送池＝核心任务奖励池全量；只过滤不可计价/战场补给/测试/提交物。"""
    tasks = getattr(game_data, "tasks", None)
    getter = getattr(tasks, "list_core_reward_item_names", None)
    if getter is None:
        return []
    try:
        names = {str(name).strip() for name in (getter() or set()) if str(name).strip()}
    except Exception:
        return []
    return sorted(name for name in names if is_pool_name(game_data, name, direction="give"))


def giftable_names(game_data: Any, pool: Sequence[str], cap: int) -> list[str]:
    """随机候选池：剔除单价已超过赠送上限的物品（数量 1 也不可送）。"""
    limit = int(cap)
    return [name for name in pool if 0 < item_price(game_data, name) <= limit]


def build_take_pool(game_data: Any, player_item_names: Sequence[str] = ()) -> list[str]:
    """索取池＝有正常来源的物品 ∪ 金币 ∪ 玩家持有名；金币固定排第一。"""
    names: set[str] = {GOLD}
    try:
        names.update(_index_for(game_data).keys())
    except Exception:
        pass
    names.update(
        str(name).strip() for name in (player_item_names or ()) if str(name).strip()
    )
    filtered = [
        name for name in names
        if is_pool_name(game_data, name, direction="take")
    ]
    filtered.sort()
    if GOLD in filtered:
        filtered.remove(GOLD)
        filtered.insert(0, GOLD)
    return filtered


def search_names(
    game_data: Any,
    pool: Sequence[str],
    keyword: str,
    *,
    limit: int = 10,
) -> list[str]:
    """池内关键词检索；按池顺序返回，最多 limit 条。"""
    kw = (keyword or "").strip().lower()
    if not kw:
        return []
    out: list[str] = []
    for name in pool:
        item = _get_item(game_data, name)
        hay = name
        if item is not None:
            hay = f"{name} {getattr(item, 'displayname', '') or ''}"
        if kw in hay.lower():
            out.append(name)
            if len(out) >= limit:
                break
    return out


def sample_names(
    pool: Sequence[str],
    *,
    count: int | None = None,
    gold_first: bool = False,
    rng: random.Random | None = None,
) -> list[str]:
    """从池中抽 10–20 条；索取池保证金币在第一位。"""
    rng = rng or random.Random()
    names = list(dict.fromkeys(pool))
    gold: list[str] = []
    if gold_first and GOLD in names:
        gold = [GOLD]
        names = [name for name in names if name != GOLD]
    if count is None:
        count = rng.randint(10, 20)
    count = max(0, min(int(count), len(names) + len(gold)))
    rest_count = max(0, count - len(gold))
    if rest_count >= len(names):
        rest = list(names)
        rng.shuffle(rest)
    else:
        rest = rng.sample(names, rest_count)
    return gold + rest


def item_category(game_data: Any, name: str) -> str:
    if name == GOLD:
        return "金币"
    item = _get_item(game_data, name)
    if item is None:
        return "其他"
    it_type = str(getattr(item, "type", "") or "").strip()
    it_use = str(getattr(item, "use", "") or "").strip()
    if it_type == "武器":
        return "武器"
    if it_type == "防具":
        return "防具"
    if it_type == "消耗品" or it_use in {"药剂", "弹夹", "食品", "食材"}:
        return "消耗品"
    if it_type == "收集品" or it_use == "材料":
        return "材料"
    return "其他"


def category_counts(game_data: Any, names: Iterable[str]) -> dict[str, int]:
    counts = {key: 0 for key in CATEGORIES}
    for name in names:
        category = item_category(game_data, name)
        counts[category] = counts.get(category, 0) + 1
    return counts


def remaining_pool_counts(
    game_data: Any,
    pool: Sequence[str],
    shown: Sequence[str],
) -> tuple[dict[str, int], int]:
    """本轮候选之外剩余池的分类计数与有效池总数。"""
    shown_set = set(shown)
    remaining = [name for name in pool if name not in shown_set]
    return category_counts(game_data, remaining), len(pool)
