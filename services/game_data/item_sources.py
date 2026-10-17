"""物品获取途径（对应游戏侧物品图鉴注释的来源口径）。

用途：凡是把「物品详细信息」交给大模型的地方，都在行尾补一句「获取途径：…」，
让模型知道这件事物**从哪来**（而不是只知道它值多少钱）。

来源口径（与图鉴注释一致，全部由已加载的注册表反查，无额外数据文件）：
- 合成：配方产物（附带配方来源表，如「烹饪」「武器合成」）；
- 商店：NPC 金币商店（图鉴里就是这个 NPC 卖）；
- K点商店；
- 关卡掉落：箱子掉落（按关卡名）；
- 任务：核心任务池里的提交物 / 奖励物。

反查索引第一次构建约 0.15s（扫 152 个关卡 XML，掉落记录仅数十条），
之后挂在 GameDataRegistry 实例上复用，故不引入启动开销。
"""

from __future__ import annotations

from typing import Any, Optional

# 索引缓存挂在注册表实例上（注册表是单例、只读，reload 会重建实例故无脏读风险）
_CACHE_ATTR = "_item_source_index_cache"


def _build_index(game_data: Any) -> dict[str, dict[str, Any]]:
    """item 名 -> {craft: [来源表], shop: [NPC], kshop: bool, stage: [关卡名], task: [角色]}"""
    index: dict[str, dict[str, Any]] = {}

    def _entry(name: str) -> dict[str, Any]:
        return index.setdefault(
            name,
            {"craft": [], "shop": [], "kshop": False, "stage": [], "task": []},
        )

    # 1. 合成配方
    crafting = getattr(game_data, "crafting", None)
    for recipe in getattr(crafting, "_recipes", []) or []:
        name = getattr(recipe, "name", "") or ""
        if not name:
            continue
        src = (getattr(recipe, "source", "") or "").strip()
        e = _entry(name)
        if src and src not in e["craft"]:
            e["craft"].append(src)
        elif not src and "合成" not in e["craft"]:
            e["craft"].append("合成")

    # 2. NPC 商店
    shops_reg = getattr(game_data, "shops", None)
    shops = getattr(shops_reg, "_shops", {}) or {}
    for npc_name in shops:
        try:
            goods = shops_reg.get_npc_shop(npc_name) or []
        except Exception:  # pragma: no cover - 数据异常时不影响其它来源
            continue
        for item_name in goods:
            if item_name:
                e = _entry(item_name)
                if npc_name not in e["shop"]:
                    e["shop"].append(npc_name)

    # 3. K 点商店
    kshop = getattr(game_data, "kshop", None)
    for kitem in getattr(kshop, "list_items", lambda: [])() or []:
        item_name = getattr(kitem, "item", "") or ""
        if item_name:
            _entry(item_name)["kshop"] = True

    # 4. 关卡箱子掉落
    stages = getattr(game_data, "stages", None)
    for (area, stage_name) in list(getattr(stages, "_stage_infos", {}) or {}):
        try:
            crates = stages.get_stage_loot(area, stage_name) or []
        except Exception:  # pragma: no cover
            continue
        for crate in crates:
            for drop in getattr(crate, "drops", []) or []:
                drop_name = getattr(drop, "name", "") or ""
                if not drop_name:
                    continue
                e = _entry(drop_name)
                if stage_name not in e["stage"]:
                    e["stage"].append(stage_name)

    # 5. 核心任务池（提交 / 奖励）
    tasks = getattr(game_data, "tasks", None)
    try:
        submit_names = set(tasks.list_core_submit_items() or ())
        reward_names = set(tasks.list_core_reward_item_names() or ())
    except Exception:  # pragma: no cover
        submit_names, reward_names = set(), set()
    for name in submit_names:
        _entry(name)["task"].append("提交")
    for name in reward_names:
        e = _entry(name)
        if "奖励" not in e["task"]:
            e["task"].append("奖励")

    return index


def _index_for(game_data: Any) -> dict[str, dict[str, Any]]:
    cached = getattr(game_data, _CACHE_ATTR, None)
    if isinstance(cached, dict):
        return cached
    index = _build_index(game_data)
    try:
        setattr(game_data, _CACHE_ATTR, index)
    except Exception:  # pragma: no cover - 只读实现上setattr失败时退化为每次重建
        pass
    return index


def _named(items: list[str], *, limit: int, unit: str) -> str:
    """「甲」/「甲、乙」/「甲 等N个…」。"""
    if not items:
        return ""
    if len(items) <= limit:
        return "、".join(items)
    return f"{items[0]} 等{len(items)}{unit}"


def format_item_sources(
    game_data: Optional[Any],
    item_name: str,
    *,
    max_shop: Optional[int] = None,
    max_stage: int = 2,
) -> str:
    """物品获取途径一句话；查不到任何来源时返回空串（不写「未知」、不留占位）。

    商店默认**全列**（数据侧每件物品最多 4 家在售，全列代价极低，且模型需要知道
    「还有哪家能买到」）；关卡掉落仍折叠成「首个 等N个关卡」。

    例：``合成（调酒）；商店（酒保、The Girl）；关卡掉落（地铁站 等3关）；任务提交或奖励``
    """
    if game_data is None or not item_name:
        return ""
    rec = _index_for(game_data).get(item_name)
    if not rec:
        return ""

    parts: list[str] = []
    craft = rec.get("craft") or []
    if craft:
        details = [c for c in craft if c and c != "合成"]
        parts.append(f"合成（{'、'.join(details)}）" if details else "合成")

    shop_rows = rec.get("shop") or []
    if shop_rows:
        shop_txt = (
            "、".join(shop_rows) if max_shop is None
            else _named(shop_rows, limit=max_shop, unit="家商店")
        )
        parts.append(f"商店（{shop_txt}）")

    if rec.get("kshop"):
        parts.append("K点商店")

    stage_txt = _named(rec.get("stage") or [], limit=max_stage, unit="个关卡")
    if stage_txt:
        parts.append(f"关卡掉落（{stage_txt}）")

    task = rec.get("task") or []
    if task:
        if len(task) == 1:
            parts.append(f"任务{task[0]}")
        else:
            parts.append("任务提交或奖励")

    return "；".join(parts)
