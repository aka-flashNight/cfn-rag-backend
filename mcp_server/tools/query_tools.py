"""
CFN-RAG MCP 查询工具集。

将 GameDataRegistry 的查询能力封装为 MCP Tool。
所有 Tool 函数签名中的参数类型和 docstring 会自动生成 JSON Schema 供 LLM 使用。
"""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from mcp_server.context import AppContext


def register_query_tools(mcp: FastMCP, ctx: AppContext) -> None:

    @mcp.tool
    def search_items(
        keyword: str,
        type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """按关键词搜索游戏物品。返回物品名称、类型、用途、等级、价格等。

        当用户询问"有哪些武器/药剂/材料"或"查找某个物品"时使用。
        可通过 type 参数过滤大分类（如 武器/防具/消耗品/收集品）。
        """
        reg = ctx.registry.items
        results = reg.search(keyword, type=type, limit=limit)
        return [_format_item(it) for it in results]

    @mcp.tool
    def get_item_detail(name: str) -> Optional[dict]:
        """获取单个物品的完整详情，包含所有属性字段。"""
        reg = ctx.registry.items
        it = reg.get_by_name(name)
        if it is None:
            return None
        return _format_item(it)

    @mcp.tool
    def search_stages(
        keyword: str,
        area: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """按关键词搜索游戏关卡。返回关卡名、所属区域、类型、解锁条件、描述。

        area 可指定区域过滤（如 废城、基地门口、副本任务）。
        """
        stages = ctx.registry.stages
        kw = (keyword or "").strip().lower()
        if not kw:
            return []

        results = []
        for (a, name), si in stages._stage_infos.items():
            if area and a != area:
                continue
            hay = f"{name} {si.type or ''} {si.description or ''}".lower()
            if kw in hay:
                results.append(_format_stage(si))
                if len(results) >= limit:
                    break
        return results

    @mcp.tool
    def get_stage_loot(area: str, stage_name: str) -> dict:
        """查询指定关卡的掉落物信息。

        返回关卡基本信息 + 各箱子类型（纸箱/资源箱/装备箱）的掉落物列表，
        包含每种物品的掉落数量范围。
        """
        stages = ctx.registry.stages
        si = stages._stage_infos.get((area, stage_name))
        if si is None:
            return {"error": f"未找到关卡: {area}/{stage_name}"}

        crates = stages.get_stage_loot(area, stage_name)
        crate_list = []
        for c in crates:
            drops = [{"name": d.name, "min_count": d.min_count, "max_count": d.max_count} for d in c.drops]
            crate_list.append({"identifier": c.identifier, "drops": drops})

        return {
            "area": si.area,
            "name": si.name,
            "type": si.type,
            "unlock_condition": si.unlock_condition,
            "description": si.description,
            "loot_crates": crate_list,
        }

    @mcp.tool
    def search_crafting(keyword: str, limit: int = 20) -> list[dict]:
        """按关键词搜索合成/制造配方。返回产物名、所需材料、来源等信息。

        当用户询问"如何合成某物品"或"有哪些烹饪/武器合成配方"时使用。
        """
        reg = ctx.registry.crafting
        recipes = reg.search(keyword, limit=limit)
        return [_format_recipe(r) for r in recipes]

    @mcp.tool
    def get_npc_shop(npc_name: str) -> dict:
        """查询指定 NPC 的商店售卖的物品列表。返回 NPC 名称和售卖物品清单。"""
        reg = ctx.registry.shops
        items = reg.get_npc_shop(npc_name)
        return {
            "npc_name": npc_name,
            "has_shop": len(items) > 0,
            "items": items,
        }

    @mcp.tool
    def list_npcs_with_shops() -> list[str]:
        """列出所有拥有商店的 NPC 名称列表。用于了解游戏中哪些 NPC 可以进行交易。"""
        reg = ctx.registry.shops
        return sorted(reg._shops.keys())

    @mcp.tool
    def search_tasks_by_npc(npc_name: str, limit: int = 20) -> list[dict]:
        """查询指定 NPC 关联的任务列表（含接取和交付任务）。返回任务 ID、标题、描述、需求等信息。"""
        reg = ctx.registry.tasks
        tasks = reg.list_by_npc(npc_name)
        results = [_format_task(t) for t in tasks[:limit]]
        return results

    @mcp.tool
    def list_agent_tasks(limit: int = 30) -> list[dict]:
        """列出所有 AI 生成的代理任务。"""
        reg = ctx.registry.tasks
        tasks = reg.list_agent_tasks()
        return [_format_task(t) for t in tasks[:limit]]

    @mcp.tool
    def get_task_detail(task_id: int) -> Optional[dict]:
        """根据任务 ID 获取任务完整详情。"""
        reg = ctx.registry.tasks
        task = reg.get_by_id(task_id)
        if task is None:
            return None
        return _format_task(task)

    @mcp.tool
    def search_items_by_level(
        min_level: int,
        max_level: int,
        type: Optional[str] = None,
        use: Optional[str] = None,
    ) -> list[dict]:
        """按等级范围查询物品，支持类型和用途过滤。

        当用户询问"10-20 级的武器有哪些"时使用。
        """
        reg = ctx.registry.items
        results = reg.find(type=type, use=use, min_level=min_level, max_level=max_level)
        return [_format_item(it) for it in results]

    @mcp.tool
    def list_stage_areas() -> list[str]:
        """列出游戏中所有关卡区域（大区名称）。用于了解关卡的整体组织结构。"""
        stages = ctx.registry.stages
        areas = sorted({a for (a, _) in stages._stage_infos.keys()})
        return areas

    @mcp.tool
    def list_stages_in_area(area: str) -> list[dict]:
        """列出指定区域下的所有关卡。返回关卡名、类型、解锁条件、描述。"""
        stages = ctx.registry.stages
        results = [
            _format_stage(si)
            for (a, _), si in stages._stage_infos.items()
            if a == area
        ]
        return results


# ---------------------------------------------------------------------------
# 格式化辅助函数：将 Pydantic 模型转为 LLM 友好的 dict
# ---------------------------------------------------------------------------

def _format_item(it) -> dict:
    return {
        "name": it.name,
        "displayname": it.displayname,
        "type": it.type,
        "use": it.use,
        "level": it.level,
        "price": it.price,
        "description": it.description,
        "weapontype": it.weapontype,
        "weight": it.weight,
    }


def _format_stage(si) -> dict:
    return {
        "area": si.area,
        "name": si.name,
        "type": si.type,
        "unlock_condition": si.unlock_condition,
        "description": si.description,
    }


def _format_recipe(r) -> dict:
    return {
        "title": r.title,
        "product": r.name,
        "price": r.price,
        "kprice": r.kprice,
        "materials": r.materials,
        "source": r.source,
    }


def _format_task(t) -> dict:
    return {
        "id": t.id,
        "title": t.title,
        "description": t.description,
        "get_npc": t.get_npc,
        "finish_npc": t.finish_npc,
        "get_requirements": t.get_requirements,
        "finish_requirements": t.finish_requirements,
        "finish_submit_items": t.finish_submit_items,
        "finish_contain_items": t.finish_contain_items,
        "rewards": t.rewards,
    }
