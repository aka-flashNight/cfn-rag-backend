"""
game:// NPC 资源 —— 将 NPC 列表暴露为 MCP Resource。
"""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_server.context import AppContext


def register_npc_resources(mcp: FastMCP, ctx: AppContext) -> None:

    @mcp.resource("game://npcs/list")
    def list_npcs_resource() -> list[dict]:
        """所有拥有商店的 NPC 列表。"""
        reg = ctx.registry.shops
        result = []
        for npc_name in sorted(reg._shops.keys()):
            shop_items = reg.get_npc_shop(npc_name)
            result.append({
                "name": npc_name,
                "shop_item_count": len(shop_items),
            })
        return result
