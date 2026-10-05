"""
game:// 物品资源 —— 将游戏物品表暴露为 MCP Resource。
"""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_server.context import AppContext
from mcp_server.tools.query_tools import _format_item


def register_item_resources(mcp: FastMCP, ctx: AppContext) -> None:

    @mcp.resource("game://items/{name}")
    def get_item_resource(name: str) -> dict | None:
        """获取指定物品的完整数据。"""
        reg = ctx.registry.items
        it = reg.get_by_name(name)
        if it is None:
            return None
        return _format_item(it)

    @mcp.resource("game://items/list")
    def list_items_resource() -> list[dict]:
        """所有物品列表（最多返回前 200 条）。"""
        reg = ctx.registry.items
        return [_format_item(it) for it in reg.items[:200]]
