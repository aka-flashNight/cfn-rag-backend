"""
game:// 合成配方资源 —— 将合成配方数据暴露为 MCP Resource。
"""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_server.context import AppContext
from mcp_server.tools.query_tools import _format_recipe


def register_crafting_resources(mcp: FastMCP, ctx: AppContext) -> None:

    @mcp.resource("game://crafting/list")
    def list_crafting_resource() -> list[dict]:
        """所有合成配方列表（最多 200 条）。"""
        reg = ctx.registry.crafting
        return [_format_recipe(r) for r in reg.search("", limit=200)]
