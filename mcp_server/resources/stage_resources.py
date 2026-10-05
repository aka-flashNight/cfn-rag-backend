"""
game:// 关卡资源 —— 将关卡数据暴露为 MCP Resource。
"""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_server.context import AppContext
from mcp_server.tools.query_tools import _format_stage


def register_stage_resources(mcp: FastMCP, ctx: AppContext) -> None:

    @mcp.resource("game://stages/list")
    def list_stages_resource() -> list[dict]:
        """所有关卡列表。"""
        stages = ctx.registry.stages
        return [
            _format_stage(si) for si in sorted(
                stages._stage_infos.values(),
                key=lambda si: (si.area, si.name),
            )
        ]

    @mcp.resource("game://stages/{area}")
    def list_stages_by_area_resource(area: str) -> list[dict]:
        """某区域下的所有关卡。"""
        stages = ctx.registry.stages
        return [
            _format_stage(si)
            for (a, _), si in stages._stage_infos.items()
            if a == area
        ]
