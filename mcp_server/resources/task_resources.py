"""
game:// 任务资源 —— 将任务数据暴露为 MCP Resource。
"""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_server.context import AppContext
from mcp_server.tools.query_tools import _format_task


def register_task_resources(mcp: FastMCP, ctx: AppContext) -> None:

    @mcp.resource("game://tasks/list")
    def list_tasks_resource() -> list[dict]:
        """所有任务列表（最多 200 条）。"""
        reg = ctx.registry.tasks
        return [_format_task(t) for t in reg.list_all_tasks()[:200]]

    @mcp.resource("game://tasks/agent")
    def list_agent_tasks_resource() -> list[dict]:
        """所有 AI 代理生成的任务。"""
        reg = ctx.registry.tasks
        return [_format_task(t) for t in reg.list_agent_tasks()]

    @mcp.resource("game://tasks/{task_id}")
    def get_task_resource(task_id: int) -> dict | None:
        """根据 ID 获取任务详情。"""
        reg = ctx.registry.tasks
        task = reg.get_by_id(task_id)
        if task is None:
            return None
        return _format_task(task)
