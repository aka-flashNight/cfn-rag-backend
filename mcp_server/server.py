"""
CFN-RAG MCP Server 入口点。

提供游戏数据查询与创作辅助能力：
- Tools: 搜索物品/关卡/配方/NPC 商店/任务
- Resources: 以结构化 URI 浏览游戏数据
- Prompts: 任务设计、对话创作、数据探索模板

使用方式:
    python -m mcp_server.server          # stdio 模式（Claude Desktop / Cursor）
    fastmcp dev mcp_server/server.py     # MCP Inspector 调试
"""

from __future__ import annotations

import logging
import sys

from fastmcp import FastMCP

from mcp_server.context import get_app_context, reset_app_context

# 配置日志到 stderr（stdio transport 下 stdout 用于 JSON-RPC）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("cfn-mcp-server")

# ---------------------------------------------------------------------------
# FastMCP 实例
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "CFN-RAG Game Data Server",
    instructions="Crazy Flash Night 游戏数据查询与创作辅助 MCP Server。提供物品、关卡、NPC 商店、任务、合成配方的查询能力。",
)


# ---------------------------------------------------------------------------
# 注册 Tools / Resources / Prompts
# ---------------------------------------------------------------------------

def _register_all() -> None:
    """初始化上下文并注册所有 MCP 组件。"""
    logger.info("正在初始化游戏数据上下文...")
    ctx = get_app_context()

    if ctx.ready:
        logger.info("游戏数据加载成功，注册 Tools 和 Resources")
    else:
        logger.warning("游戏数据未加载，MCP Server 将以最小模式运行（部分 Tool 可能不可用）")

    # Tools
    from mcp_server.tools.query_tools import register_query_tools
    register_query_tools(mcp, ctx)

    # Resources
    from mcp_server.resources.item_resources import register_item_resources
    from mcp_server.resources.stage_resources import register_stage_resources
    from mcp_server.resources.npc_resources import register_npc_resources
    from mcp_server.resources.task_resources import register_task_resources
    from mcp_server.resources.crafting_resources import register_crafting_resources

    register_item_resources(mcp, ctx)
    register_stage_resources(mcp, ctx)
    register_npc_resources(mcp, ctx)
    register_task_resources(mcp, ctx)
    register_crafting_resources(mcp, ctx)

    # Prompts
    from mcp_server.prompts.game_design_prompts import register_prompts
    register_prompts(mcp)

    logger.info("所有 MCP 组件注册完成")


_register_all()


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------

def main() -> None:
    """启动 MCP Server（stdio 模式）。"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
