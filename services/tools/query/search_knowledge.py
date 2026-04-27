"""search_knowledge tool — 基于 RAG 的自由文本检索。"""

from __future__ import annotations

from typing import Any

from services.agent_tools.handlers import execute_search_knowledge
from services.agent_tools.schemas import SEARCH_KNOWLEDGE_TOOL_PARAMETERS_SCHEMA
from services.tools.base import BaseTool, ToolContext, ToolResult


class SearchKnowledgeTool(BaseTool):
    name = "search_knowledge"
    category = "query"
    description = (
        "基于 RAG 的世界观/剧情/冷门设定自由文本检索。"
        "当你不确定某个罕见设定（阵营历史、角色背景、过往事件等）时使用；"
        "关卡/物品/NPC 的结构化属性请优先用 search_stages / search_items。"
    )
    parameters_schema = SEARCH_KNOWLEDGE_TOOL_PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result = execute_search_knowledge(args, retrieve_fn=ctx.retrieve_fn)
        return ToolResult(result_json=result)


tool = SearchKnowledgeTool()
