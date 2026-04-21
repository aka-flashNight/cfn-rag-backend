from __future__ import annotations

from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext
from services.agent_tools.schemas import SEARCH_KNOWLEDGE_TOOL_PARAMETERS_SCHEMA
from services.agent_tools.skill_tool_executors import execute_search_knowledge


DESCRIPTION = "复用现有 RAG 检索，获取设定/情报的摘要文本。"


class SearchKnowledgeSkill(BaseSkill):
    name = "search_knowledge"
    category = "query"
    description = DESCRIPTION
    parameters_schema = SEARCH_KNOWLEDGE_TOOL_PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        result = execute_search_knowledge(
            args,
            retrieve_fn=ctx.retrieve_fn,
        )
        return result, None, None


skill = SearchKnowledgeSkill()
