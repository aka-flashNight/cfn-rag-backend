from __future__ import annotations

from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext
from services.agent_tools.schemas import DRAFT_AGENT_TASK_PARAMETERS_SCHEMA
from services.agent_tools.skill_tool_executors import execute_draft_agent_task


DESCRIPTION = "生成并校验任务草案，暂存到 DB。"


class DraftAgentTaskSkill(BaseSkill):
    name = "draft_agent_task"
    category = "task"
    description = DESCRIPTION
    parameters_schema = DRAFT_AGENT_TASK_PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        result_str, draft = execute_draft_agent_task(
            args,
            pending_draft=ctx.pending_draft,
            npc_name=ctx.npc_name,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            game_data=ctx.game_data,
            rag_context_text=ctx.rag_context_text,
        )
        # 第三元 task_write_result 仅 confirm/cancel 使用
        return result_str, draft, None


skill = DraftAgentTaskSkill()
