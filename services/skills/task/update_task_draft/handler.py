from __future__ import annotations

from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext
from services.agent_tools.schemas import UPDATE_TASK_DRAFT_PARAMETERS_SCHEMA
from services.agent_tools.skill_tool_executors import execute_update_task_draft


DESCRIPTION = "局部修改已有草案并触发增量校验（仅校验变更字段）。"


class UpdateTaskDraftSkill(BaseSkill):
    name = "update_task_draft"
    category = "task"
    description = DESCRIPTION
    parameters_schema = UPDATE_TASK_DRAFT_PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        return execute_update_task_draft(
            args,
            pending_draft=ctx.pending_draft,
            npc_name=ctx.npc_name,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            game_data=ctx.game_data,
            rag_context_text=ctx.rag_context_text,
        )


skill = UpdateTaskDraftSkill()
