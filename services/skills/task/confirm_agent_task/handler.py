from __future__ import annotations

from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext
from services.agent_tools.schemas import CONFIRM_AGENT_TASK_PARAMETERS_SCHEMA
from services.agent_tools.skill_tool_executors import execute_confirm_agent_task


DESCRIPTION = (
    "玩家认可任务后调用：传入说明与接取/完成对话，与草案合并校验并写入任务系统。"
)


class ConfirmAgentTaskSkill(BaseSkill):
    name = "confirm_agent_task"
    category = "task"
    description = DESCRIPTION
    parameters_schema = CONFIRM_AGENT_TASK_PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        return execute_confirm_agent_task(
            args,
            pending_draft=ctx.pending_draft,
            npc_name=ctx.npc_name,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            game_data=ctx.game_data,
            rag_context_text=ctx.rag_context_text,
        )


skill = ConfirmAgentTaskSkill()
