from __future__ import annotations

from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext
from services.agent_tools.schemas import PREPARE_TASK_CONTEXT_PARAMETERS_SCHEMA
from services.agent_tools.skill_tool_executors import execute_prepare_task_context


DESCRIPTION = (
    "根据意向任务类型与奖励偏好筛选数据，返回该类型的完整上下文与规则说明。"
    "可选用 requirement_keywords / reward_keywords 优先展示与当前情境更相关的关卡与物品。"
)


class PrepareTaskContextSkill(BaseSkill):
    name = "prepare_task_context"
    category = "task"
    description = DESCRIPTION
    parameters_schema = PREPARE_TASK_CONTEXT_PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        result = execute_prepare_task_context(
            args,
            npc_name=ctx.npc_name,
            npc_faction=ctx.npc_faction,
            npc_challenge=ctx.npc_challenge,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            npc_states=ctx.npc_states,
            game_data=ctx.game_data,
        )
        return result, None, None


skill = PrepareTaskContextSkill()
