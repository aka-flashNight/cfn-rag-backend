"""prepare_task_context tool — 任务发布两步式流程的 Step 1。"""

from __future__ import annotations

from typing import Any

from services.agent_tools.handlers import execute_prepare_task_context
from services.agent_tools.schemas import PREPARE_TASK_CONTEXT_PARAMETERS_SCHEMA
from services.tools.base import BaseTool, ToolContext, ToolResult


class PrepareTaskContextTool(BaseTool):
    name = "prepare_task_context"
    category = "task"
    description = (
        "任务发布流程 Step 1：根据意向任务类型与奖励偏好筛选关卡/物品/NPC 候选集，"
        "并返回该任务类型的预算规则、合法值域与使用建议。"
        "可选用 requirement_keywords / reward_keywords 将相关候选排到前面。"
        "详细任务类型说明、奖励预算、协商规则等请通过 read_skill(name=\"task-publishing\") 按需查阅。"
    )
    parameters_schema = PREPARE_TASK_CONTEXT_PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
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
        return ToolResult(result_json=result)


tool = PrepareTaskContextTool()
