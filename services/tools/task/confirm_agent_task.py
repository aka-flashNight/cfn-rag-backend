"""confirm_agent_task tool — 玩家接受后写入任务文件。"""

from __future__ import annotations

from typing import Any

from services.agent_tools.handlers import execute_confirm_agent_task
from services.agent_tools.schemas import CONFIRM_AGENT_TASK_PARAMETERS_SCHEMA
from services.tools.base import BaseTool, ToolContext, ToolResult


class ConfirmAgentTaskTool(BaseTool):
    name = "confirm_agent_task"
    category = "task"
    description = (
        "玩家认可/接受任务后调用：传入最终的 title / description / 接取对话 / 完成对话，"
        "后端与草案合并校验并写入任务系统。成功后 pending_draft 会被清空。"
        "description 须与最终关卡/物品/奖励一致；对话 text 不要包含动作/神态/旁白/【...】。"
        "必须传入与待确认草案一致的 draft_id。"
    )
    parameters_schema = CONFIRM_AGENT_TASK_PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result_json, updated_draft, write_result = execute_confirm_agent_task(
            args,
            pending_draft=ctx.pending_draft,
            npc_name=ctx.npc_name,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            game_data=ctx.game_data,
            rag_context_text=ctx.rag_context_text,
        )
        return ToolResult(
            result_json=result_json,
            updated_pending_draft=updated_draft,
            task_write_result=write_result,
        )


tool = ConfirmAgentTaskTool()
