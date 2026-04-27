"""update_task_draft tool — 讨价还价 / 局部修改已有草案。"""

from __future__ import annotations

from typing import Any

from services.agent_tools.handlers import execute_update_task_draft
from services.agent_tools.schemas import UPDATE_TASK_DRAFT_PARAMETERS_SCHEMA
from services.tools.base import BaseTool, ToolContext, ToolResult


class UpdateTaskDraftTool(BaseTool):
    name = "update_task_draft"
    category = "task"
    description = (
        "局部修改已有草案并触发增量校验（仅校验变更字段）。"
        "常用于玩家讨价还价（调整奖励/提交品/持有品）或微调关卡/难度；"
        "不要传 description / get_dialogue / finish_dialogue —— 这三项仅在 confirm_agent_task 时写入。"
        "讨价还价次数上限 2；玩家大幅改任务方向时请改用 prepare_task_context + draft_agent_task 整体重拟。"
        "详细协商规则请通过 read_skill(name=\"task-bargaining\") 查阅。"
    )
    parameters_schema = UPDATE_TASK_DRAFT_PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result_json, updated_draft = execute_update_task_draft(
            args,
            pending_draft=ctx.pending_draft,
            npc_name=ctx.npc_name,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            game_data=ctx.game_data,
            rag_context_text=ctx.rag_context_text,
        )
        return ToolResult(result_json=result_json, updated_pending_draft=updated_draft)


tool = UpdateTaskDraftTool()
