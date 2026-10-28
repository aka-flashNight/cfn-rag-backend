"""confirm_agent_task tool — 玩家接受后写入任务文件。

不在任何 LLM 工具表内：玩家接受由聊天主 Agent 的路由声明 `route_task_confirm` 表达，
编排层收到后直接调用 `execute_confirm_agent_task`（同步动作）。本文件是该执行函数的
工具封装，供注册表与 dispatch 语义完整性保留。
"""

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
        outcome = execute_confirm_agent_task(
            args,
            pending_draft=ctx.pending_draft,
            npc_name=ctx.npc_name,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            game_data=ctx.game_data,
            rag_context_text=ctx.rag_context_text,
            npc_states=ctx.npc_states,
            save_info=ctx.save_info,
        )
        return ToolResult(
            result_json=outcome.result_json,
            updated_pending_draft=outcome.draft,
            task_write_result=outcome.task_write_result,
            bargain_count=outcome.bargain_count,
            draft_commit_valid=outcome.draft_commit_valid,
        )


tool = ConfirmAgentTaskTool()
