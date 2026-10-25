"""cancel_agent_task tool — 清除当前待确认草案。

v3：cancel 由 orchestrator 的同步动作（meta.act=task_cancel）直接触发，
本工具保留给兜底路径 / 子 Agent 语义完整性使用。
"""

from __future__ import annotations

from typing import Any

from services.agent_tools.handlers import execute_cancel_agent_task
from services.tools.base import BaseTool, ToolContext, ToolResult


CANCEL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "draft_id": {"type": "string"},
        "ui_hint": {
            "type": "string",
            "maxLength": 12,
            "description": "前端显示的超短提示（<=12字），为空则后端使用默认提示。",
        },
    },
    "required": ["draft_id"],
    "additionalProperties": False,
}


class CancelAgentTaskTool(BaseTool):
    name = "cancel_agent_task"
    category = "task"
    description = (
        "取消当前待确认的任务草案：玩家拒绝、讨价还价失败、你决定撤回时调用。"
        "取消后 pending_draft 清空，需要重新发布任务须走 prepare_task_context + draft_agent_task。"
    )
    parameters_schema = CANCEL_PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        outcome = execute_cancel_agent_task(
            args,
            pending_draft=ctx.pending_draft,
        )
        return ToolResult(
            result_json=outcome.result_json,
            updated_pending_draft=outcome.draft,
            bargain_count=outcome.bargain_count,
            draft_commit_valid=outcome.draft_commit_valid,
        )


tool = CancelAgentTaskTool()
