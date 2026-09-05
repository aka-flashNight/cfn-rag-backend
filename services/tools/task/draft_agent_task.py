"""draft_agent_task tool — 任务发布两步式流程的 Step 2。"""

from __future__ import annotations

from typing import Any

from services.agent_tools.handlers import execute_draft_agent_task
from services.agent_tools.schemas import DRAFT_AGENT_TASK_PARAMETERS_SCHEMA
from services.tools.base import BaseTool, ToolContext, ToolResult


class DraftAgentTaskTool(BaseTool):
    name = "draft_agent_task"
    category = "task"
    description = (
        "任务发布流程 Step 2：根据 prepare_task_context 返回的候选生成结构化任务草案并做合法性校验。"
        "成功返回 draft_id 与 draft_summary；校验失败返回全量 issues（含 root_cause/fix_hint/candidates）"
        "与当前草案快照，请按 fix_hint 逐条修正后重试。"
        "此时不应写入任务描述与接取/完成对话——它们只在 confirm_agent_task 时填入。"
    )
    parameters_schema = DRAFT_AGENT_TASK_PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        outcome = execute_draft_agent_task(
            args,
            pending_draft=ctx.pending_draft,
            npc_name=ctx.npc_name,
            player_progress=ctx.player_progress,
            npc_affinity=ctx.npc_affinity,
            bargain_count=ctx.bargain_count,
            game_data=ctx.game_data,
            rag_context_text=ctx.rag_context_text,
        )
        return ToolResult(
            result_json=outcome.result_json,
            updated_pending_draft=outcome.draft,
            bargain_count=outcome.bargain_count,
            draft_commit_valid=outcome.draft_commit_valid,
        )


tool = DraftAgentTaskTool()
