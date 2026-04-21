from __future__ import annotations

import json
from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext
from services.agent_tools.skill_tool_executors import execute_cancel_agent_task


DESCRIPTION = "取消当前待确认的任务草案。"

CANCEL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "draft_id": {"type": "string"},
        "ui_hint": {"type": "string", "maxLength": 12},
    },
    "required": ["draft_id"],
    "additionalProperties": False,
}


class CancelAgentTaskSkill(BaseSkill):
    name = "cancel_agent_task"
    category = "task"
    description = DESCRIPTION
    parameters_schema = CANCEL_PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        return execute_cancel_agent_task(
            args,
            pending_draft=ctx.pending_draft,
        )


skill = CancelAgentTaskSkill()
