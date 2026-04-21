from __future__ import annotations

from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext
from services.agent_tools.skill_tool_executors import execute_update_npc_mood


DESCRIPTION = (
    "在每次以 NPC 身份回复玩家后调用，用于上报本次对话的好感度变化与当前情绪。"
    "好感度变化取值范围为 -5 到 5，常规对话可传 0；情绪必须从当前 NPC 的可用情绪标签中选择。"
)

UPDATE_NPC_MOOD_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "favorability_change": {
            "type": "integer",
            "description": "本次对话对玩家的好感度变化，范围 -5 到 5。",
        },
        "emotion": {
            "type": "string",
            "description": "当前回复对应的情绪标签，用于立绘展示。",
        },
    },
    "required": ["favorability_change", "emotion"],
}


class UpdateNpcMoodSkill(BaseSkill):
    name = "update_npc_mood"
    category = "mood"
    description = DESCRIPTION
    parameters_schema = UPDATE_NPC_MOOD_PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        result = execute_update_npc_mood(args)
        return result, None, None


skill = UpdateNpcMoodSkill()
