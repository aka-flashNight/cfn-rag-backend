"""update_npc_mood tool — 情绪与好感度变化上报。"""

from __future__ import annotations

from typing import Any

from services.agent_tools.handlers import execute_update_npc_mood
from services.tools.base import BaseTool, ToolContext, ToolResult


UPDATE_NPC_MOOD_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "favorability_change": {
            "type": "integer",
            "description": "本次对话对玩家的好感度变化，范围 -5 到 5，常规为 0。",
        },
        "emotion": {
            "type": "string",
            "description": "当前回复对应的情绪标签，必须从当前 NPC 的可用情绪列表中选择。",
        },
    },
    "required": ["favorability_change", "emotion"],
}


class UpdateNpcMoodTool(BaseTool):
    name = "update_npc_mood"
    category = "mood"
    description = (
        "上报本次回复的好感度变化与情绪标签。必须在**决策/路由阶段**调用（与其它工具同一轮），"
        "**绝不**留到对话生成阶段再调用——生成阶段 tools 不会再暴露本工具。"
        "favorability_change 范围 -5~5（常规 0），emotion 必须从当前 NPC 的可用情绪列表中选。"
        "一次对话内可多次调用，仅最后一次有效。"
    )
    parameters_schema = UPDATE_NPC_MOOD_PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(result_json=execute_update_npc_mood(args))


tool = UpdateNpcMoodTool()
