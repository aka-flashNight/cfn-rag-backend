"""route_task_update — 路由到后台任务系统，按玩家新条件修改待确认草案。"""

from __future__ import annotations

from typing import Any

from services.tools.route._routing_base import RoutingTool


class RouteTaskUpdateTool(RoutingTool):
    name = "route_task_update"
    description = (
        "把本轮交给后台任务系统，按玩家的新条件修改**当前待确认的委托草案**"
        "（讨价还价、换奖励物品、调奖励数量、改难度等小幅调整）。"
        "你愿意让步时用本工具；不愿意让步时**不要调用任何工具**，"
        "直接以你的口吻拒绝并问玩家到底接还是不接。"
        "当你调用本工具时，本轮**只发这个工具调用，不要输出任何台词**：改完后新方案会回到你手上，"
        "那时你再向玩家说明并重新征询是否接取。"
        "玩家一句话同时表达「接受 + 要调整」（如「我接，但奖励多点」）时，"
        "先用本工具调整，等他认可后再 route_task_confirm。"
        "note 写玩家提出的新条件，一句话。"
        "无需调用工具时正常输出台词。"
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "maxLength": 60,
                "description": "玩家的新条件，一句话，不超过 60 字。",
            },
        },
        "required": ["note"],
        "additionalProperties": False,
    }


tool = RouteTaskUpdateTool()
