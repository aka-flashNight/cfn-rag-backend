"""route_task_bargain_reject — 拒绝讨价还价但保留草案，纯信号路由。"""

from __future__ import annotations

from typing import Any

from services.tools.route._routing_base import RoutingTool


class RouteTaskBargainRejectTool(RoutingTool):
    name = "route_task_bargain_reject"
    description = (
        "拒绝玩家的讨价还价，但**不取消草案**——条件不变，再问玩家接还是不接。"
        "你不愿意让步时用本工具（而不是 route_task_cancel，那是玩家主动放弃）。"
        "当你调用本工具时，本轮**只发这个工具调用，不要输出任何台词**："
        "后台会把拒绝指示交回给你，那时你再以角色口吻拒绝、重申条件、征询是否接取，"
        "并留一句结束语（closing）。"
        "无需调用工具时正常输出台词。"
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "maxLength": 60,
                "description": "玩家提出的条件（你拒绝的那个），一句话，不超过 60 字。",
            },
        },
        "required": ["note"],
        "additionalProperties": False,
    }


tool = RouteTaskBargainRejectTool()