"""route_search — 路由到后台检索系统，查证设定/关卡/物品信息。"""

from __future__ import annotations

from typing import Any

from services.tools.route._routing_base import RoutingTool


class RouteSearchTool(RoutingTool):
    name = "route_search"
    description = (
        "把本轮交给后台资料检索系统：需要查证世界观设定、角色往事、"
        "关卡或物品的具体信息才能回答玩家时调用。"
        "本轮**只发这个工具调用，不要输出任何台词**：查到后结果会回到你手上，"
        "那时你以自己的口吻讲给玩家；查不到就自然带过，不要编造设定。"
        "query 用不超过 40 字写清要查什么。"
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "maxLength": 40,
                "description": "查询要点，不超过 40 字。",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }


tool = RouteSearchTool()
