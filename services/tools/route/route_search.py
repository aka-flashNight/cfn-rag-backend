"""route_search — 直接检索后台资料库（检索子 Agent 的第一步）。"""

from __future__ import annotations

from typing import Any

from services.tools.route._routing_base import RoutingTool


class RouteSearchTool(RoutingTool):
    name = "route_search"
    description = (
        "直接检索后台资料库：需要查证世界观设定、角色往事、"
        "关卡或物品的具体信息才能回答玩家时调用。"
        "keywords 写清要查什么；sources 选定检索范围"
        "（knowledge=世界观/剧情/角色/任务对话，items=物品，stages=关卡），"
        "不填或填 all 表示全部，多个范围取并集。"
        "deep_search 默认 false，不推荐开启：常规检索通常已经足够；"
        "只有一次检索明显不够、需要多轮深入核对时才设为 true，"
        "并同时填写 query（只给后台深入检索使用）。开启后会多轮检索，耗时和花费更高。"
        "本轮**只发这个工具调用，不要输出任何台词**：检索结果会直接回到你手上，"
        "那时你以自己的口吻讲给玩家；查不到就自然带过，不要编造设定。"
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "maxLength": 80,
                "description": "检索关键词或要点，不超过 80 字。",
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["knowledge", "items", "stages", "all"],
                },
                "description": "检索范围，可多选；不填或填 [\"all\"] 表示全部。",
            },
            "deep_search": {
                "type": "boolean",
                "description": (
                    "是否深度检索，默认 false，不推荐开启。"
                    "开启后委派后台检索子 Agent 多轮深入查询，耗时和花费更高，"
                    "此时必须同时填写 query。"
                ),
            },
            "query": {
                "type": "string",
                "maxLength": 40,
                "description": "深度检索时给后台子 Agent 的查询要点，不超过 40 字；deep_search=false 时不需要填写。",
            },
        },
        "required": ["keywords"],
        "additionalProperties": False,
    }


tool = RouteSearchTool()
