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
        "只有你认为一次检索明显不够、查询内容复杂到必须需要多轮深入核对时才设为 true，"
        "并同时填写 query（只给后台深入检索使用）。开启后会多轮检索，耗时和花费更高。"
        "可以把给玩家看的等待提示填进 ui_hint（≤12 字），以NPC身份按你这一轮实际在做什么写："
        "本来就是你自己该知道、只是想起来的（往事、亲历、常识）用「回忆中」「回想中」「沉思中」"
        "「整理思绪中」这类；确实要翻查外部记录或资料才用「查资料中」「翻查记录中」「打听中」这类。"
        "要符合剧情和人物设定，不要涉及后端名词；拿不准就省略，玩家会看到默认文案。"
        "当你调用本工具时，本轮**只发这个工具调用，不要输出任何台词**：检索结果会直接回到你手上，"
        "那时你以自己的口吻讲给玩家；查不到就自然带过，不要编造设定。"
        "无需调用工具时正常输出台词。"
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
            "ui_hint": {
                "type": "string",
                "maxLength": 12,
                "description": (
                    "给玩家看的等待提示，不超过 12 字。按信息来源选词："
                    "自己想得起来的事用「回忆中」「回想中」「沉思中」「整理思绪中」；"
                    "需要翻查外部资料用「查资料中」「翻查记录中」「打听中」。"
                    "不要涉及后端名词；拿不准就省略。"
                ),
            },
        },
        "required": ["keywords"],
        "additionalProperties": False,
    }


tool = RouteSearchTool()
