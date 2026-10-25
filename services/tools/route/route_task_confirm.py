"""route_task_confirm — 路由到后台任务系统，正式发布待确认草案。"""

from __future__ import annotations

from services.tools.route._routing_base import _UI_HINT_SCHEMA, RoutingTool


class RouteTaskConfirmTool(RoutingTool):
    name = "route_task_confirm"
    description = (
        "把当前待确认的委托草案正式**发布**给玩家。"
        "仅在玩家明确接受时调用：「我接了」「接」「成交」「就它了」「干」这类话，"
        "以及玩家点击「接」按钮，都算明确接受。"
        "玩家已经点头时**必须本轮直接调用**，不要再追问「你确认吗/你点头吗」，"
        "也不要复述一遍条件再等一次确认——草案本来就只等他点头一次。"
        "调整完成后玩家再次明确接受（「可以」「行，就这样」「接了」）时同样必须直接调用。"
        "可以把给玩家看的进度提示填进 ui_hint（≤12 字）：以NPC的身份、按你这一轮实际在做的事写，"
        "要符合剧情和人物设定，不要涉及后端名词；格式参考「派发委托中」「登记委托中」，"
        "拿不准就省略，玩家会看到默认文案。"
        "当你调用本工具时，本轮**只发这个工具调用，不要输出任何台词**：发布完成后你再向玩家交代。"
        "无需调用工具时正常输出台词。"
    )
    parameters_schema = _UI_HINT_SCHEMA


tool = RouteTaskConfirmTool()
