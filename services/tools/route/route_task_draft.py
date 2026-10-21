"""route_task_draft — 路由到后台任务系统，拟一份新的委托草案。"""

from __future__ import annotations

from typing import Any

from services.tools.route._routing_base import RoutingTool

_HEAD = (
    "把本轮交给后台任务系统，为玩家拟定一份**新的**委托草案。"
    "仅在玩家明确向你索要委托/工作/差事，或按你的身份和当下情景你决定主动派活时调用。"
)
# 只在「当前可切磋」时拼进描述：不可切磋时描述里一个「切磋/对战」字样都不出现，
# 避免模型据此向玩家承诺打不了的一场。可行性判定与 prepare 的切磋目标同源
# （orchestrator.context 里算好的 player_can_challenge）。
_CHALLENGE = (
    "玩家提出想跟你切磋、对战、比划，或直接跟你动手，也属于可以路由的情形："
    "note 里写明他要切磋/对战即可，后台会按切磋类任务拟。"
)
_TAIL = (
    "当你调用本工具时，本轮**只发这个工具调用，不要输出任何台词**：具体做什么、去哪做、给多少报酬"
    "全部由后台系统拟定，拟好后结果会回到你手上，那时你再向玩家说明。"
    "note 写一句话的诉求要点（玩家想要什么、你缺什么、他提到过哪里），"
    "不要写物品数量和报酬数字——那些由后台系统按规则定。"
    "已有待确认草案时：玩家只是讨价还价/小幅调整请用 route_task_update，"
    "只有他要求彻底换一个任务方向才用本工具。"
    "无需调用工具时正常输出台词。"
)


def build_description(*, player_can_challenge: bool = False) -> str:
    """路由判据文本：可切磋时多一句「切磋/对战也能路由」，否则原样。"""
    return _HEAD + (_CHALLENGE if player_can_challenge else "") + _TAIL


class RouteTaskDraftTool(RoutingTool):
    name = "route_task_draft"
    description = build_description()
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "maxLength": 60,
                "description": "玩家诉求要点，一句话，不超过 60 字；不含数量/金额等具体数字。",
            },
        },
        "required": ["note"],
        "additionalProperties": False,
    }

    def to_openai_tool(
        self, *, player_can_challenge: bool | None = None,
    ) -> dict[str, Any]:
        """默认（无入参）输出静态描述；可切磋时换成带切磋判据的版本。"""
        schema = super().to_openai_tool()
        if player_can_challenge:
            schema["function"]["description"] = build_description(
                player_can_challenge=True,
            )
        return schema


tool = RouteTaskDraftTool()
