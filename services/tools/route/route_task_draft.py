"""route_task_draft — 路由到后台任务系统，拟一份新的委托草案。"""

from __future__ import annotations

from typing import Any

from services.tools.route._routing_base import RoutingTool


class RouteTaskDraftTool(RoutingTool):
    name = "route_task_draft"
    description = (
        "把本轮交给后台任务系统，为玩家拟定一份**新的**委托草案。"
        "仅在玩家明确向你索要委托/工作/差事，或按你的身份和当下情景你决定主动派活时调用。"
        "本轮**只发这个工具调用，不要输出任何台词**：具体做什么、去哪做、给多少报酬"
        "全部由后台系统拟定，拟好后结果会回到你手上，那时你再向玩家说明。"
        "note 写一句话的诉求要点（玩家想要什么、你缺什么、他提到过哪里），"
        "不要写物品数量和报酬数字——那些由后台系统按规则定。"
        "已有待确认草案时：玩家只是讨价还价/小幅调整请用 route_task_update，"
        "只有他要求彻底换一个任务方向才用本工具。"
    )
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


tool = RouteTaskDraftTool()
