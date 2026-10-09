"""route_task_cancel — 路由到后台任务系统，撤销待确认草案。"""

from __future__ import annotations

from services.tools.route._routing_base import _NO_ARG_SCHEMA, RoutingTool


class RouteTaskCancelTool(RoutingTool):
    name = "route_task_cancel"
    description = (
        "撤销当前待确认的委托草案。玩家明确拒绝/放弃这份委托时调用；"
        "只有玩家态度恶劣、或你的性格强硬、或语境确实合适时，"
        "才在拒绝讨价还价的同时用本工具直接取消。"
        "**正常情况下玩家讨价还价而你不想让步，不要调用本工具**——"
        "直接说话拒绝让步，并问他到底接还是不接。"
        "本轮**只发这个工具调用，不要输出任何台词**：撤销后你再向玩家收场。"
    )
    parameters_schema = _NO_ARG_SCHEMA


tool = RouteTaskCancelTool()
