"""路由工具基类。

路由工具是**声明**不是操作：编排层在流结束后读取 tool_calls 直接分派（启动子 Agent
或跑同步动作），从不经过 ``ToolRegistry.dispatch``。``run()`` 只为满足 BaseTool
抽象接口而存在——被误调用时给出明确状态，好过静默返回空串让调用方以为成功了。

路由判据全部写在子类的 ``description`` 里：聊天主 Agent 的工具表就是它的路由提示词，
不再另发一份 meta act 协议说明。
"""

from __future__ import annotations

import json
from typing import Any

from services.tools.base import BaseTool, ToolContext, ToolResult


class RoutingTool(BaseTool):
    """无业务副作用的路由声明工具。"""

    category = "route"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(result_json=json.dumps(
            {"status": "routing_declaration", "message": f"{self.name} 由编排层处理，不应被执行"},
            ensure_ascii=False,
        ))


_NO_ARG_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
