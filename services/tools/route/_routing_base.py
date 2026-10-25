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

#: 带「模型自带进度提示」的路由工具 schema（confirm/cancel 用）。
#: 提示会直接显示给玩家当等待文案，所以限制长度与口径；缺省时后台回落固定文案。
_UI_HINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ui_hint": {
            "type": "string",
            "maxLength": 12,
            "description": (
                "给玩家看的进度提示，一句话、不超过 12 字。"
                "以NPC的身份，按你这一轮实际在做的事写，要符合剧情和人物设定，不要涉及后端名词；"
                "格式参考「派发委托中」「登记委托中」「撤销委托中」这类动宾短语。"
                "不要写任务内容、报酬数字或玩家还看不到的信息；拿不准就省略。"
            ),
        },
    },
    "additionalProperties": False,
}
