"""edit_npc_dialogues tool — 按编号把整组修改落到本存档的对话覆盖层。

执行＝用最近一次 search_npc_dialogues 写入 ctx 的编号映射构造 ``DialogueEditor``
调 ``apply()``（写入侧唯一入口，全部打回守卫都在 editor 里，本工具不重复实现）；
打回原因原文作为工具结果回给模型，模型据此修正后重新提交。

触发来源（reason / source_task_id / source_trigger）与存档槽位由委派方经 ctx 注入，
本工具只透传：槽位钉死在委派受理当刻，子 Agent 跑完 900 秒也不会写进别的存档。
"""

from __future__ import annotations

import json
from typing import Any

from services.dialogue.editor import DialogueEditor
from services.dialogue.view import build_view
from services.tools.base import BaseTool, ToolContext, ToolResult

_LINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "说话人：你自己（填你的名字）或玩家（固定填 $PC）。"},
        "text": {"type": "string", "description": "这行台词。"},
        "emotion": {
            "type": "string",
            "description": "这行台词的情绪，从你的可用情绪标签里选一个；语气平实、没有表情的行留空。",
        },
    },
    "required": ["name", "text"],
}


class EditNpcDialoguesTool(BaseTool):
    name = "edit_npc_dialogues"
    category = "task"
    description = (
        "按编号整组提交你的对话修改，写进当前存档立即生效。\n"
        "- 只能按编号整组修改；不需要改的行请原样抄回，不能删除原有信息；\n"
        "- 只输出其中有需要修改的组，可以改台词，可以向后加台词；\n"
        "- 每行填说话人 name 与台词 text，情绪可选填 emotion；\n"
        "- 任务对话只能在本组内改台词、在组尾加台词，不能新增组；"
        "日常对话除改已有组外，可以用 new_groups 直接新增一组；\n"
        "- 修改被打回时结果里会给出原因，按原因修正后重新提交。"
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "description": "要修改的组，每项＝编号 ＋ 该组每行的 name/text。",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "查询结果里该组的编号。",
                        },
                        "lines": {
                            "type": "array",
                            "description": "该组修改后的全部台词行（按原行序，新增行排在组尾）。",
                            "items": _LINE_SCHEMA,
                        },
                    },
                    "required": ["number", "lines"],
                },
            },
            "new_groups": {
                "type": "array",
                "description": "新增的日常对话组（可选），每项＝该组每行的 name/text。",
                "items": {
                    "type": "object",
                    "properties": {
                        "lines": {
                            "type": "array",
                            "description": "新组的全部台词行。",
                            "items": _LINE_SCHEMA,
                        },
                    },
                    "required": ["lines"],
                },
            },
        },
        "required": ["edits"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        npc = (ctx.npc_name or "").strip()
        view = build_view(game_data=ctx.game_data, slot_key=ctx.slot_key)
        editor = DialogueEditor(view, ctx.dialogue_index or [])
        outcome = editor.apply(
            npc=npc,
            edits=args.get("edits"),
            new_groups=args.get("new_groups"),
            reason=ctx.dialogue_reason,
            source_task_id=ctx.dialogue_source_task_id,
            source_trigger=ctx.dialogue_source_trigger,
        )
        if outcome.changed:
            status = "ok"
        elif outcome.rejected:
            status = "rejected"
        else:
            status = "noop"
        payload = {
            "status": status,
            "applied": outcome.applied,
            "rejected": outcome.rejected,
        }
        return ToolResult(result_json=json.dumps(payload, ensure_ascii=False))


tool = EditNpcDialoguesTool()
