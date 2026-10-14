"""search_npc_dialogues tool — 查询当前 NPC 可修改的对话组（编号清单）。

执行＝构造合并视图（``services/dialogue/view.py``）调 ``select()``，
把 ``render()`` 的编号清单文本直接作为工具结果返回给模型；编号映射经
``ToolResult.dialogue_index`` 传给本轮/后续轮的 edit_npc_dialogues。
存档槽位由 ctx 钉死（委派受理当刻那个），ctx 没给才回落到当前绑定。

范围口径逐字照抄 ``services/dialogue/view.py`` 模块 docstring（用户裁定 2026-10-13），
改这里必须先改数据层口径。
"""

from __future__ import annotations

from typing import Any

from services.dialogue.view import ALL_SCOPES, build_view
from services.tools.base import BaseTool, ToolContext, ToolResult

_SCOPE_PARAM_DESCRIPTION = (
    "查询范围。⚠ 这里是多选，不是单选——一次查询可以同时选多个范围，"
    "结果按范围分桶拼接后统一编号。四个范围：\n"
    "- daily_all 日常全量（推荐）：该 NPC 的全部日常对话组（基础 ∪ 本存档覆盖层）；\n"
    "- task_recent 任务近期（推荐）：未完成、非 Agent 的任务里，最近的主线 5 组＋支线 5 组；\n"
    "- keyword 关键词（可选）：在未被上面选中范围覆盖的组里逐个搜关键词取并集，上限 30 组；"
    "选此范围时在 keywords 里给出关键词；\n"
    "- task_all 任务全量：该 NPC 说话的全部未完成任务对话组；与 task_recent 冲突，"
    "两者都选时采用 task_recent。不推荐。除非是非常重大的关系变化、导致所有对话都必须"
    "全部修改时，才选中任务全量；否则不建议选。\n"
    "范围选择建议：日常全量 + 较多关键词通常足够。"
)


class SearchNpcDialoguesTool(BaseTool):
    name = "search_npc_dialogues"
    category = "query"
    description = (
        "查询你（当前 NPC）可修改的对话组，返回带编号的候选清单。"
        "编号仅本次查询结果内有效，之后按编号提交整组修改时使用。"
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "scopes": {
                "type": "array",
                "items": {"type": "string", "enum": list(ALL_SCOPES)},
                "description": _SCOPE_PARAM_DESCRIPTION,
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "关键词列表，仅在 scopes 选了 keyword 时使用。",
            },
        },
        "required": ["scopes"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        npc = (ctx.npc_name or "").strip()
        if not npc:
            return ToolResult(result_json="当前没有对话对象，查不到可修改的对话组。")
        view = build_view(game_data=ctx.game_data, slot_key=ctx.slot_key)
        selection = view.select(
            npc,
            scopes=args.get("scopes"),
            keywords=args.get("keywords"),
        )
        return ToolResult(
            result_json=selection.render(),
            dialogue_index=selection.index_entries(),
        )


tool = SearchNpcDialoguesTool()
