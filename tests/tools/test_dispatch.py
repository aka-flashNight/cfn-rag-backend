"""ToolRegistry.dispatch / dispatch_batch 测试（修 B3：同批并行 + pipeline 状态继承）。"""

from __future__ import annotations

import asyncio
import json

from services.tools.base import (
    BaseTool,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


def _tc(name: str, args: dict | None = None, call_id: str = "c0") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)},
    }


class _RecordTool(BaseTool):
    """记录执行并可选修改草案的工具。"""

    def __init__(self, name: str, category: str = "query", mutate: bool = False):
        self.name = name
        self.category = category
        self.description = "test"
        self.parameters_schema = {"type": "object", "properties": {}}
        self.mutate = mutate
        self.seen_ctx: list[ToolContext] = []

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.seen_ctx.append(ctx)
        if self.mutate:
            return ToolResult(
                result_json=json.dumps({"status": "draft_created"}),
                updated_pending_draft={**(ctx.pending_draft or {}), "draft_id": "d1"},
                draft_commit_valid=True,
                bargain_count=0,
            )
        return ToolResult(result_json="{}")


class _UpdateTool(BaseTool):
    def __init__(self):
        self.name = "update_task_draft"
        self.category = "task"
        self.description = "test"
        self.parameters_schema = {"type": "object", "properties": {}}
        self.seen_ctx: list[ToolContext] = []

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.seen_ctx.append(ctx)
        return ToolResult(
            result_json=json.dumps({"status": "draft_updated"}),
            updated_pending_draft={**(ctx.pending_draft or {}), "updated": True},
            draft_commit_valid=True,
            bargain_count=1,
        )


def test_dispatch_unknown_tool_returns_error_json():
    registry = ToolRegistry()
    result = asyncio.run(registry.dispatch("nope", {}, ToolContext()))
    payload = json.loads(result.result_json)
    assert payload["status"] == "error"


def test_dispatch_wraps_tool_exception():
    class Boom(BaseTool):
        name = "boom"
        category = "query"
        description = ""
        parameters_schema = {"type": "object"}

        def run(self, args, ctx):
            raise ValueError("炸了")

    registry = ToolRegistry()
    registry.register(Boom())
    result = asyncio.run(registry.dispatch("boom", {}, ToolContext()))
    payload = json.loads(result.result_json)
    assert payload["status"] == "error" and "炸了" in payload["message"]


def test_batch_pipeline_state_inheritance():
    """同批 draft → update：update 能看到 draft 写入的 pending_draft（pipeline 组间串行）。"""
    registry = ToolRegistry()
    draft = _RecordTool("draft_agent_task", category="task", mutate=True)
    update = _UpdateTool()
    registry.register(draft)
    registry.register(update)

    results = asyncio.run(registry.dispatch_batch(
        [_tc("draft_agent_task", call_id="c1"), _tc("update_task_draft", call_id="c2")],
        ToolContext(),
    ))
    by_id = {call_id: res for call_id, res, _ctx in results}
    assert json.loads(by_id["c1"].result_json)["status"] == "draft_created"
    assert json.loads(by_id["c2"].result_json)["status"] == "draft_updated"
    # update 工具看到的 ctx 已含 draft 的产物
    assert update.seen_ctx[0].pending_draft == {"draft_id": "d1"}
    assert update.seen_ctx[0].draft_commit_valid is True


def test_batch_same_group_tools_run_in_parallel_group():
    """同 pipeline 序的工具（skills 元工具）同组执行，全部完成且共享起始 ctx。"""
    registry = ToolRegistry()
    a = _RecordTool("list_skills", category="system")
    b = _RecordTool("read_skill", category="system")
    c = _RecordTool("read_skill_file", category="system")
    for t in (a, b, c):
        registry.register(t)

    results = asyncio.run(registry.dispatch_batch(
        [_tc("list_skills", call_id="c1"), _tc("read_skill", call_id="c2"), _tc("read_skill_file", call_id="c3")],
        ToolContext(),
    ))
    assert len(results) == 3
    assert {t.seen_ctx[0] is not None for t in (a, b, c)} == {True}


def test_batch_malformed_arguments_become_empty_dict():
    registry = ToolRegistry()
    tool = _RecordTool("search_items")
    registry.register(tool)
    results = asyncio.run(registry.dispatch_batch(
        [{"id": "c1", "type": "function", "function": {"name": "search_items", "arguments": "{bad json"}}],
        ToolContext(),
    ))
    assert len(results) == 1
    assert tool.seen_ctx[0] is not None
