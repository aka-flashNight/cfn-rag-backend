"""
Atomic tool contract + registry.

与 Anthropic 2026 Skills 规范配合：

- 本模块里的 ``BaseTool`` 是 OpenAI function calling 的"function"，硬编码 JSON schema。
- ``services/skills/``（YAML frontmatter + Markdown body）是"复合流程/领域知识"，
  通过 ``list_skills / read_skill / read_skill_file`` 三个元工具按需加载。
- Tools 与 Skills 完全解耦：同一个工具可能被多个 Skill 文档引用，同一个 Skill 可能只是纯文档、不对应任何独立工具。

v3：工具只服务后台子 Agent（TaskRunner / SearchRunner）；聊天主 Agent 不持有任何工具（决策走 meta 行）。
``ToolRegistry.dispatch`` 为 async；同批 tool_calls 按 task pipeline 分组并行执行（asyncio.gather，修 B3 串行）。
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from services.game_data.registry import GameDataRegistry
    from services.skills import SkillRegistry


Category = Literal["task", "query", "system"]

_VALID_CATEGORIES: tuple[str, ...] = ("task", "query", "system")


# ---------------------------------------------------------------------------
# Runtime context injected at dispatch time
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Tool 执行期上下文（由子 Agent 在每轮工具执行前构造注入）。"""

    npc_name: str = ""
    npc_faction: str = ""
    npc_challenge: Optional[str] = None
    player_progress: int = 1
    npc_affinity: int = 0
    npc_states: Optional[dict[str, Any]] = None
    game_data: Optional["GameDataRegistry"] = None
    pending_draft: Optional[dict[str, Any]] = None
    # 草案业务状态（bargain_count/_draft_commit_valid 独立于草案 JSON 传递，修 D5）
    bargain_count: int = 0
    draft_commit_valid: bool = False
    retrieve_fn: Optional[Callable[[str], str]] = None
    rag_context_text: Optional[str] = None
    # list_skills / read_skill / read_skill_file 三个元工具运行时需要 skill 注册表
    skill_registry: Optional["SkillRegistry"] = None


@dataclass
class ToolResult:
    """Tool.run() 返回结构化结果（供子 Agent 提取状态更新）。"""

    result_json: str
    updated_pending_draft: Optional[dict[str, Any]] = None
    task_write_result: Optional[str] = None
    # draft/update 工具回传的草案业务状态（None = 未触碰）
    bargain_count: Optional[int] = None
    draft_commit_valid: Optional[bool] = None


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """单个原子工具：名称、类别、JSON Schema、同步执行入口（CPU 轻操作，不跑 IO）。"""

    name: str
    category: Category
    description: str
    parameters_schema: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行工具并返回 ToolResult。"""


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Tool 注册表：按名称索引，支持按 category / names 过滤生成 OpenAI tools schema 列表。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        if tool.category not in _VALID_CATEGORIES:
            raise ValueError(
                f"tool {tool.name!r} has invalid category: {tool.category!r}"
            )
        self._tools[tool.name] = tool

    def discover(self, package: str = "services.tools") -> None:
        """
        递归扫描 ``services/tools/<category>/<name>.py`` 自动注册。

        每个 tool 模块必须顶层导出 ``tool``（BaseTool 实例）。
        多次调用幂等：已存在的 tool 会跳过，不会抛异常，方便热重载场景。
        """
        pkg = importlib.import_module(package)
        pkg_paths = [Path(p) for p in getattr(pkg, "__path__", [])]
        if not pkg_paths:
            return

        for base in pkg_paths:
            for category_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                if category_dir.name.startswith("_"):
                    continue
                for tool_py in sorted(
                    p for p in category_dir.iterdir()
                    if p.is_file()
                    and p.suffix == ".py"
                    and not p.name.startswith("_")
                ):
                    mod_name = (
                        f"{package}.{category_dir.name}.{tool_py.stem}"
                    )
                    try:
                        mod = importlib.import_module(mod_name)
                    except Exception as e:  # pragma: no cover
                        raise ImportError(
                            f"加载 tool 模块失败: {mod_name}: {e}"
                        ) from e
                    tool_obj = getattr(mod, "tool", None)
                    if not isinstance(tool_obj, BaseTool):
                        continue
                    if tool_obj.name in self._tools:
                        continue
                    self.register(tool_obj)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def all_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_openai_tools(
        self,
        *,
        categories: Optional[Iterable[Category]] = None,
        names: Optional[Iterable[str]] = None,
    ) -> list[dict[str, Any]]:
        cat_set = set(categories) if categories is not None else None
        name_set = set(names) if names is not None else None

        out: list[dict[str, Any]] = []
        for t in self._tools.values():
            if cat_set is not None and t.category not in cat_set:
                continue
            if name_set is not None and t.name not in name_set:
                continue
            out.append(t.to_openai_tool())
        return out

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolResult:
        """执行单个工具（async；工具本体是同步 CPU 轻操作，直接调用）。"""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": f"未知工具: {tool_name}"},
                ensure_ascii=False,
            ))
        try:
            return tool.run(tool_args or {}, ctx)
        except Exception as e:
            logger.exception("tool %s 执行异常", tool_name)
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": f"{tool_name} 执行异常: {e}"},
                ensure_ascii=False,
            ))

    async def dispatch_batch(
        self,
        tool_calls: list[dict[str, Any]],
        ctx: ToolContext,
    ) -> list[tuple[str, ToolResult, "ToolContext"]]:
        """执行同批 tool_calls：按 task pipeline 分组，组内并行（asyncio.gather）、组间串行。

        - 并行消除了旧 tool_executor 的逐个串行（修 B3）；
        - 组间串行保证依赖正确：prepare(10) → draft(20) → update(30) 等任务状态变更
          不会与依赖它的工具同批乱序（draft 产生的 pending_draft 需先落再跑 update）；
        - 返回 [(tool_call_id, ToolResult, 本调用后的 ctx)]，ctx 按组间顺序继承
          （draft/update 对草案的修改向后续组传递）。
        """
        normalized: list[tuple[str, str, dict[str, Any]]] = []
        for tc in tool_calls or []:
            func = tc.get("function", tc) if isinstance(tc, dict) else {}
            name = str(func.get("name", "") or "")
            call_id = str(tc.get("id", "") or "")
            raw_args = func.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            normalized.append((call_id, name, args))

        groups = _group_by_pipeline_order([name for _, name, _ in normalized])
        results: list[tuple[str, ToolResult, ToolContext]] = []
        working_ctx = ctx
        for group_indexes in groups:
            coros = [
                self.dispatch(normalized[i][1], normalized[i][2], working_ctx)
                for i in group_indexes
            ]
            group_results = await asyncio.gather(*coros)
            for i, res in zip(group_indexes, group_results):
                call_id, name, _args = normalized[i]
                # 先继承本结果的状态变更再绑定 ctx：调用方（终态检测）能看到
                # draft/update 工具写入的 pending_draft / bargain_count
                working_ctx = _inherit_ctx(working_ctx, res)
                results.append((call_id, res, working_ctx))
        return results


def _inherit_ctx(ctx: ToolContext, res: ToolResult) -> ToolContext:
    """工具修改了草案/业务状态时，派生新的 ctx 供后续工具看到。"""
    if (
        res.updated_pending_draft is None
        and res.bargain_count is None
        and res.draft_commit_valid is None
    ):
        return ctx
    import dataclasses

    updates: dict[str, Any] = {}
    if res.updated_pending_draft is not None:
        updates["pending_draft"] = res.updated_pending_draft
    if res.bargain_count is not None:
        updates["bargain_count"] = res.bargain_count
    if res.draft_commit_valid is not None:
        updates["draft_commit_valid"] = res.draft_commit_valid
    return dataclasses.replace(ctx, **updates)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        r = ToolRegistry()
        r.discover(package="services.tools")
        _registry = r
    return _registry


def set_tool_registry(registry: ToolRegistry | None) -> None:
    """测试注入用。"""
    global _registry
    _registry = registry


# Pipeline ordering for the task family（同批 tool_calls 分组并行用）
# draft 永远在 update 之前、prepare 在最前；同组内工具互不依赖可并行。
_TASK_TOOL_PIPELINE_ORDER: dict[str, int] = {
    "list_skills": 5,
    "read_skill": 5,
    "read_skill_file": 5,
    "prepare_task_context": 10,
    "search_knowledge": 15,
    "search_stages": 16,
    "search_items": 17,
    "draft_agent_task": 20,
    "update_task_draft": 30,
    "confirm_agent_task": 40,
    "cancel_agent_task": 50,
}


def _pipeline_order(name: str) -> int:
    return _TASK_TOOL_PIPELINE_ORDER.get(name, 1000)


def _group_by_pipeline_order(names: list[str]) -> list[list[int]]:
    """按 pipeline 序把索引分组：同序号（可并行）为一组，组间按序号升序。"""
    order: list[tuple[int, int]] = [
        (_pipeline_order(n), i) for i, n in enumerate(names)
    ]
    order.sort()
    groups: list[list[int]] = []
    cur_key: int | None = None
    for key, i in order:
        if key != cur_key:
            groups.append([])
            cur_key = key
        groups[-1].append(i)
    return groups


def sort_tool_calls_for_pipeline(pending_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同轮 tool_calls 按 task pipeline 排序（兼容旧入口；新代码用 dispatch_batch 分组并行）。"""
    if len(pending_calls) <= 1:
        return pending_calls

    def _name(tc: dict[str, Any]) -> str:
        func_info = tc.get("function", tc)
        return str(func_info.get("name", "") or "")

    return [
        tc
        for _, tc in sorted(
            enumerate(pending_calls),
            key=lambda pair: (
                _pipeline_order(_name(pair[1])),
                pair[0],
            ),
        )
    ]
