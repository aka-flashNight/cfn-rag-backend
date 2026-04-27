"""
Atomic tool contract + registry.

与 Anthropic 2026 Skills 规范配合：

- 本模块里的 ``BaseTool`` 是 OpenAI function calling 的"function"，硬编码 JSON schema。
- ``services/skills/``（YAML frontmatter + Markdown body）是"复合流程/领域知识"，
  通过 ``list_skills / read_skill / read_skill_file`` 三个元工具按需加载。
- Tools 与 Skills 完全解耦：同一个工具可能被多个 Skill 文档引用，同一个 Skill 可能只是纯文档、不对应任何独立工具。
"""

from __future__ import annotations

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


Category = Literal["task", "query", "mood", "system"]


# ---------------------------------------------------------------------------
# Runtime context injected at dispatch time
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Tool 执行期上下文（由 LangGraph tool_executor 节点在调用前注入）。"""

    npc_name: str = ""
    npc_faction: str = ""
    npc_challenge: Optional[str] = None
    player_progress: int = 1
    npc_affinity: int = 0
    npc_states: Optional[dict[str, Any]] = None
    game_data: Optional["GameDataRegistry"] = None
    pending_draft: Optional[dict[str, Any]] = None
    retrieve_fn: Optional[Callable[[str], str]] = None
    rag_context_text: Optional[str] = None
    # list_skills / read_skill / read_skill_file 三个元工具运行时需要 skill 注册表
    skill_registry: Optional["SkillRegistry"] = None


@dataclass
class ToolResult:
    """Tool.run() 返回结构化结果（供 dispatcher 提取 state 更新）。"""

    result_json: str
    updated_pending_draft: Optional[dict[str, Any]] = None
    task_write_result: Optional[str] = None

    def as_tuple(
        self,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        return self.result_json, self.updated_pending_draft, self.task_write_result


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """单个原子工具：名称、类别、JSON Schema、同步执行入口。"""

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
        if tool.category not in ("task", "query", "mood", "system"):
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

    def dispatch(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: ToolContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        tool = self._tools.get(tool_name)
        if tool is None:
            return (
                json.dumps(
                    {"status": "error", "message": f"未知工具: {tool_name}"},
                    ensure_ascii=False,
                ),
                None,
                None,
            )
        try:
            result = tool.run(tool_args or {}, ctx)
        except Exception as e:
            logger.exception("tool %s 执行异常", tool_name)
            return (
                json.dumps(
                    {"status": "error", "message": f"{tool_name} 执行异常: {e}"},
                    ensure_ascii=False,
                ),
                None,
                None,
            )
        return result.as_tuple()


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


# Pipeline ordering for the task family（同一批 tool_calls 内 sort 用）
# draft 永远在 confirm 之前、prepare 在最前。
_TASK_TOOL_PIPELINE_ORDER: dict[str, int] = {
    "prepare_task_context": 10,
    "search_knowledge": 15,
    "search_stages": 16,
    "search_items": 17,
    "list_skills": 18,
    "read_skill": 19,
    "read_skill_file": 19,
    "draft_agent_task": 20,
    "update_task_draft": 30,
    "confirm_agent_task": 40,
    "cancel_agent_task": 50,
}


def sort_tool_calls_for_pipeline(pending_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一轮 tool_calls 按 task pipeline 排序，避免 confirm 出现在 draft 之前。"""
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
                _TASK_TOOL_PIPELINE_ORDER.get(_name(pair[1]), 1000),
                pair[0],
            ),
        )
    ]
