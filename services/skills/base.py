"""
Anthropic 风格 Skills：统一契约与 OpenAI Function 工具定义生成。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

@dataclass
class SkillDispatchContext:
    """工具执行期上下文（由 LangGraph tool_executor 注入）。"""

    npc_name: str = ""
    npc_faction: str = ""
    npc_challenge: Optional[str] = None
    player_progress: int = 1
    npc_affinity: int = 0
    npc_states: Optional[dict[str, Any]] = None
    game_data: Any = None
    pending_draft: Optional[dict[str, Any]] = None
    retrieve_fn: Any = None
    rag_context_text: Optional[str] = None


Category = Literal["task", "query", "mood", "system"]


class BaseSkill(ABC):
    """单个 skill：名称、类别、参数 JSON Schema、描述、同步执行入口。"""

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
    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        """
        返回与 tool_executor.dispatch_tool_call 一致：
        (result_json_str, updated_pending_draft, task_write_result)
        """


class SkillRegistry:
    """Skill 注册表：按名称索引，支持按 category 过滤生成 tools 列表。"""

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    def all_skills(self) -> list[BaseSkill]:
        return list(self._skills.values())

    def get_openai_tools(
        self,
        categories: Optional[set[Category]] = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in self._skills.values():
            if categories is not None and s.category not in categories:
                continue
            out.append(s.to_openai_tool())
        return out

    def dispatch(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        skill = self._skills.get(tool_name)
        if skill is None:
            return (
                json.dumps(
                    {"status": "error", "message": f"未知工具: {tool_name}"},
                    ensure_ascii=False,
                ),
                None,
                None,
            )
        return skill.run(tool_args, ctx)
