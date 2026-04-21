"""
Anthropic 风格 Skills：统一契约与 OpenAI Function 工具定义生成。
"""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

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
        # 记录 handler 模块所在目录，便于读取 SKILL.md
        self._skill_dirs: dict[str, Path] = {}

    def register(self, skill: BaseSkill, *, module_dir: Optional[Path] = None) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill
        if module_dir is not None:
            self._skill_dirs[skill.name] = module_dir

    def discover(
        self,
        package: str = "services.skills",
        *,
        priority: Optional[Iterable[str]] = None,
    ) -> None:
        """
        递归扫描 ``services/skills/<category>/<skill_name>/handler.py`` 并自动注册。

        每个 handler 模块必须顶层导出名为 ``skill`` 的 :class:`BaseSkill` 实例。
        注册顺序：
            1. ``priority`` 中列出的 skill 名称按给定顺序优先注册；
            2. 其余 skill 按目录遍历顺序追加（对主模型影响很小，主要是一致性）。
        """
        pkg = importlib.import_module(package)
        pkg_paths = [Path(p) for p in getattr(pkg, "__path__", [])]
        if not pkg_paths:
            return

        found: dict[str, tuple[BaseSkill, Path]] = {}
        for base in pkg_paths:
            for category_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                if category_dir.name.startswith("_"):
                    continue
                for skill_dir in sorted(
                    p for p in category_dir.iterdir() if p.is_dir()
                ):
                    handler_py = skill_dir / "handler.py"
                    if not handler_py.is_file():
                        continue
                    mod_name = (
                        f"{package}.{category_dir.name}.{skill_dir.name}.handler"
                    )
                    try:
                        mod = importlib.import_module(mod_name)
                    except Exception as e:  # pragma: no cover - 启动期兜底
                        raise ImportError(
                            f"加载 skill 模块失败: {mod_name}: {e}"
                        ) from e
                    skill_obj = getattr(mod, "skill", None)
                    if not isinstance(skill_obj, BaseSkill):
                        continue
                    found[skill_obj.name] = (skill_obj, skill_dir)

        ordered: list[str] = []
        if priority:
            for name in priority:
                if name in found and name not in ordered:
                    ordered.append(name)
        for name in found.keys():
            if name not in ordered:
                ordered.append(name)

        for name in ordered:
            skill_obj, skill_dir = found[name]
            if name in self._skills:
                continue
            self.register(skill_obj, module_dir=skill_dir)

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

    def get_skill_doc(self, name: str) -> Optional[str]:
        """
        读取指定 skill 的 SKILL.md **完整正文**，供 agent 按需加载（渐进式披露）。

        找不到 skill 或未提供 SKILL.md 时返回 ``None``；调用方负责 fallback。
        """
        if name not in self._skills:
            return None
        dir_path = self._skill_dirs.get(name)
        if dir_path is None:
            return None
        md_path = dir_path / "SKILL.md"
        if not md_path.is_file():
            return None
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception:
            return None

    def get_system_prompt_fragment(self, names: Iterable[str]) -> str:
        """
        拼接指定 skill 的 SKILL.md **前两段**为 system prompt 片段。

        - 第一段：一句话 description（LLM 用于选择工具）
        - 第二段：何时触发 / 不触发
        找不到 SKILL.md 或段落缺失时，回退使用 skill.description。
        """
        parts: list[str] = []
        for name in names:
            skill = self._skills.get(name)
            if skill is None:
                continue
            fallback = (skill.description or "").strip()
            dir_path = self._skill_dirs.get(name)
            text = ""
            if dir_path is not None:
                md_path = dir_path / "SKILL.md"
                if md_path.is_file():
                    try:
                        text = md_path.read_text(encoding="utf-8")
                    except Exception:
                        text = ""
            if text:
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                first_two = "\n\n".join(paragraphs[:2]) if paragraphs else fallback
            else:
                first_two = fallback
            if first_two:
                parts.append(f"### {name}\n{first_two}")
        return "\n\n".join(parts)

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
