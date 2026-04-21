"""
Skills 注册表（Anthropic 风格目录 + OpenAI Function 工具定义）。

目录契约（详见 plan 路线2 §2.1）::

    services/skills/
      <category>/                # task | query | mood | system
        <skill_name>/
          SKILL.md                # 前两段：description + 触发/不触发
          handler.py              # 顶层导出名为 `skill` 的 BaseSkill 实例

``SkillRegistry.discover()`` 会递归扫描上述目录并自动注册，"新增文件夹即扩展"。
task 流水线相关 skill 给予固定优先顺序，避免模型在工具列表顺序敏感时先 confirm 后 draft。

使用方式::

    from services.skills import get_skill_registry
    tools = get_skill_registry().get_openai_tools()
"""

from __future__ import annotations

from services.skills.base import BaseSkill, SkillDispatchContext, SkillRegistry

# 顺序仅是"偏好"：任务流水线优先，list_skills 放最后，其余 skill 按目录顺序追加。
# 新增文件夹会被 discover() 自动识别并追加到末尾，无需改本文件。
_PRIORITY_ORDER: tuple[str, ...] = (
    "prepare_task_context",
    "search_knowledge",
    "search_stages",
    "search_items",
    "draft_agent_task",
    "update_task_draft",
    "confirm_agent_task",
    "cancel_agent_task",
    "update_npc_mood",
    "list_skills",
)

_registry: SkillRegistry | None = None


def _build_registry() -> SkillRegistry:
    r = SkillRegistry()
    r.discover(package="services.skills", priority=_PRIORITY_ORDER)
    return r


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


__all__ = [
    "BaseSkill",
    "SkillDispatchContext",
    "SkillRegistry",
    "get_skill_registry",
]
