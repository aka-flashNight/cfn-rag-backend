"""
Skills 注册表（Anthropic 风格目录 + OpenAI Function 工具定义）。

使用方式::

    from services.skills import get_skill_registry
    tools = get_skill_registry().get_openai_tools()
"""

from __future__ import annotations

from services.skills.base import BaseSkill, SkillDispatchContext, SkillRegistry

_registry: SkillRegistry | None = None


def _build_registry() -> SkillRegistry:
    r = SkillRegistry()
    # 顺序影响部分模型对工具列表的阅读；任务流水线优先，list_skills 放最后
    from services.skills.task.prepare_task_context.handler import skill as sk_prepare
    from services.skills.query.search_knowledge.handler import skill as sk_search_kn
    from services.skills.query.search_stages.handler import skill as sk_stages
    from services.skills.query.search_items.handler import skill as sk_items
    from services.skills.task.draft_agent_task.handler import skill as sk_draft
    from services.skills.task.update_task_draft.handler import skill as sk_update
    from services.skills.task.confirm_agent_task.handler import skill as sk_confirm
    from services.skills.task.cancel_agent_task.handler import skill as sk_cancel
    from services.skills.mood.update_npc_mood.handler import skill as sk_mood
    from services.skills.system.list_skills.handler import skill as sk_list

    for sk in (
        sk_prepare,
        sk_search_kn,
        sk_stages,
        sk_items,
        sk_draft,
        sk_update,
        sk_confirm,
        sk_cancel,
        sk_mood,
        sk_list,
    ):
        r.register(sk)
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
