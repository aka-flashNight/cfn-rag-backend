"""
Anthropic-style Agent Skills registry (YAML frontmatter + Markdown body).

参考规范：https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview（2026.04）

每个 skill 一个目录，目录内必有 ``SKILL.md``；
可选 ``references/`` 存放附加材料（供 ``read_skill_file`` 按需加载）。

启动时 ``SkillRegistry.discover()`` 只会：
1. 扫描 ``services/skills/<skill-name>/SKILL.md``；
2. 解析 YAML frontmatter 得 ``name / description``；
3. 记录 body + references 路径（body 延迟加载，不默认塞 system prompt）。

Level 1: ``list_skills`` 返回 {name, description} 简表进入上下文
Level 2: ``read_skill(skill_name)`` 返回完整 body
Level 3: ``read_skill_file(skill_name, file)`` 返回 references/xxx 附件

当前状态（2026-10-10）：本通道**暂时未使用**——各子 Agent 的规则已全部前置内联，
``list_skills / read_skill / read_skill_file`` 三个元工具不在任何 LLM 工具表内。
整目录与元工具、技能文档一并保留，以备后续可能的复杂 SOP 流程场景重启
（例如「更改游戏对话」的子 Agent 可能需要多步流程说明）。
"""

from __future__ import annotations

from services.skills.base import Skill, SkillRegistry, get_skill_registry

__all__ = [
    "Skill",
    "SkillRegistry",
    "get_skill_registry",
]
