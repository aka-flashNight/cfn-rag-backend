from __future__ import annotations

import json
from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext


DESCRIPTION = (
    "列出当前可用的工具（skills）。默认返回名称 + 简介；"
    "传入 skill_name 时返回该 skill 的完整 SKILL.md 正文（按需加载详细触发条件、示例与边界）。"
)

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "可选，只列出这些类别：task/query/mood/system。",
        },
        "skill_name": {
            "type": "string",
            "description": (
                "可选。指定某个 skill 名称时，返回其 SKILL.md 完整正文"
                "（当某个工具的简介不足以判断如何使用时，按需拉取详细文档）。"
            ),
        },
    },
    "additionalProperties": False,
}


class ListSkillsSkill(BaseSkill):
    name = "list_skills"
    category = "system"
    description = DESCRIPTION
    parameters_schema = PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        from services.skills import get_skill_registry

        reg = get_skill_registry()

        # 渐进式披露：按名字取完整 SKILL.md 正文
        wanted_name = (args.get("skill_name") or "").strip()
        if wanted_name:
            skill = reg.get(wanted_name)
            if skill is None:
                return (
                    json.dumps(
                        {
                            "status": "error",
                            "message": f"未知 skill: {wanted_name}",
                            "available": [s.name for s in reg.all_skills()],
                        },
                        ensure_ascii=False,
                    ),
                    None,
                    None,
                )
            doc = reg.get_skill_doc(wanted_name) or ""
            return (
                json.dumps(
                    {
                        "status": "ok",
                        "skill": {
                            "name": skill.name,
                            "category": skill.category,
                            "description": skill.description or "",
                            "doc": doc or "（该 skill 未提供 SKILL.md 详细正文）",
                        },
                    },
                    ensure_ascii=False,
                ),
                None,
                None,
            )

        want = args.get("categories")
        cats: Optional[set[str]] = None
        if isinstance(want, list) and want:
            cats = {str(c).strip() for c in want if str(c).strip()}

        lines: list[dict[str, str]] = []
        for s in reg.all_skills():
            if cats is not None and s.category not in cats:
                continue
            lines.append(
                {
                    "name": s.name,
                    "category": s.category,
                    "description": (s.description or "")[:200],
                }
            )
        return (
            json.dumps(
                {
                    "status": "ok",
                    "skills": lines,
                    "hint": (
                        "如需某个 skill 的详细用法，"
                        "再次调用 list_skills 时传入 skill_name='<name>' 即可获取完整文档。"
                    ),
                },
                ensure_ascii=False,
            ),
            None,
            None,
        )


skill = ListSkillsSkill()
