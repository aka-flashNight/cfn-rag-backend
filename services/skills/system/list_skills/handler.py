from __future__ import annotations

import json
from typing import Any, Optional

from services.skills.base import BaseSkill, SkillDispatchContext


DESCRIPTION = "列出当前可用的工具（skills）名称与简短说明，供自检。"

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "可选，只列出这些类别：task/query/mood/system。",
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
                {"status": "ok", "skills": lines},
                ensure_ascii=False,
            ),
            None,
            None,
        )


skill = ListSkillsSkill()
