"""list_skills tool — Anthropic Skills 渐进式披露 Level 1：skills 简表。"""

from __future__ import annotations

import json
from typing import Any

from services.tools.base import BaseTool, ToolContext, ToolResult


PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "可选，过滤 skill 大类。目前支持："
                "task-publishing / task-bargaining / knowledge-search / mood-tracking / skill-discovery。"
                "不传则返回全部 skills。"
            ),
        },
    },
    "additionalProperties": False,
}


class ListSkillsTool(BaseTool):
    name = "list_skills"
    category = "system"
    description = (
        "渐进式披露 Level 1：列出当前可用的 skills（流程/领域知识文档）简表，"
        "每条仅包含 name + description（~300 字）。"
        "如需某个 skill 的完整流程指引，之后调用 read_skill(skill_name=...)。"
    )
    parameters_schema = PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.skill_registry is None:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "skill registry 未注入。"},
                ensure_ascii=False,
            ))
        cats_raw = args.get("categories")
        cats: list[str] | None = None
        if isinstance(cats_raw, list) and cats_raw:
            cats = [str(c).strip() for c in cats_raw if str(c).strip()]

        skills = ctx.skill_registry.index(categories=cats)
        return ToolResult(result_json=json.dumps(
            {
                "status": "ok",
                "skills": skills,
                "hint": (
                    "渐进式披露流程："
                    "看到相关 skill 时用 read_skill(skill_name=\"...\") 拉取完整正文；"
                    "若 read_skill 返回里提到 references/ 附件，可用 read_skill_file(skill_name=..., file=\"references/xxx.md\") 继续深入。"
                ),
            },
            ensure_ascii=False,
        ))


tool = ListSkillsTool()
