"""read_skill tool — 渐进式披露 Level 2：读取 SKILL.md 完整正文。"""

from __future__ import annotations

import json
from typing import Any

from services.tools.base import BaseTool, ToolContext, ToolResult


PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill_name": {
            "type": "string",
            "description": "由 list_skills 返回的 skill 名称（小写 + 短横），如 'task-publishing'。",
        },
    },
    "required": ["skill_name"],
    "additionalProperties": False,
}


class ReadSkillTool(BaseTool):
    name = "read_skill"
    category = "system"
    description = (
        "渐进式披露 Level 2：读取指定 skill 的 SKILL.md 完整正文（流程说明 / 约束 / 示例）。"
        "调用前请先用 list_skills 确认 skill 名称；"
        "正文内若提到 references/ 附件，再用 read_skill_file 按需加载。"
    )
    parameters_schema = PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.skill_registry is None:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "skill registry 未注入。"},
                ensure_ascii=False,
            ))
        name = (args.get("skill_name") or "").strip()
        if not name:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "skill_name 不能为空。"},
                ensure_ascii=False,
            ))
        skill = ctx.skill_registry.get(name)
        if skill is None:
            return ToolResult(result_json=json.dumps(
                {
                    "status": "error",
                    "message": f"未知 skill: {name}",
                    "available": [s.name for s in ctx.skill_registry.all_skills()],
                },
                ensure_ascii=False,
            ))
        refs = ctx.skill_registry.list_reference_files(name)
        return ToolResult(result_json=json.dumps(
            {
                "status": "ok",
                "skill": {
                    "name": skill.name,
                    "description": skill.description,
                    "body": skill.body,
                    "references": refs,
                },
                "hint": (
                    "如需更深入的参考资料，可用 read_skill_file(skill_name, file=\"references/xxx.md\")。"
                    if refs else None
                ),
            },
            ensure_ascii=False,
        ))


tool = ReadSkillTool()
