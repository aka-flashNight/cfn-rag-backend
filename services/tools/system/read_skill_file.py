"""read_skill_file tool — 渐进式披露 Level 3：读取 skill 目录下 references/ 附件。"""

from __future__ import annotations

import json
from typing import Any

from services.tools.base import BaseTool, ToolContext, ToolResult


PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill_name": {
            "type": "string",
            "description": "skill 名称（与 read_skill 一致）。",
        },
        "file": {
            "type": "string",
            "description": (
                "相对路径，必须以 'references/' 开头并指向 skill 目录内的文件；"
                "严禁绝对路径或含 '..' 的路径。"
            ),
        },
    },
    "required": ["skill_name", "file"],
    "additionalProperties": False,
}


class ReadSkillFileTool(BaseTool):
    name = "read_skill_file"
    category = "system"
    description = (
        "渐进式披露 Level 3：读取指定 skill 的 references/ 附件全文（扩展规则、详细分类表等）。"
        "调用前应先通过 read_skill 确认附件列表。"
    )
    parameters_schema = PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.skill_registry is None:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "skill registry 未注入。"},
                ensure_ascii=False,
            ))
        name = (args.get("skill_name") or "").strip()
        rel_path = (args.get("file") or "").strip()
        if not name or not rel_path:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "skill_name / file 不能为空。"},
                ensure_ascii=False,
            ))
        content = ctx.skill_registry.read_reference(name, rel_path)
        if content is None:
            return ToolResult(result_json=json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"读取失败：skill={name}, file={rel_path}。"
                        "请确认 skill 存在且 file 是 references/ 下的合法相对路径。"
                    ),
                },
                ensure_ascii=False,
            ))
        return ToolResult(result_json=json.dumps(
            {"status": "ok", "skill_name": name, "file": rel_path, "content": content},
            ensure_ascii=False,
        ))


tool = ReadSkillFileTool()
