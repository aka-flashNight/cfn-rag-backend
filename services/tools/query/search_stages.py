"""search_stages tool — 按关卡名关键词查询结构化关卡信息。"""

from __future__ import annotations

import json
from typing import Any

from services.game_entity_prompts import format_stage_embedding_text
from services.tools.base import BaseTool, ToolContext, ToolResult


PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stage_name_keyword": {
            "type": "string",
            "description": "关卡名称或片段关键词（支持中文子串）。",
        },
        "area": {
            "type": "string",
            "description": "可选，限定大区（如「废城」「副本任务」）。",
        },
    },
    "required": ["stage_name_keyword"],
    "additionalProperties": False,
}


class SearchStagesTool(BaseTool):
    name = "search_stages"
    category = "query"
    description = (
        "按关卡名关键词查询关卡结构化信息（大区、解锁条件、主线前置、推荐等级、难度描述等）。"
        "最多返回 8 条匹配。比 search_knowledge 更精确，优先使用。"
    )
    parameters_schema = PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        kw = (args.get("stage_name_keyword") or "").strip()
        area_filter = (args.get("area") or "").strip() or None
        if not kw:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "stage_name_keyword 不能为空"},
                ensure_ascii=False,
            ))
        gd = ctx.game_data
        if gd is None:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "游戏数据未加载。"},
                ensure_ascii=False,
            ))
        kw_lower = kw.lower()
        matches: list[dict[str, Any]] = []
        for (area, name), si in gd.stages._stage_infos.items():
            if area_filter and area != area_filter:
                continue
            if kw_lower not in name.lower() and kw_lower not in area.lower():
                continue
            line = format_stage_embedding_text(si)
            matches.append({
                "area": area,
                "name": name,
                "unlock_condition": si.unlock_condition,
                "detail_line": line,
            })
            if len(matches) >= 8:
                break
        return ToolResult(result_json=json.dumps(
            {"status": "ok", "count": len(matches), "stages": matches},
            ensure_ascii=False,
        ))


tool = SearchStagesTool()
