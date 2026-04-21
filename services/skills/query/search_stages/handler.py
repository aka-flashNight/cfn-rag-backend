from __future__ import annotations

import json
from typing import Any, Optional

from services.game_entity_prompts import format_stage_embedding_text
from services.skills.base import BaseSkill, SkillDispatchContext


DESCRIPTION = "按关卡名称关键词查询关卡结构化信息（大区、解锁条件、描述等）。"

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stage_name_keyword": {
            "type": "string",
            "description": "关卡名称或片段关键词。",
        },
        "area": {
            "type": "string",
            "description": "可选，限定大区（如「废城」）。",
        },
    },
    "required": ["stage_name_keyword"],
    "additionalProperties": False,
}


class SearchStagesSkill(BaseSkill):
    name = "search_stages"
    category = "query"
    description = DESCRIPTION
    parameters_schema = PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        kw = (args.get("stage_name_keyword") or "").strip()
        area_filter = (args.get("area") or "").strip() or None
        if not kw:
            return (
                json.dumps(
                    {"status": "error", "message": "stage_name_keyword 不能为空"},
                    ensure_ascii=False,
                ),
                None,
                None,
            )
        gd = ctx.game_data
        if gd is None:
            return (
                json.dumps(
                    {"status": "error", "message": "游戏数据未加载。"},
                    ensure_ascii=False,
                ),
                None,
                None,
            )
        kw_lower = kw.lower()
        matches: list[dict[str, Any]] = []
        for (area, name), si in gd.stages._stage_infos.items():
            if area_filter and area != area_filter:
                continue
            if kw_lower not in name.lower() and kw_lower not in area.lower():
                continue
            line = format_stage_embedding_text(si)
            matches.append(
                {
                    "area": area,
                    "name": name,
                    "unlock_condition": si.unlock_condition,
                    "detail_line": line,
                }
            )
            if len(matches) >= 8:
                break
        return (
            json.dumps(
                {
                    "status": "ok",
                    "count": len(matches),
                    "stages": matches,
                },
                ensure_ascii=False,
            ),
            None,
            None,
        )


skill = SearchStagesSkill()
