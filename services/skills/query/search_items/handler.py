from __future__ import annotations

import json
from typing import Any, Optional

from services.game_entity_prompts import compute_reward_tags, format_item_prompt_line
from services.skills.base import BaseSkill, SkillDispatchContext


DESCRIPTION = "按物品名称关键词查询物品详情（类型、等级、单价、用途等）。"

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "item_name_keyword": {
            "type": "string",
            "description": "物品名称或片段关键词。",
        },
        "category": {
            "type": "string",
            "description": "可选，物品类型过滤（与数据文件 type 字段一致）。",
        },
    },
    "required": ["item_name_keyword"],
    "additionalProperties": False,
}


class SearchItemsSkill(BaseSkill):
    name = "search_items"
    category = "query"
    description = DESCRIPTION
    parameters_schema = PARAMETERS_SCHEMA

    def run(
        self,
        args: dict[str, Any],
        ctx: SkillDispatchContext,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        kw = (args.get("item_name_keyword") or "").strip()
        cat = (args.get("category") or "").strip() or None
        if not kw:
            return (
                json.dumps(
                    {"status": "error", "message": "item_name_keyword 不能为空"},
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
        items = gd.items.search(kw, type=cat, limit=10)
        rows: list[dict[str, Any]] = []
        for it in items:
            tags = compute_reward_tags(it, gd.equipment_mods)
            line = format_item_prompt_line(it, reward_tags=tags, price=it.price)
            rows.append(
                {
                    "name": it.name,
                    "type": it.type,
                    "level": it.level,
                    "price": it.price,
                    "detail_line": line,
                }
            )
        return (
            json.dumps(
                {"status": "ok", "count": len(rows), "items": rows},
                ensure_ascii=False,
            ),
            None,
            None,
        )


skill = SearchItemsSkill()
