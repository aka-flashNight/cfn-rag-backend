"""search_items tool — 按物品名关键词查询结构化物品信息。"""

from __future__ import annotations

import json
from typing import Any

from services.game_entity_prompts import compute_reward_tags, format_item_prompt_line
from services.tools.base import BaseTool, ToolContext, ToolResult


PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "item_name_keyword": {
            "type": "string",
            "description": "物品名称或片段关键词。",
        },
        "category": {
            "type": "string",
            "description": "可选，物品类型过滤（与数据文件 type 字段一致，如「药剂」「武器」）。",
        },
    },
    "required": ["item_name_keyword"],
    "additionalProperties": False,
}


class SearchItemsTool(BaseTool):
    name = "search_items"
    category = "query"
    description = (
        "按物品名关键词查询物品详情（类型、等级、单价、可作为哪种 reward_type 等）。"
        "最多返回 10 条。任务发布时选具体奖励物品前建议先用此工具确认物品属性/单价。"
    )
    parameters_schema = PARAMETERS_SCHEMA

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        kw = (args.get("item_name_keyword") or "").strip()
        cat = (args.get("category") or "").strip() or None
        if not kw:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "item_name_keyword 不能为空"},
                ensure_ascii=False,
            ))
        gd = ctx.game_data
        if gd is None:
            return ToolResult(result_json=json.dumps(
                {"status": "error", "message": "游戏数据未加载。"},
                ensure_ascii=False,
            ))
        items = gd.items.search(kw, type=cat, limit=10)
        rows: list[dict[str, Any]] = []
        for it in items:
            tags = compute_reward_tags(it, gd.equipment_mods)
            line = format_item_prompt_line(it, reward_tags=tags, price=it.price)
            rows.append({
                "name": it.name,
                "type": it.type,
                "level": it.level,
                "price": it.price,
                "detail_line": line,
            })
        return ToolResult(result_json=json.dumps(
            {"status": "ok", "count": len(rows), "items": rows},
            ensure_ascii=False,
        ))


tool = SearchItemsTool()
