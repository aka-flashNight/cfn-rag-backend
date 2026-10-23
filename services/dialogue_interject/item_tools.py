"""物品工具 schema、参数解析与固定文案。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GIVE_TOOL = "give_item_to_player"
TAKE_TOOL = "request_item_from_player"
CANCEL_TOOL = "cancel_item_request"
ITEM_TOOL_NAMES = (GIVE_TOOL, TAKE_TOOL, CANCEL_TOOL)

_ITEM_ARGS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "精确物品名；不确定就留空"},
        "quantity": {"type": "integer", "minimum": 1},
        "keyword": {"type": "string", "description": "搜索用关键词"},
        "instruction": {"type": "string", "description": "你要找什么样的物品"},
    },
    "required": ["quantity"],
    "additionalProperties": False,
}


def item_tool_schemas(names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """按注册矩阵取 schema；names=None 返回全部三个。"""
    wanted = set(names or ITEM_TOOL_NAMES)
    schemas = [
        {
            "type": "function",
            "function": {
                "name": GIVE_TOOL,
                "description": (
                    "你决定送给玩家一件物品时调用。name 只能填你确定的精确物品名；"
                    "不确定就留空，用 keyword/instruction 描述你要找的东西。quantity 必填。"
                    "调用这一轮不要输出台词。"
                ),
                "parameters": _ITEM_ARGS,
            },
        },
        {
            "type": "function",
            "function": {
                "name": TAKE_TOOL,
                "description": (
                    "你决定向玩家索要/询问是否给一件物品时调用。name 只能填你确定的精确物品名；"
                    "不确定就留空，用 keyword/instruction 描述你要找的东西。quantity 必填。"
                    "调用这一轮不要输出台词。"
                ),
                "parameters": _ITEM_ARGS,
            },
        },
        {
            "type": "function",
            "function": {
                "name": CANCEL_TOOL,
                "description": "玩家明确拒绝当前索取时调用，取消这次索取。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]
    return [schema for schema in schemas if schema["function"]["name"] in wanted]


def is_item_tool(name: str) -> bool:
    return (name or "").strip() in ITEM_TOOL_NAMES


@dataclass(frozen=True)
class ItemToolArgs:
    name: str
    quantity: int
    keyword: str
    instruction: str


def parse_item_arguments(args: Any) -> ItemToolArgs:
    if isinstance(args, str):
        try:
            import json

            args = json.loads(args or "{}")
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    name = str(args.get("name") or "").strip()[:100]
    keyword = str(args.get("keyword") or "").strip()[:40]
    instruction = str(args.get("instruction") or "").strip()[:200]
    try:
        quantity = int(args.get("quantity") or 0)
    except Exception:
        quantity = 0
    if quantity < 1:
        raise ValueError("quantity 必须是正整数")
    if not (name or keyword or instruction):
        raise ValueError("name/keyword/instruction 至少填一个")
    return ItemToolArgs(
        name=name,
        quantity=min(quantity, 9999),
        keyword=keyword,
        instruction=instruction,
    )


def is_equipment_item(game_data: Any, name: str) -> bool:
    reg = getattr(game_data, "items", None)
    item = None
    getter = getattr(reg, "get_by_name", None)
    if getter is not None:
        try:
            item = getter(name)
        except Exception:
            item = None
    if item is not None and str(getattr(item, "type", "") or "") in {"武器", "防具"}:
        return True
    mods = getattr(game_data, "equipment_mods", None)
    try:
        return bool(mods and mods.is_plugin(name))
    except Exception:
        return False


def take_instruction(name: str, quantity: int) -> str:
    return (
        f"你现在要索取物品：{quantity}个{name}，该物品确认有效，可以索取。"
        "现在，请用符合你身份且符合上下文的语句，向玩家索要物品或询问玩家是否要把该物品给你。"
    )


def take_not_found_text(name: str, quantity: int, keyword: str, instruction: str) -> str:
    detail = "名称/数量/关键词/指示：" + "；".join([
        f"名称={name or '空'}",
        f"数量={quantity}",
        f"关键词={keyword or '空'}",
        f"指示={instruction or '空'}",
    ])
    return (
        "未找到有效的物品，你要求的索取任务执行失败，无法索取，"
        "请在不能索取物品的前提下继续聊天。" + detail
    )


def gift_not_found_text() -> str:
    return "未找到合适的赠送物品，这次赠送没有发生；请以角色口吻拒绝或换个话题。"


def with_item_info(text: str, info: str) -> str:
    info = (info or "").strip()
    return f"{text}\n物品信息：\n{info}" if info else text
