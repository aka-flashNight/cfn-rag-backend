"""跨对话物品往来账本的读写与提示词块。"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from services.memory.long_term import scope_key

logger = logging.getLogger(__name__)

GIVE = "give"       # NPC → 玩家
RECEIVE = "receive"  # 玩家 → NPC

#: 提示词块标题（协议 §7.5.7）
LEDGER_TITLE = "【物品往来记忆（跨对话）】"


def _format_list(rows: Iterable[dict[str, Any]], direction: str) -> str:
    names = [
        f"{int(row.get('quantity') or 0)}个{row.get('item_name')}"
        for row in rows
        if row.get("direction") == direction and int(row.get("quantity") or 0) > 0
    ]
    return "，".join(names) if names else "无"


def format_ledger_block(rows: list[dict[str, Any]]) -> str:
    return "\n".join([
        LEDGER_TITLE,
        f"你曾经送给玩家以下物品：{_format_list(rows, GIVE)}",
        f"玩家曾送给你以下物品：{_format_list(rows, RECEIVE)}",
    ])


async def record_item_ledger(
    memory: Any,
    *,
    slot_key_value: str | None,
    npc: str,
    direction: str,
    item_name: str,
    quantity: int,
) -> None:
    """写跨对话账本；失败只记日志，不打断本轮（物品已经真实转移）。"""
    if memory is None:
        return
    try:
        await memory.npc_item_ledger_upsert(
            scope_key(slot_key_value),
            npc,
            direction,
            item_name,
            int(quantity),
        )
    except Exception as exc:  # noqa: BLE001 - 账本失败不能反向破坏已完成动作
        logger.warning(
            "[interject] 物品账本写入失败 scope=%s npc=%s direction=%s item=%s qty=%s: %s",
            scope_key(slot_key_value), npc, direction, item_name, quantity, exc,
        )


async def load_item_ledger_block(
    memory: Any,
    *,
    slot_key_value: str | None,
    npc: str,
) -> str:
    if memory is None:
        return ""
    try:
        rows = await memory.npc_item_ledger_list(scope_key(slot_key_value), npc)
    except Exception as exc:  # noqa: BLE001 - 读失败按无记录处理，不阻断对话
        logger.warning("[interject] 物品账本读取失败 npc=%s: %s", npc, exc)
        return ""
    return format_ledger_block(list(rows or []))
