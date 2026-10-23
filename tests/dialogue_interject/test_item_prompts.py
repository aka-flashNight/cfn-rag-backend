"""P6：物品工具说明的按轮变体、任务推辞、终端拒绝与账本注入位置。"""

from __future__ import annotations

from services.dialogue_interject.context import build_user_context
from services.dialogue_interject.prompts import (
    build_daily_system_prompt,
    build_task_system_prompt,
    tool_matrix_for,
)
from services.npc.manager import NPCState
from services.orchestrator.prompts import build_static_system, build_user_shared_core
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import make_request


def _daily(**kwargs) -> str:
    return build_daily_system_prompt(npc_name=fx.NPC, state=NPCState(), **kwargs)


def test_daily_tool_guide_variants():
    base = _daily()
    assert "give_item_to_player" not in base
    assert "cancel_item_request" not in base

    with_items = _daily(item_tools=True)
    assert "give_item_to_player" in with_items
    assert "request_item_from_player" in with_items
    assert "cancel_item_request" not in with_items
    assert "口头答应不等于交付" in with_items

    with_claim = _daily(has_item_claim=True)
    assert "cancel_item_request" in with_claim
    assert "give_item_to_player" not in with_claim
    assert "request_item_from_player" not in with_claim
    assert "route_task_draft" not in with_claim
    assert "route_search" in with_claim

    tool_free = _daily(item_tools=True, include_tool_guide=False)
    assert "【工具（本轮可用）】" not in tool_free
    assert "give_item_to_player" not in tool_free
    assert "【输出格式（最高优先级）】" in tool_free


def test_task_prompt_refuses_items_without_tool_names():
    system = build_task_system_prompt(npc_name=fx.NPC, state=NPCState())
    assert "【物品往来】" in system
    assert "推辞" in system
    assert "不要当真" in system
    assert "give_item_to_player" not in system
    assert "request_item_from_player" not in system

    tool_free = build_task_system_prompt(
        npc_name=fx.NPC, state=NPCState(), include_tool_guide=False,
    )
    assert "【工具（本轮可用）】" not in tool_free
    assert "【物品往来】" in tool_free


def test_tool_matrix_item_states():
    daily = tool_matrix_for("daily", item_tools=True)
    assert "give_item_to_player" in daily
    assert "request_item_from_player" in daily
    assert "cancel_item_request" not in daily

    with_draft = tool_matrix_for("daily", item_tools=True, has_draft=True)
    assert "give_item_to_player" not in with_draft
    assert "route_task_confirm" in with_draft

    with_claim = tool_matrix_for("daily", item_tools=True, has_item_claim=True)
    assert with_claim == (
        "route_search", "update_long_term_memory", "cancel_item_request",
    )
    assert "give_item_to_player" not in tool_matrix_for("task", item_tools=True)


def test_embedded_terminal_prompt_has_item_refusal():
    system = build_static_system(npc_name=fx.NPC, state=NPCState(), embedded=True)
    assert "不能接收玩家给你的东西" in system
    assert "物品往来记忆里没有记录" in system

    standalone = build_static_system(npc_name=fx.NPC, state=NPCState())
    assert "不能接收玩家给你的东西" not in standalone


def test_item_ledger_block_sits_after_long_term_memory():
    shared = build_user_shared_core(
        long_term_block="【长期记忆】\n记着一件事。",
        item_ledger_block="【物品往来记忆（跨对话）】\n你曾经送给玩家以下物品：2个药。",
        summary_str="【早期摘要】\n摘要。",
    )
    assert shared.index("【长期记忆】") < shared.index("【物品往来记忆（跨对话）】")
    assert shared.index("【物品往来记忆（跨对话）】") < shared.index("【早期摘要】")


def test_daily_user_context_places_item_blocks_after_long_term():
    request = make_request(fx.daily_request())
    text = build_user_context(
        request,
        long_term_block="【长期记忆】",
        item_memory_block="【物品往来记忆（跨对话）】",
        gift_detail_block="【本组已赠送】",
        item_recent_block="【最近物品往来】",
        pending_claim_block="【待处理的索取】",
    )
    order = [
        "【长期记忆】",
        "【物品往来记忆（跨对话）】",
        "【本组已赠送】",
        "【最近物品往来】",
        "【待处理的索取】",
        "【本组已播历史】",
    ]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)
