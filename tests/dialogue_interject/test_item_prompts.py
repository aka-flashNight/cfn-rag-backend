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
    assert "玩家向你要东西：给不给由你决定" in with_items
    assert "玩家说要送你东西：必须调用 request_item_from_player" in with_items
    assert "拒绝也要承认玩家开过口" in with_items
    assert "没收到就说没收到" in with_items
    # 通用场景说明必须保留，方向约束只做追加。
    assert "关系不错、想答谢或讨好玩家、手头宽裕等情景下" in with_items
    assert "好感度低时更倾向于拒绝" in with_items
    assert "拒绝就不调用工具" in with_items
    assert "通常要东西应走委托" in with_items
    assert "玩家主动要给你且不要报酬" in with_items
    assert "玩家声称已经送了" in with_items
    assert "你想要但不想付报酬" in with_items
    assert "只是价值不高的小东西" in with_items

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
    assert "玩家主动说要送你东西时" not in system

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
    # 任务模式按状态注册：无 claim 可赠送/索取，有 claim 只留取消，直连轮无物品工具。
    assert tool_matrix_for("task") == ("route_search", "update_long_term_memory")
    assert tool_matrix_for("task", item_tools=True) == (
        "route_search", "update_long_term_memory",
        "give_item_to_player", "request_item_from_player",
    )
    assert tool_matrix_for("task", item_tools=True, has_item_claim=True) == (
        "route_search", "update_long_term_memory", "cancel_item_request",
    )
    assert "give_item_to_player" not in tool_matrix_for("task", has_item_claim=True)


def test_task_prompt_item_states_are_isolated():
    """任务 system 按程序选定的状态注入：不写缺席状态、不写否定句、不写机制说明。"""
    give_take = build_task_system_prompt(
        npc_name=fx.NPC, state=NPCState(), item_state="give_take",
    )
    assert "【物品往来】" not in give_take
    assert "推辞" not in give_take
    assert "give_item_to_player" in give_take
    assert "request_item_from_player" in give_take
    assert "cancel_item_request" not in give_take
    assert "只有办理结果确认已交付" in give_take

    claim = build_task_system_prompt(
        npc_name=fx.NPC, state=NPCState(), item_state="claim",
    )
    assert "【物品往来】" not in claim
    assert "cancel_item_request" in claim
    assert "give_item_to_player" not in claim
    assert "request_item_from_player" not in claim
    assert "当前有一笔待处理的索取" in claim

    result = build_task_system_prompt(
        npc_name=fx.NPC, state=NPCState(), item_state="result",
    )
    assert "【物品往来】" not in result
    for name in ("give_item_to_player", "request_item_from_player", "cancel_item_request"):
        assert name not in result
    assert "route_search" in result

    # 工具结果后的 tool-free system 同样不带推辞块，也不出现物品工具名。
    for state in ("give_take", "claim", "result"):
        tool_free = build_task_system_prompt(
            npc_name=fx.NPC, state=NPCState(), item_state=state,
            include_tool_guide=False,
        )
        assert "【工具（本轮可用）】" not in tool_free
        assert "【物品往来】" not in tool_free
        assert "give_item_to_player" not in tool_free


def test_task_messages_carry_item_scene_blocks():
    """任务 user 现场与日常共用物品块：确认后模型能看到待处理索取。"""
    from types import SimpleNamespace

    from services.dialogue_interject.prompts import build_task_messages

    blocks = SimpleNamespace(
        system_prompt="SYS",
        rag_context_text="",
        long_term_block="【长期记忆】",
        item_memory_block="【物品往来记忆（跨对话）】",
        gift_detail_block="【本组已赠送】",
        item_recent_block="【最近物品往来】",
        pending_claim_block="【待处理的索取】",
    )
    text = build_task_messages(make_request(fx.task_request()), blocks)[1]["content"]
    positions = [
        text.index("【长期记忆】"),
        text.index("【物品往来记忆（跨对话）】"),
        text.index("【本组已赠送】"),
        text.index("【最近物品往来】"),
        text.index("【待处理的索取】"),
        text.index("【本组已播历史对话】"),
    ]
    assert positions == sorted(positions)


def test_embedded_terminal_prompt_has_item_refusal():
    system = build_static_system(npc_name=fx.NPC, state=NPCState(), embedded=True)
    assert "不能接收玩家给你的东西" in system
    assert "物品往来记忆里没有记录" in system
    assert "request_item_from_player" not in system

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
        "【本组已播历史对话】",
    ]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_item_tool_descriptions_distinguish_directions():
    from services.dialogue_interject.item_tools import item_tool_schemas

    schemas = {
        schema["function"]["name"]: schema["function"]["description"]
        for schema in item_tool_schemas()
    }
    assert "玩家向你要东西" in schemas["give_item_to_player"]
    assert "你决定送给玩家一件物品时调用" in schemas["give_item_to_player"]
    assert "好感度低时倾向于拒绝" in schemas["give_item_to_player"]
    assert "拒绝时不调用本工具" in schemas["give_item_to_player"]
    assert "玩家主动要给你东西时" in schemas["request_item_from_player"]
    assert "你决定向玩家索要或询问是否给一件物品时调用" in schemas["request_item_from_player"]
    assert "玩家→你" in schemas["request_item_from_player"]


def test_confirm_instruction_names_publish_npc_only_when_different():
    from services.dialogue_interject.tools import _confirm_instruction

    same = _confirm_instruction(same_npc=True, get_npc="Andy Law")
    other = _confirm_instruction(same_npc=False, get_npc="researcher")

    assert "台词里要指明具体名字" not in same
    assert "「researcher」" in other
    assert "台词里要指明具体名字和大概位置，让玩家知道找谁接任务" in other
