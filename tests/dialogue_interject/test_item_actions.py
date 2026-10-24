"""P6：索取选项直连、任务选项直连与 action_id 幂等。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.item_actions import (
    _run_item_take_action,
    _run_task_action,
)
from services.dialogue_interject.item_claims import ITEM_CLAIMS
from services.dialogue_interject.tools import DailyToolRuntime
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import make_request
from tests.dialogue_interject.test_item_pool_value import make_game_data
from tests.dialogue_interject.test_item_tools import FakeMemory, _items


def _blocks(game_data=None, memory=None) -> DailyBlocks:
    return DailyBlocks(
        system_prompt="SYS",
        tool_free_system_prompt="FREE_SYS",
        npc_state=SimpleNamespace(
            faction="A兵团", titles=[], favorability=0, challenge=None,
        ),
        npc_states={},
        game_data=game_data or make_game_data(_items()),
        registry=object(),
        memory=memory or FakeMemory(),
        session_id=fx.DAILY_DIALOGUE_ID,
        slot_key="slot-1",
        progress_stage=1,
        turn_no=1,
    )


def _item_request(kind: str, **fields):
    payload = fx.daily_request(action={"kind": kind, **fields})
    return make_request(payload)


@pytest.mark.asyncio
async def test_item_take_accept_records_receive_and_closes_claim():
    claim = ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID,
        npc=fx.NPC,
        item_name="M4A1",
        quantity=1,
        turn_no=1,
    )
    memory = FakeMemory()
    request = _item_request(
        "item_take_accept",
        action_id="act-accept-1",
        claim_id=claim.claim_id,
        item_name="M4A1",
        quantity=1,
        item_level=6,
    )

    result, instruction, closed = await _run_item_take_action(request, _blocks(memory=memory))

    assert "玩家把 1 个 M4A1" in result
    assert "强化度6" in result
    assert "经过了较高成本的强化" in result
    assert "名称：M4A1" in result
    assert "收下这件物品" in instruction
    assert memory.ledger == [("slot-1", fx.NPC, "receive", "M4A1", 1)]
    assert ITEM_CLAIMS.pending(fx.DAILY_DIALOGUE_ID) is None
    assert closed is not None and closed.data["state"] == "closed"
    assert "玩家刚刚给了你 1 个 M4A1" in ITEM_CLAIMS.recent_block(fx.DAILY_DIALOGUE_ID)

    # 同一 action_id 重试：不再重复记账
    result2, _, _ = await _run_item_take_action(request, _blocks(memory=memory))
    assert result2 == result
    assert memory.ledger == [("slot-1", fx.NPC, "receive", "M4A1", 1)]


@pytest.mark.asyncio
async def test_item_take_accept_mismatch_keeps_claim():
    claim = ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID,
        npc=fx.NPC,
        item_name="普通hp药剂",
        quantity=2,
        turn_no=1,
    )
    memory = FakeMemory()
    request = _item_request(
        "item_take_accept",
        action_id="act-mismatch",
        claim_id=claim.claim_id,
        item_name="M4A1",
        quantity=2,
    )

    result, _, closed = await _run_item_take_action(request, _blocks(memory=memory))

    assert "对不上" in result
    assert closed is None
    assert memory.ledger == []
    assert ITEM_CLAIMS.pending(fx.DAILY_DIALOGUE_ID) is claim


@pytest.mark.asyncio
async def test_item_take_decline_closes_and_records_failure():
    claim = ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID,
        npc=fx.NPC,
        item_name="M4A1",
        quantity=1,
        turn_no=1,
    )
    memory = FakeMemory()
    request = _item_request(
        "item_take_decline",
        action_id="act-decline",
        claim_id=claim.claim_id,
        item_name="M4A1",
        quantity=1,
    )

    result, _, closed = await _run_item_take_action(request, _blocks(memory=memory))

    assert "玩家拒绝把M4A1交给你" in result
    assert closed is not None and closed.data["state"] == "closed"
    assert ITEM_CLAIMS.pending(fx.DAILY_DIALOGUE_ID) is None
    assert memory.ledger == []
    assert "玩家拒绝向你赠送M4A1" in ITEM_CLAIMS.recent_block(fx.DAILY_DIALOGUE_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_factory", [fx.task_request, fx.daily_request], ids=["task", "daily"])
async def test_task_accept_direct_loads_pending_draft(monkeypatch, request_factory):
    row = SimpleNamespace(draft={"draft_id": "d1"}, draft_id="d1", bargain_count=0)
    captured = {}

    class DraftMemory(FakeMemory):
        async def get_draft(self, session_id):
            return row

    async def _accept(self):
        captured["row"] = self.blocks.pending_draft_row
        return "【办理结果】已发布", "收尾"

    monkeypatch.setattr(DailyToolRuntime, "run_task_accept", _accept)
    request = make_request(request_factory(action={
        "kind": "task_accept",
        "action_id": "task-act-draft",
    }))

    await _run_task_action(request, _blocks(memory=DraftMemory()), llm=None)

    assert captured["row"] is row


@pytest.mark.asyncio
@pytest.mark.parametrize("request_factory", [fx.task_request, fx.daily_request], ids=["task", "daily"])
async def test_task_accept_direct_is_idempotent(monkeypatch, request_factory):
    calls = []

    async def _accept(self):
        calls.append(1)
        return "【办理结果】这份委托已经正式派给玩家了。", "收尾一句话"

    monkeypatch.setattr(DailyToolRuntime, "run_task_accept", _accept)
    memory = FakeMemory()
    request = make_request(request_factory(action={
        "kind": "task_accept",
        "action_id": "task-act-1",
    }))
    blocks = _blocks(memory=memory)

    first = await _run_task_action(request, blocks, llm=None)
    second = await _run_task_action(request, blocks, llm=None)

    assert first == second
    assert len(calls) == 1
    saved = memory.actions["task-act-1"]
    assert saved["kind"] == "task_accept"
    assert saved["result"]["result_block"] == first[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_factory", [fx.task_request, fx.daily_request], ids=["task", "daily"])
async def test_task_decline_direct_uses_cancel(monkeypatch, request_factory):
    calls = []

    async def _decline(self):
        calls.append(1)
        return "【办理结果】这份委托已经撤销。", "收尾一句话"

    monkeypatch.setattr(DailyToolRuntime, "run_task_decline", _decline)
    request = make_request(request_factory(action={
        "kind": "task_decline",
        "action_id": "task-decline-1",
    }))

    result = await _run_task_action(request, _blocks(), llm=None)

    assert result == ("【办理结果】这份委托已经撤销。", "收尾一句话")
    assert len(calls) == 1
