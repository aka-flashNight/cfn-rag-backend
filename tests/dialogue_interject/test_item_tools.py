"""P6：物品工具注册矩阵、执行链路与旁路 SSE。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import services.dialogue_interject.tools as tools_module
from services.dialogue_interject.actions import ACTIONS, ActionResult
from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.item_claims import ITEM_CLAIMS
from services.dialogue_interject.item_picker import ItemPicker
from services.dialogue_interject.llm import stream_daily_llm
from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.schemas import (
    ActionPayload,
    ItemClaimPayload,
    StreamAction,
    StreamItemClaim,
)
from services.dialogue_interject.tools import (
    DailyToolRuntime,
    ToolExecutionResult,
    ToolProgress,
)
from services.game_data.models import Item
from services.llm import LLMConfig, StreamEvent
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import collect_events, make_request
from tests.dialogue_interject.test_item_pool_value import make_game_data

GIVE_CALL = {
    "type": "function",
    "id": "call_give",
    "function": {"name": "give_item_to_player", "arguments": '{"name":"普通hp药剂","quantity":2}'},
}
TAKE_CALL = {
    "type": "function",
    "id": "call_take",
    "function": {"name": "request_item_from_player", "arguments": '{"name":"普通hp药剂","quantity":3}'},
}
CANCEL_CALL = {
    "type": "function",
    "id": "call_cancel",
    "function": {"name": "cancel_item_request", "arguments": "{}"},
}


class FakeRegistry:
    def __init__(self):
        self.names = None

    def get_openai_tools(self, names=None):
        self.names = list(names or [])
        return [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in self.names
        ]


class FakeMemory:
    def __init__(self):
        self.ledger = []
        self.actions = {}

    async def npc_item_ledger_upsert(self, scope, npc, direction, item_name, quantity):
        self.ledger.append((scope, npc, direction, item_name, quantity))

    async def dialogue_action_get(self, action_id):
        return self.actions.get(action_id)

    async def dialogue_action_save(self, action_id, session_id, kind, result):
        if action_id in self.actions:
            return False
        self.actions[action_id] = {
            "action_id": action_id,
            "session_id": session_id,
            "kind": kind,
            "result": result,
            "created_at": 0.0,
        }
        return True


def _items():
    return [
        Item(name="普通hp药剂", type="消耗品", use="药剂", price=250),
        Item(name="M4A1", type="武器", use="步枪", price=500),
        Item(name="稀有芯片", type="收集品", use="材料", price=30000),
    ]


def _game_data():
    return make_game_data(_items(), rewards=["普通hp药剂", "M4A1", "稀有芯片", "金币"])


def _request(**overrides):
    payload = fx.daily_request(**overrides)
    return make_request(payload)


def _blocks(game_data=None, memory=None, registry=None, **kwargs) -> DailyBlocks:
    return DailyBlocks(
        system_prompt="SYS",
        tool_free_system_prompt="FREE_SYS",
        npc_state=SimpleNamespace(
            faction="A兵团", titles=[], favorability=0, challenge=None,
        ),
        npc_states={},
        game_data=game_data or _game_data(),
        registry=registry or object(),
        memory=memory or FakeMemory(),
        session_id=fx.DAILY_DIALOGUE_ID,
        slot_key="slot-1",
        progress_stage=1,
        turn_no=1,
        **kwargs,
    )


def _patch_picker(monkeypatch, picked):
    async def _pick(self, **_kwargs):
        return picked

    monkeypatch.setattr(ItemPicker, "pick", _pick)


async def _drain(queue):
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def _execute_gift(runtime, call, *, actual_quantity=None, success=True, error=""):
    """跑赠送工具并模拟 Host 回执：收到 action 旁路事件后立刻 resolve。"""
    task = asyncio.create_task(runtime.execute([call]))
    events = []
    resolved = False
    while not task.done():
        try:
            event = await asyncio.wait_for(runtime.progress_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        events.append(event)
        if event.action is not None and not resolved:
            ACTIONS.resolve(ActionResult(
                action_id=event.action.data["action_id"],
                success=success,
                error=error,
                actual_quantity=actual_quantity,
            ))
            resolved = True
    events.extend(await _drain(runtime.progress_queue))
    return await task, events


# ---------------------------------------------------------------------------
# 注册矩阵
# ---------------------------------------------------------------------------


def test_daily_tools_include_give_take_when_no_claim_or_draft():
    registry = FakeRegistry()
    runtime = DailyToolRuntime(_request(), _blocks(registry=registry))
    names = [item["function"]["name"] for item in runtime.tools()]

    assert "give_item_to_player" in names
    assert "request_item_from_player" in names
    assert "cancel_item_request" not in names
    assert "route_task_draft" in names


def test_pending_claim_only_registers_cancel_and_drops_task_tools():
    ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID,
        npc=fx.NPC,
        item_name="普通hp药剂",
        quantity=2,
        turn_no=1,
    )
    registry = FakeRegistry()
    runtime = DailyToolRuntime(_request(), _blocks(registry=registry))
    names = [item["function"]["name"] for item in runtime.tools()]

    assert "cancel_item_request" in names
    assert "give_item_to_player" not in names
    assert "request_item_from_player" not in names
    assert "route_task_draft" not in names
    assert "route_task_confirm" not in names
    assert "route_search" in names


def test_pending_draft_drops_item_tools():
    registry = FakeRegistry()
    blocks = _blocks(
        registry=registry,
        pending_draft_row=SimpleNamespace(draft={"draft_id": "d1"}, bargain_count=0),
    )
    names = [item["function"]["name"] for item in DailyToolRuntime(_request(), blocks).tools()]

    assert "give_item_to_player" not in names
    assert "request_item_from_player" not in names
    assert "route_task_confirm" in names


def test_task_mode_registers_no_item_tools():
    registry = FakeRegistry()
    blocks = _blocks(registry=registry, task_publish_disabled=True)
    request = make_request(fx.task_request())
    names = [item["function"]["name"] for item in DailyToolRuntime(request, blocks).tools()]

    assert "give_item_to_player" not in names
    assert "request_item_from_player" not in names
    assert "cancel_item_request" not in names


def test_direct_action_request_registers_no_item_tools():
    registry = FakeRegistry()
    request = _request(action={
        "kind": "item_take_accept",
        "action_id": "act-1",
        "claim_id": "claim-1",
        "item_name": "普通hp药剂",
        "quantity": 2,
    })
    names = [item["function"]["name"] for item in DailyToolRuntime(request, _blocks(registry=registry)).tools()]

    assert "give_item_to_player" not in names
    assert "request_item_from_player" not in names


# ---------------------------------------------------------------------------
# 赠送
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gift_success_emits_action_and_records_ledger(monkeypatch):
    _patch_picker(monkeypatch, "普通hp药剂")
    memory = FakeMemory()
    runtime = DailyToolRuntime(_request(), _blocks(memory=memory))

    result, events = await _execute_gift(runtime, GIVE_CALL)

    actions = [ev.action for ev in events if ev.action is not None]
    assert len(actions) == 1
    payload = actions[0].data
    assert payload["kind"] == "item_grant"
    assert payload["item_name"] == "普通hp药剂"
    assert payload["quantity"] == 2
    assert payload["requested_quantity"] == 2

    assert result.requires_followup is True
    assert result.system_prompt == "FREE_SYS"
    assert "已成功赠送 2 个 普通hp药剂" in result.result_block
    assert "名称：普通hp药剂" in result.result_block
    assert memory.ledger == [("slot-1", fx.NPC, "give", "普通hp药剂", 2)]
    assert "第1轮：2个普通hp药剂" in ITEM_CLAIMS.gift_detail_block(fx.DAILY_DIALOGUE_ID)


@pytest.mark.asyncio
async def test_gift_over_cap_reduces_quantity(monkeypatch):
    _patch_picker(monkeypatch, "普通hp药剂")
    memory = FakeMemory()
    runtime = DailyToolRuntime(_request(), _blocks(memory=memory))
    call = {
        "type": "function",
        "id": "call_give",
        "function": {
            "name": "give_item_to_player",
            "arguments": '{"name":"普通hp药剂","quantity":200}',
        },
    }

    result, events = await _execute_gift(runtime, call)

    payload = next(ev.action.data for ev in events if ev.action is not None)
    assert payload["quantity"] == 108
    assert payload["requested_quantity"] == 200
    assert "原请求 200，因价值上限调整" in result.result_block
    assert memory.ledger == [("slot-1", fx.NPC, "give", "普通hp药剂", 108)]


@pytest.mark.asyncio
async def test_gift_value_exceeded_rejects_without_action(monkeypatch):
    _patch_picker(monkeypatch, "稀有芯片")
    memory = FakeMemory()
    runtime = DailyToolRuntime(_request(), _blocks(memory=memory))
    call = {
        "type": "function",
        "id": "call_give",
        "function": {
            "name": "give_item_to_player",
            "arguments": '{"name":"稀有芯片","quantity":1}',
        },
    }

    result, events = await _execute_gift(runtime, call)

    assert all(ev.action is None for ev in events)
    assert "因价值过高，赠送失败" in result.result_block
    assert memory.ledger == []
    assert "赠送稀有芯片失败" in ITEM_CLAIMS.recent_block(fx.DAILY_DIALOGUE_ID)


@pytest.mark.asyncio
async def test_gift_not_found_rejects(monkeypatch):
    _patch_picker(monkeypatch, None)
    runtime = DailyToolRuntime(_request(), _blocks())

    result, events = await _execute_gift(runtime, GIVE_CALL)

    assert all(ev.action is None for ev in events)
    assert "未找到合适的赠送物品" in result.result_block


@pytest.mark.asyncio
async def test_late_gift_ack_still_records_ledger(monkeypatch):
    """等待超时后才到的成功回执：不补台词，但物品已发放，必须补记账本。"""
    _patch_picker(monkeypatch, "普通hp药剂")
    monkeypatch.setattr(tools_module, "ACTION_WAIT_TIMEOUT_S", 0.05)
    memory = FakeMemory()
    runtime = DailyToolRuntime(_request(), _blocks(memory=memory))

    task = asyncio.create_task(runtime.execute([GIVE_CALL]))
    action_id = None
    while not task.done():
        try:
            event = await asyncio.wait_for(runtime.progress_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        if event.action is not None:
            action_id = event.action.data["action_id"]
    result = await task

    assert "没有收到游戏侧确认" in result.result_block
    assert memory.ledger == []

    ACTIONS.resolve(ActionResult(action_id=action_id, success=True, actual_quantity=2))
    await asyncio.sleep(0.01)
    assert memory.ledger == [("slot-1", fx.NPC, "give", "普通hp药剂", 2)]
    assert "第1轮：2个普通hp药剂" in ITEM_CLAIMS.gift_detail_block(fx.DAILY_DIALOGUE_ID)


@pytest.mark.asyncio
async def test_gift_host_failure_does_not_claim_success(monkeypatch):
    _patch_picker(monkeypatch, "普通hp药剂")
    memory = FakeMemory()
    runtime = DailyToolRuntime(_request(), _blocks(memory=memory))

    result, _ = await _execute_gift(runtime, GIVE_CALL, success=False, error="背包已满")

    assert "赠送失败：背包已满" in result.result_block
    assert "已成功赠送" not in result.result_block
    assert memory.ledger == []


# ---------------------------------------------------------------------------
# 索取 / 取消
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_take_creates_pending_claim(monkeypatch):
    _patch_picker(monkeypatch, "普通hp药剂")
    runtime = DailyToolRuntime(
        _request(player_item_names=["普通hp药剂"]), _blocks(),
    )

    result = await runtime.execute([TAKE_CALL])
    events = await _drain(runtime.progress_queue)

    claims = [ev.claim for ev in events if ev.claim is not None]
    assert len(claims) == 1
    assert claims[0].data["state"] == "pending"
    assert claims[0].data["item_name"] == "普通hp药剂"
    assert claims[0].data["quantity"] == 3

    claim = ITEM_CLAIMS.pending(fx.DAILY_DIALOGUE_ID)
    assert claim is not None and claim.item_name == "普通hp药剂" and claim.quantity == 3
    assert "你现在要索取物品：3个普通hp药剂" in result.result_block
    assert result.system_prompt == "FREE_SYS"


@pytest.mark.asyncio
async def test_take_equipment_forces_single_quantity(monkeypatch):
    _patch_picker(monkeypatch, "M4A1")
    call = {
        "type": "function",
        "id": "call_take",
        "function": {
            "name": "request_item_from_player",
            "arguments": '{"name":"M4A1","quantity":3}',
        },
    }
    runtime = DailyToolRuntime(
        _request(player_item_names=["M4A1"]), _blocks(),
    )

    result = await runtime.execute([call])

    claim = ITEM_CLAIMS.pending(fx.DAILY_DIALOGUE_ID)
    assert claim is not None and claim.quantity == 1
    assert "数量固定为 1 件" in result.result_block


@pytest.mark.asyncio
async def test_cancel_item_closes_claim():
    claim = ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID,
        npc=fx.NPC,
        item_name="普通hp药剂",
        quantity=2,
        turn_no=1,
    )
    runtime = DailyToolRuntime(_request(), _blocks())

    result = await runtime.execute([CANCEL_CALL])
    events = await _drain(runtime.progress_queue)

    closed = [ev.claim for ev in events if ev.claim is not None]
    assert len(closed) == 1
    assert closed[0].data["state"] == "closed"
    assert closed[0].data["claim_id"] == claim.claim_id
    assert ITEM_CLAIMS.pending(fx.DAILY_DIALOGUE_ID) is None
    assert "取消了这次索取" in result.result_block


@pytest.mark.asyncio
async def test_only_one_item_action_per_round(monkeypatch):
    _patch_picker(monkeypatch, None)
    runtime = DailyToolRuntime(_request(), _blocks())

    result = await runtime.execute([GIVE_CALL, TAKE_CALL])

    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert "未找到合适的赠送物品" in tool_messages[0]["content"]
    assert tool_messages[1]["content"] == "已忽略（一轮只处理一个物品动作）"


# ---------------------------------------------------------------------------
# 流循环：旁路事件与最终无工具 system
# ---------------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        for text, tool_calls in (self.rounds.pop(0) if self.rounds else []):
            if text:
                yield StreamEvent(kind="content", text=text)
            if tool_calls is not None:
                yield StreamEvent(kind="finish", tool_calls=tool_calls)


class ActionProgressExecutor:
    def __init__(self, progress, result: ToolExecutionResult):
        self.progress = progress
        self.result = result
        self.progress_queue: asyncio.Queue[ToolProgress] = asyncio.Queue()

    def tools(self):
        return [{"type": "function", "function": {"name": "give_item_to_player", "parameters": {}}}]

    async def execute(self, tool_calls, *, round_text=""):
        self.progress_queue.put_nowait(self.progress)
        return self.result


def _config() -> LLMConfig:
    return LLMConfig(api_key="test-key", api_base="http://unit.test/v1", model_name="m")


@pytest.mark.asyncio
async def test_stream_forwards_action_and_swaps_to_tool_free_system():
    client = ScriptedLLM([
        [(None, [GIVE_CALL])],
        [('{"add":"拿着吧。","emo":"普通"}\n', None)],
    ])
    executor = ActionProgressExecutor(
        ToolProgress(action=StreamAction(ActionPayload(
            action_id="a1", npc=fx.NPC, item_name="普通hp药剂",
            quantity=2, requested_quantity=2,
        ))),
        ToolExecutionResult(
            requires_followup=True,
            messages=[{"role": "user", "content": "【办理结果】已成功赠送"}],
            result_block="【办理结果】已成功赠送",
            instruction="说最终的话",
            system_prompt="FREE_SYS",
        ),
    )
    factory = lambda request: stream_daily_llm(  # noqa: E731
        request,
        client=client,
        blocks=DailyBlocks(system_prompt="SYS", npc_emotions=("普通",)),
        tool_executor=executor,
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)

    assert [event.event for event in events] == [
        "status", "status", "action", "line", "done",
    ]
    assert events[2].data["item_name"] == "普通hp药剂"
    assert events[3].data["text"] == "拿着吧。"
    assert client.requests[1].tools is None
    assert client.requests[1].messages[0]["content"] == "FREE_SYS"


@pytest.mark.asyncio
async def test_stream_forwards_claim_pending():
    client = ScriptedLLM([
        [(None, [TAKE_CALL])],
        [('{"add":"把药给我。","emo":"普通"}\n', None)],
    ])
    executor = ActionProgressExecutor(
        ToolProgress(claim=StreamItemClaim(ItemClaimPayload(
            state="pending", claim_id="c1", npc=fx.NPC,
            item_name="普通hp药剂", quantity=3,
        ))),
        ToolExecutionResult(
            requires_followup=True,
            messages=[{"role": "user", "content": "【办理结果】索取"}],
            result_block="【办理结果】索取",
            instruction="开口要",
            system_prompt="FREE_SYS",
        ),
    )
    factory = lambda request: stream_daily_llm(  # noqa: E731
        request,
        client=client,
        blocks=DailyBlocks(system_prompt="SYS", npc_emotions=("普通",)),
        tool_executor=executor,
        llm_config=_config(),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)

    assert [event.event for event in events] == [
        "status", "status", "item_claim", "line", "done",
    ]
    assert events[2].data["state"] == "pending"
    assert events[2].data["claim_id"] == "c1"
