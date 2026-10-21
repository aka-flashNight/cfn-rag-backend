"""插话 status 阶段：confirm/cancel 归 publishing，draft/update 仍归 drafting。

游戏侧按 status.stage 判定「草案确认轮」：只有 drafting 才弹「接不接」；
确认/撤销落地阶段必须是 publishing，并通过 ui_hint 换成「派发/撤销」文案。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import services.dialogue_interject.tools as tools_module
from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.llm import _status_stage_for
from services.dialogue_interject.schemas import StatusPayload
from services.dialogue_interject.tools import DailyToolRuntime
from services.npc.manager import NPCState
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import make_request

STATE = NPCState(
    favorability=3, relationship_level="熟悉", sex="男",
    emotions=["普通"], faction="A兵团", titles=["军需官"],
)
DRAFT = {"draft_id": "d1", "title": "城西补给线"}
DRAFT_ROW = SimpleNamespace(draft=DRAFT, draft_id="d1", bargain_count=0)


def _call(name: str) -> dict:
    return {"type": "function", "id": f"call_{name}", "function": {"name": name, "arguments": "{}"}}


class FakeMemory:
    async def delete_draft(self, session_id: str, *, draft_id: str) -> None:
        return None


def _blocks(**overrides) -> DailyBlocks:
    payload = dict(
        system_prompt="SYS",
        npc_state=STATE,
        npc_states={fx.NPC: STATE},
        memory=FakeMemory(),
        session_id=fx.DAILY_DIALOGUE_ID,
        pending_draft_row=DRAFT_ROW,
        pending_draft_summary="城西补给线：去城西看一眼，带回货物。",
        progress_stage=2,
    )
    payload.update(overrides)
    return DailyBlocks(**payload)


def _runtime(blocks: DailyBlocks) -> DailyToolRuntime:
    return DailyToolRuntime(make_request(fx.daily_request()), blocks)


def _drain_hints(runtime: DailyToolRuntime) -> list[str]:
    hints: list[str] = []
    while not runtime.progress_queue.empty():
        hints.append(runtime.progress_queue.get_nowait().ui_hint)
    return hints


# ---------------------------------------------------------------------------
# stage 映射
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", ["route_task_confirm", "route_task_cancel"])
def test_confirm_and_cancel_are_publishing(tool):
    assert _status_stage_for([_call(tool)], mode="daily") == "publishing"


@pytest.mark.parametrize("tool", ["route_task_draft", "route_task_update"])
def test_draft_and_update_stay_drafting(tool):
    assert _status_stage_for([_call(tool)], mode="daily") == "drafting"


def test_search_and_task_mode_unchanged():
    assert _status_stage_for([_call("route_search")], mode="daily") == "searching"
    assert _status_stage_for([_call("route_task_confirm")], mode="task") == ""
    assert _status_stage_for([_call("route_search")], mode="task") == "searching"


def test_publishing_is_a_legal_stage_value():
    assert StatusPayload(stage="publishing").stage == "publishing"


# ---------------------------------------------------------------------------
# ui_hint 透传
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confirm_pushes_dispatch_hint(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "execute_confirm_agent_task",
        lambda *a, **k: SimpleNamespace(payload={"status": "confirmed", "task_id": 7}),
    )
    runtime = _runtime(_blocks())
    await runtime._run_confirm()
    assert _drain_hints(runtime) == ["派发委托中"]


@pytest.mark.asyncio
async def test_cancel_pushes_withdraw_hint(monkeypatch):
    monkeypatch.setattr(
        tools_module, "execute_cancel_agent_task", lambda *a, **k: None,
    )
    runtime = _runtime(_blocks())
    await runtime._run_cancel()
    assert _drain_hints(runtime) == ["撤销委托中"]


@pytest.mark.asyncio
async def test_confirm_without_draft_does_not_push_hint():
    runtime = _runtime(_blocks(pending_draft_row=None))
    await runtime._run_confirm()
    assert _drain_hints(runtime) == []
