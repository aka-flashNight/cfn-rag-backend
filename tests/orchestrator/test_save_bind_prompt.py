"""B2-1 编排层接入：绑定存档投影 → save_info → 会话态块玩家名字行。

- bind 且角色名存在 → user shared core（messages[1]）出现「玩家的名字是「X」。」；
- 未绑定 → 不出现该行（standalone 行为与现状一致，FR-9）；
- 名字行位于玩家身份行之后、好感行之前；save_info 收全 {slot_key, character, progress}。
"""

from __future__ import annotations

import pytest

from services.integration.state import get_integration_state
from services.orchestrator.prompts import build_session_state_block
from services.orchestrator.turn import TurnOrchestrator
from tests.orchestrator.conftest import NPC_NAME, collect

NAME_LINE = "玩家的名字是「林三」。"


@pytest.fixture
def integration_state():
    st = get_integration_state()
    st.unbind()
    yield st
    st.unbind()


async def _one_turn(env, **kwargs) -> TurnOrchestrator:
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = TurnOrchestrator(
        session_id=env.session_id, npc_name=NPC_NAME, query="你好", deps=env.deps, **kwargs,
    )
    await collect(orch)
    return orch


def _shared_core(env) -> str:
    return env.fake.stream_requests[0].messages[1]["content"]


async def test_bind_fills_save_info_and_injects_name_line(env, integration_state):
    integration_state.bind(
        slot_key="slot_a",
        character={"name": "林三"},
        progress={"tasks_finished": [200001], "task_chains_progress": {"主线甲": 3}},
    )
    orch = await _one_turn(env)

    assert orch._ctx.save_info == {
        "slot_key": "slot_a",
        "character": {"name": "林三"},
        "progress": {"tasks_finished": [200001], "task_chains_progress": {"主线甲": 3}},
    }
    assert NAME_LINE in _shared_core(env)


async def test_unbound_no_name_line_and_save_info_none(env, integration_state):
    orch = await _one_turn(env)

    assert orch._ctx.save_info is None
    assert "玩家的名字是" not in _shared_core(env)


async def test_name_line_between_identity_and_favorability(env, integration_state):
    integration_state.bind(slot_key="slot_a", character={"name": "林三"})
    await _one_turn(env, player_identity="男，佣兵，跑单帮的")

    shared = _shared_core(env)
    identity_idx = shared.index("玩家的身份是：男，佣兵，跑单帮的")
    name_idx = shared.index(NAME_LINE)
    favor_idx = shared.index("好感度")
    assert identity_idx < name_idx < favor_idx


def test_session_state_block_save_info_shapes():
    """名字行只在投影里真有非空名字时出现；缺名字/空名字/未绑定一律不加行。"""
    def block(save_info):
        return build_session_state_block(favorability=30, relationship_level="熟悉", save_info=save_info)

    assert NAME_LINE in block({"slot_key": "s", "character": {"name": " 林三 "}, "progress": None})
    assert "玩家的名字是" not in block(None)
    assert "玩家的名字是" not in block({"slot_key": "s", "character": None, "progress": None})
    assert "玩家的名字是" not in block({"slot_key": "s", "character": {}, "progress": None})
    assert "玩家的名字是" not in block({"slot_key": "s", "character": {"name": "   "}, "progress": None})
