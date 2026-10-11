"""B2-1 编排层接入：绑定存档投影 → save_info → 会话态块玩家名字行；#83 场景行与阶段推导。

- bind 且角色名存在 → user shared core（messages[1]）出现「玩家的名字是「X」。」；
- 未绑定 → 不出现该行（standalone 行为与现状一致，FR-9）；
- 名字行位于玩家身份行之后、好感行之前；save_info 收全 {slot_key, character, progress, scene, player}；
- 场景行：非战斗给地图 label（帧标签归一，缺标签回落原文），战斗中给关卡名（inCombat 门控）；
- 阶段推导：绑定存档能推出阶段时覆盖请求携带的 progress_stage。
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
        "scene": None,
        "player": None,
    }
    assert NAME_LINE in _shared_core(env)


async def test_bind_scene_line_and_stage_override(env, integration_state):
    """场景行＝归一 label；主线链值推导阶段覆盖请求携带的 progress_stage。"""
    integration_state.bind(
        slot_key="slot_a",
        character={"name": "林三"},
        progress={"tasks_finished": [], "task_chains_progress": {"主线": 22}},
        scene={"stageFlag": "地图-食堂", "stageName": "", "entrance": "出生地", "inCombat": False},
        player={"level": 12},
    )
    orch = await _one_turn(env, progress_stage=6)

    assert orch._ctx.progress_stage == 2  # 完成地铁站 → 堕落城，覆盖请求值 6
    assert orch._ctx.stage_cfg.name == "堕落城"
    shared = _shared_core(env)
    assert "玩家当前所在场景：食堂。" in shared  # 帧标签 地图-食堂 → label 食堂
    assert orch._ctx.save_info["scene"]["label"] == "食堂"


async def test_bind_combat_scene_line_uses_stage_name(env, integration_state):
    """战斗中（inCombat）给关卡名；stageFlag 不参与，防非战斗残留值串线。"""
    integration_state.bind(
        slot_key="slot_a",
        character={"name": "林三"},
        scene={"stageFlag": "基地1层", "stageName": "深入禁区", "entrance": "出生地", "inCombat": True},
    )
    await _one_turn(env)

    shared = _shared_core(env)
    assert "玩家当前正在关卡「深入禁区」中战斗。" in shared
    assert "玩家当前所在场景" not in shared


async def test_unbound_no_name_line_and_save_info_none(env, integration_state):
    orch = await _one_turn(env)

    assert orch._ctx.save_info is None
    assert "玩家的名字是" not in _shared_core(env)
    assert "玩家当前所在场景" not in _shared_core(env)


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


def test_session_state_block_scene_line_shapes():
    """场景行：label 优先、缺标签回落帧标签；战斗给关卡名；空段/未绑定不加行。"""
    def block(scene):
        return build_session_state_block(
            favorability=30, relationship_level="熟悉",
            save_info={"slot_key": "s", "character": None, "progress": None, "scene": scene},
        )

    assert "玩家当前所在场景：食堂。" in block(
        {"stageFlag": "地图-食堂", "label": "食堂", "inCombat": False})
    assert "玩家当前所在场景：基地1层。" in block({"stageFlag": "基地1层", "inCombat": False})
    assert "玩家当前正在关卡「深入禁区」中战斗。" in block(
        {"stageFlag": "基地1层", "stageName": "深入禁区", "inCombat": True})
    # 战斗但关卡名缺（残留清空）→ 回落地图行
    assert "玩家当前所在场景：基地1层。" in block(
        {"stageFlag": "基地1层", "stageName": "  ", "inCombat": True})
    # 非战斗但 stageName 有残留值 → 仍按地图行，不出现关卡文案
    assert "关卡" not in block({"stageFlag": "基地1层", "stageName": "深入禁区", "inCombat": False})
    for empty in (None, {}, {"stageFlag": "", "stageName": "", "label": "", "inCombat": False}):
        assert "玩家当前所在场景" not in block(empty)
        assert "玩家当前正在关卡" not in block(empty)
    assert build_session_state_block(favorability=1, save_info=None).find("玩家当前") == -1
