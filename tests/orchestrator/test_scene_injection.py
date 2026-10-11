"""B8-1 编排层接入：聊天主 Agent 的 system 静态区场景块与场景补充。

- 场景块进 messages[0]（system），位于同阵营块之前；
- 场景补充（同阵营块未覆盖的同场景 db 记录）紧跟在同阵营块之后；
- 无驻地 NPC（lochan）不拼场景块；map_registry 缺失时整体降级为空。
"""

from __future__ import annotations

from services.npc.manager import NPCManager, NPCState
from services.orchestrator.turn import TurnOrchestrator
from services.npc.scene_context import MULTI_SCENE_HEADER
from tests.orchestrator.conftest import collect

# 基地车库真实成员（数据侧占位顺序：Pig，Boy，King，冷兵器商人）
GARAGE_STATES = {
    "Pig": NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-右前锋", "野猪"]),
    "Boy": NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-队长"]),
    "King": NPCState(sex="女", faction="A兵团元老", titles=["BPK战队-左前锋"]),
    "冷兵器商人": NPCState(sex="男", faction="商人", titles=["传统卫道士"]),
}


class _NoMapRegistry:
    """包装真实注册表但隐藏 map_registry（模拟地图数据缺失的降级路径）。"""

    def __init__(self, inner):
        self._inner = inner
        self.map_registry = None

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def _system_prompt_of(env, npc_name: str, states: dict[str, NPCState]) -> str:
    """换 NPC 管理器与发布会话，跑一轮纯聊天，返回请求 #1 的 system 文本。"""
    env.deps.npc_manager = NPCManager(
        state=dict(states), state_path=env.tmp_path / "scene_state.json",
    )
    session = env.store.create_session_sync(npc_name=npc_name, title="场景注入测试")
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = TurnOrchestrator(
        session_id=session.session_id, npc_name=npc_name, query="你好", deps=env.deps,
    )
    await collect(orch)
    return env.fake.stream_requests[0].messages[0]["content"]


async def test_scene_block_and_supplement_order(env):
    """Pig（A兵团元老）：场景块 → 同阵营块（Boy/King）→ 场景补充（冷兵器商人）。"""
    system = await _system_prompt_of(env, "Pig", GARAGE_STATES)

    assert "你所在场景：基地车库（基地场景-基地车库）。" in system
    assert "同场景npc（均在你视线范围内）：Boy，King，冷兵器商人。" in system

    same_hdr = "同阵营角色（设定供参考，不要替他们发言）："
    assert same_hdr in system
    assert "「Boy」（性别：男）（阵营：A兵团元老）（身份或称呼：BPK战队-队长）" in system
    assert "「King」（性别：女）（阵营：A兵团元老）（身份或称呼：BPK战队-左前锋）" in system

    # 不同阵营的同场景 NPC 走场景补充（同阵营块正下方）
    supp_line = "「冷兵器商人」（性别：男）（阵营：商人）（身份或称呼：传统卫道士）"
    assert supp_line in system

    scene_idx = system.index("你所在场景：")
    same_idx = system.index(same_hdr)
    supp_idx = system.index(supp_line)
    assert scene_idx < same_idx < supp_idx

    # 同阵营已列者不重复补进场景补充；名单里也不混入地图名（无重复行）
    assert system.count("「Boy」") == 1
    assert system.count("「King」") == 1


async def test_single_location_without_others(env):
    system = await _system_prompt_of(
        env, "厨师", {"厨师": NPCState(sex="男", faction="A兵团", titles=["机器人厨师"])},
    )
    assert "你所在场景：食堂（地图-食堂）。" in system
    assert "同场景无其他npc。" in system


async def test_multi_location_npc_gets_annotated_header(env):
    system = await _system_prompt_of(
        env, "前治安官", {"前治安官": NPCState(sex="男", faction="闲杂人等", titles=["警长"])},
    )
    assert MULTI_SCENE_HEADER in system
    assert (
        "基地大厅（基地场景-基地1层）：同场景npc（均在你视线范围内）："
        "杀马特，宝石线人，黑铁会外交部长，学生妹，幸存老兵，迷之盔甲君，爱国青年。"
    ) in system
    assert (
        "第一防线（地图-第一防线防区）：同场景npc（均在你视线范围内）："
        "artist，soldier，武器大师。"
    ) in system
    # 前治安官与同场景 NPC 无同阵营关系、db 记录也未提供 → 无同阵营块/补充行
    assert "同阵营角色" not in system
    assert "「杀马特」" not in system
    assert "「soldier」" not in system


async def test_no_location_npc_gets_no_scene_block(env):
    system = await _system_prompt_of(
        env, "lochan", {"lochan": NPCState(sex="女", faction="闲杂人等", titles=["谜之兔女郎"])},
    )
    assert "你所在场景" not in system
    assert "同场景" not in system


async def test_missing_map_registry_degrades_without_scene_block(env, monkeypatch):
    monkeypatch.setattr(env.deps, "game_data", _NoMapRegistry(env.game_data))
    system = await _system_prompt_of(env, "Pig", GARAGE_STATES)
    assert "你所在场景" not in system
    assert "同场景" not in system
