"""B8-1 编排层接入：聊天主 Agent 的 system 静态区场景块与场景补充。

- 场景块进 messages[0]（system），位于同阵营块之前；
- 场景补充（同阵营块未覆盖的同场景 db 记录）紧跟在同阵营块之后；
- 无驻地 NPC（lochan）不拼场景块；map_registry 缺失时整体降级为空；
- 同阵营块与「对话中提到的其他角色」块每行行尾补「；所在场景：…。」（句号收尾，
  免得弱模型把下一行接读成同一句），有驻地才补、无驻地不补（第三批）。
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

# 同阵营取样：Boy 有驻地、lochan 无驻地（场景后缀「有才补」的两端）。
# 阵营必须非「闲杂人等」——那是 SKIP_FACTION_SAME_CAMP，同阵营块整体不出。
SAME_CAMP_STATES = {
    "Pig": NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-右前锋"]),
    "Boy": NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-队长"]),
    "lochan": NPCState(sex="女", faction="A兵团元老", titles=["谜之兔女郎"]),
}

# 提及取样：武器大师双驻地、lochan 无驻地
MENTION_STATES = {
    **GARAGE_STATES,
    "武器大师": NPCState(sex="男", faction="闲杂人等", titles=["武器大师"]),
    "lochan": NPCState(sex="女", faction="闲杂人等", titles=["谜之兔女郎"]),
}


class _NoMapRegistry:
    """包装真实注册表但隐藏 map_registry（模拟地图数据缺失的降级路径）。"""

    def __init__(self, inner):
        self._inner = inner
        self.map_registry = None

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def _messages_of(
    env, npc_name: str, states: dict[str, NPCState], query: str = "你好",
) -> list[dict]:
    """换 NPC 管理器与发布会话，跑一轮纯聊天，返回请求 #1 的 messages。"""
    env.deps.npc_manager = NPCManager(
        state=dict(states), state_path=env.tmp_path / "scene_state.json",
    )
    session = env.store.create_session_sync(npc_name=npc_name, title="场景注入测试")
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = TurnOrchestrator(
        session_id=session.session_id, npc_name=npc_name, query=query, deps=env.deps,
    )
    await collect(orch)
    return env.fake.stream_requests[0].messages


async def _system_prompt_of(
    env, npc_name: str, states: dict[str, NPCState], query: str = "你好",
) -> str:
    """请求 #1 的 system 文本（静态区：场景块／同阵营块／场景补充）。"""
    return (await _messages_of(env, npc_name, states, query))[0]["content"]


async def _shared_core_of(
    env, npc_name: str, states: dict[str, NPCState], query: str = "你好",
) -> str:
    """请求 #1 的 user 共享核文本（提及 NPC 块在其中，玩家当轮发言不在）。"""
    return (await _messages_of(env, npc_name, states, query))[1]["content"]


def _only_line(text: str, prefix: str) -> str:
    """取 text 中唯一以 prefix 开头的整行，用于行级逐字口径校验。"""
    hits = [line for line in text.splitlines() if line.startswith(prefix)]
    assert len(hits) == 1, f"{prefix!r} 命中 {len(hits)} 行，期望 1 行"
    return hits[0]


async def test_scene_block_and_supplement_order(env):
    """Pig（A兵团元老）：场景块 → 同阵营块（Boy/King）→ 场景补充（冷兵器商人）。"""
    system = await _system_prompt_of(env, "Pig", GARAGE_STATES)

    assert "你所在场景：基地车库（基地场景-基地车库）。" in system
    assert "同场景npc（均在你视线范围内）：Boy，King，冷兵器商人。" in system

    same_hdr = "同阵营角色（设定供参考，不要替他们发言）："
    assert same_hdr in system
    assert _only_line(system, "「Boy」") == (
        "「Boy」（性别：男）（阵营：A兵团元老）（身份或称呼：BPK战队-队长）"
        "；所在场景：基地车库（基地场景-基地车库）。"
    )
    assert _only_line(system, "「King」") == (
        "「King」（性别：女）（阵营：A兵团元老）（身份或称呼：BPK战队-左前锋）"
        "；所在场景：基地车库（基地场景-基地车库）。"
    )

    # 不同阵营的同场景 NPC 走场景补充（同阵营块正下方）；补充行不加场景后缀
    supp_line = "「冷兵器商人」（性别：男）（阵营：商人）（身份或称呼：传统卫道士）"
    assert _only_line(system, "「冷兵器商人」") == supp_line

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
    # 地图缺载时同阵营行同样不带场景后缀
    assert _only_line(system, "「Boy」") == (
        "「Boy」（性别：男）（阵营：A兵团元老）（身份或称呼：BPK战队-队长）"
    )


async def test_same_faction_rows_carry_scene_only_when_residence_known(env):
    """同阵营块：有驻地的行尾补「；所在场景：…」，无驻地的行原样收在称呼括号。"""
    system = await _system_prompt_of(env, "Pig", SAME_CAMP_STATES)
    assert _only_line(system, "「Boy」") == (
        "「Boy」（性别：男）（阵营：A兵团元老）（身份或称呼：BPK战队-队长）"
        "；所在场景：基地车库（基地场景-基地车库）。"
    )
    assert _only_line(system, "「lochan」") == (
        "「lochan」（性别：女）（阵营：A兵团元老）（身份或称呼：谜之兔女郎）"
    )


async def test_mentioned_npc_rows_carry_scene_only_when_residence_known(env):
    """提及 NPC 块（user 共享核）同样逐行补场景；无驻地者不加。"""
    core = await _shared_core_of(
        env, "Pig", MENTION_STATES, query="武器大师和 lochan 最近有消息吗？",
    )
    assert "【对话中提到的其他角色（设定供参考）】" in core
    assert _only_line(core, "「武器大师」") == (
        "「武器大师」（性别：男）（阵营：闲杂人等）（身份或称呼：武器大师）；所在场景："
        "健身房（基地场景-健身房）、第一防线（地图-第一防线防区）。"
    )
    assert _only_line(core, "「lochan」") == (
        "「lochan」（性别：女）（阵营：闲杂人等）（身份或称呼：谜之兔女郎）"
    )
