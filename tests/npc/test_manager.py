"""NPCManager 测试（06 §1.1/§5）：并发好感度无丢失更新、防抖落盘、appearance/未知字段 forward-compatible、
按存档作用域隔离（B3-1）。"""

from __future__ import annotations

import asyncio
import json

import pytest

from services.integration.state import get_integration_state
from services.npc.manager import NPCManager, NPCState


def _write_state(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def bound_slot():
    """绑定一个存档槽位，用例结束后解绑（NPC 状态作用域随绑定切换）。"""
    state = get_integration_state()

    def _bind(slot_key: str):
        state.bind(slot_key)
        return slot_key

    yield _bind
    state.unbind()


async def test_load_existing_state(tmp_path):
    state_path = tmp_path / "npc_state_db.json"
    _write_state(state_path, {
        "铁匠": {
            "favorability": 55,
            "relationship_level": "朋友",
            "sex": "男",
            "emotions": ["普通", "微笑"],
            "faction": "王国",
            "titles": ["大师铁匠"],
        }
    })
    manager = await NPCManager.load(state_path=state_path)
    state = await manager.get("铁匠")
    assert state.favorability == 55
    assert state.relationship_level == "朋友"
    assert state.emotions == ["普通", "微笑"]
    assert state.faction == "王国"
    assert state.titles == ["大师铁匠"]


async def test_relationship_level_transitions(tmp_path):
    manager = await NPCManager.load(state_path=tmp_path / "s.json")
    st = await manager.update_favorability("NPC", 25)
    assert st.relationship_level == "熟悉"      # 25 ∈ [20,50)
    st = await manager.update_favorability("NPC", 30)
    assert st.relationship_level == "朋友"       # 55 ∈ [50,80)
    st = await manager.update_favorability("NPC", 30)
    assert st.relationship_level == "生死之交"   # 85 ≥ 80
    st = await manager.update_favorability("NPC", -500)
    assert st.favorability == 0 and st.relationship_level == "陌生"


async def test_concurrent_updates_no_lost_writes(tmp_path):
    """同 NPC 两会话并发聊天 20 轮，好感度无丢失更新（06 §5 验收项，修 E1）。"""
    manager = await NPCManager.load(state_path=tmp_path / "s.json")

    async def bump(delta):
        await manager.update_favorability("铁匠", delta)

    await asyncio.gather(*[bump(1) for _ in range(20)])
    state = await manager.get("铁匠")
    assert state.favorability == 20  # 旧版 last-write-wins 会远小于 20

    # 并发混合正负增量
    await asyncio.gather(*[bump(3) for _ in range(10)], *[bump(-1) for _ in range(10)])
    state = await manager.get("铁匠")
    assert state.favorability == 20 + 30 - 10


async def test_debounce_flush_writes_file(tmp_path):
    state_path = tmp_path / "s.json"
    _write_state(state_path, {"铁匠": {"favorability": 10, "relationship_level": "陌生"}})
    manager = await NPCManager.load(state_path=state_path)
    await manager.update_favorability("铁匠", 5)
    await manager.flush()

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["global"]["铁匠"]["favorability"] == 15
    # 防抖任务应已空闲（dirty 已清）
    assert manager._dirty is False


async def test_stop_flushes_pending_writes(tmp_path):
    state_path = tmp_path / "s.json"
    manager = await NPCManager.load(state_path=state_path)
    await manager.update_favorability("铁匠", 7)
    await manager.stop()  # 进程退出前 flush
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["global"]["铁匠"]["favorability"] == 7


async def test_appearance_and_unknown_fields_forward_compatible(tmp_path):
    """appearance 读取容忍、写回保留；未知字段原样写回（forward-compatible）。"""
    state_path = tmp_path / "s.json"
    _write_state(state_path, {
        "铁匠": {
            "favorability": 30,
            "appearance": "络腮胡，满脸煤灰的中年铁匠",
            "custom_future_field": {"keep": True},
        }
    })
    manager = await NPCManager.load(state_path=state_path)
    st = await manager.get("铁匠")
    assert st.appearance == "络腮胡，满脸煤灰的中年铁匠"
    assert st.extra == {"custom_future_field": {"keep": True}}

    await manager.update_favorability("铁匠", 1)
    await manager.flush()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["global"]["铁匠"]["appearance"] == "络腮胡，满脸煤灰的中年铁匠"
    assert raw["global"]["铁匠"]["custom_future_field"] == {"keep": True}
    assert raw["global"]["铁匠"]["favorability"] == 31


async def test_init_from_missing_file_creates_state(tmp_path, monkeypatch):
    """文件不存在 → 从对话数据兜底初始化 NPC 列表并落盘。"""
    from services.npc import manager as manager_mod

    dialogues_dir = tmp_path / "res" / "data" / "dialogues"
    dialogues_dir.mkdir(parents=True)
    (dialogues_dir / "list.xml").write_text("<root><items>d.xml</items></root>", encoding="utf-8")
    (dialogues_dir / "d.xml").write_text(
        "<root><Dialogues><Name>铁匠</Name>"
        "<Dialogue><SubDialogue><Name>铁匠</Name><Text>欢迎光临</Text></SubDialogue></Dialogue>"
        "</Dialogues></root>",
        encoding="utf-8",
    )

    monkeypatch.setattr(manager_mod, "find_resources_directory", lambda: tmp_path / "res")
    manager = await NPCManager.load(state_path=tmp_path / "s.json")
    state = await manager.get("铁匠")
    assert state.favorability == 0
    assert state.relationship_level == "陌生"
    # 玩家占位符不进状态库
    assert "$PC" not in await manager.all_states()


async def test_corrupted_file_starts_empty(tmp_path):
    state_path = tmp_path / "s.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{broken json", encoding="utf-8")
    manager = await NPCManager.load(state_path=state_path)
    assert isinstance(manager, NPCManager)


# ---------------------------------------------------------------------------
# 按存档作用域（B3-1）：只有好感度与关系等级隔离，NPC 设定仍来自 global
# ---------------------------------------------------------------------------

async def test_bound_slot_without_record_keeps_settings_favorability(tmp_path, bound_slot):
    """新存档里没动过的 NPC 沿用设定里的初始好感（A 兵团 25/熟悉、其余 10/陌生），
    归零会让每个新存档都从「谁都不认识」开始；阵营/头衔/性别等设定照旧。"""
    state_path = tmp_path / "s.json"
    _write_state(state_path, {
        "铁匠": {
            "favorability": 85,
            "relationship_level": "生死之交",
            "sex": "男",
            "faction": "A兵团",
            "titles": ["大师铁匠"],
            "emotions": ["普通", "微笑"],
        }
    })
    manager = await NPCManager.load(state_path=state_path)

    bound_slot("slot_1")
    st = await manager.get("铁匠")
    assert st.favorability == 85
    assert st.relationship_level == "生死之交"
    assert st.sex == "男"
    assert st.faction == "A兵团"
    assert st.titles == ["大师铁匠"]
    assert st.emotions == ["普通", "微笑"]

    # 第一次改动从设定初值起算，且只落进本存档
    updated = await manager.update_favorability("铁匠", -10)
    assert updated.favorability == 75 and updated.relationship_level == "朋友"
    await manager.flush()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["saves"]["slot_1"]["铁匠"] == {"favorability": 75, "relationship_level": "朋友"}
    assert raw["global"]["铁匠"]["favorability"] == 85


async def test_two_slots_do_not_share_favorability(tmp_path, bound_slot):
    state_path = tmp_path / "s.json"
    manager = await NPCManager.load(state_path=state_path)

    bound_slot("slot_1")
    await manager.update_favorability("铁匠", 60)
    bound_slot("slot_2")
    await manager.update_favorability("铁匠", 10)

    bound_slot("slot_1")
    st1 = await manager.get("铁匠")
    assert st1.favorability == 60 and st1.relationship_level == "朋友"
    bound_slot("slot_2")
    st2 = await manager.get("铁匠")
    assert st2.favorability == 10 and st2.relationship_level == "陌生"

    await manager.flush()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["saves"]["slot_1"]["铁匠"] == {"favorability": 60, "relationship_level": "朋友"}
    assert raw["saves"]["slot_2"]["铁匠"] == {"favorability": 10, "relationship_level": "陌生"}
    # 存档侧只落这两个字段，设定不重复存
    assert set(raw["saves"]["slot_1"]["铁匠"]) == {"favorability", "relationship_level"}


async def test_unbound_writes_global_and_bound_leaves_it_alone(tmp_path, bound_slot):
    """未绑定存档＝独立那一份，写 global；绑定存档的改动不污染 global。"""
    state_path = tmp_path / "s.json"
    manager = await NPCManager.load(state_path=state_path)

    await manager.update_favorability("铁匠", 25)
    bound_slot("slot_1")
    await manager.update_favorability("铁匠", 10)
    await manager.flush()

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["global"]["铁匠"]["favorability"] == 25
    assert raw["global"]["铁匠"]["relationship_level"] == "熟悉"
    # 绑定档从 global 的 25 起算，但增量只落进 saves
    assert raw["saves"]["slot_1"]["铁匠"]["favorability"] == 35


async def test_all_states_follows_current_scope(tmp_path, bound_slot):
    state_path = tmp_path / "s.json"
    _write_state(state_path, {"铁匠": {"favorability": 70}, "药师": {"favorability": 40}})
    manager = await NPCManager.load(state_path=state_path)

    assert (await manager.all_states())["铁匠"].favorability == 70

    bound_slot("slot_1")
    states = await manager.all_states()
    assert set(states) == {"铁匠", "药师"}
    # 该存档一条记录都没有＝全部沿用设定初值
    assert states["铁匠"].favorability == 70 and states["药师"].favorability == 40

    await manager.update_favorability("铁匠", 10)
    bound_slot("slot_2")
    assert (await manager.all_states())["铁匠"].favorability == 70


async def test_legacy_flat_file_read_as_global(tmp_path, bound_slot):
    """旧版平铺结构整体视为 global 读入，不做额外迁移；下次落盘自然写成新结构。"""
    state_path = tmp_path / "s.json"
    _write_state(state_path, {"铁匠": {"favorability": 55, "relationship_level": "朋友", "faction": "王国"}})
    manager = await NPCManager.load(state_path=state_path)

    st = await manager.get("铁匠")
    assert st.favorability == 55 and st.faction == "王国"

    await manager.flush()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["global"]["铁匠"]["favorability"] == 55
    assert raw["saves"] == {}


async def test_new_structure_roundtrip_preserves_both_scopes(tmp_path, bound_slot):
    state_path = tmp_path / "s.json"
    _write_state(state_path, {
        "global": {"铁匠": {"favorability": 30, "relationship_level": "熟悉", "faction": "A兵团"}},
        "saves": {"slot_1": {"铁匠": {"favorability": 80, "relationship_level": "生死之交"}}},
    })
    manager = await NPCManager.load(state_path=state_path)

    bound_slot("slot_1")
    st = await manager.get("铁匠")
    assert st.favorability == 80 and st.relationship_level == "生死之交"
    assert st.faction == "A兵团"

    await manager.update_favorability("铁匠", -100)
    await manager.flush()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["saves"]["slot_1"]["铁匠"]["favorability"] == 0
    assert raw["global"]["铁匠"]["favorability"] == 30
    assert raw["global"]["铁匠"]["faction"] == "A兵团"


async def test_seed_fills_missing_npcs_and_fields_without_touching_progress(tmp_path):
    """种子只补缺：已有好感/关系/人工字段不变，saves 原样保留，新 NPC 按种子补入。"""
    state_path = tmp_path / "npc_state_db.json"
    seed_path = tmp_path / "seed.json"
    _write_state(state_path, {
        "global": {
            "铁匠": {
                "favorability": 42,
                "relationship_level": "熟悉",
                "emotions": ["普通"],
                "titles": ["老铁匠"],
            }
        },
        "saves": {
            "slot_1": {"铁匠": {"favorability": 42, "relationship_level": "熟悉"}}
        },
    })
    _write_state(seed_path, {
        "global": {
            "铁匠": {
                "favorability": 10,
                "relationship_level": "陌生",
                "sex": "男",
                "faction": "王国",
                "emotions": ["普通", "微笑"],
                "titles": ["大师铁匠"],
                "challenge": "试炼",
            },
            "新NPC": {
                "favorability": 25,
                "relationship_level": "熟悉",
                "emotions": ["普通"],
                "faction": "A兵团",
            },
        },
        "saves": {},
    })

    manager = await NPCManager.load(state_path=state_path, seed_path=seed_path)

    blacksmith = await manager.get("铁匠")
    assert blacksmith.favorability == 42                      # 不重置好感
    assert blacksmith.relationship_level == "熟悉"
    assert blacksmith.sex == "男"                              # 空字段补齐
    assert blacksmith.faction == "王国"
    assert blacksmith.challenge == "试炼"
    assert blacksmith.emotions == ["普通", "微笑"]             # 列表取并集
    assert blacksmith.titles == ["老铁匠", "大师铁匠"]

    new_npc = await manager.get("新NPC")
    assert new_npc.favorability == 25 and new_npc.relationship_level == "熟悉"

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["saves"]["slot_1"]["铁匠"]["favorability"] == 42


async def test_seed_matches_existing_name_case_insensitively(tmp_path):
    """扫描名 boy 与已有 Boy 视作同一 NPC，不产生大小写重复条目。"""
    state_path = tmp_path / "npc_state_db.json"
    seed_path = tmp_path / "seed.json"
    _write_state(state_path, {"Boy": {
        "favorability": 30, "relationship_level": "熟悉", "emotions": ["普通"],
    }})
    _write_state(seed_path, {"global": {"boy": {
        "favorability": 10,
        "relationship_level": "陌生",
        "faction": "A兵团元老",
        "emotions": ["微笑"],
    }}, "saves": {}})

    manager = await NPCManager.load(state_path=state_path, seed_path=seed_path)

    boy = await manager.get("Boy")
    assert boy.favorability == 30 and boy.faction == "A兵团元老"
    assert boy.emotions == ["普通", "微笑"]
    assert await manager.get("boy") is not None                 # 大小写归一命中同一条
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert list(raw["global"]).count("Boy") == 1
    assert "boy" not in raw["global"]


async def test_seed_initializes_missing_runtime_file(tmp_path):
    """运行时文件不存在时直接用种子起库，并按新结构落盘。"""
    state_path = tmp_path / "npc_state_db.json"
    seed_path = tmp_path / "seed.json"
    _write_state(seed_path, {"global": {"铁匠": {
        "favorability": 25, "relationship_level": "熟悉", "faction": "A兵团",
    }}, "saves": {}})

    manager = await NPCManager.load(state_path=state_path, seed_path=seed_path)

    assert (await manager.get("铁匠")).favorability == 25
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["global"]["铁匠"]["favorability"] == 25
    assert raw["saves"] == {}
