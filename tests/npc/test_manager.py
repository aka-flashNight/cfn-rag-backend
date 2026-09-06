"""NPCManager 测试（06 §1.1/§5）：并发好感度无丢失更新、防抖落盘、appearance/未知字段 forward-compatible。"""

from __future__ import annotations

import asyncio
import json

import pytest

from services.npc.manager import NPCManager, NPCState


def _write_state(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


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
    assert raw["铁匠"]["favorability"] == 15
    # 防抖任务应已空闲（dirty 已清）
    assert manager._dirty is False


async def test_stop_flushes_pending_writes(tmp_path):
    state_path = tmp_path / "s.json"
    manager = await NPCManager.load(state_path=state_path)
    await manager.update_favorability("铁匠", 7)
    await manager.stop()  # 进程退出前 flush
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["铁匠"]["favorability"] == 7


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
    assert raw["铁匠"]["appearance"] == "络腮胡，满脸煤灰的中年铁匠"
    assert raw["铁匠"]["custom_future_field"] == {"keep": True}
    assert raw["铁匠"]["favorability"] == 31


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
