"""B2-2/B2-3 按存档写盘：bind→槽目录（文件级隔离）；未 bind→共享文件；id 取号＝共享 ∪ 本存档。

用 tmp data_root 的假注册表，绝不触碰真实数据目录。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.agent_tools.task_tools import write_confirmed_agent_task_files
from services.integration.state import get_integration_state

NPC = "铁匠"


class _FakeTasks:
    def list_all_tasks(self):
        return []


class _FakeGameData:
    def __init__(self, root: Path) -> None:
        self.data_root = Path(root)
        self.tasks = _FakeTasks()


@pytest.fixture
def state():
    st = get_integration_state()
    st.unbind()
    yield st
    st.unbind()


@pytest.fixture
def fake_game_data(tmp_path) -> _FakeGameData:
    return _FakeGameData(tmp_path)


def _draft(title: str = "测试委托") -> dict:
    return {
        "title": title,
        "description": "帮我收点废铁。",
        "npc_name": NPC,
        "get_dialogue": [{"name": NPC, "text": "把废铁收来。"}],
        "finish_dialogue": [{"name": NPC, "text": "干得不错。"}],
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slot_paths(root: Path, slot: str) -> tuple[Path, Path]:
    # root＝data_root（tmp_path 就是 data 目录本身）
    d = root / "rag" / "agent_saves" / slot
    return d / "agent_tasks.json", d / "agent_text.json"


def _shared_paths(root: Path) -> tuple[Path, Path]:
    return root / "task" / "agent_tasks.json", root / "task" / "text" / "agent_text.json"


def test_bound_writes_to_slot_dir_only(state, fake_game_data, tmp_path):
    state.bind(slot_key="slot_a")
    msg, task_id = write_confirmed_agent_task_files(
        draft=_draft(), npc_name_fallback=NPC, game_data=fake_game_data,
    )

    assert task_id == 200001
    tasks_path, text_path = _slot_paths(tmp_path, "slot_a")
    assert tasks_path.exists() and text_path.exists()

    tasks_doc = _read(tasks_path)
    assert [t["id"] for t in tasks_doc["tasks"]] == [200001]
    assert tasks_doc["tasks"][0]["get_conversation"] == "$AGENT_GET_200001"
    text_doc = _read(text_path)
    assert text_doc["$AGENT_TITLE_200001"] == "测试委托"
    assert text_doc["$AGENT_GET_200001"][0]["text"] == "把废铁收来。"

    # 共享文件零新增（正式数据目录不被混入）
    shared_tasks, shared_text = _shared_paths(tmp_path)
    assert not shared_tasks.exists() and not shared_text.exists()
    # task_publish_version 照旧 bump（形式不变）
    assert _read(tmp_path / "rag" / "sync_state.json")["task_publish_version"] == 1
    assert str(task_id) in msg


def test_two_slots_content_isolated(state, fake_game_data, tmp_path):
    state.bind(slot_key="slot_a")
    write_confirmed_agent_task_files(draft=_draft("委托甲"), npc_name_fallback=NPC, game_data=fake_game_data)
    state.bind(slot_key="slot_b")
    write_confirmed_agent_task_files(draft=_draft("委托乙"), npc_name_fallback=NPC, game_data=fake_game_data)

    a_tasks = _read(_slot_paths(tmp_path, "slot_a")[0])
    b_tasks = _read(_slot_paths(tmp_path, "slot_b")[0])
    a_text = _read(_slot_paths(tmp_path, "slot_a")[1])
    b_text = _read(_slot_paths(tmp_path, "slot_b")[1])

    # 文件级隔离：各槽只含自己的任务（跨槽不复用 id 也不互相渗透）
    assert [t["title"] for t in a_tasks["tasks"]] == ["$AGENT_TITLE_200001"]
    assert [t["title"] for t in b_tasks["tasks"]] == ["$AGENT_TITLE_200001"]
    assert a_text["$AGENT_TITLE_200001"] == "委托甲"
    assert b_text["$AGENT_TITLE_200001"] == "委托乙"
    assert set(a_text) == {"$AGENT_TITLE_200001", "$AGENT_DESCRIPTION_200001", "$AGENT_GET_200001", "$AGENT_FINISH_200001"}
    assert set(b_text) == set(a_text)


def test_id_scan_covers_shared_union_slot(state, fake_game_data, tmp_path):
    shared_tasks, _ = _shared_paths(tmp_path)
    shared_tasks.parent.mkdir(parents=True, exist_ok=True)
    shared_tasks.write_text(
        json.dumps({"tasks": [{"id": 200005}]}, ensure_ascii=False), encoding="utf-8",
    )
    state.bind(slot_key="slot_a")
    # 槽内手工预置一条更大的 id，验证扫描把本存档文件也纳入
    slot_tasks, _ = _slot_paths(tmp_path, "slot_a")
    slot_tasks.parent.mkdir(parents=True, exist_ok=True)
    slot_tasks.write_text(
        json.dumps({"tasks": [{"id": 200007}]}, ensure_ascii=False), encoding="utf-8",
    )

    _, task_id = write_confirmed_agent_task_files(
        draft=_draft(), npc_name_fallback=NPC, game_data=fake_game_data,
    )
    assert task_id == 200008
    ids = sorted(t["id"] for t in _read(slot_tasks)["tasks"])
    assert ids == [200007, 200008]


def test_id_scan_reads_shared_when_slot_empty(state, fake_game_data, tmp_path):
    shared_tasks, _ = _shared_paths(tmp_path)
    shared_tasks.parent.mkdir(parents=True, exist_ok=True)
    shared_tasks.write_text(
        json.dumps({"tasks": [{"id": 200042}]}, ensure_ascii=False), encoding="utf-8",
    )
    state.bind(slot_key="slot_a")

    _, task_id = write_confirmed_agent_task_files(
        draft=_draft(), npc_name_fallback=NPC, game_data=fake_game_data,
    )
    assert task_id == 200043


def test_unbound_writes_shared_files(state, fake_game_data, tmp_path):
    _, task_id = write_confirmed_agent_task_files(
        draft=_draft(), npc_name_fallback=NPC, game_data=fake_game_data,
    )

    assert task_id == 200001
    shared_tasks, shared_text = _shared_paths(tmp_path)
    assert shared_tasks.exists() and shared_text.exists()
    assert _read(shared_tasks)["tasks"][0]["id"] == 200001
    # 槽目录零新增
    assert not (tmp_path / "rag" / "agent_saves").exists()


def test_invalid_slot_key_fail_closed(state, fake_game_data, tmp_path):
    state.bind(slot_key="bad/../key")
    with pytest.raises(ValueError):
        write_confirmed_agent_task_files(
            draft=_draft(), npc_name_fallback=NPC, game_data=fake_game_data,
        )
    # 既不写槽目录、也不回退共享文件
    assert not (tmp_path / "rag" / "agent_saves").exists()
    assert not _shared_paths(tmp_path)[0].exists()


def test_write_triggers_index_refresh(state, fake_game_data, monkeypatch):
    """B2-7 接线：写盘成功后（锁外）触发语料刷新检查。"""
    from services.agent_tools import task_tools as task_tools_mod

    calls: list[int] = []
    monkeypatch.setattr(task_tools_mod, "refresh_index_if_stale", lambda: calls.append(1))

    write_confirmed_agent_task_files(
        draft=_draft(), npc_name_fallback=NPC, game_data=fake_game_data,
    )
    assert calls == [1]
