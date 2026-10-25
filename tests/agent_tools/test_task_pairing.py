"""两段式配对（回接取 NPC 领奖）的写盘、校验与 handler 回归。

覆盖：两条连续 id / 8 个文本键 / 固定 1000 经验与真实奖励分流 /
任务二 get_requirements=[id1]（绝不经过 77 clamp）/ 条件全留任务一 /
标记只挂任务一 / 单次 bump / id 边界降级与越界守卫 / 原子回滚。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.agent_tools import handlers, task_tools
from services.agent_tools.handlers import (
    coerce_return_for_reward,
    execute_confirm_agent_task,
    validate_confirm_dialogues,
)
from services.agent_tools.task_tools import write_confirmed_agent_task_files
from services.agent_tools.validator import ValidationReport
from services.integration.state import get_integration_state

NPC_A = "药师"      # 接取 NPC（任务一发布、任务二完成）
NPC_B = "铁匠"      # 完成 NPC（任务一完成、任务二接取）

RETURN_GET = [{"name": NPC_B, "emotion": "", "text": "你回去找药师领赏吧。"}]
RETURN_FINISH = [{"name": NPC_A, "emotion": "", "text": "辛苦了，这是你的报酬。"}]


class _FakeTask:
    def __init__(self, title: str) -> None:
        self.title = title


class _FakeTasks:
    def __init__(self, titles: list[str] | None = None) -> None:
        self._titles = list(titles or [])

    def list_all_tasks(self):
        return [_FakeTask(t) for t in self._titles]


class _FakeGameData:
    def __init__(self, root: Path, titles: list[str] | None = None) -> None:
        self.data_root = Path(root)
        self.tasks = _FakeTasks(titles)


@pytest.fixture
def state():
    st = get_integration_state()
    st.unbind()
    yield st
    st.unbind()


@pytest.fixture
def fake_game_data(tmp_path) -> _FakeGameData:
    return _FakeGameData(tmp_path)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _shared_paths(root: Path) -> tuple[Path, Path]:
    return root / "task" / "agent_tasks.json", root / "task" / "text" / "agent_text.json"


def _slot_paths(root: Path, slot: str) -> tuple[Path, Path]:
    d = root / "rag" / "agent_saves" / slot
    return d / "agent_tasks.json", d / "agent_text.json"


def _draft(
    title: str = "送货给铁匠",
    get_npc: str = NPC_A,
    finish_npc: str = NPC_B,
) -> dict:
    return {
        "title": title,
        "description": "把这封信送到铁匠手里。",
        "npc_name": get_npc,
        "get_npc": get_npc,
        "finish_npc": finish_npc,
        "rewards": [{"item_name": "金币", "count": 20000}],
        "finish_requirements": [],
        "finish_submit_items": [],
        "finish_contain_items": [],
        "get_dialogue": [{"name": get_npc, "emotion": "", "text": "送到铁匠手里。"}],
        "finish_dialogue": [{"name": finish_npc, "emotion": "", "text": "信我收到了。"}],
    }


def _write_paired(draft: dict, game_data: _FakeGameData, **over):
    kwargs = dict(
        draft=draft,
        npc_name_fallback=NPC_A,
        game_data=game_data,
        return_for_reward=True,
        return_description="回到药师处复命，领取这份委托的报酬。",
        return_get_dialogue=RETURN_GET,
        return_finish_dialogue=RETURN_FINISH,
    )
    kwargs.update(over)
    return write_confirmed_agent_task_files(**kwargs)


# ---------------------------------------------------------------------------
# writer：配对形状
# ---------------------------------------------------------------------------

def test_paired_writes_two_tasks_with_eight_text_keys(state, fake_game_data, tmp_path):
    msg, task_id = _write_paired(
        _draft(), fake_game_data,
        dialogue_rewrite=True,
        dialogue_rewrite_instruction="以后对他更客气",
        publish_session_id="sess-1",
    )

    assert task_id == 200001
    tasks_path, text_path = _shared_paths(tmp_path)
    tasks = _read(tasks_path)["tasks"]
    assert [t["id"] for t in tasks] == [200001, 200002]
    first, second = tasks

    # 任务一：NPC 对调、固定 1000 经验、next_task_id 指向任务二、条件全在这
    assert first["get_npc"] == NPC_A and first["finish_npc"] == NPC_B
    assert first["rewards"] == ["经验值#1000"]
    assert first["next_task_id"] == 200002
    assert first["chain"] == "委托"
    assert first["title"] == "$AGENT_TITLE_200001"

    # 任务二：NPC 对调回来、真实奖励、前置只有任务一、条件全空
    assert second["get_npc"] == NPC_B and second["finish_npc"] == NPC_A
    assert second["get_requirements"] == [200001]
    assert second["get_requirements"] != [77]
    assert second["rewards"] == ["金币#20000"]
    assert second["finish_requirements"] == []
    assert second["finish_submit_items"] == []
    assert second["finish_contain_items"] == []
    assert second["chain"] == "委托"
    assert "next_task_id" not in second

    # 对话重写标记只挂任务一
    assert first["dialogue_rewrite"] is True
    assert first["publish_session_id"] == "sess-1"
    assert "dialogue_rewrite" not in second

    text = _read(text_path)
    assert set(text) == {
        "$AGENT_TITLE_200001", "$AGENT_DESCRIPTION_200001",
        "$AGENT_GET_200001", "$AGENT_FINISH_200001",
        "$AGENT_TITLE_200002", "$AGENT_DESCRIPTION_200002",
        "$AGENT_GET_200002", "$AGENT_FINISH_200002",
    }
    assert text["$AGENT_TITLE_200001"] == "送货给铁匠"
    assert text["$AGENT_TITLE_200002"] == "送货给铁匠·回访"
    assert text["$AGENT_DESCRIPTION_200002"] == "回到药师处复命，领取这份委托的报酬。"
    assert text["$AGENT_GET_200002"][0]["name"] == NPC_B
    assert text["$AGENT_FINISH_200002"][0]["name"] == NPC_A

    # 两条任务 + 8 个文本键一次性提交：版本号只 +1
    assert _read(tmp_path / "rag" / "sync_state.json")["task_publish_version"] == 1
    assert "200002" in msg


def test_paired_uses_fallback_description_when_model_gives_none(state, fake_game_data, tmp_path):
    _write_paired(_draft(), fake_game_data, return_description="   ")
    _, text_path = _shared_paths(tmp_path)
    assert _read(text_path)["$AGENT_DESCRIPTION_200002"] == (
        "回到药师处复命，领取这份委托的报酬。"
    )


def test_paired_to_bound_slot_writes_both_records(state, fake_game_data, tmp_path):
    state.bind(slot_key="slot_a")
    _write_paired(_draft(), fake_game_data)

    tasks_path, text_path = _slot_paths(tmp_path, "slot_a")
    assert [t["id"] for t in _read(tasks_path)["tasks"]] == [200001, 200002]
    assert len(_read(text_path)) == 8
    # 共享文件零新增
    assert not _shared_paths(tmp_path)[0].exists()


def test_single_task_shape_unchanged(state, fake_game_data, tmp_path):
    write_confirmed_agent_task_files(
        draft=_draft(get_npc=NPC_A, finish_npc=NPC_A),
        npc_name_fallback=NPC_A,
        game_data=fake_game_data,
    )
    tasks_path, text_path = _shared_paths(tmp_path)

    record = _read(tasks_path)["tasks"][0]
    assert set(record) == {
        "id", "title", "description", "get_requirements", "get_conversation",
        "get_npc", "finish_requirements", "finish_submit_items",
        "finish_contain_items", "finish_conversation", "finish_npc",
        "rewards", "announcement", "chain",
    }
    assert record["rewards"] == ["金币#20000"]
    assert set(_read(text_path)) == {
        "$AGENT_TITLE_200001", "$AGENT_DESCRIPTION_200001",
        "$AGENT_GET_200001", "$AGENT_FINISH_200001",
    }


# ---------------------------------------------------------------------------
# writer：id 边界
# ---------------------------------------------------------------------------

def test_paired_ids_skip_finished_and_active_history(state, fake_game_data, tmp_path):
    state.bind(slot_key="slot_a", progress={
        "tasks_finished": [200001, 200003],
        "tasks_active": [200002],
    })

    msg, task_id = _write_paired(_draft(), fake_game_data)

    # 两个 id 各自避让历史占用：200001-200003 已用，任务一 200004、回访 200005。
    assert task_id == 200004
    tasks = _read(_slot_paths(tmp_path, "slot_a")[0])["tasks"]
    assert [t["id"] for t in tasks] == [200004, 200005]
    assert tasks[0]["next_task_id"] == 200005
    assert "任务 200004 与回访任务 200005" in msg


def test_id2_overflow_downgrades_to_single_task(state, tmp_path):
    game_data = _FakeGameData(tmp_path)
    tasks_path, _ = _shared_paths(tmp_path)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps({"tasks": [{"id": 299999}]}), encoding="utf-8")

    msg, task_id = _write_paired(_draft(), game_data)

    assert task_id == 300000
    tasks = _read(tasks_path)["tasks"]
    assert [t["id"] for t in tasks] == [299999, 300000]
    record = tasks[-1]
    # 降级：真实奖励全部留在唯一那条任务上，绝不保留固定的 1000 经验
    assert record["rewards"] == ["金币#20000"]
    assert "next_task_id" not in record
    assert "回访任务因 id 区间不足未创建" in msg


def test_id1_overflow_raises_and_writes_nothing(state, tmp_path):
    game_data = _FakeGameData(tmp_path)
    tasks_path, text_path = _shared_paths(tmp_path)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps({"tasks": [{"id": 300000}]}), encoding="utf-8")

    with pytest.raises(ValueError):
        _write_paired(_draft(), game_data)

    assert _read(tasks_path)["tasks"] == [{"id": 300000}]
    assert not text_path.exists()


# ---------------------------------------------------------------------------
# writer：原子回滚
# ---------------------------------------------------------------------------

def _seed_files(root: Path) -> tuple[Path, Path]:
    tasks_path, text_path = _shared_paths(root)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(
        json.dumps({"tasks": [{"id": 200001, "title": "旧任务"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    text_path.write_text(
        json.dumps({"$AGENT_TITLE_200001": "旧任务"}, ensure_ascii=False), encoding="utf-8",
    )
    return tasks_path, text_path


def test_tasks_write_failure_restores_both_files(state, tmp_path, monkeypatch):
    game_data = _FakeGameData(tmp_path)
    tasks_path, text_path = _seed_files(tmp_path)
    real_write = task_tools._atomic_write_text
    failed = {"done": False}

    def flaky(path, text):
        if Path(path).name == "agent_tasks.json" and not failed["done"]:
            failed["done"] = True
            raise OSError("磁盘写失败")
        return real_write(path, text)

    monkeypatch.setattr(task_tools, "_atomic_write_text", flaky)

    with pytest.raises(OSError):
        _write_paired(_draft(), game_data)

    assert _read(tasks_path)["tasks"] == [{"id": 200001, "title": "旧任务"}]
    assert _read(text_path) == {"$AGENT_TITLE_200001": "旧任务"}
    assert not (tmp_path / "rag" / "sync_state.json").exists()


def test_bump_failure_restores_files_and_keeps_version(state, tmp_path, monkeypatch):
    game_data = _FakeGameData(tmp_path)
    tasks_path, text_path = _seed_files(tmp_path)
    sync_path = tmp_path / "rag" / "sync_state.json"
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    sync_path.write_text(json.dumps({"task_publish_version": 5}), encoding="utf-8")

    monkeypatch.setattr(
        task_tools, "_bump_task_publish_version",
        lambda _path: (_ for _ in ()).throw(OSError("版本写失败")),
    )

    with pytest.raises(OSError):
        _write_paired(_draft(), game_data)

    assert _read(tasks_path)["tasks"] == [{"id": 200001, "title": "旧任务"}]
    assert _read(text_path) == {"$AGENT_TITLE_200001": "旧任务"}
    assert _read(sync_path)["task_publish_version"] == 5


# ---------------------------------------------------------------------------
# writer：标题去重
# ---------------------------------------------------------------------------

def test_return_title_built_on_renamed_first_title(state, tmp_path):
    game_data = _FakeGameData(tmp_path, titles=["送货给铁匠", "送货给铁匠 II·回访"])
    _, text_path = _shared_paths(tmp_path)

    _write_paired(_draft(), game_data)

    text = _read(text_path)
    assert text["$AGENT_TITLE_200001"] == "送货给铁匠 II"
    # 任务二基于改名后的标题，再 + "·回访"，与已有标题冲突后再去重
    assert text["$AGENT_TITLE_200002"] == "送货给铁匠 II·回访 II"


# ---------------------------------------------------------------------------
# 说话人校验（纯函数）
# ---------------------------------------------------------------------------

def _dialogue_args(**over) -> dict:
    args = {
        "return_for_reward": True,
        "get_dialogue": [{"name": NPC_A, "text": "交给你了。"}],
        "finish_dialogue": [{"name": NPC_B, "text": "收到了。"}],
        "return_get_dialogue": [{"name": NPC_B, "text": "回去领赏吧。"}],
        "return_finish_dialogue": [{"name": NPC_A, "text": "报酬拿好。"}],
    }
    args.update(over)
    return args


def test_valid_pairing_has_no_issues():
    assert validate_confirm_dialogues(_dialogue_args(), get_npc=NPC_A, finish_npc=NPC_B) == []


def test_flag_true_with_same_npc_is_issue():
    issues = validate_confirm_dialogues(_dialogue_args(), get_npc=NPC_A, finish_npc=NPC_A)
    assert any(i["field"] == "return_for_reward" for i in issues)


def test_flag_true_with_empty_return_groups_is_issue():
    issues = validate_confirm_dialogues(
        _dialogue_args(return_get_dialogue=[], return_finish_dialogue=[]),
        get_npc=NPC_A, finish_npc=NPC_B,
    )
    fields = {i["field"] for i in issues}
    assert {"return_get_dialogue", "return_finish_dialogue"} <= fields
    # 报错要能直接回给模型：期望的 NPC 名与 $PC 都写在里面
    assert any(NPC_B in i["message"] or NPC_B in i["fix_hint"] for i in issues)
    assert any("$PC" in i["fix_hint"] for i in issues)


@pytest.mark.parametrize("field, wrong_speaker", [
    ("get_dialogue", NPC_B),
    ("finish_dialogue", NPC_A),
    ("return_get_dialogue", NPC_A),
    ("return_finish_dialogue", NPC_B),
])
def test_wrong_speaker_per_field_is_issue(field, wrong_speaker):
    args = _dialogue_args()
    args[field] = [{"name": wrong_speaker, "text": "走错场了。"}]
    issues = validate_confirm_dialogues(args, get_npc=NPC_A, finish_npc=NPC_B)
    assert [i["field"] for i in issues] == [field]


def test_player_name_is_always_legal():
    args = _dialogue_args(
        get_dialogue=[{"name": "$PC", "text": "交给我吧。"}],
        return_get_dialogue=[{"name": "$PC", "text": "我这就回去。"}],
    )
    assert validate_confirm_dialogues(args, get_npc=NPC_A, finish_npc=NPC_B) == []


def test_flag_false_skips_return_groups():
    args = _dialogue_args(
        return_for_reward=False, return_get_dialogue=[], return_finish_dialogue=[],
    )
    assert validate_confirm_dialogues(args, get_npc=NPC_A, finish_npc=NPC_B) == []


@pytest.mark.parametrize("value, expected", [
    (True, True),
    ("true", True),
    ("TRUE", True),
    (" false ", False),
    ("yes", False),
    (1, False),
    (None, False),
])
def test_coerce_return_for_reward(value, expected):
    assert coerce_return_for_reward(value) is expected


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

@pytest.fixture
def handler_env(monkeypatch):
    """跳过草案校验与摘要（只测两段式校验与写盘分流）。"""
    monkeypatch.setattr(handlers, "validate_task_draft", lambda *a, **k: ValidationReport())
    monkeypatch.setattr(handlers, "_detailed_draft_summary", lambda *a, **k: "")


def _confirm_args(**over) -> dict:
    args = {
        "draft_id": "d1",
        "title": "送货给铁匠",
        "description": "把这封信送到铁匠手里。",
        "get_dialogue": [{"name": NPC_A, "text": "送到铁匠手里。"}],
        "finish_dialogue": [{"name": NPC_B, "text": "信我收到了。"}],
        "return_for_reward": True,
        "return_description": "回到药师处复命，领取这份委托的报酬。",
        "return_get_dialogue": RETURN_GET,
        "return_finish_dialogue": RETURN_FINISH,
    }
    args.update(over)
    return args


def _pending_draft() -> dict:
    draft = _draft()
    draft["draft_id"] = "d1"
    return draft


def test_handler_rejects_bad_dialogue_without_writing(handler_env, state, tmp_path):
    game_data = _FakeGameData(tmp_path)
    outcome = execute_confirm_agent_task(
        _confirm_args(finish_dialogue=[{"name": NPC_A, "text": "说错人了。"}]),
        pending_draft=_pending_draft(),
        npc_name=NPC_A,
        game_data=game_data,
    )

    payload = json.loads(outcome.result_json)
    assert payload["status"] == "validation_failed"
    assert payload["message"] == "两段式对话校验未通过，无法确认。"
    assert payload["issues"]
    assert outcome.draft is not None  # 草案保留
    assert not _shared_paths(tmp_path)[0].exists()


def test_handler_returns_first_id_and_merges_warnings(handler_env, state, tmp_path):
    game_data = _FakeGameData(tmp_path)
    outcome = execute_confirm_agent_task(
        _confirm_args(),
        pending_draft=_pending_draft(),
        npc_name=NPC_A,
        game_data=game_data,
        generation_warnings=["两段式回访对话校验未通过，本次按单任务发布。"],
    )

    payload = json.loads(outcome.result_json)
    assert payload["status"] == "confirmed"
    assert payload["task_id"] == 200001
    assert payload["warnings"] == ["两段式回访对话校验未通过，本次按单任务发布。"]

    tasks = _read(_shared_paths(tmp_path)[0])["tasks"]
    assert [t["id"] for t in tasks] == [200001, 200002]
