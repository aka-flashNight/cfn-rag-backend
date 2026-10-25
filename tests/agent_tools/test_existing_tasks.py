"""existing_tasks：当前 NPC 已发布的 Agent 委托（任务名＋通关条件）。

为什么要直读磁盘：``TaskRegistry`` 排除了 ``agent_tasks.json``，
``game_data.tasks`` 里根本看不到 Agent 任务（旧写法就算恢复也拿不到数据）。
读取范围＝**本存档的 Agent 委托**（绑定存档时只读本存档，不并入共享落点、不看其它存档；
未绑定时读共享文件），字段恰好是 ``title``/``completion``。
非 Agent 任务（主线/原有任务）不参与——避重只在本存档的 Agent 任务内部发生。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.agent_tools.context_builder import (
    MAX_EXISTING_AGENT_TASKS,
    _build_existing_agent_tasks,
    prepare_task_context,
)
from services.integration.save_paths import agent_save_task_paths, shared_task_paths
from services.integration.state import get_integration_state

NPC = "铁匠"
OTHER = "药师"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _task(task_id: int, get_npc: str = NPC, **extra) -> dict:
    return {
        "id": task_id,
        "title": f"$AGENT_TITLE_{task_id}",
        "get_npc": get_npc,
        "finish_npc": get_npc,
        "chain": "委托",
        **extra,
    }


@pytest.fixture
def state():
    st = get_integration_state()
    st.unbind()
    yield st
    st.unbind()


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "data"
    tasks_path, text_path = shared_task_paths(root)
    _write(tasks_path, {"tasks": [
        _task(200001, finish_requirements=["堕落城#普通"]),
        _task(200002, get_npc=OTHER),
    ]})
    _write(text_path, {
        "$AGENT_TITLE_200001": "取回祖传铁锤",
        "$AGENT_TITLE_200002": "照看药圃",
    })
    return root


def _game_data(root: Path, tasks=None) -> SimpleNamespace:
    return SimpleNamespace(data_root=root, tasks=tasks)


# ---------------------------------------------------------------------------
# 读取口径
# ---------------------------------------------------------------------------

def test_only_current_npc_and_fields_are_title_completion(data_root, state):
    items = _build_existing_agent_tasks(_game_data(data_root), NPC)
    assert items == [{
        "title": "取回祖传铁锤",
        "completion": "任务达成内容（玩家达成）：通关堕落城（普通）",
    }]
    # 字段恰好这两个：不带 id / type / 说明 / 报酬
    assert set(items[0]) == {"title", "completion"}


def test_bound_slot_ignores_shared_landing(data_root, state):
    """绑定存档后只读本存档：共享落点里同 NPC 的旧委托不再进避重名单。"""
    tasks_path, text_path = agent_save_task_paths(data_root, "slot_x")
    _write(tasks_path, {"tasks": [_task(200005, finish_requirements=["医院#困难"])]})
    _write(text_path, {"$AGENT_TITLE_200005": "本存档的委托"})
    state.bind(slot_key="slot_x")

    items = _build_existing_agent_tasks(_game_data(data_root), NPC)
    assert [it["title"] for it in items] == ["本存档的委托"]
    assert "通关医院（困难）" in items[0]["completion"]


def test_other_saves_are_invisible(data_root, state):
    """其它存档的 Agent 委托不参与避重：只看当前绑定的那个存档。"""
    for slot, tid, title in (
        ("slot_a", 200006, "A 存档的委托"),
        ("slot_b", 200007, "B 存档的委托"),
    ):
        tasks_path, text_path = agent_save_task_paths(data_root, slot)
        _write(tasks_path, {"tasks": [_task(tid)]})
        _write(text_path, {f"$AGENT_TITLE_{tid}": title})

    state.bind(slot_key="slot_a")
    assert [it["title"] for it in _build_existing_agent_tasks(_game_data(data_root), NPC)] == [
        "A 存档的委托",
    ]
    state.bind(slot_key="slot_b")
    assert [it["title"] for it in _build_existing_agent_tasks(_game_data(data_root), NPC)] == [
        "B 存档的委托",
    ]


def test_standalone_reads_shared_only(data_root, state):
    tasks_path, text_path = agent_save_task_paths(data_root, "slot_y")
    _write(tasks_path, {"tasks": [_task(200003)]})
    _write(text_path, {"$AGENT_TITLE_200003": "只有存档有的委托"})

    # 未绑定：槽位文件完全看不到
    assert [it["title"] for it in _build_existing_agent_tasks(_game_data(data_root), NPC)] == [
        "取回祖传铁锤",
    ]
    # 绑定后只看到本存档，共享落点完全不参与
    state.bind(slot_key="slot_y")
    titles = [it["title"] for it in _build_existing_agent_tasks(_game_data(data_root), NPC)]
    assert titles == ["只有存档有的委托"]


def test_missing_title_is_skipped(data_root, state):
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(200001), _task(200004)]})
    _write(text_path, {"$AGENT_TITLE_200001": "取回祖传铁锤"})

    items = _build_existing_agent_tasks(_game_data(data_root), NPC)
    assert [it["title"] for it in items] == ["取回祖传铁锤"]


def test_mainline_tasks_are_out_of_scope(data_root, state):
    """主线任务在 game_data.tasks 里，不在 agent_tasks.json —— 天然不出现。"""

    class _MainlineTasks:
        def list_by_npc(self, npc):  # 旧写法会读它；新实现根本不碰
            raise AssertionError("existing_tasks 不应读 game_data.tasks")

        def list_all_tasks(self):
            raise AssertionError("existing_tasks 不应读 game_data.tasks")

    items = _build_existing_agent_tasks(_game_data(data_root, _MainlineTasks()), NPC)
    assert [it["title"] for it in items] == ["取回祖传铁锤"]


def test_missing_data_root_or_bad_json_degrades_to_empty(tmp_path):
    assert _build_existing_agent_tasks(_game_data(tmp_path / "nowhere"), NPC) == []
    tasks_path, _ = shared_task_paths(tmp_path)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text("{broken", encoding="utf-8")
    assert _build_existing_agent_tasks(_game_data(tmp_path), NPC) == []


def test_builder_writes_nothing(data_root, state):
    before = {
        str(p.relative_to(data_root)): p.stat().st_size
        for p in sorted(data_root.rglob("*")) if p.is_file()
    }
    _build_existing_agent_tasks(_game_data(data_root), NPC)
    after = {
        str(p.relative_to(data_root)): p.stat().st_size
        for p in sorted(data_root.rglob("*")) if p.is_file()
    }
    assert after == before


def test_builder_stops_one_past_the_cap(data_root, state):
    ids = list(range(200001, 200001 + MAX_EXISTING_AGENT_TASKS + 3))
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(i) for i in ids]})
    _write(text_path, {f"$AGENT_TITLE_{i}": f"委托{i}" for i in ids})

    items = _build_existing_agent_tasks(_game_data(data_root), NPC)
    assert len(items) == MAX_EXISTING_AGENT_TASKS + 1  # 只多留一条，用于判 truncated


# ---------------------------------------------------------------------------
# prepare 侧：15 条上限 + truncated
# ---------------------------------------------------------------------------

def test_prepare_caps_at_15_and_sets_truncated(game_data, monkeypatch):
    import services.agent_tools.context_builder as cb

    fake_items = [
        {"title": f"委托{i}", "completion": "任务达成内容（玩家达成）：无"}
        for i in range(MAX_EXISTING_AGENT_TASKS + 1)
    ]
    monkeypatch.setattr(cb, "_build_existing_agent_tasks", lambda *a, **k: list(fake_items))

    raw = prepare_task_context(
        task_type="资源收集",
        reward_types={"regular": ["金币"], "optional": []},
        npc_name=NPC,
        player_progress=1,
        game_data=game_data,
    )
    context = json.loads(raw)
    assert len(context["existing_tasks"]) == MAX_EXISTING_AGENT_TASKS
    assert context["truncated"] is True


def test_prepare_keeps_empty_existing_tasks_key(game_data, monkeypatch):
    """没有历史委托时键仍在（空数组），不删键——任务 Agent 的规则要能引用它。"""
    import services.agent_tools.context_builder as cb

    monkeypatch.setattr(cb, "_build_existing_agent_tasks", lambda *a, **k: [])
    raw = prepare_task_context(
        task_type="资源收集",
        reward_types={"regular": ["金币"], "optional": []},
        npc_name=NPC,
        player_progress=1,
        game_data=game_data,
    )
    context = json.loads(raw)
    assert context["existing_tasks"] == []
    assert "truncated" not in context


# ---------------------------------------------------------------------------
# 选型指南里的避重约束
# ---------------------------------------------------------------------------

def test_selection_guide_forbids_duplicate_tasks():
    from services.subagents.prompts import PREPARE_SELECTION_GUIDE

    assert "existing_tasks 是该 NPC 已发布过的 Agent 委托（任务名＋通关条件）" in PREPARE_SELECTION_GUIDE
    assert "不得再发布同类任务" in PREPARE_SELECTION_GUIDE
    assert "即使玩家要求重发" in PREPARE_SELECTION_GUIDE
