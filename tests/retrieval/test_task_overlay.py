"""语料装配与指纹只含基础数据（作者裁定 2026-10-17）。

AI 终端写的任务文件——发布委托的 agent_tasks.json / text/agent_text.json（未绑定时在
data/task 下、绑定后在 data/rag/agent_saves/<slot>/ 槽内）——既不装配成节点也不计入
指纹，索引因此与存档无关。改写文本的查询期替换见 test_overlay_query.py。

全部用 tmp 资源树，不触碰真实数据目录；绑定态用全局单例，fixture 里复位。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.integration.state import get_integration_state
from services.retrieval.loader import compute_corpus_fingerprint, load_task_nodes

#: resources fixture 的两条任务展开出的四个基础节点
BASE_IDS = frozenset({"task-101-get-0", "task-101-get-1", "task-101-finish-0", "task-102-get-0"})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def state():
    st = get_integration_state()
    st.unbind()
    yield st
    st.unbind()


@pytest.fixture
def resources(tmp_path) -> Path:
    """最小共享任务语料：两条任务、四个台词节点。"""
    _write_json(
        tmp_path / "data" / "task" / "mainline_tasks.json",
        {
            "tasks": [
                {"id": 101, "get_conversation": "$T101_GET", "finish_conversation": "$T101_FINISH"},
                {"id": 102, "get_conversation": "$T102_GET"},
            ]
        },
    )
    _write_json(
        tmp_path / "data" / "task" / "text" / "mainline_text.json",
        {
            "$T101_GET": [
                {"name": "铁匠", "text": "拿废铁来。"},
                {"name": "铁匠", "text": "动作快点。"},
            ],
            "$T101_FINISH": [{"name": "铁匠", "text": "干得不错。"}],
            "$T102_GET": [{"name": "酒保", "text": "帮我找个人。"}],
        },
    )
    return tmp_path


def _write_agent_task_files(tasks_path: Path, text_path: Path, task_id: int,
                            get_text: str, finish_text: str) -> None:
    """写一对委托文件（共享侧 text 落在 text/ 子目录，槽内平铺——与真实写盘路径一致）。"""
    _write_json(
        tasks_path,
        {
            "tasks": [
                {
                    "id": task_id,
                    "get_conversation": f"$AGENT_GET_{task_id}",
                    "finish_conversation": f"$AGENT_FINISH_{task_id}",
                }
            ]
        },
    )
    _write_json(
        text_path,
        {
            f"$AGENT_GET_{task_id}": [{"name": "铁匠", "text": get_text}],
            f"$AGENT_FINISH_{task_id}": [{"name": "铁匠", "text": finish_text}],
        },
    )


def _ids(nodes) -> set[str]:
    return {n.id for n in nodes}


# ---------------------------------------------------------------------------
# 稳定节点 id
# ---------------------------------------------------------------------------

def test_task_ids_use_stable_format(resources):
    nodes = load_task_nodes(resources)
    assert _ids(nodes) == BASE_IDS
    by_id = {n.id: n for n in nodes}
    assert by_id["task-101-get-0"].text == "拿废铁来。"
    assert by_id["task-101-get-0"].character == "铁匠"
    assert by_id["task-101-get-0"].type == "task"


def test_task_without_id_falls_back_to_conversation_key(resources):
    _write_json(
        resources / "data" / "task" / "guide_tasks.json",
        {"tasks": [{"get_conversation": "$G_ONLY_GET"}]},
    )
    _write_json(
        resources / "data" / "task" / "text" / "guide_text.json",
        {"$G_ONLY_GET": [{"name": "教官", "text": "先跟我来。"}]},
    )
    assert "task-$G_ONLY_GET-get-0" in _ids(load_task_nodes(resources))


# ---------------------------------------------------------------------------
# AI 终端写的文件不进语料
# ---------------------------------------------------------------------------

def test_shared_agent_files_never_enter_corpus(resources):
    """未绑定态的发布委托落点（data/task 下）不装配：写多少次都不改节点集合。"""
    _write_agent_task_files(
        resources / "data" / "task" / "agent_tasks.json",
        resources / "data" / "task" / "text" / "agent_text.json",
        200005, "新委托。", "新委托完成了。",
    )
    assert _ids(load_task_nodes(resources)) == BASE_IDS


def test_slot_agent_files_never_enter_corpus(resources, state):
    """绑定后槽内的委托文件同样不装配——索引与存档无关，换槽不重建。"""
    slot_dir = resources / "data" / "rag" / "agent_saves" / "slot_a"
    _write_agent_task_files(
        slot_dir / "agent_tasks.json", slot_dir / "agent_text.json",
        200001, "甲存档的委托。", "甲存档完成了。",
    )

    state.bind(slot_key="slot_a")
    assert _ids(load_task_nodes(resources)) == BASE_IDS

    state.bind(slot_key="slot_b")
    assert _ids(load_task_nodes(resources)) == BASE_IDS


def test_slot_text_override_does_not_change_corpus(resources, state):
    """槽内同名 key 不改基础节点：改写只发生在检索之后，语料装配始终是基础文本。"""
    _write_json(
        resources / "data" / "rag" / "agent_saves" / "slot_a" / "agent_text.json",
        {"$T101_GET": [{"name": "铁匠", "text": "覆盖后的台词。"}]},
    )
    state.bind(slot_key="slot_a")
    by_id = {n.id: n for n in load_task_nodes(resources)}
    assert by_id["task-101-get-0"].text == "拿废铁来。"
    assert "task-101-get-1" in by_id


# ---------------------------------------------------------------------------
# 指纹与存档无关
# ---------------------------------------------------------------------------

def test_fingerprint_ignores_agent_written_files(resources, state):
    base = compute_corpus_fingerprint(resources)

    _write_agent_task_files(
        resources / "data" / "task" / "agent_tasks.json",
        resources / "data" / "task" / "text" / "agent_text.json",
        200005, "新委托。", "新委托完成了。",
    )
    assert compute_corpus_fingerprint(resources) == base

    slot_dir = resources / "data" / "rag" / "agent_saves" / "slot_a"
    _write_agent_task_files(
        slot_dir / "agent_tasks.json", slot_dir / "agent_text.json",
        200001, "甲委托。", "甲完成。",
    )
    state.bind(slot_key="slot_a")
    assert compute_corpus_fingerprint(resources) == base

    state.bind(slot_key="slot_b")
    assert compute_corpus_fingerprint(resources) == base


def test_fingerprint_still_tracks_base_corpus(resources):
    """基础语料本身的变化仍要计入，否则游戏数据更新后索引永不追平。"""
    base = compute_corpus_fingerprint(resources)
    _write_json(
        resources / "data" / "task" / "mainline_tasks.json",
        {"tasks": [{"id": 101, "get_conversation": "$T101_GET"}]},
    )
    assert compute_corpus_fingerprint(resources) != base
