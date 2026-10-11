"""B2-7 语料增读：绑定存档的任务覆盖层、稳定节点 id、指纹作用域。

全部用 tmp 资源树，不触碰真实数据目录；绑定态用全局单例，fixture 里复位。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.integration.state import get_integration_state
from services.retrieval.loader import compute_corpus_fingerprint, load_task_nodes


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


def _write_slot(tmp_path: Path, slot: str, task_id: int, get_text: str, finish_text: str) -> None:
    slot_dir = tmp_path / "data" / "rag" / "agent_saves" / slot
    _write_json(
        slot_dir / "agent_tasks.json",
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
        slot_dir / "agent_text.json",
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
    assert _ids(nodes) == {
        "task-101-get-0",
        "task-101-get-1",
        "task-101-finish-0",
        "task-102-get-0",
    }
    by_id = {n.id: n for n in nodes}
    assert by_id["task-101-get-0"].text == "拿废铁来。"
    assert by_id["task-101-get-0"].character == "铁匠"
    assert by_id["task-101-get-0"].type == "task"


def test_new_earlier_sorted_file_does_not_shift_ids(resources):
    """新增排序在前的任务文件（agent_tasks.json 排首）时，既有节点 id 不漂移。"""
    before = _ids(load_task_nodes(resources))
    _write_json(
        resources / "data" / "task" / "agent_tasks.json",
        {"tasks": [{"id": 200005, "get_conversation": "$AGENT_GET_200005"}]},
    )
    _write_json(
        resources / "data" / "task" / "text" / "agent_text.json",
        {"$AGENT_GET_200005": [{"name": "铁匠", "text": "新委托。"}]},
    )
    after = _ids(load_task_nodes(resources))
    assert before <= after
    assert "task-200005-get-0" in after


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
# 绑定存档覆盖层
# ---------------------------------------------------------------------------

def test_overlay_unbound_absent_bound_present(resources, state):
    _write_slot(resources, "slot_a", 200001, "甲存档的委托。", "甲存档完成了。")
    _write_slot(resources, "slot_b", 200002, "乙存档的委托。", "乙存档完成了。")

    ids_unbound = _ids(load_task_nodes(resources))
    assert "task-200001-get-0" not in ids_unbound
    assert "task-200002-get-0" not in ids_unbound

    state.bind(slot_key="slot_a")
    nodes = load_task_nodes(resources)
    ids = _ids(nodes)
    by_id = {n.id: n for n in nodes}
    assert {"task-200001-get-0", "task-200001-finish-0"} <= ids
    assert by_id["task-200001-get-0"].text == "甲存档的委托。"
    assert "task-200002-get-0" not in ids  # 其他槽的文件不增读

    state.bind(slot_key="slot_b")
    assert "task-200002-get-0" in _ids(load_task_nodes(resources))


def test_overlay_slot_text_overrides_shared_same_key(resources, state):
    """槽内与共享同名文本 key：槽文本生效（覆盖层后加载，存档优先）。"""
    slot_dir = resources / "data" / "rag" / "agent_saves" / "slot_a"
    _write_json(
        slot_dir / "agent_text.json",
        {"$T101_GET": [{"name": "铁匠", "text": "覆盖后的台词。"}]},
    )
    state.bind(slot_key="slot_a")
    nodes = {n.id: n for n in load_task_nodes(resources)}
    assert nodes["task-101-get-0"].text == "覆盖后的台词。"
    assert "task-101-get-1" not in nodes  # 覆盖数组以槽内为准（一条）


def test_overlay_missing_files_silent(resources, state):
    state.bind(slot_key="slot_empty")  # 槽目录不存在
    assert _ids(load_task_nodes(resources)) == {
        "task-101-get-0",
        "task-101-get-1",
        "task-101-finish-0",
        "task-102-get-0",
    }


# ---------------------------------------------------------------------------
# 指纹作用域
# ---------------------------------------------------------------------------

def test_fingerprint_scoped_to_bound_slot(resources, state):
    base = compute_corpus_fingerprint(resources)
    # 未绑定时：其他槽文件写入不影响
    _write_slot(resources, "slot_b", 200002, "乙委托。", "乙完成。")
    assert compute_corpus_fingerprint(resources) == base

    # 绑定空槽：无增量，指纹不变
    state.bind(slot_key="slot_a")
    assert compute_corpus_fingerprint(resources) == base

    # 绑定到已有文件的槽：指纹变化
    state.bind(slot_key="slot_b")
    with_files = compute_corpus_fingerprint(resources)
    assert with_files != base

    # 本槽文件内容修改：指纹再变
    _write_slot(resources, "slot_b", 200002, "乙委托改长了。", "乙完成改长了。")
    assert compute_corpus_fingerprint(resources) != with_files

    # 解绑：回到基线
    state.unbind()
    assert compute_corpus_fingerprint(resources) == base
