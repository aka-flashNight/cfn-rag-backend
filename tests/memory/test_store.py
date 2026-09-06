"""MemoryStore 测试（06 §1.2/§5）：单连接 + WAL、统一 ChatMessage、草案 bargain_count 独立列、过期计数。"""

from __future__ import annotations

import pytest

from services.memory.store import ChatMessage, MemoryStore, TaskDraftRow


@pytest.fixture
def store(tmp_path):
    return MemoryStore(db_path=tmp_path / "memory.db")


def test_wal_mode_enabled(store, tmp_path):
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert store._db_path.exists()
    assert (tmp_path / "memory.db-wal").exists() or mode == "wal"


def test_single_connection_reused(store):
    """同一连接对象跨操作复用（修 E4：不再每操作新建连接）。"""
    conn_before = store._conn
    store.add_message_sync("s1", "user", "你好")
    store.get_history_sync("s1")
    assert store._conn is conn_before


def test_message_roundtrip_and_ordering(store):
    for i in range(5):
        store.add_message_sync("s1", "user" if i % 2 == 0 else "assistant", f"消息{i}")
    hist_asc = store.get_history_sync("s1", limit=10, order="asc")
    hist_desc = store.get_history_sync("s1", limit=10, order="desc")
    assert [m.content for m in hist_asc] == [f"消息{i}" for i in range(5)]
    assert [m.content for m in hist_desc] == [f"消息{i}" for i in range(4, -1, -1)]
    assert isinstance(hist_asc[0], ChatMessage)
    assert store.count_messages_sync("s1") == 5
    # 分页：取最新 2 条（正序）
    page = store.get_history_sync("s1", limit=2, order="asc")
    assert [m.content for m in page] == ["消息3", "消息4"]


def test_message_role_validation(store):
    with pytest.raises(ValueError):
        store.add_message_sync("s1", "system", "不该出现")
    with pytest.raises(ValueError):
        store.add_message_sync("", "user", "空 session")


def test_summary_upsert(store):
    assert store.get_summary_sync("s1") is None
    store.save_summary_sync("s1", "第一版摘要", 10)
    store.save_summary_sync("s1", "整合后的摘要", 20)
    assert store.get_summary_sync("s1") == "整合后的摘要"


def test_session_crud_and_cascade(store):
    info = store.create_session_sync("铁匠", "新会话")
    store.update_session_title_sync(info.session_id, "改名会话")
    sessions = store.list_sessions_sync()
    assert len(sessions) == 1 and sessions[0].title == "改名会话"

    store.add_message_sync(info.session_id, "user", "你好")
    store.save_summary_sync(info.session_id, "摘要", 1)
    store.upsert_draft_sync(info.session_id, {"draft_id": "d1"})
    store.delete_session_sync(info.session_id)
    with pytest.raises(ValueError):
        store.delete_session_sync(info.session_id)
    assert store.get_history_sync(info.session_id) == []
    assert store.get_summary_sync(info.session_id) is None
    assert store.get_draft_sync(info.session_id) is None


def test_draft_upsert_and_partial_update(store):
    draft_id = store.upsert_draft_sync("s1", {"draft_id": "d1", "npc_name": "铁匠", "title": "收集任务"})
    assert draft_id == "d1"
    row = store.get_draft_sync("s1")
    assert isinstance(row, TaskDraftRow)
    assert row.draft["title"] == "收集任务"

    updated = store.update_partial_sync("s1", "d1", {"title": "讨价还价后的任务"})
    assert updated.draft["title"] == "讨价还价后的任务"
    assert updated.draft["npc_name"] == "铁匠"
    # draft_id 不匹配 → 不更新
    assert store.update_partial_sync("s1", "wrong-id", {"title": "x"}) is None


def test_bargain_count_is_column_not_json(store):
    """bargain_count 独立成列，不混进草案 JSON（修 D5）。"""
    store.upsert_draft_sync("s1", {"draft_id": "d1", "title": "任务"}, bargain_count=0)
    store.update_partial_sync("s1", "d1", {"title": "任务v2"}, bargain_count=1)
    row = store.get_draft_sync("s1")
    assert row.bargain_count == 1
    assert "bargain_count" not in row.draft
    assert "_draft_commit_valid" not in row.draft


def test_draft_internal_fields_stripped(store):
    """内部字段 _draft_commit_valid 不入库（修 D5）。"""
    store.upsert_draft_sync("s1", {"draft_id": "d1", "_draft_commit_valid": True, "bargain_count": 9})
    row = store.get_draft_sync("s1")
    assert "_draft_commit_valid" not in row.draft
    assert "bargain_count" not in row.draft
    assert row.bargain_count == 0  # 默认 0，而非混入 JSON 的 9


def test_draft_delete_with_id_guard(store):
    store.upsert_draft_sync("s1", {"draft_id": "d1"})
    assert store.delete_draft_sync("s1", draft_id="wrong") is False
    assert store.delete_draft_sync("s1", draft_id="d1") is True
    assert store.get_draft_sync("s1") is None


def test_rounds_without_task_counters(store):
    assert store.get_rounds_without_task_sync("s1") == 0
    assert store.increment_rounds_without_task_sync("s1") == 1
    assert store.increment_rounds_without_task_sync("s1") == 2
    store.reset_rounds_without_task_sync("s1")
    assert store.get_rounds_without_task_sync("s1") == 0
