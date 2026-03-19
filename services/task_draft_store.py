from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.memory_manager import DB_PATH


_TABLE_INIT_LOCK = threading.Lock()
_TABLE_INITIALIZED = False


def _ensure_table_sync() -> None:
    """
    确保 session_task_drafts 表结构存在。

    注意：不依赖 MemoryManager 的 init 逻辑，保证这个模块可独立使用。
    """

    global _TABLE_INITIALIZED
    if _TABLE_INITIALIZED:
        return

    with _TABLE_INIT_LOCK:
        if _TABLE_INITIALIZED:
            return

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_task_drafts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    draft_id TEXT NOT NULL,
                    npc_name TEXT NOT NULL DEFAULT '',
                    draft_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_task_drafts_updated_at
                ON session_task_drafts(updated_at)
                """
            )
            # 4.5：连续 N 次 ask 未调用任务相关工具时清除草案，按 ask 轮次计数
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_ask_counters (
                    session_id TEXT NOT NULL PRIMARY KEY,
                    rounds_without_task INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
            _TABLE_INITIALIZED = True
        finally:
            conn.close()


@dataclass(frozen=True)
class TaskDraftStoreRow:
    session_id: str
    draft_id: str
    npc_name: str
    draft: Dict[str, Any]
    created_at: float
    updated_at: float


class SessionTaskDraftStore:
    """
    session_task_drafts 表持久化访问层：
    - CRUD：按 session_id 存储/读取/更新/删除“待确认任务草案”
    - 局部更新：update_partial 会仅更新 modify_fields 中出现的顶层字段
      （通过读取 draft_json -> merge -> 写回实现）
    """

    def __init__(self, *, db_path: Any = None) -> None:
        self._db_path = db_path or DB_PATH

    async def _ensure_table(self) -> None:
        await asyncio.to_thread(_ensure_table_sync)

    # ---------------------------------------------------------------------
    # CRUD: Create / Read
    # ---------------------------------------------------------------------
    async def upsert_draft(self, *, session_id: str, draft: Dict[str, Any]) -> str:
        """
        创建或覆盖草案（按 session_id 唯一）。

        Returns:
            draft_id
        """
        session_id = (session_id or "").strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not isinstance(draft, dict):
            raise ValueError("draft 必须是 dict。")

        await self._ensure_table()

        draft_id = str(draft.get("draft_id") or "").strip()
        if not draft_id:
            # draft_agent_task 在 tool_executor 里会生成 draft_id
            draft_id = str(int(time.time() * 1000))
            draft = {**draft, "draft_id": draft_id}

        npc_name = str(draft.get("npc_name") or "").strip()
        now = time.time()
        draft_json = json.dumps(draft, ensure_ascii=False)

        def _inner() -> None:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO session_task_drafts
                        (session_id, draft_id, npc_name, draft_json, created_at, updated_at)
                    VALUES
                        (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        draft_id = excluded.draft_id,
                        npc_name = excluded.npc_name,
                        draft_json = excluded.draft_json,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, draft_id, npc_name, draft_json, now, now),
                )
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_inner)
        return draft_id

    async def get_draft_by_session_id(self, session_id: str) -> Optional[TaskDraftStoreRow]:
        session_id = (session_id or "").strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")

        await self._ensure_table()

        def _inner() -> Optional[TaskDraftStoreRow]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT session_id, draft_id, npc_name, draft_json, created_at, updated_at
                    FROM session_task_drafts
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                draft_json = row["draft_json"]
                try:
                    draft = json.loads(draft_json) if draft_json else {}
                except Exception:
                    # 不应发生：回退为“空草案但保留 raw JSON”
                    draft = {"_draft_json_raw": draft_json}

                return TaskDraftStoreRow(
                    session_id=str(row["session_id"]),
                    draft_id=str(row["draft_id"]),
                    npc_name=str(row["npc_name"] or ""),
                    draft=draft if isinstance(draft, dict) else {},
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                )
            finally:
                conn.close()

        return await asyncio.to_thread(_inner)

    async def get_draft_json_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = await self.get_draft_by_session_id(session_id)
        return row.draft if row else None

    # ---------------------------------------------------------------------
    # CRUD: Update / Delete
    # ---------------------------------------------------------------------
    async def update_partial(
        self,
        *,
        session_id: str,
        draft_id: Optional[str],
        modify_fields: Dict[str, Any],
    ) -> Optional[TaskDraftStoreRow]:
        """
        局部更新草案：
        - 从数据库读取当前 draft_json
        - 顶层 merge modify_fields
        - 写回 draft_json

        Returns:
            更新后的草案行（若不存在/raft_id 不匹配则返回 None）
        """
        session_id = (session_id or "").strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not isinstance(modify_fields, dict):
            raise ValueError("modify_fields 必须是 dict。")

        await self._ensure_table()
        modify_fields = dict(modify_fields)
        now = time.time()

        def _inner() -> Optional[TaskDraftStoreRow]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT session_id, draft_id, npc_name, draft_json, created_at, updated_at
                    FROM session_task_drafts
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                current_draft_id = str(row["draft_id"] or "").strip()
                if (
                    draft_id is not None
                    and str(draft_id).strip()
                    and str(draft_id).strip() != current_draft_id
                ):
                    return None

                draft_json = row["draft_json"]
                try:
                    current_draft = json.loads(draft_json) if draft_json else {}
                except Exception:
                    current_draft = {"_draft_json_raw": draft_json}

                if not isinstance(current_draft, dict):
                    current_draft = {}

                # 顶层合并：modify_fields 出现的字段会替换原值
                current_draft.update(modify_fields)
                new_draft_id = str(current_draft.get("draft_id") or current_draft_id).strip()
                new_npc_name = (
                    str(current_draft.get("npc_name") or row["npc_name"] or "").strip()
                )

                new_draft_json = json.dumps(current_draft, ensure_ascii=False)

                cur.execute(
                    """
                    UPDATE session_task_drafts
                    SET
                        draft_id = ?,
                        npc_name = ?,
                        draft_json = ?,
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (new_draft_id, new_npc_name, new_draft_json, now, session_id),
                )
                conn.commit()

                return TaskDraftStoreRow(
                    session_id=session_id,
                    draft_id=new_draft_id,
                    npc_name=new_npc_name,
                    draft=current_draft,
                    created_at=float(row["created_at"]),
                    updated_at=now,
                )
            finally:
                conn.close()

        return await asyncio.to_thread(_inner)

    async def delete_by_session_id(self, session_id: str, *, draft_id: Optional[str] = None) -> bool:
        """
        删除草案（按 session_id）。

        若提供 draft_id，则只在匹配当前草案时删除，避免误删。
        """
        session_id = (session_id or "").strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")

        await self._ensure_table()

        def _inner() -> bool:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                if draft_id is not None and str(draft_id).strip():
                    cur.execute(
                        """
                        DELETE FROM session_task_drafts
                        WHERE session_id = ? AND draft_id = ?
                        """,
                        (session_id, str(draft_id).strip()),
                    )
                else:
                    cur.execute(
                        """
                        DELETE FROM session_task_drafts
                        WHERE session_id = ?
                        """,
                        (session_id,),
                    )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

        return await asyncio.to_thread(_inner)

    async def clear_session(self, session_id: str) -> bool:
        """别名：删除指定会话的草案。"""
        return await self.delete_by_session_id(session_id)

    # ---------------------------------------------------------------------
    # 4.5：草案自动过期（连续 3 次 ask 未提及任务则清除）
    # ---------------------------------------------------------------------

    _TASK_RELATED_TOOL_NAMES = frozenset({
        "prepare_task_context",
        "draft_agent_task",
        "update_task_draft",
        "confirm_agent_task",
        "cancel_agent_task",
    })

    async def get_rounds_without_task(self, session_id: str) -> int:
        """返回该 session 当前「连续未调用任务相关工具的 ask 轮数」。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return 0
        await self._ensure_table()

        def _inner() -> int:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT rounds_without_task FROM session_ask_counters
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()

        return await asyncio.to_thread(_inner)

    async def increment_rounds_without_task(self, session_id: str) -> int:
        """本 ask 未调用任务相关工具：轮数 +1，返回新值。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return 0
        await self._ensure_table()
        now = time.time()

        def _inner() -> int:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO session_ask_counters (session_id, rounds_without_task, updated_at)
                    VALUES (?, 1, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        rounds_without_task = rounds_without_task + 1,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, now),
                )
                conn.commit()
                cur.execute(
                    "SELECT rounds_without_task FROM session_ask_counters WHERE session_id = ?",
                    (session_id,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 1
            finally:
                conn.close()

        return await asyncio.to_thread(_inner)

    async def reset_rounds_without_task(self, session_id: str) -> None:
        """本 ask 调用了任务相关工具：轮数归零。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return
        await self._ensure_table()
        now = time.time()

        def _inner() -> None:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO session_ask_counters (session_id, rounds_without_task, updated_at)
                    VALUES (?, 0, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        rounds_without_task = 0,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, now),
                )
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_inner)


_GLOBAL_STORE: SessionTaskDraftStore | None = None
_GLOBAL_STORE_LOCK = threading.Lock()


def get_session_task_draft_store() -> SessionTaskDraftStore:
    """获取全局单例 Store。"""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        return _GLOBAL_STORE
    with _GLOBAL_STORE_LOCK:
        if _GLOBAL_STORE is not None:
            return _GLOBAL_STORE
        _GLOBAL_STORE = SessionTaskDraftStore()
        return _GLOBAL_STORE

