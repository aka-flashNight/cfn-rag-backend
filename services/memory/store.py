"""会话存储（SQLite 单连接 + WAL，修 E4）：消息 / 会话 / 摘要 / 任务草案 / ask 计数。

对应 docs/v3-developer/06-存储启动与打包瘦身.md §1.2：
- 单连接 sqlite3.connect(check_same_thread=False) + threading.Lock + WAL，不再每操作新建连接；
- 统一返回 dataclass（ChatMessage），消灭「dict 列表 vs dataclass 双形状」；
- 草案过期：连续 N 次 ask 未触碰任务工具即删除，N 由 draft_keep_turns 配置；
- bargain_count 独立成列，不再混进草案 JSON（修 D5），内部字段 _draft_commit_valid 同样不入库。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from services.game_data.paths import (
    RESOURCE_FOLDER_NAMES,
    pick_existing_or_default_resource_root,
)

logger = logging.getLogger(__name__)

# 摘要触发间隔（消息条数）
SUMMARIZE_INTERVAL = 30

# 不允许混入草案 JSON 的内部/业务分离字段（修 D5）
_DRAFT_INTERNAL_FIELDS = ("bargain_count", "_draft_commit_valid")


def get_db_path() -> Path:
    """memory.db 路径（resources/tools/memory.db，开发环境与 exe 运行时同规则）。"""
    if hasattr(sys, "_MEIPASS") or getattr(sys, "frozen", False):
        base_dir = Path(os.path.dirname(sys.executable))
    else:
        base_dir = Path(__file__).resolve().parent.parent.parent

    for name in RESOURCE_FOLDER_NAMES:  # 情况1: base_dir/<resources>/tools/memory.db
        db_path = base_dir / name / "tools" / "memory.db"
        if db_path.parent.exists():
            return db_path

    cwd = Path(os.getcwd()).resolve()  # 情况2: 当前工作目录
    for name in RESOURCE_FOLDER_NAMES:
        db_path = cwd / name / "tools" / "memory.db"
        if db_path.parent.exists():
            return db_path

    for name in RESOURCE_FOLDER_NAMES:  # 情况3: 工作目录父目录
        db_path = cwd.parent / name / "tools" / "memory.db"
        if db_path.parent.exists():
            return db_path

    root = pick_existing_or_default_resource_root(base_dir)
    db_path = root / "tools" / "memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@dataclass
class ChatMessage:
    """统一消息形状（API 层与实现一致）。"""

    id: int
    role: str
    content: str
    timestamp: float


@dataclass
class TaskDraftRow:
    session_id: str
    draft_id: str
    npc_name: str
    draft: dict[str, Any]
    bargain_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class SessionInfo:
    session_id: str
    npc_name: str
    title: str
    created_at: float


class MemoryStore:
    """SQLite 存储访问层（同步实现 + asyncio.to_thread 包装）。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else get_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    # ------------------------------------------------------------------
    # 表结构
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_session_time ON chat_history(session_id, timestamp)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    npc_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    summary TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_task_drafts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    draft_id TEXT NOT NULL,
                    npc_name TEXT NOT NULL DEFAULT '',
                    draft_json TEXT NOT NULL,
                    bargain_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_ask_counters (
                    session_id TEXT NOT NULL PRIMARY KEY,
                    rounds_without_task INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._migrate_legacy_schema(cur)
            self._conn.commit()

    def _migrate_legacy_schema(self, cur: sqlite3.Cursor) -> None:
        """v3 前的旧库升级：session_task_drafts 补 bargain_count 独立列（修 D5），
        并把旧草案 JSON 内混入的 bargain_count / _draft_commit_valid 剥离。"""
        cur.execute("PRAGMA table_info(session_task_drafts)")
        cols = {str(r[1]) for r in cur.fetchall()}
        if not cols or "bargain_count" in cols:
            return
        cur.execute(
            "ALTER TABLE session_task_drafts ADD COLUMN bargain_count INTEGER NOT NULL DEFAULT 0"
        )
        rows = cur.execute("SELECT id, draft_json FROM session_task_drafts").fetchall()
        for row_id, draft_json in rows:
            try:
                draft = json.loads(draft_json) if draft_json else {}
            except json.JSONDecodeError:
                continue
            if isinstance(draft, dict) and draft.get("bargain_count") is not None:
                try:
                    bc = int(draft["bargain_count"])
                except (TypeError, ValueError):
                    bc = 0
                cur.execute(
                    "UPDATE session_task_drafts SET bargain_count = ? WHERE id = ?", (bc, row_id)
                )
        # 剥离旧草案 JSON 中的内部字段
        cur.execute(
            "UPDATE session_task_drafts SET draft_json = REPLACE(draft_json, '\"_draft_commit_valid\": true, ', '')"
        )
        cur.execute(
            "UPDATE session_task_drafts SET draft_json = REPLACE(draft_json, ', \"_draft_commit_valid\": false', '')"
        )
        logger.info("旧库迁移完成：session_task_drafts 补 bargain_count 列（%d 行回填）", len(rows))

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------

    def add_message_sync(self, session_id: str, role: str, content: str) -> int:
        """新增一条消息，返回该会话消息总数（调用方据此触发摘要）。"""
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if role not in {"user", "assistant"}:
            raise ValueError("role 必须是 'user' 或 'assistant'。")
        ts = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, ts),
            )
            cur.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (session_id,))
            count = int(cur.fetchone()[0])
            self._conn.commit()
            return count

    def count_messages_sync(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (session_id.strip(),))
            return int(cur.fetchone()[0])

    def get_history_sync(
        self,
        session_id: str,
        limit: int = SUMMARIZE_INTERVAL,
        offset: int = 0,
        order: Literal["asc", "desc"] = "asc",
    ) -> list[ChatMessage]:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if limit <= 0:
            return []
        offset = max(0, offset)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, role, content, timestamp FROM chat_history
                WHERE session_id = ? ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            )
            rows = cur.fetchall()
        messages = [
            ChatMessage(id=int(r["id"]), role=str(r["role"]), content=str(r["content"]), timestamp=float(r["timestamp"]))
            for r in rows
        ]
        return list(reversed(messages)) if order == "asc" else messages

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    def get_summary_sync(self, session_id: str) -> str | None:
        session_id = session_id.strip()
        if not session_id:
            return None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
        return str(row["summary"]) if row else None

    def save_summary_sync(self, session_id: str, summary: str, message_count: int) -> None:
        ts = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_summaries (session_id, summary, message_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    message_count = excluded.message_count,
                    updated_at = excluded.updated_at
                """,
                (session_id.strip(), summary, message_count, ts),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    def create_session_sync(self, npc_name: str, title: str) -> SessionInfo:
        npc_name = npc_name.strip()
        title = title.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")
        if not title:
            raise ValueError("title 不能为空。")
        info = SessionInfo(session_id=str(uuid.uuid4()), npc_name=npc_name, title=title, created_at=time.time())
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO sessions (session_id, npc_name, title, created_at) VALUES (?, ?, ?, ?)",
                (info.session_id, info.npc_name, info.title, info.created_at),
            )
            self._conn.commit()
        return info

    def list_sessions_sync(self) -> list[SessionInfo]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT session_id, npc_name, title, created_at FROM sessions ORDER BY created_at DESC, session_id ASC"
            )
            rows = cur.fetchall()
        return [
            SessionInfo(session_id=str(r["session_id"]), npc_name=str(r["npc_name"]), title=str(r["title"]), created_at=float(r["created_at"]))
            for r in rows
        ]

    def update_session_title_sync(self, session_id: str, title: str) -> SessionInfo:
        session_id = session_id.strip()
        title = title.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not title:
            raise ValueError("title 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT session_id, npc_name, title, created_at FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"会话 '{session_id}' 不存在。")
            cur.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id))
            self._conn.commit()
        return SessionInfo(
            session_id=str(row["session_id"]),
            npc_name=str(row["npc_name"]),
            title=title,
            created_at=float(row["created_at"]),
        )

    def delete_session_sync(self, session_id: str) -> None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            if cur.fetchone() is None:
                raise ValueError(f"会话 '{session_id}' 不存在。")
            cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM session_task_drafts WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM session_ask_counters WHERE session_id = ?", (session_id,))
            self._conn.commit()

    # ------------------------------------------------------------------
    # 任务草案（bargain_count 独立列，修 D5）
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_draft(draft: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in draft.items() if k not in _DRAFT_INTERNAL_FIELDS}

    def upsert_draft_sync(
        self, session_id: str, draft: dict[str, Any], *, bargain_count: int | None = None
    ) -> str:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not isinstance(draft, dict):
            raise ValueError("draft 必须是 dict。")
        draft = self._clean_draft(dict(draft))
        draft_id = str(draft.get("draft_id") or "").strip() or str(int(time.time() * 1000))
        draft.setdefault("draft_id", draft_id)
        npc_name = str(draft.get("npc_name") or "").strip()
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            if bargain_count is None:
                cur.execute(
                    "SELECT bargain_count FROM session_task_drafts WHERE session_id = ?", (session_id,)
                )
                row = cur.fetchone()
                bc = int(row["bargain_count"]) if row else 0
            else:
                bc = int(bargain_count)
            cur.execute(
                """
                INSERT INTO session_task_drafts
                    (session_id, draft_id, npc_name, draft_json, bargain_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    draft_id = excluded.draft_id,
                    npc_name = excluded.npc_name,
                    draft_json = excluded.draft_json,
                    bargain_count = excluded.bargain_count,
                    updated_at = excluded.updated_at
                """,
                (session_id, draft_id, npc_name, json.dumps(draft, ensure_ascii=False), bc, now, now),
            )
            self._conn.commit()
        return draft_id

    def get_draft_sync(self, session_id: str) -> TaskDraftRow | None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT session_id, draft_id, npc_name, draft_json, bargain_count, created_at, updated_at
                FROM session_task_drafts WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        draft_json = row["draft_json"]
        try:
            draft = json.loads(draft_json) if draft_json else {}
        except json.JSONDecodeError:
            draft = {"_draft_json_raw": draft_json}
        if isinstance(draft, dict):
            draft = self._clean_draft(draft)  # 旧库可能残留内部字段
        return TaskDraftRow(
            session_id=str(row["session_id"]),
            draft_id=str(row["draft_id"]),
            npc_name=str(row["npc_name"] or ""),
            draft=draft if isinstance(draft, dict) else {},
            bargain_count=int(row["bargain_count"] or 0),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def update_partial_sync(
        self,
        session_id: str,
        draft_id: str | None,
        modify_fields: dict[str, Any],
        *,
        bargain_count: int | None = None,
    ) -> TaskDraftRow | None:
        """局部更新草案：顶层 merge modify_fields；draft_id 不匹配返回 None。"""
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not isinstance(modify_fields, dict):
            raise ValueError("modify_fields 必须是 dict。")
        modify_fields = self._clean_draft(dict(modify_fields))
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT session_id, draft_id, npc_name, draft_json, bargain_count, created_at
                FROM session_task_drafts WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            current_draft_id = str(row["draft_id"] or "").strip()
            if draft_id is not None and str(draft_id).strip() and str(draft_id).strip() != current_draft_id:
                return None
            try:
                current_draft = json.loads(row["draft_json"]) if row["draft_json"] else {}
            except json.JSONDecodeError:
                current_draft = {}
            if not isinstance(current_draft, dict):
                current_draft = {}
            current_draft.update(modify_fields)
            current_draft = self._clean_draft(current_draft)
            new_draft_id = str(current_draft.get("draft_id") or current_draft_id).strip()
            new_npc_name = str(current_draft.get("npc_name") or row["npc_name"] or "").strip()
            bc = int(row["bargain_count"] or 0) if bargain_count is None else int(bargain_count)
            cur.execute(
                """
                UPDATE session_task_drafts
                SET draft_id = ?, npc_name = ?, draft_json = ?, bargain_count = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    new_draft_id,
                    new_npc_name,
                    json.dumps(current_draft, ensure_ascii=False),
                    bc,
                    now,
                    session_id,
                ),
            )
            self._conn.commit()
            return TaskDraftRow(
                session_id=session_id,
                draft_id=new_draft_id,
                npc_name=new_npc_name,
                draft=current_draft,
                bargain_count=bc,
                created_at=float(row["created_at"]),
                updated_at=now,
            )

    def delete_draft_sync(self, session_id: str, *, draft_id: str | None = None) -> bool:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            if draft_id is not None and str(draft_id).strip():
                cur.execute(
                    "DELETE FROM session_task_drafts WHERE session_id = ? AND draft_id = ?",
                    (session_id, str(draft_id).strip()),
                )
            else:
                cur.execute("DELETE FROM session_task_drafts WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # 草案过期计数（连续 N 次 ask 未触碰任务工具即删除）
    # ------------------------------------------------------------------

    # 任务相关工具名（触碰即视为「任务轮」，计数归零）
    TASK_RELATED_TOOL_NAMES = frozenset(
        {
            "prepare_task_context",
            "draft_agent_task",
            "update_task_draft",
            "confirm_agent_task",
            "cancel_agent_task",
        }
    )

    def get_rounds_without_task_sync(self, session_id: str) -> int:
        session_id = session_id.strip()
        if not session_id:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT rounds_without_task FROM session_ask_counters WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def increment_rounds_without_task_sync(self, session_id: str) -> int:
        session_id = session_id.strip()
        if not session_id:
            return 0
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
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
            self._conn.commit()
            cur.execute(
                "SELECT rounds_without_task FROM session_ask_counters WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
        return int(row[0]) if row else 1

    def reset_rounds_without_task_sync(self, session_id: str) -> None:
        session_id = session_id.strip()
        if not session_id:
            return
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_ask_counters (session_id, rounds_without_task, updated_at)
                VALUES (?, 0, ?)
                ON CONFLICT(session_id) DO UPDATE SET rounds_without_task = 0, updated_at = excluded.updated_at
                """,
                (session_id, now),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# asyncio 包装（API 层 await 调用）
# ---------------------------------------------------------------------------

def _make_async(attr: str):
    async def _call(self: MemoryStore, *args, **kwargs):
        return await asyncio.to_thread(getattr(self, attr), *args, **kwargs)
    _call.__name__ = attr.removesuffix("_sync")
    _call.__doc__ = f"async 版本：{attr}"
    return _call


for _name in [
    "add_message",
    "count_messages",
    "get_history",
    "get_summary",
    "save_summary",
    "create_session",
    "list_sessions",
    "update_session_title",
    "delete_session",
    "upsert_draft",
    "get_draft",
    "update_partial",
    "delete_draft",
    "get_rounds_without_task",
    "increment_rounds_without_task",
    "reset_rounds_without_task",
]:
    setattr(MemoryStore, _name, _make_async(f"{_name}_sync"))


_MEMORY_STORE: MemoryStore | None = None
_STORE_LOCK = threading.Lock()


def get_memory_store() -> MemoryStore:
    """全局单例（startup 初始化；未初始化时懒加载兜底）。"""
    global _MEMORY_STORE
    if _MEMORY_STORE is not None:
        return _MEMORY_STORE
    with _STORE_LOCK:
        if _MEMORY_STORE is None:
            _MEMORY_STORE = MemoryStore()
        return _MEMORY_STORE


def set_memory_store(store: MemoryStore | None) -> None:
    global _MEMORY_STORE
    with _STORE_LOCK:
        _MEMORY_STORE = store
