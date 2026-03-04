from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
DB_PATH: Path = DATA_DIR / "memory.db"


class MemoryManager:
    """
    本地单机游戏 NPC 对话记忆管理器，基于 sqlite3。
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @classmethod
    async def create(cls) -> "MemoryManager":
        """
        创建并初始化 MemoryManager，确保数据库与表存在。
        """

        manager = cls(DB_PATH)
        await manager._init_db()
        return manager

    async def _init_db(self) -> None:
        """
        初始化 sqlite 数据库与表结构。
        """

        def _inner() -> None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
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
                    "CREATE INDEX IF NOT EXISTS idx_chat_session_time "
                    "ON chat_history(session_id, timestamp)"
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
                    "CREATE INDEX IF NOT EXISTS idx_sessions_session_id "
                    "ON sessions(session_id)"
                )

                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_inner)

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        新增一条对话消息。
        """

        session_id = session_id.strip()
        role = role.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if role not in {"user", "assistant"}:
            raise ValueError("role 必须是 'user' 或 'assistant'。")

        ts: float = time.time()

        def _inner() -> None:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO chat_history (session_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, role, content, ts),
                )
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_inner)

    async def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取指定会话最近的对话记录，按时间正序返回。
        """

        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if limit <= 0:
            return []

        def _inner() -> List[Dict[str, Any]]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT id, role, content, timestamp
                    FROM chat_history
                    WHERE session_id = ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()
            finally:
                conn.close()

            # 目前 rows 是倒序（最新在前），需要反转成正序
            result: List[Dict[str, Any]] = []
            for row in reversed(rows):
                result.append(
                    {
                        "id": int(row["id"]),
                        "role": str(row["role"]),
                        "content": str(row["content"]),
                        "timestamp": float(row["timestamp"]),
                    }
                )
            return result

        return await asyncio.to_thread(_inner)

    async def create_session(self, npc_name: str, title: str) -> Dict[str, Any]:
        """
        创建一个新的会话，并返回其元信息。
        """

        npc_name = npc_name.strip()
        title = title.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")
        if not title:
            raise ValueError("title 不能为空。")

        session_id: str = str(uuid.uuid4())
        created_at: float = time.time()

        def _inner() -> None:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO sessions (session_id, npc_name, title, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, npc_name, title, created_at),
                )
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_inner)

        return {
            "session_id": session_id,
            "npc_name": npc_name,
            "title": title,
            "created_at": created_at,
        }

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有已存在的会话。
        """

        def _inner() -> List[Dict[str, Any]]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT session_id, npc_name, title, created_at
                    FROM sessions
                    ORDER BY created_at DESC, session_id ASC
                    """
                )
                rows = cur.fetchall()
            finally:
                conn.close()

            result: List[Dict[str, Any]] = []
            for row in rows:
                result.append(
                    {
                        "session_id": str(row["session_id"]),
                        "npc_name": str(row["npc_name"]),
                        "title": str(row["title"]),
                        "created_at": float(row["created_at"]),
                    }
                )
            return result

        return await asyncio.to_thread(_inner)

