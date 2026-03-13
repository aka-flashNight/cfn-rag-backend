from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List


def is_packaged_environment() -> bool:
    """检查是否在PyInstaller打包环境中运行"""
    return hasattr(sys, '_MEIPASS') or getattr(sys, 'frozen', False)


def get_db_path() -> Path:
    """
    获取memory.db的路径。

    数据库现在存储在resources/tools/memory.db，这样可以避免打包后数据库被删除的问题。

    目录结构：
        开发环境：
            父目录/
            ├── resources/tools/memory.db  <-- 数据库存储在这里
            └── cfn-rag-backend/           <-- 本项目
                └── ...

        打包后：
            部署目录/
            ├── resources/tools/memory.db  <-- 数据库存储在这里
            └── CFN-RAG.exe                <-- 打包后的exe
    """
    # 获取基础目录
    if is_packaged_environment():
        # 打包环境：exe所在目录
        base_dir = Path(os.path.dirname(sys.executable))
    else:
        # 开发环境：当前文件所在目录的父目录的父目录
        # memory_manager.py 位置: cfn-rag-backend/services/memory_manager.py
        # resources 位置: cfn-rag-backend/../resources
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent.parent

    # 情况1: base_dir/resources/tools/memory.db（打包后或resources和项目同级）
    db_path = base_dir / "resources" / "tools" / "memory.db"
    if db_path.parent.exists():
        return db_path

    # 情况2: 开发环境特殊情况，resources在当前工作目录
    cwd = Path(os.getcwd()).resolve()
    cwd_db_path = cwd / "resources" / "tools" / "memory.db"
    if cwd_db_path.parent.exists():
        return cwd_db_path

    # 情况3: 尝试查找工作目录的父目录
    parent_cwd_db_path = cwd.parent / "resources" / "tools" / "memory.db"
    if parent_cwd_db_path.parent.exists():
        return parent_cwd_db_path

    # 如果都找不到，使用默认路径（base_dir），并创建目录
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


DB_PATH: Path = get_db_path()


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
            # 确保数据库目录存在
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
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

    async def update_session_title(self, session_id: str, title: str) -> Dict[str, Any]:
        """
        更新指定会话的标题。

        Args:
            session_id: 会话 ID
            title: 新的标题

        Returns:
            更新后的会话信息

        Raises:
            ValueError: 当会话不存在或标题为空时
        """
        session_id = session_id.strip()
        title = title.strip()

        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not title:
            raise ValueError("title 不能为空。")

        def _inner() -> Dict[str, Any]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                # 检查会话是否存在
                cur.execute(
                    "SELECT session_id, npc_name, title, created_at FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"会话 '{session_id}' 不存在。")

                # 更新标题
                cur.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (title, session_id),
                )
                conn.commit()

                return {
                    "session_id": str(row["session_id"]),
                    "npc_name": str(row["npc_name"]),
                    "title": title,
                    "created_at": float(row["created_at"]),
                }
            finally:
                conn.close()

        return await asyncio.to_thread(_inner)

    async def delete_session(self, session_id: str) -> None:
        """
        删除指定的会话及其所有聊天记录。

        Args:
            session_id: 会话 ID

        Raises:
            ValueError: 当会话不存在时
        """
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")

        def _inner() -> bool:
            conn = sqlite3.connect(self._db_path)
            try:
                cur = conn.cursor()
                # 检查会话是否存在
                cur.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                if cur.fetchone() is None:
                    raise ValueError(f"会话 '{session_id}' 不存在。")

                # 删除会话元数据
                cur.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                # 删除相关聊天记录
                cur.execute(
                    "DELETE FROM chat_history WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_inner)

