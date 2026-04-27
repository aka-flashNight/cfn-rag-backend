"""
数据库后端抽象：会话、消息、任务草案的持久化接口。

- SqliteBackend：封装现有 MemoryManager + SessionTaskDraftStore（local profile）
- PostgresBackend：SQLAlchemy async + asyncpg（server profile，见 postgres_db.py）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from core.config import Settings


# ---------------------------------------------------------------------------
# 共享数据结构
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    session_id: str
    npc_name: str
    title: str
    created_at: float


@dataclass
class ChatMessage:
    id: int
    role: str
    content: str
    timestamp: float


@dataclass
class TaskDraft:
    session_id: str
    draft_id: str
    npc_name: str
    draft_json: str
    created_at: float
    updated_at: float


# ---------------------------------------------------------------------------
# 存储接口 Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStore(Protocol):
    """会话元数据持久化接口。"""

    async def create_session(self, npc_name: str, title: str) -> SessionInfo:
        ...

    async def list_sessions(self) -> list[SessionInfo]:
        ...

    async def delete_session(self, session_id: str) -> None:
        ...

    async def update_title(self, session_id: str, title: str) -> SessionInfo:
        ...


@runtime_checkable
class MessageStore(Protocol):
    """聊天消息持久化接口。"""

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        llm_config: dict | None = None,
        npc_name: str = "",
        summarize_interval: int = 30,
    ) -> None:
        ...

    async def get_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
    ) -> list[ChatMessage]:
        ...

    async def get_summary(self, session_id: str) -> str | None:
        ...

    async def save_summary(
        self,
        session_id: str,
        summary: str,
        message_count: int,
    ) -> None:
        ...


@runtime_checkable
class DraftStore(Protocol):
    """任务草案持久化接口。"""

    async def save_draft(
        self,
        session_id: str,
        draft_id: str,
        npc_name: str,
        draft_json: str,
    ) -> None:
        ...

    async def get_draft(self, session_id: str) -> TaskDraft | None:
        ...

    async def delete_draft(self, session_id: str) -> None:
        ...

    async def increment_ask_counter(self, session_id: str) -> int:
        ...

    async def reset_ask_counter(self, session_id: str) -> None:
        ...

    async def check_auto_expiry(self, session_id: str, max_rounds: int = 3) -> bool:
        """检查是否超过 N 轮未涉及任务，返回 True 表示已触发自动清除。"""
        ...


# ---------------------------------------------------------------------------
# SqliteBackend：基于现有 MemoryManager + SessionTaskDraftStore 的适配器
# ---------------------------------------------------------------------------


class SqliteBackend:
    """SQLite 持久化后端（local profile 默认）。

    直接封装项目现有的 ``MemoryManager`` 和 ``SessionTaskDraftStore``，
    对外暴露统一的 SessionStore / MessageStore / DraftStore 接口。

    重要：此适配器不修改现有模块的内部行为，仅做委托调用。
    """

    def __init__(self) -> None:
        self._memory: Any = None
        self._draft_store: Any = None

    async def _get_memory(self) -> Any:
        if self._memory is None:
            from services.memory_manager import MemoryManager

            self._memory = await MemoryManager.create()
        return self._memory

    async def _get_draft_store(self) -> Any:
        if self._draft_store is None:
            from services.task_draft_store import SessionTaskDraftStore

            self._draft_store = SessionTaskDraftStore()
        return self._draft_store

    # ---- SessionStore ----

    async def create_session(self, npc_name: str, title: str) -> SessionInfo:
        m = await self._get_memory()
        result = await m.create_session(npc_name, title)
        return SessionInfo(**result)

    async def list_sessions(self) -> list[SessionInfo]:
        m = await self._get_memory()
        results = await m.list_sessions()
        return [SessionInfo(**r) for r in results]

    async def delete_session(self, session_id: str) -> None:
        m = await self._get_memory()
        await m.delete_session(session_id)

    async def update_title(self, session_id: str, title: str) -> SessionInfo:
        m = await self._get_memory()
        result = await m.update_session_title(session_id, title)
        return SessionInfo(**result)

    # ---- MessageStore ----

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        llm_config: dict | None = None,
        npc_name: str = "",
        summarize_interval: int = 30,
    ) -> None:
        m = await self._get_memory()
        await m.add_message(
            session_id, role, content,
            llm_config=llm_config,
            npc_name=npc_name,
            summarize_interval=summarize_interval,
        )

    async def get_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
    ) -> list[ChatMessage]:
        m = await self._get_memory()
        results = await m.get_history(session_id, limit=limit, offset=offset, order=order)
        return [ChatMessage(**r) for r in results]

    async def get_summary(self, session_id: str) -> str | None:
        m = await self._get_memory()
        return await m.get_summary(session_id)

    async def save_summary(
        self,
        session_id: str,
        summary: str,
        message_count: int,
    ) -> None:
        m = await self._get_memory()
        # MemoryManager 的保存摘要方法是私有的 _save_summary
        await m._save_summary(session_id, summary, message_count)

    # ---- DraftStore ----

    async def save_draft(
        self,
        session_id: str,
        draft_id: str,
        npc_name: str,
        draft_json: str,
    ) -> None:
        ds = await self._get_draft_store()
        import json

        draft_dict = json.loads(draft_json)
        draft_dict.setdefault("draft_id", draft_id)
        draft_dict.setdefault("npc_name", npc_name)
        await ds.upsert_draft(session_id=session_id, draft=draft_dict)

    async def get_draft(self, session_id: str) -> TaskDraft | None:
        ds = await self._get_draft_store()

        raw = await ds.get_draft_json_by_session_id(session_id)
        if raw is None:
            return None
        draft_id = raw.get("draft_id", "")
        draft_json = raw.get("_raw_json", "")
        if not draft_json:
            import json
            draft_json = json.dumps(raw, ensure_ascii=False)
        return TaskDraft(
            session_id=session_id,
            draft_id=draft_id,
            npc_name=raw.get("npc_name", ""),
            draft_json=draft_json,
            created_at=raw.get("created_at", 0.0),
            updated_at=raw.get("updated_at", 0.0),
        )

    async def delete_draft(self, session_id: str) -> None:
        ds = await self._get_draft_store()
        await ds.delete_by_session_id(session_id)

    async def increment_ask_counter(self, session_id: str) -> int:
        ds = await self._get_draft_store()
        return await ds.increment_rounds_without_task(session_id)

    async def reset_ask_counter(self, session_id: str) -> None:
        ds = await self._get_draft_store()
        await ds.reset_rounds_without_task(session_id)

    async def check_auto_expiry(self, session_id: str, max_rounds: int = 3) -> bool:
        ds = await self._get_draft_store()
        rounds = await ds.get_rounds_without_task(session_id)
        if rounds >= max_rounds:
            await ds.delete_by_session_id(session_id)
            await ds.reset_rounds_without_task(session_id)
            return True
        return False


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_session_store(settings: Settings) -> SessionStore:
    backend = settings.effective("db_backend")
    if backend == "postgres":
        from services.storage.postgres_db import PostgresBackend

        return PostgresBackend(url=settings.postgres_url)
    return SqliteBackend()


def create_message_store(settings: Settings) -> MessageStore:
    backend = settings.effective("db_backend")
    if backend == "postgres":
        from services.storage.postgres_db import PostgresBackend

        return PostgresBackend(url=settings.postgres_url)
    return SqliteBackend()


def create_draft_store(settings: Settings) -> DraftStore:
    backend = settings.effective("db_backend")
    if backend == "postgres":
        from services.storage.postgres_db import PostgresBackend

        return PostgresBackend(url=settings.postgres_url)
    return SqliteBackend()
