"""
Postgres 数据库后端（路线四 · Server Profile）。

基于 SQLAlchemy 2.0 async + asyncpg，实现
SessionStore / MessageStore / DraftStore 三个 Protocol。

表结构：
- chat_history（消息）
- sessions（会话）
- session_summaries（摘要）
- task_drafts（任务草案）
- ask_counters（草案轮次计数器）
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from services.storage.db import ChatMessage, SessionInfo, TaskDraft
from services.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLAlchemy ORM 模型
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class ChatHistoryRow(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(Float, nullable=False)


class SessionRow(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True)
    npc_name = Column(String(128), nullable=False)
    title = Column(String(256), nullable=False)
    created_at = Column(Float, nullable=False)


class SessionSummaryRow(Base):
    __tablename__ = "session_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True)
    summary = Column(Text, nullable=False)
    message_count = Column(Integer, nullable=False)
    updated_at = Column(Float, nullable=False)


class TaskDraftRow(Base):
    __tablename__ = "task_drafts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True)
    draft_id = Column(String(64), nullable=False)
    npc_name = Column(String(128), nullable=False, default="")
    draft_json = Column(Text, nullable=False)
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)


class AskCounterRow(Base):
    __tablename__ = "ask_counters"

    session_id = Column(String(64), primary_key=True)
    rounds_without_task = Column(Integer, nullable=False, default=0)
    updated_at = Column(Float, nullable=False)


# ---------------------------------------------------------------------------
# PostgresBackend
# ---------------------------------------------------------------------------


class PostgresBackend:
    """Postgres 持久化后端，同时实现 SessionStore / MessageStore / DraftStore。"""

    def __init__(self, url: str) -> None:
        self._url = url
        self._engine: Any = None
        self._sessionmaker: Any = None

    async def _ensure_engine(self) -> Any:
        if self._engine is None:
            self._engine = create_async_engine(self._url, echo=False)
            self._sessionmaker = async_sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )
        return self._engine

    async def _ensure_tables(self) -> None:
        engine = await self._ensure_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def _session(self) -> AsyncSession:
        if self._sessionmaker is None:
            raise RuntimeError("PostgresBackend 未初始化，请先调用 _ensure_engine()")
        return self._sessionmaker()

    async def init(self) -> None:
        await self._ensure_engine()
        await self._ensure_tables()

    # ---- SessionStore ----

    async def create_session(self, npc_name: str, title: str) -> SessionInfo:
        with LatencyTracker("pg.create_session"):
            await self._ensure_tables()
            sid = uuid.uuid4().hex
            now = time.time()
            async with self._session() as s:
                row = SessionRow(
                    session_id=sid,
                    npc_name=npc_name,
                    title=title,
                    created_at=now,
                )
                s.add(row)
                await s.commit()
            return SessionInfo(session_id=sid, npc_name=npc_name, title=title, created_at=now)

    async def list_sessions(self) -> list[SessionInfo]:
        await self._ensure_tables()
        async with self._session() as s:
            from sqlalchemy import select, desc
            stmt = select(SessionRow).order_by(desc(SessionRow.created_at))
            result = await s.execute(stmt)
            rows = result.scalars().all()
            return [SessionInfo(
                session_id=r.session_id,
                npc_name=r.npc_name,
                title=r.title,
                created_at=r.created_at,
            ) for r in rows]

    async def delete_session(self, session_id: str) -> None:
        await self._ensure_tables()
        async with self._session() as s:
            from sqlalchemy import delete as sa_delete
            await s.execute(sa_delete(ChatHistoryRow).where(ChatHistoryRow.session_id == session_id))
            await s.execute(sa_delete(SessionRow).where(SessionRow.session_id == session_id))
            await s.execute(sa_delete(SessionSummaryRow).where(SessionSummaryRow.session_id == session_id))
            await s.execute(sa_delete(TaskDraftRow).where(TaskDraftRow.session_id == session_id))
            await s.execute(sa_delete(AskCounterRow).where(AskCounterRow.session_id == session_id))
            await s.commit()

    async def update_title(self, session_id: str, title: str) -> SessionInfo:
        await self._ensure_tables()
        async with self._session() as s:
            from sqlalchemy import select
            stmt = select(SessionRow).where(SessionRow.session_id == session_id)
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                raise ValueError(f"会话不存在: {session_id}")
            row.title = title
            await s.commit()
            return SessionInfo(
                session_id=row.session_id,
                npc_name=row.npc_name,
                title=row.title,
                created_at=row.created_at,
            )

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
        with LatencyTracker("pg.add_message"):
            await self._ensure_tables()
            async with self._session() as s:
                row = ChatHistoryRow(
                    session_id=session_id,
                    role=role,
                    content=content,
                    timestamp=time.time(),
                )
                s.add(row)
                await s.commit()

            # 检查是否需要触发摘要（与 MemoryManager 一致）
            from sqlalchemy import select, func
            async with self._session() as s:
                count_stmt = select(func.count(ChatHistoryRow.id)).where(
                    ChatHistoryRow.session_id == session_id
                )
                count_result = await s.execute(count_stmt)
                count = count_result.scalar() or 0

            if count > 0 and count % summarize_interval == 0:
                import asyncio
                asyncio.create_task(self._safe_summarize(
                    session_id, npc_name, llm_config, summarize_interval,
                ))

    async def _safe_summarize(
        self,
        session_id: str,
        npc_name: str,
        llm_config: dict | None,
        interval: int,
    ) -> None:
        # Postgres 版的摘要逻辑（与 MemoryManager._safe_summarize 对应）
        # 当前保持简洁：仅记录日志。完整实现需接入 LLM 调用
        logger.debug("Postgres summary trigger: session=%s, npc=%s", session_id, npc_name)

    async def get_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
    ) -> list[ChatMessage]:
        with LatencyTracker("pg.get_history"):
            await self._ensure_tables()
            from sqlalchemy import select, desc as sa_desc, asc as sa_asc
            order_fn = sa_desc if order == "desc" else sa_asc
            stmt = (
                select(ChatHistoryRow)
                .where(ChatHistoryRow.session_id == session_id)
                .order_by(order_fn(ChatHistoryRow.timestamp))
                .offset(offset)
                .limit(limit)
            )
            async with self._session() as s:
                result = await s.execute(stmt)
                rows = result.scalars().all()
                # 倒序时再反转（与 MemoryManager 行为一致）
                if order == "desc":
                    rows = list(reversed(rows))
                return [ChatMessage(
                    id=r.id, role=r.role, content=r.content, timestamp=r.timestamp,
                ) for r in rows]

    async def get_summary(self, session_id: str) -> str | None:
        await self._ensure_tables()
        from sqlalchemy import select
        stmt = select(SessionSummaryRow).where(SessionSummaryRow.session_id == session_id)
        async with self._session() as s:
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            return row.summary if row else None

    async def save_summary(self, session_id: str, summary: str, message_count: int) -> None:
        await self._ensure_tables()
        from sqlalchemy import select
        async with self._session() as s:
            stmt = select(SessionSummaryRow).where(SessionSummaryRow.session_id == session_id)
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is not None:
                row.summary = summary
                row.message_count = message_count
                row.updated_at = time.time()
            else:
                s.add(SessionSummaryRow(
                    session_id=session_id,
                    summary=summary,
                    message_count=message_count,
                    updated_at=time.time(),
                ))
            await s.commit()

    # ---- DraftStore ----

    async def save_draft(
        self,
        session_id: str,
        draft_id: str,
        npc_name: str,
        draft_json: str,
    ) -> None:
        with LatencyTracker("pg.save_draft"):
            await self._ensure_tables()
            now = time.time()
            from sqlalchemy import select
            async with self._session() as s:
                stmt = select(TaskDraftRow).where(TaskDraftRow.session_id == session_id)
                result = await s.execute(stmt)
                row = result.scalar_one_or_none()
                if row is not None:
                    row.draft_id = draft_id
                    row.npc_name = npc_name
                    row.draft_json = draft_json
                    row.updated_at = now
                else:
                    s.add(TaskDraftRow(
                        session_id=session_id,
                        draft_id=draft_id,
                        npc_name=npc_name,
                        draft_json=draft_json,
                        created_at=now,
                        updated_at=now,
                    ))
                await s.commit()

    async def get_draft(self, session_id: str) -> TaskDraft | None:
        await self._ensure_tables()
        from sqlalchemy import select
        stmt = select(TaskDraftRow).where(TaskDraftRow.session_id == session_id)
        async with self._session() as s:
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return TaskDraft(
                session_id=row.session_id,
                draft_id=row.draft_id,
                npc_name=row.npc_name,
                draft_json=row.draft_json,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    async def delete_draft(self, session_id: str) -> None:
        await self._ensure_tables()
        from sqlalchemy import delete as sa_delete
        async with self._session() as s:
            await s.execute(sa_delete(TaskDraftRow).where(TaskDraftRow.session_id == session_id))
            await s.commit()

    async def increment_ask_counter(self, session_id: str) -> int:
        await self._ensure_tables()
        now = time.time()
        from sqlalchemy import select
        async with self._session() as s:
            stmt = select(AskCounterRow).where(AskCounterRow.session_id == session_id)
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is not None:
                row.rounds_without_task += 1
                row.updated_at = now
                val = row.rounds_without_task
            else:
                s.add(AskCounterRow(
                    session_id=session_id,
                    rounds_without_task=1,
                    updated_at=now,
                ))
                val = 1
            await s.commit()
            return val

    async def reset_ask_counter(self, session_id: str) -> None:
        await self._ensure_tables()
        from sqlalchemy import select
        async with self._session() as s:
            stmt = select(AskCounterRow).where(AskCounterRow.session_id == session_id)
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is not None:
                row.rounds_without_task = 0
                row.updated_at = time.time()
                await s.commit()

    async def check_auto_expiry(self, session_id: str, max_rounds: int = 3) -> bool:
        await self._ensure_tables()
        from sqlalchemy import select
        stmt = select(AskCounterRow).where(AskCounterRow.session_id == session_id)
        async with self._session() as s:
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return False
            if row.rounds_without_task >= max_rounds:
                from sqlalchemy import delete as sa_delete
                await s.execute(sa_delete(TaskDraftRow).where(TaskDraftRow.session_id == session_id))
                await s.execute(sa_delete(AskCounterRow).where(AskCounterRow.session_id == session_id))
                await s.commit()
                return True
            return False

    # ---- 兼容 MemoryManager API 的方法别名 ----

    async def update_session_title(self, session_id: str, title: str) -> dict:
        """MemoryManager 兼容方法（等同于 update_title）。"""
        result = await self.update_title(session_id, title)
        return {
            "session_id": result.session_id,
            "title": result.title,
        }

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
