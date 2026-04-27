#!/usr/bin/env python
"""
SQLite → PostgreSQL 数据迁移脚本。

用法::

    python scripts/migrate_sqlite_to_pg.py \
        --sqlite-path resources/tools/memory.db \
        --pg-url postgresql+asyncpg://postgres:postgres@localhost:5432/cfn_rag
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="SQLite → PostgreSQL 数据迁移")
    p.add_argument("--sqlite-path", default="resources/tools/memory.db")
    p.add_argument("--pg-url", default="postgresql+asyncpg://postgres:postgres@localhost:5432/cfn_rag")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read_sqlite(db_path: str) -> dict[str, list[dict]]:
    """读取 SQLite 中所有数据。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    tables = {
        "chat_history":     "SELECT * FROM chat_history",
        "sessions":         "SELECT * FROM sessions",
        "session_summaries":"SELECT * FROM session_summaries",
        "session_task_drafts": "SELECT * FROM session_task_drafts",
        "session_ask_counters": "SELECT * FROM session_ask_counters",
    }

    data = {}
    for name, sql in tables.items():
        try:
            cur.execute(sql)
            data[name] = [dict(r) for r in cur.fetchall()]
            print(f"[SQLite] {name}: {len(data[name])} 行")
        except sqlite3.OperationalError:
            print(f"[SQLite] {name}: 表不存在，跳过")
            data[name] = []
    conn.close()
    return data


async def write_postgres(url: str, data: dict, dry_run: bool):
    if dry_run:
        print("[迁移] --dry-run，不上传数据")
        return

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    engine = create_async_engine(url, echo=False)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from services.storage.postgres_db import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # sessions
    async with maker() as s:
        from services.storage.postgres_db import SessionRow
        for row in data.get("sessions", []):
            s.add(SessionRow(
                session_id=row["session_id"],
                npc_name=row.get("npc_name", ""),
                title=row.get("title", ""),
                created_at=row.get("created_at", time.time()),
            ))
        await s.commit()
    print(f"[迁移] sessions: {len(data.get('sessions', []))} 行已写入")

    # chat_history（可能很多行，分批写入）
    messages = data.get("chat_history", [])
    batch_size = 500
    from services.storage.postgres_db import ChatHistoryRow
    for i in range(0, len(messages), batch_size):
        async with maker() as s:
            for row in messages[i : i + batch_size]:
                s.add(ChatHistoryRow(
                    session_id=row["session_id"],
                    role=row.get("role", ""),
                    content=row.get("content", ""),
                    timestamp=row.get("timestamp", time.time()),
                ))
            await s.commit()
        print(f"\r[迁移] chat_history: {min(i + batch_size, len(messages))}/{len(messages)}", end="", flush=True)
    print(f"\n[迁移] chat_history: {len(messages)} 行已写入")

    # session_summaries
    async with maker() as s:
        from services.storage.postgres_db import SessionSummaryRow
        for row in data.get("session_summaries", []):
            s.add(SessionSummaryRow(
                session_id=row["session_id"],
                summary=row.get("summary", ""),
                message_count=row.get("message_count", 0),
                updated_at=row.get("updated_at", time.time()),
            ))
        await s.commit()
    print(f"[迁移] session_summaries: {len(data.get('session_summaries', []))} 行已写入")

    # task_drafts（从 session_task_drafts 映射）
    drafts = data.get("session_task_drafts", [])
    async with maker() as s:
        from services.storage.postgres_db import TaskDraftRow
        for row in drafts:
            s.add(TaskDraftRow(
                session_id=row.get("session_id", ""),
                draft_id=row.get("draft_id", ""),
                npc_name=row.get("npc_name", ""),
                draft_json=row.get("draft_json", "{}"),
                created_at=row.get("created_at", time.time()),
                updated_at=row.get("updated_at", time.time()),
            ))
        await s.commit()
    print(f"[迁移] task_drafts: {len(drafts)} 行已写入")

    # ask_counters
    counters = data.get("session_ask_counters", [])
    async with maker() as s:
        from services.storage.postgres_db import AskCounterRow
        for row in counters:
            s.add(AskCounterRow(
                session_id=row.get("session_id", ""),
                rounds_without_task=row.get("rounds_without_task", 0),
                updated_at=row.get("updated_at", time.time()),
            ))
        await s.commit()
    print(f"[迁移] ask_counters: {len(counters)} 行已写入")

    await engine.dispose()
    print("[迁移] 完成 ✓")


async def main():
    args = parse_args()
    print(f"[迁移] SQLite: {args.sqlite_path}")
    print(f"[迁移] PG:     {args.pg_url}")

    data = read_sqlite(args.sqlite_path)
    total_rows = sum(len(v) for v in data.values())
    print(f"[迁移] 总行数: {total_rows}")
    await write_postgres(args.pg_url, data, args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
