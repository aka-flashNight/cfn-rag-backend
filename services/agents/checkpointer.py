"""
LangGraph Checkpointer 工厂（路线四 Profile 切换）。

路线三已在 ``services/agents/graph.py`` 完成完整的 AsyncSqliteSaver 接入与
HITL v2 挂起点。本模块将其抽取为 profile-aware 工厂函数：

- Local profile: ``AsyncSqliteSaver``（默认，与路线三行为一致）
- Server / Postgres: ``AsyncPostgresSaver``（需 langgraph-checkpoint-postgres）
- Server / Redis: ``AsyncRedisSaver``（需 langgraph-checkpoint-redis）

调用方（graph.py）无需关心底层实现，只需调用 ``build_checkpointer()``。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径解析（与原来 graph.py 中的逻辑一致）
# ---------------------------------------------------------------------------


def _default_checkpoint_path() -> Path:
    from services.memory_manager import DB_PATH

    return DB_PATH.parent / "langgraph_checkpoint.sqlite"


def resolve_checkpoint_db_path() -> Path:
    """决定本地 checkpoint 的 sqlite 路径。

    - 优先读 env ``CFN_LANGGRAPH_CHECKPOINT_DB``
    - 否则放在 memory.db 的同级目录 ``langgraph_checkpoint.sqlite``
    """
    settings = get_settings()
    if settings.checkpoint_db_path:
        p = Path(settings.checkpoint_db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return _default_checkpoint_path()


# ---------------------------------------------------------------------------
# Checkpointer 构建
# ---------------------------------------------------------------------------


async def build_checkpointer() -> Any:
    """根据当前 profile 创建 LangGraph checkpointer。

    Returns:
        一个 async context manager 的 __aenter__ 结果（即 saver 实例），
        可直接传给 ``graph.compile(checkpointer=...)``。
    """
    settings = get_settings()
    backend = settings.effective("checkpoint_backend")

    if backend == "postgres":
        return await _build_postgres_saver(settings)
    elif backend == "redis":
        return await _build_redis_saver(settings)
    else:
        return await _build_sqlite_saver()


async def _build_sqlite_saver() -> Any:
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少 langgraph-checkpoint-sqlite 依赖。请在当前 venv 里运行：\n"
            "    pip install langgraph-checkpoint-sqlite\n"
            "若 requirements.txt 已更新过，重跑 `pip install -r requirements.txt` 也可。"
        ) from e

    db_path = resolve_checkpoint_db_path()
    logger.info("LangGraph Checkpointer (sqlite): %s", db_path)
    cm = AsyncSqliteSaver.from_conn_string(str(db_path))
    return await cm.__aenter__()


async def _build_postgres_saver(settings: Any) -> Any:
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少 langgraph-checkpoint-postgres 依赖。请在当前 venv 里运行：\n"
            "    pip install langgraph-checkpoint-postgres\n"
        ) from e

    logger.info("LangGraph Checkpointer (postgres): %s", settings.postgres_url)
    cm = AsyncPostgresSaver.from_conn_string(settings.postgres_url)
    saver = await cm.__aenter__()
    await saver.setup()
    return saver


async def _build_redis_saver(settings: Any) -> Any:
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少 langgraph-checkpoint-redis 依赖。请在当前 venv 里运行：\n"
            "    pip install langgraph-checkpoint-redis\n"
        ) from e

    logger.info("LangGraph Checkpointer (redis): %s", settings.redis_url)
    cm = AsyncRedisSaver.from_conn_string(settings.redis_url)
    return await cm.__aenter__()
