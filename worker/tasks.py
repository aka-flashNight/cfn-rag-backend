"""Worker 任务定义。

每个函数都是一个 arq 任务，通过 ``arq_pool.enqueue_job()`` 投递。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("worker.tasks")


# ---------------------------------------------------------------------------
# 任务 ID 常量（供 Web 进程 enqueue 使用）
# ---------------------------------------------------------------------------

REBUILD_INDEX = "rebuild_vector_index"
RUN_EVAL_RETRIEVER = "run_eval_retriever"
RUN_EVAL_RAG = "run_eval_rag"
HEALTH_CHECK = "health_check"


# ---------------------------------------------------------------------------
# 任务实现
# ---------------------------------------------------------------------------


async def rebuild_vector_index(ctx: dict) -> dict:
    """重建知识库向量索引。

    流程：
    1. 确保嵌入模型已加载
    2. 调用 ``rebuild_vector_index()`` 重建索引
    3. 若配置了 Qdrant，同步迁移向量数据

    Returns:
        {"status": "ok", "nodes": 1234, "elapsed": 45.6}
    """
    t0 = time.monotonic()
    logger.info("开始重建向量索引...")

    from core.config import get_settings

    settings = get_settings()

    # 离线加载嵌入模型
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from ai_engine.game_data_loader import ensure_embed_model, rebuild_vector_index as rebuild

    ensure_embed_model(offline=True)
    index = await asyncio.to_thread(rebuild)

    # 统计节点数
    from ai_engine.game_data_loader import iter_docstore_nodes

    nodes = list(iter_docstore_nodes(index))
    node_count = len(nodes)
    elapsed = time.monotonic() - t0

    logger.info("向量索引重建完成 — %s 节点，耗时 %.1fs", node_count, elapsed)
    return {"status": "ok", "nodes": node_count, "elapsed": round(elapsed, 1)}


async def run_eval_retriever(ctx: dict, *, sample: int = 20, mode: str = "dense") -> dict:
    """触发检索评估。

    Args:
        sample: 评估样本数
        mode: 检索模式（dense / bm25 / hybrid）
    """
    t0 = time.monotonic()
    logger.info("开始检索评估 — sample=%s mode=%s", sample, mode)

    try:
        from evals.runners.run_all import run_retriever_eval

        result = await asyncio.to_thread(run_retriever_eval, sample=sample, mode=mode)
        elapsed = time.monotonic() - t0
        return {"status": "ok", "result": result, "elapsed": round(elapsed, 1)}
    except ImportError:
        logger.warning("evals 模块不可用，跳过检索评估")
        return {"status": "skipped", "reason": "evals 模块不可用"}


async def run_eval_rag(ctx: dict, *, sample: int = 20) -> dict:
    """触发 RAG 端到端评估（Ragas）。

    Args:
        sample: 评估样本数
    """
    t0 = time.monotonic()
    logger.info("开始 RAG 评估 — sample=%s", sample)

    try:
        from evals.runners.run_all import run_rag_eval

        result = await asyncio.to_thread(run_rag_eval, sample=sample)
        elapsed = time.monotonic() - t0
        return {"status": "ok", "result": result, "elapsed": round(elapsed, 1)}
    except ImportError:
        logger.warning("evals 模块不可用，跳过 RAG 评估")
        return {"status": "skipped", "reason": "evals 模块不可用"}


async def health_check(ctx: dict) -> dict:
    """Worker 健康检查。"""
    from core.config import get_settings

    settings = get_settings()
    return {
        "status": "ok",
        "profile": settings.deployment_profile,
        "db": settings.effective("db_backend"),
        "cache": settings.effective("cache_backend"),
        "vector": settings.effective("vector_backend"),
    }
