"""后台 Worker 服务（路线四）。

基于 arq（Redis-based 异步任务队列）实现的后台任务处理器。

任务列表：
- ``rebuild_vector_index`` — 重建知识库向量索引
- ``run_eval_retriever`` — 触发检索评估
- ``run_eval_rag`` — 触发 RAG 端到端评估
- ``health_check`` — Worker 健康检查

注意：arq/redis 为可选依赖（server profile），仅在 ``main()`` 调用时才导入。
       这样 local profile 下不需要安装这些依赖也能正常运行。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("worker")


async def _startup(ctx: dict) -> None:
    """Worker 启动回调。"""
    from core.config import get_settings

    settings = get_settings()
    logger.info(
        "Worker 启动 — profile=%s db=%s cache=%s vector=%s",
        settings.deployment_profile,
        settings.effective("db_backend"),
        settings.effective("cache_backend"),
        settings.effective("vector_backend"),
    )


async def _shutdown(ctx: dict) -> None:
    """Worker 关闭回调。"""
    logger.info("Worker 正在关闭...")


def main():
    """启动 arq Worker（命令行入口）。

    仅在调用时才导入 arq 相关依赖，避免 local profile 下因缺少
    arq/redis 而导致导入错误。
    """
    import logging.config

    from worker.settings import create_worker_settings
    from worker.tasks import HEALTH_CHECK, REBUILD_INDEX, RUN_EVAL_RAG, RUN_EVAL_RETRIEVER

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    class WorkerConfig:
        functions = [
            REBUILD_INDEX,
            RUN_EVAL_RETRIEVER,
            RUN_EVAL_RAG,
            HEALTH_CHECK,
        ]
        on_startup = _startup
        on_shutdown = _shutdown
        redis_settings = create_worker_settings()
        max_jobs = 4
        job_timeout = 3600
        keep_result = 3600
        poll_delay = 0.5

    from arq.worker import run_worker

    run_worker(WorkerConfig)


if __name__ == "__main__":
    main()
