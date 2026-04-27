import logging
import os
import sys
from contextlib import asynccontextmanager

# 在导入 llama_index 之前设置，避免 tiktoken 编码问题
os.environ["LLAMA_INDEX_CACHE_DIR"] = ".llamaindex_cache"

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api import api_router
from core.config import Settings, get_settings
from core.exceptions import register_exception_handlers
from core.startup import run_startup_tasks

# ---------------------------------------------------------------------------
# 结构化日志（路线四 · 观测性）
# ---------------------------------------------------------------------------


def _setup_logging(settings: Settings) -> None:
    """根据 profile 配置日志格式。

    - local profile: 标准控制台格式（人类可读）
    - server profile: JSON 格式（适合 ELK / Loki 采集）
    """
    if settings.is_server_profile:
        try:
            import structlog

            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
            )
            logging.getLogger("uvicorn.access").handlers = []
            logging.getLogger("uvicorn.access").addHandler(
                logging.StreamHandler(sys.stderr)
            )
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Prometheus 指标（路线四 · 观测性）
# ---------------------------------------------------------------------------

def _setup_metrics(app: FastAPI, settings: Settings) -> None:
    """为 server profile 启用 Prometheus /metrics 端点。"""
    if not settings.is_server_profile:
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
        )
        instrumentator.add(
            # 自定义指标：活跃 SSE 连接数
            instrumentator.metrics.gauge(
                name="cfn_active_sse_connections",
                documentation="当前活跃的 SSE 长连接数",
            )
        )
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
        print("[观测性] Prometheus /metrics 端点已启用")
    except ImportError:
        print("[观测性] prometheus-fastapi-instrumentator 未安装，跳过 /metrics")


# ---------------------------------------------------------------------------
# 应用生命周期
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理。

    在应用启动时执行初始化任务，在应用关闭时清理资源。
    """
    # 启动时的初始化任务
    await run_startup_tasks()

    yield

    # 关闭时的清理任务
    settings = get_settings()
    if settings.is_server_profile:
        print("[关闭] 后端服务正在关闭...")
    else:
        print("[关闭] 后端服务正在关闭...")


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    FastAPI 应用工厂，负责注册中间件、路由和全局异常处理。
    """

    settings: Settings = get_settings()

    _setup_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(api_router, prefix="/api")

    _setup_metrics(app, settings)

    return app


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

app: FastAPI = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7077, reload=True)
