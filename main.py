import os
from contextlib import asynccontextmanager

# 在导入 llama_index 之前设置，避免 tiktoken 编码问题
os.environ["LLAMA_INDEX_CACHE_DIR"] = ".llamaindex_cache"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import api_router
from core.config import Settings, get_settings
from core.exceptions import register_exception_handlers
from core.startup import run_startup_tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理。

    在应用启动时执行初始化任务，在应用关闭时清理资源。
    """
    # 启动时的初始化任务
    await run_startup_tasks()

    yield

    # 关闭时的清理任务（如有需要）
    print("[关闭] 后端服务正在关闭...")


def create_app() -> FastAPI:
    """
    FastAPI 应用工厂，负责注册中间件、路由和全局异常处理。
    """

    settings: Settings = get_settings()

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

    return app


app: FastAPI = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7077, reload=True)
