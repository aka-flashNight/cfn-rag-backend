import os

# 在导入 llama_index 之前设置，避免 tiktoken 编码问题
os.environ["LLAMA_INDEX_CACHE_DIR"] = ".llamaindex_cache"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import api_router
from core.config import Settings, get_settings
from core.exceptions import register_exception_handlers


def create_app() -> FastAPI:
    """
    FastAPI 应用工厂，负责注册中间件、路由和全局异常处理。
    """

    settings: Settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
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
