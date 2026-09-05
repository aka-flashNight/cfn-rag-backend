from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import api_router
from core.config import get_settings
from core.exceptions import register_exception_handlers
from core.startup import run_startup_tasks, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时并行初始化（检索/NPC/存储/摘要 worker），关闭时落盘收尾。"""
    await run_startup_tasks()

    yield

    try:
        from services.npc.manager import get_npc_manager

        manager = await get_npc_manager()
        await manager.flush()  # 进程退出前落盘好感度（06 §1.1）
    except Exception:
        pass
    print("[关闭] 后端服务正在关闭...")


def create_app() -> FastAPI:
    """FastAPI 应用工厂：中间件、路由、全局异常处理。"""
    setup_logging()
    settings = get_settings()

    app = FastAPI(
        title="CFN-RAG Backend",
        lifespan=lifespan,
    )

    # 本地单用户形态：launcher 同源反代为主，CORS 保持宽松但凭证组合无效（S10 记录在案）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(api_router, prefix="/api")

    return app


app: FastAPI = create_app()

if __name__ == "__main__":
    import uvicorn

    # reload=False（修 F3：旧版 __main__ 硬编码 reload=True，与 launcher 不一致）
    uvicorn.run("main:app", host="127.0.0.1", port=7077, reload=False)
