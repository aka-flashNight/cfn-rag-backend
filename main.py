from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import api_router
from core.config import get_settings
from core.exceptions import register_exception_handlers
from core.startup import run_startup_tasks, setup_logging

logger = logging.getLogger(__name__)


class SPAStaticFiles(StaticFiles):
    """前端静态层（B1-1）：`/api/*` 由先注册的路由优先命中；未命中的非 API 路径回退 index.html。"""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            normalized = path.replace("\\", "/").lstrip("/")
            if exc.status_code != 404 or normalized == "api" or normalized.startswith("api/"):
                raise
            return await super().get_response("index.html", scope)


def _find_frontend_dist() -> Path | None:
    """定位前端 dist：打包内嵌（_MEIPASS/dist）→ 开发目录（本文件旁）→ exe 目录旁。"""
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "dist")
    candidates.append(Path(__file__).resolve().parent / "dist")
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "dist")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


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

    # 前端页面由 7077 同源伺服（内嵌面板 iframe 需要固定 origin；静态层必须最后挂）
    dist_dir = _find_frontend_dist()
    if dist_dir is not None:
        app.mount("/", SPAStaticFiles(directory=dist_dir, html=True), name="frontend")
    else:
        logger.warning("未找到前端 dist 目录，7077 只伺服 API（面板 iframe 将不可用）")

    return app


app: FastAPI = create_app()

if __name__ == "__main__":
    import uvicorn

    # reload=False（修 F3：旧版 __main__ 硬编码 reload=True，与 launcher 不一致）
    uvicorn.run("main:app", host="127.0.0.1", port=7077, reload=False)
