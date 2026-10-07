from fastapi import APIRouter

from core.config import Settings, get_settings
from services.integration.state import frontend_url_for, get_integration_state
from .game_api import router as game_router
from .assets_api import router as assets_router
from .integration_api import router as integration_router

api_router: APIRouter = APIRouter()


@api_router.get("/health", summary="健康检查")
async def health_check(settings: Settings = get_settings()) -> dict:
    """
    健康检查接口。集成字段（mode/bound_slot/frontend_url）供游戏 Host 判定与面板取址；
    原 status/app 字段保留兼容。
    """

    snapshot = get_integration_state().snapshot()
    mode = snapshot["mode"]
    return {
        "status": "ok",
        "app": "CFN-RAG Backend",
        "version": settings.app_version,
        "mode": mode,
        "bound_slot": snapshot["bound_slot"],
        "frontend_url": frontend_url_for(mode),
    }


api_router.include_router(game_router, prefix="/game", tags=["GameRAG"])
api_router.include_router(assets_router, prefix="/assets", tags=["Assets"])
api_router.include_router(integration_router, prefix="/integration", tags=["Integration"])
