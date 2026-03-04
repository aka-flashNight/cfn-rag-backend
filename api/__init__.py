from fastapi import APIRouter

from core.config import Settings, get_settings
from .knowledge_api import router as knowledge_router
from .game_api import router as game_router
from .assets_api import router as assets_router

api_router: APIRouter = APIRouter()


@api_router.get("/health", summary="健康检查")
async def health_check(settings: Settings = get_settings()) -> dict[str, str]:
    """
    基础健康检查接口。
    """

    return {
        "status": "ok",
        "app": settings.app_name,
    }


api_router.include_router(knowledge_router, prefix="/knowledge", tags=["Knowledge"])
api_router.include_router(game_router, prefix="/game", tags=["GameRAG"])
api_router.include_router(assets_router, prefix="/assets", tags=["Assets"])
