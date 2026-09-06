"""静态资源 API（v3：立绘全面走游戏 manifest 查表，docs/v3-developer/07 §6）。

- GET /avatar/{npc_name}：NPC 立绘（「普通」情绪）——manifest 查表 + bounds 裁剪 PNG；
- GET /illustration/{npc_name}/{emotion}：指定情绪立绘（缺情绪走 manifest 回退链）。

旧体系（illustration.zip 解压 / SWF+FFDec 导出 / profiles 回退）已整体删除；
游戏内立绘展示由游戏 web 端自己的 manifest 逻辑负责，本后端仅为独立前端供图。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from services.portraits import get_portrait_png

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

_PNG_MEDIA_TYPE = "image/png"


@router.get("/avatar/{npc_name}", summary="获取 NPC 头像（manifest 查表，「普通」情绪）")
async def get_avatar(npc_name: str) -> Response:
    """返回指定 NPC 的立绘本体图（bounds 裁剪 PNG）；查表无命中时 404。"""
    return _portrait_response(npc_name, "普通")


@router.get(
    "/illustration/{npc_name}/{emotion}",
    summary="获取 NPC 情绪立绘（manifest 查表 + bounds 裁剪，情绪缺省走回退链）",
)
async def get_illustration(npc_name: str, emotion: str) -> Response:
    """返回指定 NPC + 情绪的立绘 PNG；该情绪缺失时按 manifest 回退链兜底，仍无则 404。"""
    return _portrait_response(npc_name, emotion)


def _portrait_response(npc_name: str, emotion: str) -> Response:
    """manifest 查表 → bounds 裁剪 PNG；查表器缺失（无图模式）/查不到角色/主角 → 404。"""
    try:
        png_bytes = get_portrait_png(npc_name, emotion)
    except Exception as exc:
        logger.warning("立绘接口处理失败: %s %s %s", npc_name, emotion, exc)
        png_bytes = None
    if png_bytes is None:
        raise HTTPException(status_code=404, detail="立绘资源不存在")
    return Response(content=png_bytes, media_type=_PNG_MEDIA_TYPE)
