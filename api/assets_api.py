"""静态资源 API（v3：立绘切换游戏 manifest 查表，docs/v3-developer/07 §6）。

- GET /avatar/{npc_name}：**头像**——沿用原位置原文件
  （resources/flashswf/portraits/profiles/{npc}.png），与立绘体系无关；
- GET /illustration/{npc_name}/{emotion}：**立绘**——游戏项目对话立绘 manifest
  查表（角色归一化 + 情绪回退链）→ bounds 裁剪 → PNG。旧 illustration.zip
  解压 / SWF+FFDec 导出 / 旧 `名称#情绪.webp` 体系随 D9 整体删除；立绘画布
  格式（1024×576 / 775×1000 RGBA）与本体位置（manifest 的 asset.bounds）由
  查表协议给出，本接口返回裁剪后的本体图。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from services.game_data.paths import find_resources_directory
from services.portraits import get_portrait_png

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

_PNG_MEDIA_TYPE = "image/png"


@router.get("/avatar/{npc_name}", summary="获取 NPC 头像（原 profiles 目录，与立绘体系无关）")
async def get_avatar(npc_name: str) -> FileResponse:
    """返回指定 NPC 的头像 PNG（resources/flashswf/portraits/profiles/{npc}.png）。"""
    avatar_path = (
        find_resources_directory() / "flashswf" / "portraits" / "profiles" / f"{npc_name}.png"
    )
    if not avatar_path.is_file():
        raise HTTPException(status_code=404, detail="头像资源不存在")
    return FileResponse(path=str(avatar_path), media_type=_PNG_MEDIA_TYPE)


@router.get(
    "/illustration/{npc_name}/{emotion}",
    summary="获取 NPC 情绪立绘（manifest 查表 + bounds 裁剪，情绪缺省走回退链）",
)
async def get_illustration(npc_name: str, emotion: str) -> Response:
    """返回指定 NPC + 情绪的立绘本体图 PNG；该情绪缺失时按 manifest 回退链兜底，仍无则 404。"""
    try:
        png_bytes = get_portrait_png(npc_name, emotion)
    except Exception as exc:
        logger.warning("立绘接口处理失败: %s %s %s", npc_name, emotion, exc)
        png_bytes = None
    if png_bytes is None:
        raise HTTPException(status_code=404, detail="立绘资源不存在")
    return Response(content=png_bytes, media_type=_PNG_MEDIA_TYPE)
