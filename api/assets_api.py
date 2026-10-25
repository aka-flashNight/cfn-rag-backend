"""静态资源 API（v3：立绘切换游戏 manifest 查表，docs/v3-developer/07 §6）。

- GET /avatar/{npc_name}：**头像**——沿用原位置原文件
  （resources/flashswf/portraits/profiles/{npc}.png），与立绘体系无关；
- GET /illustration/{npc_name}/{emotion}：**立绘**——游戏项目对话立绘 manifest
  查表（角色归一化 + 情绪回退链）定位后**原始文件直出**，不做任何裁剪/重编码
  （与旧版行为一致）。查表是必须的：p_*/e_* 为内容 hash，文件名不可猜测。
  裁剪/缩放仅用于大模型输入（services/portraits/provider.get_portrait_data_url）。
  旧 illustration.zip 解压 / SWF+FFDec 导出 / 旧 `名称#情绪.webp` 体系随 D9 删除。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from services.game_data.paths import find_resources_directory
from services.portraits import get_portrait_source_path

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
    summary="获取 NPC 情绪立绘（manifest 查表定位，原始文件直出，情绪缺省走回退链）",
)
async def get_illustration(npc_name: str, emotion: str) -> FileResponse:
    """返回指定 NPC + 情绪的立绘**原始 PNG 文件**（不做裁剪）。

    该情绪缺失时按 manifest 回退链兜底；查表器缺失（无图模式）/
    查不到角色/主角（heroKeys 无静态图）/文件缺失 → 404。
    """
    try:
        source = get_portrait_source_path(npc_name, emotion)
    except Exception as exc:
        logger.warning("立绘接口处理失败: %s %s %s", npc_name, emotion, exc)
        source = None
    if source is None:
        raise HTTPException(status_code=404, detail="立绘资源不存在")
    return FileResponse(path=str(source), media_type=_PNG_MEDIA_TYPE)
