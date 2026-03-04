from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router: APIRouter = APIRouter()


def _get_resources_dir() -> Path:
    """
    获取resources目录路径。
    resources是外部项目文件夹，和本项目放在同一目录下。
    """
    # 1. 检查环境变量（由launcher.py设置）
    env_path = os.environ.get('CFN_RESOURCES_DIR')
    if env_path:
        return Path(env_path)

    # 2. 检查是否在PyInstaller打包环境
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        resources_path = exe_dir / "resources"
        if resources_path.exists():
            return resources_path
        raise FileNotFoundError(f"打包环境未找到resources目录: {resources_path}")

    # 3. 开发环境：resources在父目录
    # 当前文件位置: cfn-rag-backend/api/assets_api.py
    # resources位置: cfn-rag-backend/../resources
    project_dir = Path(__file__).resolve().parent.parent  # cfn-rag-backend
    parent_dir = project_dir.parent
    resources_path = parent_dir / "resources"

    if resources_path.exists():
        return resources_path

    # 如果父目录没有，再检查同级目录
    sibling_path = project_dir / "resources"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(f"开发环境未找到resources目录")


@router.get("/avatar/{npc_name}", summary="获取 NPC 头像")
async def get_avatar(npc_name: str) -> FileResponse:
    """
    返回指定 NPC 的头像 PNG。
    """

    resources_dir = _get_resources_dir()
    avatar_path: Path = (
        resources_dir / "flashswf" / "portraits" / "profiles" / f"{npc_name}.png"
    )
    if not avatar_path.is_file():
        raise HTTPException(status_code=404, detail="头像资源不存在")

    return FileResponse(path=str(avatar_path), media_type="image/png")


@router.get("/illustration/{npc_name}/{emotion}", summary="获取 NPC 情绪立绘")
async def get_illustration(npc_name: str, emotion: str) -> FileResponse:
    """
    返回指定 NPC + 情绪的立绘 PNG；找不到时回退到“普通”，仍失败则 404。
    """

    resources_dir = _get_resources_dir()
    base_dir: Path = resources_dir / "flashswf" / "portraits" / "illustration"

    primary: Path = base_dir / f"{npc_name}#{emotion}.png"
    if primary.is_file():
        return FileResponse(path=str(primary), media_type="image/png")

    fallback: Path = base_dir / f"{npc_name}#普通.png"
    if fallback.is_file():
        return FileResponse(path=str(fallback), media_type="image/png")

    raise HTTPException(status_code=404, detail="立绘资源不存在")

