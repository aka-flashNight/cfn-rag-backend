from __future__ import annotations

import asyncio
import sys
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.game_data.paths import find_resources_directory

# v3 注：立绘体系按 D9 改为游戏 manifest 查表（P7/窗口③）。此处保留 avatar/illustration
# 静态文件路由与 illustration.zip 解压；SWF/FFDec 导出链路已废弃（文件与脚本已删除）。


router: APIRouter = APIRouter()


def _get_project_root() -> Path:
    """打包后为 bundle 根（_MEIPASS，tools/scripts 在此），开发环境为项目根目录。"""
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def _get_exe_or_project_dir() -> Path:
    """打包后为 exe 所在目录（illustration.zip 放此），开发环境为项目根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def _get_illustration_zip_path() -> Path | None:
    """与 exe/项目同目录的 illustration.zip，不存在则返回 None。"""
    p = _get_exe_or_project_dir() / "illustration.zip"
    return p if p.is_file() else None


def _get_illustration_extract_target() -> Path:
    """解压 illustration.zip 的目标目录：游戏资源根/flashswf/portraits/illustration。"""
    return find_resources_directory() / "flashswf" / "portraits" / "illustration"


def _extract_illustration_zip() -> tuple[bool, str]:
    """
    若存在 illustration.zip 则解压到 resources/.../illustration（覆盖），无需 Java。
    返回 (success, message)。
    """
    zip_path = _get_illustration_zip_path()
    if zip_path is None:
        return False, "未找到 illustration.zip"
    target_dir = _get_illustration_extract_target()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        # 使用默认编码（UTF-8）：请用 UTF-8 制作 illustration.zip，避免解压后文件名乱码
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        return True, f"已从 {zip_path.name} 解压到 {target_dir}"
    except Exception as e:
        return False, str(e)


class ExportIllustrationsRequest(BaseModel):
    """立绘导出请求：仅 overwrite 由前端传入。"""
    overwrite: bool = False


class ExportIllustrationsResponse(BaseModel):
    """立绘导出结果，前端可根据 message 与 source 展示提示。"""
    success: bool
    processed: int
    total: int
    error: str | None
    message: str | None = None
    source: str | None = None  # "zip" | "swf"，便于前端区分解压来源


def _get_resources_dir() -> Path:
    """与全局逻辑一致：resources 或 CrazyFlashNight 资源根目录。"""
    return find_resources_directory()


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


def _illustration_paths(base_dir: Path, npc_name: str, emotion: str) -> list[tuple[Path, str]]:
    """返回 (路径, media_type) 列表：先尝试 WebP，再尝试 PNG。"""
    return [
        (base_dir / f"{npc_name}#{emotion}.webp", "image/webp"),
        (base_dir / f"{npc_name}#{emotion}.png", "image/png"),
        (base_dir / f"{npc_name}#普通.webp", "image/webp"),
        (base_dir / f"{npc_name}#普通.png", "image/png"),
    ]


@router.get("/illustration/{npc_name}/{emotion}", summary="获取 NPC 情绪立绘")
async def get_illustration(npc_name: str, emotion: str) -> FileResponse:
    """
    返回指定 NPC + 情绪的立绘（优先 WebP，其次 PNG）；找不到时回退到“普通”，仍失败则 404。
    """

    resources_dir = _get_resources_dir()
    base_dir: Path = resources_dir / "flashswf" / "portraits" / "illustration"

    for path, media_type in _illustration_paths(base_dir, npc_name, emotion):
        if path.is_file():
            return FileResponse(path=str(path), media_type=media_type)

    raise HTTPException(status_code=404, detail="立绘资源不存在")


# 供前端展示的 503 与成功/失败文案
MSG_NO_ZIP_NO_JAVA = (
    "未检测到立绘拓展包或 Java 环境。请下载 illustration.zip，"
    "并将其与 exe 放在同一目录（程序会自动解压），"
    "或手动解压到 resources\\flashswf\\portraits\\illustration；"
    "若要从 SWF 导成立绘，请安装 JRE 并将 ffdec.jar 放入 tools 目录后重试。"
)
MSG_NO_JAVA_HAS_FFDEC = (
    "未检测到立绘拓展包，且未检测到 Java 环境。"
    "请下载 illustration.zip 与 exe 放在同一目录，或解压到 resources\\flashswf\\portraits\\illustration；"
    "或安装 JRE 后可从 SWF 导成立绘。"
)


MSG_EXPORT_UNAVAILABLE = (
    "未检测到立绘拓展包。请下载 illustration.zip 并将其与 exe 放在同一目录"
    "（程序会自动解压），或手动解压到 resources\\flashswf\\portraits\\illustration。"
    "（从 SWF 导出立绘的功能已下线，立绘改由游戏项目 manifest 体系提供）"
)


@router.post(
    "/export-illustrations",
    response_model=ExportIllustrationsResponse,
    summary="立绘就绪（仅支持 illustration.zip 解压；SWF 导出已下线）",
)
async def export_illustrations(body: ExportIllustrationsRequest) -> ExportIllustrationsResponse:
    """有 illustration.zip 则解压（很快）；否则 503（SWF/FFDec 链路已按 D9 废弃）。"""
    _ = body
    zip_path = _get_illustration_zip_path()
    if zip_path is None:
        raise HTTPException(status_code=503, detail=MSG_EXPORT_UNAVAILABLE)
    ok, msg = await asyncio.to_thread(_extract_illustration_zip)
    if ok:
        return ExportIllustrationsResponse(
            success=True,
            processed=1,
            total=1,
            error=None,
            message="已从 illustration.zip 解压立绘完成。",
            source="zip",
        )
    return ExportIllustrationsResponse(
        success=False,
        processed=0,
        total=0,
        error=msg,
        message=msg,
        source="zip",
    )

