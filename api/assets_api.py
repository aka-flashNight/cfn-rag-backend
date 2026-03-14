from __future__ import annotations

import asyncio
import os
import sys
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.startup import _get_ffdec_path, _has_java
from scripts.extract_portraits_from_swf import run_extract


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
    """解压 illustration.zip 的目标目录：resources/flashswf/portraits/illustration。"""
    base = _get_exe_or_project_dir()
    return base / "resources" / "flashswf" / "portraits" / "illustration"


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


@router.post(
    "/export-illustrations",
    response_model=ExportIllustrationsResponse,
    summary="立绘就绪（从 zip 解压或从 SWF 导出）",
)
async def export_illustrations(body: ExportIllustrationsRequest) -> ExportIllustrationsResponse:
    """
    优先使用 illustration.zip：若与 exe 同目录存在 illustration.zip，则解压（很快）。
    否则若有 tools/ffdec.jar 且本机有 Java，则从 SWF 导成立绘（约需数分钟）。
    前端建议：请求超时设长（如 5 分钟），等待期间展示「正在准备立绘，请稍候（从 SWF 导出约需数分钟）」；
    返回后根据 success、message、source 展示结果，503 时用 detail 展示指引。
    """
    # 1. 有 zip 则只做解压，不检查 Java
    zip_path = _get_illustration_zip_path()
    if zip_path is not None:
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

    # 2. 无 zip：从 SWF 导出，需要 FFDec + Java
    project_root = _get_project_root()
    ffdec_path = _get_ffdec_path(project_root)
    has_java = _has_java()

    if ffdec_path is None or not ffdec_path.exists():
        raise HTTPException(status_code=503, detail=MSG_NO_ZIP_NO_JAVA)
    if not has_java:
        raise HTTPException(status_code=503, detail=MSG_NO_JAVA_HAS_FFDEC)

    # 有 zip 时已返回；此处为无 zip、有 FFDec 且有 Java，开始从 SWF 导出（耗时长），默认输出 WebP
    result = await asyncio.to_thread(
        run_extract,
        ffdec_path=ffdec_path,
        resources_dir=None,
        overwrite=body.overwrite,
        zoom=4,
        smooth=True,
        crop_rect=None,
        only_npc=None,
        webp=True,
        webp_quality=0.85,
    )
    success = bool(result.get("success"))
    processed = int(result.get("processed", 0))
    total = int(result.get("total", 0))
    err = result.get("error")
    if success:
        msg = f"已从 SWF 导成立绘完成，共处理 {processed}/{total} 个。"
    else:
        msg = err or "从 SWF 导成立绘失败。"
    return ExportIllustrationsResponse(
        success=success,
        processed=processed,
        total=total,
        error=err,
        message=msg,
        source="swf",
    )

