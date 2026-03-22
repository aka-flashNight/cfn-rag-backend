from __future__ import annotations

import os
import sys
from pathlib import Path

# 游戏资源根目录的常见文件夹名：优先 resources，其次为 GitHub 克隆仓库默认目录名 CrazyFlashNight
RESOURCE_FOLDER_NAMES: tuple[str, ...] = ("resources", "CrazyFlashNight")


def _pick_resources_under(parent: Path) -> Path | None:
    """在 parent 下按顺序查找 RESOURCE_FOLDER_NAMES 中第一个已存在的目录。"""
    for name in RESOURCE_FOLDER_NAMES:
        p = (parent / name).resolve()
        if p.exists() and p.is_dir():
            return p
    return None


def pick_existing_or_default_resource_root(parent: Path) -> Path:
    """
    在 parent 下选择「默认落盘」用的资源根目录（用于新建 tools 等子路径）：
    若已存在名为 resources 的目录则用之；否则若存在列表中其它已存在的别名目录（如 CrazyFlashNight）则用之；
    若均不存在，则返回 parent/resources（由调用方 mkdir 创建）。
    """
    for name in RESOURCE_FOLDER_NAMES:
        p = (parent / name).resolve()
        if p.is_dir():
            return p
    return (parent / RESOURCE_FOLDER_NAMES[0]).resolve()


def is_packaged_environment() -> bool:
    """
    是否处于 PyInstaller 打包环境。
    参考 launcher.py 的判断逻辑。
    """

    return hasattr(sys, "_MEIPASS") or getattr(sys, "frozen", False)


def find_resources_directory() -> Path:
    """
    定位游戏资源根目录（非交互版），对应原 `resources` 文件夹内容。

    约定与 launcher.py 一致：
    - 开发环境：与项目（cfn-rag-backend）同级或位于项目根下
    - 打包环境：与 exe 同级

    若不存在名为 `resources` 的文件夹，则按相同规则尝试 `CrazyFlashNight`（游戏仓库默认克隆目录名）。

    同时支持通过环境变量 `CFN_RESOURCES_DIR` 显式指定。
    """

    env = os.environ.get("CFN_RESOURCES_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p

    if is_packaged_environment():
        candidates = [Path(sys.executable).resolve().parent, Path.cwd()]
    else:
        # services/game_data/paths.py -> parents[2] = <project_root>（cfn-rag-backend）
        project_root = Path(__file__).resolve().parents[2]
        candidates = [project_root.parent, project_root, Path.cwd()]

    for parent in candidates:
        found = _pick_resources_under(parent)
        if found is not None:
            return found

    raise FileNotFoundError(
        "未找到游戏资源文件夹（已依次尝试 resources 与 CrazyFlashNight）。"
        "请将其与项目/EXE 同级或置于项目根目录下，"
        "或设置环境变量 CFN_RESOURCES_DIR 指向该资源根目录。"
    )


def get_game_data_root() -> Path:
    """
    返回游戏数据根目录：<resources>/data
    """

    resources_dir = find_resources_directory()
    data_dir = (resources_dir / "data").resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"resources 已找到但 data 目录不存在: {data_dir}")
    return data_dir

