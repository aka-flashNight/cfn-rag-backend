from __future__ import annotations

import os
import sys
from pathlib import Path


def is_packaged_environment() -> bool:
    """
    是否处于 PyInstaller 打包环境。
    参考 launcher.py 的判断逻辑。
    """

    return hasattr(sys, "_MEIPASS") or getattr(sys, "frozen", False)


def find_resources_directory() -> Path:
    """
    定位 resources 目录（非交互版）。

    约定与 launcher.py 一致：
    - 开发环境：resources 与项目（cfn-rag-backend）同级
    - 打包环境：resources 与 exe 同级

    同时支持通过环境变量 `CFN_RESOURCES_DIR` 显式指定。
    """

    env = os.environ.get("CFN_RESOURCES_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p

    if is_packaged_environment():
        base_dir = Path(sys.executable).resolve().parent
    else:
        # 开发环境：resources 与项目根目录同级
        # services/game_data/paths.py -> services/game_data -> services -> <project_root>
        project_root = Path(__file__).resolve().parents[2]
        base_dir = project_root.parent

    p1 = (base_dir / "resources").resolve()
    if p1.exists() and p1.is_dir():
        return p1

    # 兜底：当前工作目录下
    p2 = (Path.cwd() / "resources").resolve()
    if p2.exists() and p2.is_dir():
        return p2

    raise FileNotFoundError(
        "未找到 resources 文件夹。请确保 resources 与项目/EXE 同级，"
        "或设置环境变量 CFN_RESOURCES_DIR 指向 resources 目录。"
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

