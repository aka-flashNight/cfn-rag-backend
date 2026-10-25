#!/usr/bin/env python3
"""构建独立体验包 CFN-RAG-Full-v<版本>.zip。

面向没有游戏本体的用户：解压后双击版本化 exe 即可聊天，无需 Python、外部数据、
立绘包或预生成索引。包内只放后端运行实际读取的游戏数据子集：

- data/ 下的对话、任务、情报、术语表、关卡、物品、商店、合成、单位、怪物属性、地图
- docs/story/ 世界观设定
- launcher/web/assets/dialogue-portraits/ 对话立绘与 manifest
- flashswf/portraits/profiles/ NPC 头像

不会放入：向量索引、data/rag/ 运行期增量、存档、无关美术/音频/开发文档。
Agent 任务文件使用空种子，避免把开发机的共享任务带进分发包。

用法：
  python scripts/build_full_zip.py --game-dir <游戏项目根> --version 3.0.0
  python scripts/build_full_zip.py --game-dir <游戏项目根> --exe CFN-RAG.exe --version 3.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXE_NAME = "CFN-RAG.exe"
STAGING_DIR = PROJECT_ROOT / "build_temp" / "full_zip_staging"

# 后端运行实际读取的 data 子目录（services/game_data + services/retrieval/loader）
DATA_SUBDIRS: tuple[str, ...] = (
    "dialogues",
    "task",
    "intelligence",
    "intelligence_h5",
    "glossary",
    "stages",
    "items",
    "shops",
    "kshop",
    "crafting",
    "units",
    "enemy_properties",
    "map",
)

# 开发/运行期文件，不进分发
IGNORE_NAMES = {
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}
IGNORE_SUFFIXES = (".log", ".tmp", ".bak", ".pyc", ".pyo")
TASK_EXCLUDE_FILES = {
    "mercenary_tasks_old.json",  # 检索 denylist 内的旧版文件
    "agent_tasks.json",          # 用空种子替换
    "序号登记.txt",              # 开发台账
}
TASK_TEXT_EXCLUDE_FILES = {
    "agent_text.json",           # 用空种子替换
}


def _ignore(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in IGNORE_NAMES or name.endswith(IGNORE_SUFFIXES):
            ignored.add(name)
    return ignored


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"[错误] {label}不存在: {path}")


def _copy_tree(src: Path, dst: Path, *, ignore_extra: set[str] | None = None) -> None:
    def ignore(dir_path: str, names: list[str]) -> set[str]:
        result = _ignore(dir_path, names)
        if ignore_extra and Path(dir_path).resolve() == src.resolve():
            result |= ignore_extra
        return result

    shutil.copytree(src, dst, ignore=ignore)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def _seed_agent_files(content_root: Path) -> None:
    task_root = content_root / "data" / "task"
    text_root = task_root / "text"
    text_root.mkdir(parents=True, exist_ok=True)

    backup_tasks = PROJECT_ROOT / "backup_resources" / "data" / "task" / "agent_tasks.json"
    if backup_tasks.is_file():
        shutil.copy2(backup_tasks, task_root / "agent_tasks.json")
    else:
        _write_json(task_root / "agent_tasks.json", {"tasks": []})
    _write_json(text_root / "agent_text.json", {})


def build_zip(
    *,
    game_dir: Path,
    exe_path: Path,
    version: str,
    output_path: Path,
) -> Path:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit(f"[错误] 版本号应为 x.y.z: {version!r}")
    if not exe_path.is_file():
        raise SystemExit(f"[错误] 未找到 exe: {exe_path}（先运行 scripts/build_exe.py）")

    _require_dir(game_dir, "游戏项目根")
    _require_dir(game_dir / "data", "游戏 data 目录")
    _require_dir(game_dir / "docs" / "story", "docs/story 目录")
    portrait_dir = game_dir / "launcher" / "web" / "assets" / "dialogue-portraits"
    avatar_dir = game_dir / "flashswf" / "portraits" / "profiles"
    _require_dir(portrait_dir, "对话立绘目录")
    _require_dir(avatar_dir, "头像目录")
    if not (portrait_dir / "manifest.json").is_file():
        raise SystemExit(f"[错误] 立绘 manifest 缺失: {portrait_dir / 'manifest.json'}")

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    top_dir = STAGING_DIR / f"CFN-RAG-v{version}"
    content_root = top_dir / "CrazyFlashNight"
    content_root.mkdir(parents=True)

    for rel in DATA_SUBDIRS:
        src = game_dir / "data" / rel
        if not src.is_dir():
            raise SystemExit(f"[错误] 缺少 data 子目录: {src}")
        extra: set[str] = set()
        if rel == "task":
            extra |= TASK_EXCLUDE_FILES
        _copy_tree(src, content_root / "data" / rel, ignore_extra=extra)

    text_dir = content_root / "data" / "task" / "text"
    for name in TASK_TEXT_EXCLUDE_FILES:
        target = text_dir / name
        if target.exists():
            target.unlink()

    _copy_tree(game_dir / "docs" / "story", content_root / "docs" / "story")
    _copy_tree(portrait_dir, content_root / "launcher" / "web" / "assets" / "dialogue-portraits")
    _copy_tree(avatar_dir, content_root / "flashswf" / "portraits" / "profiles")

    _seed_agent_files(content_root)

    exe_name = f"CFN-RAG-v{version}.exe"
    packaged_exe = top_dir / exe_name
    shutil.copy2(exe_path, packaged_exe)

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in STAGING_DIR.rglob("*") if p.is_file())
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for path in files:
            zf.write(path, path.relative_to(STAGING_DIR).as_posix())

    # 自检：分发包里不能出现运行期状态或索引
    with zipfile.ZipFile(output_path) as zf:
        names = zf.namelist()
    forbidden = [
        name for name in names
        if "data/rag/" in name or "vector_index" in name or name.endswith((".log", ".db"))
    ]
    if forbidden:
        raise SystemExit(f"[错误] zip 内出现运行期文件: {forbidden[:10]}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    exe_mb = packaged_exe.stat().st_size / (1024 * 1024)
    print("=" * 60)
    print(f"独立体验包: {output_path}")
    print(f"  体积: {size_mb:.1f} MB（{len(files)} 个文件）")
    print(f"  内置 exe: {exe_name}  {exe_mb:.1f} MB")
    print(f"  exe SHA-256: {_sha256(packaged_exe)}")
    print(f"  zip SHA-256: {_sha256(output_path)}")
    print(f"  顶层结构: {exe_name} + CrazyFlashNight/")
    print("=" * 60)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 CFN-RAG Full 独立体验 zip")
    parser.add_argument(
        "--game-dir",
        default="",
        help="游戏项目根（含 data/、docs/story/、launcher/、flashswf/）",
    )
    parser.add_argument("--exe", default=str(PROJECT_ROOT / DEFAULT_EXE_NAME), help="单文件 exe 路径")
    parser.add_argument("--version", default="3.0.0", help="版本号（x.y.z）")
    parser.add_argument(
        "--output",
        default="",
        help="输出 zip 路径（默认项目根 CFN-RAG-Full-v<版本>.zip）",
    )
    args = parser.parse_args()

    game_dir = Path(args.game_dir).expanduser().resolve() if args.game_dir else None
    if game_dir is None:
        print("[错误] 请用 --game-dir 指定游戏项目根（或设置环境变量后手动传入）")
        sys.exit(1)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else PROJECT_ROOT / f"CFN-RAG-Full-v{args.version}.zip"
    )
    try:
        build_zip(
            game_dir=game_dir,
            exe_path=Path(args.exe).expanduser().resolve(),
            version=args.version,
            output_path=output,
        )
    finally:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
