#!/usr/bin/env python3
"""CFN-RAG.exe 打包脚本（v3 重写，对应 docs/v3-developer/06-存储启动与打包瘦身.md §4）。

形态：PyInstaller onefile（launcher.py 为入口，单实例互斥 + 起后端 + 前端静态服务 + /api 反代）。

打包内容（仅三类 + 启动资源）：
- dist/                          前端静态产物
- models/bge-small-zh-v1.5-onnx-int8/   int8 嵌入模型（~55MB）
- backup_resources/              npc_state_db / agent_tasks / agent_text 首启种子
- scripts/icon.ico、scripts/loading_audio.mp3   启动闪屏/提示音

不再打入：tools/（ffdec 已随 D9 废弃）、原始 HF 模型目录、scripts/ 其余、evals、tests。

hidden imports 重点：onnxruntime / tokenizers / uvicorn 标准件；
excludes 重点：torch、transformers、llama_index、langchain 系（06 §4 包体主战场）。

用法：
  python scripts/build_exe.py              # 日常打包
  python scripts/build_exe.py --no-splash  # 不带闪屏（可省 ~数 MB）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR_NAME = "bge-small-zh-v1.5-onnx-int8"
EXE_NAME = "CFN-RAG"


def _collect_tool_modules() -> list[str]:
    """枚举 services.tools/<category>/<name>.py（discover 运行时动态导入，
    PyInstaller 静态分析看不到，必须显式进 hiddenimports）。"""
    import importlib
    import pkgutil

    names: list[str] = []
    pkg = importlib.import_module("services.tools")
    for _imp, category, ispkg in pkgutil.iter_modules(list(getattr(pkg, "__path__", []))):
        if not ispkg or category.startswith("_"):
            continue
        sub = importlib.import_module(f"services.tools.{category}")
        for _imp2, mod, _ispkg2 in pkgutil.iter_modules(list(getattr(sub, "__path__", []))):
            if not mod.startswith("_"):
                names.append(f"services.tools.{category}.{mod}")
    return sorted(names)

HIDDEN_IMPORTS = [
    # 运行时框架
    "main",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "pydantic",
    "pydantic_settings",
    # 检索栈
    "onnxruntime",
    "tokenizers",
    "rank_bm25",
    "numpy",
    # LLM / IO
    "openai",
    "httpx",
    "aiofiles",
    "PIL",
    "PIL.Image",
    "PIL.ImageOps",
    "pypdf",
    "docx",
    "docx2txt",
]

# 06 §4：显式排除重依赖（未安装时 exclude 无副作用）
EXCLUDES = [
    "torch",
    "transformers",
    "llama_index",
    "llama_index.core",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langgraph",
    "numpy.tests",
    "numpy.f2py",
    "numpy.random._examples",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "jupyter",
    "pytest",
    "modelscope",
    "datasets",
    "optimum",
]


def _sh(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd])


def build_spec(no_splash: bool) -> str:
    """生成 PyInstaller spec（onefile：无 COLLECT）。"""
    hidden = list(HIDDEN_IMPORTS) + _collect_tool_modules()
    datas = [
        (str(PROJECT_ROOT / "dist"), "dist"),
        (str(PROJECT_ROOT / "models" / MODEL_DIR_NAME), f"models/{MODEL_DIR_NAME}"),
        (str(PROJECT_ROOT / "backup_resources"), "backup_resources"),
        (str(PROJECT_ROOT / "scripts" / "icon.ico"), "scripts"),
        (str(PROJECT_ROOT / "scripts" / "loading_audio.mp3"), "scripts"),
    ]
    missing = [src for src, _ in datas if not Path(src).exists()]
    if missing:
        print(f"[错误] 打包数据缺失: {missing}")
        print("  dist/（前端构建产物）与 models/（int8 模型）必须先就绪。")
        sys.exit(1)

    data_str = "\n".join(f"        ({src!r}, {dst!r})," for src, dst in datas)
    hidden_str = "\n".join(f"        {name!r}," for name in hidden)
    excludes_str = "\n".join(f"        {name!r}," for name in EXCLUDES)
    icon = str(PROJECT_ROOT / "scripts" / "icon.ico")

    splash_block = ""
    splash_arg = ""
    if not no_splash:
        splash_block = f"""
splash = Splash(
    {icon!r},
    binaries=a.binaries,
    datas=a.datas,
    minify_script=True,
    always_on_top=True,
)
"""
        splash_arg = "\n    splash,"

    return f'''# -*- mode: python ; coding: utf-8 -*-
# 由 scripts/build_exe.py 生成（06 §4：onefile + 依赖裁剪）

a = Analysis(
    [{str(PROJECT_ROOT / "launcher.py")!r}],
    pathex=[{str(PROJECT_ROOT)!r}],
    binaries=[],
    datas=[
{data_str}
    ],
    hiddenimports=[
{hidden_str}
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
{excludes_str}
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)
{splash_block}
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,{splash_arg}
    [],
    name={EXE_NAME!r},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon={icon!r},
)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="CFN-RAG.exe 打包（PyInstaller onefile）")
    parser.add_argument("--no-splash", action="store_true", help="不打包启动闪屏")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    print("=" * 50)
    print("CFN-RAG 打包（v3 / onefile）")
    print(f"项目根: {PROJECT_ROOT}")
    print("=" * 50)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[错误] 未安装 pyinstaller（requirements-dev）：pip install pyinstaller")
        sys.exit(1)

    # 清理旧产物
    for folder in ("build", "build_temp"):
        shutil.rmtree(PROJECT_ROOT / folder, ignore_errors=True)
    old_exe = PROJECT_ROOT / f"{EXE_NAME}.exe"
    if old_exe.exists():
        try:
            old_exe.unlink()
            print("已删除旧 exe")
        except PermissionError:
            print("[错误] CFN-RAG.exe 正在运行，无法覆盖，请先退出再打包")
            sys.exit(1)

    spec_path = PROJECT_ROOT / "build_temp" / f"{EXE_NAME}.spec"
    spec_path.parent.mkdir(exist_ok=True)
    spec_path.write_text(build_spec(args.no_splash), encoding="utf-8")
    print(f"spec 已生成: {spec_path}")

    print("\n开始打包（可能需要几分钟）...")
    _sh([sys.executable, "-m", "PyInstaller", str(spec_path), "--clean", "--noconfirm"])

    dist_exe = PROJECT_ROOT / "dist" / f"{EXE_NAME}.exe"
    if not dist_exe.exists():
        print("[错误] 打包产物缺失")
        sys.exit(1)
    shutil.move(str(dist_exe), str(old_exe))

    size_mb = old_exe.stat().st_size / (1024 * 1024)
    shutil.rmtree(PROJECT_ROOT / "build", ignore_errors=True)
    shutil.rmtree(PROJECT_ROOT / "build_temp", ignore_errors=True)

    print("\n" + "=" * 50)
    print(f"打包完成: {old_exe}  ({size_mb:.1f} MB)")
    print(f"体积预算: ≤160MB（06 §4；int8 模型按 55MB 口径复核）")
    print("=" * 50)
    print("冒烟提示：将 exe 与 resources（游戏资源）同目录放置后双击；")
    print("立绘 manifest 随游戏项目根（launcher/web/assets/dialogue-portraits/）自动探测，")
    print("或以环境变量 CFN_GAME_PROJECT_DIR 指定游戏项目根。")
    if size_mb > 160:
        print("[警告] 超出 160MB 预算——可回退 int8_full 量化档（22.8MB，见实施手记 §4）")
        sys.exit(2)


if __name__ == "__main__":
    main()
