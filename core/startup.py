"""
后端启动初始化模块

统一管理所有需要在后端启动时执行的初始化任务：
1. 加载嵌入模型（异步后台加载）
2. 检查并生成 NPC 状态数据库
3. 初始化数据库
"""

from __future__ import annotations

import asyncio
import os
import sys
import zipfile
from pathlib import Path

# 模型加载状态
_EMBED_MODEL_LOADING = False
_EMBED_MODEL_LOADED = False
_EMBED_MODEL_LOAD_TASK: asyncio.Task | None = None

# 索引构建状态
_INDEX_BUILT = False
_INDEX_BUILDING = False

# 立绘预处理状态
_PORTRAITS_CHECKED = False


def get_npc_state_db_path() -> Path:
    """
    获取 npc_state_db.json 的路径
    """
    # 获取 resources 目录
    resources_dir = _get_resources_dir()
    return resources_dir / "data" / "rag" / "npc_state_db.json"


def _get_resources_dir() -> Path:
    """
    获取 resources 目录路径
    """
    # 1. 检查环境变量（由 launcher.py 设置）
    env_path = os.environ.get('CFN_RESOURCES_DIR')
    if env_path:
        return Path(env_path)

    # 2. 检查是否在 PyInstaller 打包环境
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        resources_path = exe_dir / "resources"
        if resources_path.exists():
            return resources_path

    # 3. 开发环境：resources 在父目录
    # 当前文件位置: cfn-rag-backend/core/startup.py
    # resources 位置: cfn-rag-backend/../resources
    project_dir = Path(__file__).resolve().parent.parent
    parent_dir = project_dir.parent
    resources_path = parent_dir / "resources"

    if resources_path.exists():
        return resources_path

    # 检查同级目录
    sibling_path = project_dir / "resources"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        f"未找到 resources 目录。\n"
        f"已查找: {resources_path} 和 {sibling_path}"
    )


def _get_portraits_dir() -> Path:
    """
    获取 resources/flashswf/portraits 目录路径。
    """
    resources_dir = _get_resources_dir()
    portraits_dir = resources_dir / "flashswf" / "portraits"
    return portraits_dir


def _has_valid_illustrations() -> bool:
    """
    检查是否已经存在至少一张符合命名规则的立绘：
    <NPC名>#<情绪>.png
    """
    try:
        portraits_dir = _get_portraits_dir()
        illustration_dir = portraits_dir / "illustration"
        if not illustration_dir.exists():
            return False

        # 简单判定：存在任何包含 '#' 的 png 文件即可视为已初始化
        for p in illustration_dir.glob("*.png"):
            if "#" in p.stem:
                return True
        return False
    except Exception as e:
        print(f"[初始化] 检查立绘时出错，将在后续重试: {e}")
        return False


def _has_java() -> bool:
    """
    检测当前环境是否可用 Java（JRE 即可）。
    FFDec 的 .jar 与 .exe 均依赖 Java 运行，立绘解压前需通过此检查。
    """
    import subprocess
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _get_exe_or_project_dir() -> Path:
    """打包后为 exe 所在目录，开发环境为项目根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def _extract_illustration_zip_if_present() -> None:
    """
    若与 exe/项目同目录存在 illustration.zip，则解压到
    resources/flashswf/portraits/illustration（覆盖）。不检查 Java，解压很快。
    """
    base = _get_exe_or_project_dir()
    zip_path = base / "illustration.zip"
    if not zip_path.is_file():
        return
    target_dir = base / "resources" / "flashswf" / "portraits" / "illustration"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print(f"[初始化] 已从 illustration.zip 解压立绘到 {target_dir}")
    except Exception as e:
        print(f"[初始化] illustration.zip 解压失败: {e}")


def _get_ffdec_path(project_root: Path) -> Path | None:
    """
    从 tools 目录获取 FFDec 命令行工具路径。仅使用完整版 JAR（exe 为启动器，不需放入）。

    优先顺序：
    1) ffdec.jar（推荐：从官方 ZIP 解压的完整 JAR）
    2) 任意 ffdec_<版本>.jar（如 ffdec_25.1.3.jar）

    下载：https://github.com/jindrapetrik/jpexs-decompiler/releases
    解压 ffdec_*.zip，将包内的 ffdec.jar 或 ffdec_<版本>.jar 放入 tools 即可。
    """
    tools_dir = project_root / "tools"
    if not tools_dir.exists():
        return None

    full_jar = tools_dir / "ffdec.jar"
    if full_jar.exists():
        return full_jar
    for p in sorted(tools_dir.glob("ffdec_*.jar")):
        return p
    return None


def _run_portraits_extraction_blocking() -> None:
    """
    在后台线程中执行 SWF 立绘预处理脚本（阻塞当前线程，但由线程池执行）。

    触发条件：
      - tools 下存在 ffdec.jar 或 ffdec_<版本>.jar
      - 当前环境有可用 Java（FFDec 依赖 JRE）
      - 尚未检测到任何有效立绘 PNG
    """
    global _PORTRAITS_CHECKED

    if _PORTRAITS_CHECKED:
        return

    _PORTRAITS_CHECKED = True

    # 如果已经有立绘，直接返回
    if _has_valid_illustrations():
        print("[初始化] 已检测到立绘资源，跳过 SWF 预处理。")
        return

    project_root = Path(__file__).resolve().parent.parent
    tools_ffdec = _get_ffdec_path(project_root)

    if tools_ffdec is None:
        print(
            "[初始化] 未找到 FFDec 命令行工具，跳过 SWF 立绘预处理。\n"
            "        请将 ffdec-cli.exe 或 ffdec-cli.jar 放到项目 tools 目录下。"
        )
        return

    if not _has_java():
        print(
            "[初始化] 未检测到 Java 环境，跳过 SWF 立绘预处理。\n"
            "        FFDec 需要 JRE，请安装 Java 后将 java 加入 PATH，或配置 JAVA_HOME。"
        )
        return

    script_path = project_root / "scripts" / "extract_portraits_from_swf.py"
    if not script_path.exists():
        print(f"[初始化] 未找到立绘预处理脚本: {script_path}，跳过。")
        return

    print(
        "[初始化] 检测到缺失立绘，正在后台执行 SWF 立绘预处理脚本，"
        "该过程可能需要数十秒到数分钟，请耐心等待。"
    )

    # 直接以当前 Python 解释器调用脚本，继承环境变量（包括 CFN_RESOURCES_DIR）
    import subprocess

    try:
        cmd = [
            sys.executable,
            str(script_path),
            "--ffdec-path",
            str(tools_ffdec),
        ]
        # 在独立进程中阻塞执行；由于调用发生在线程池，不会阻塞事件循环
        completed = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if completed.returncode == 0:
            print("[初始化] SWF 立绘预处理完成 ✓")
        else:
            print(
                "[初始化] SWF 立绘预处理失败，脚本返回非零退出码。\n"
                f"命令: {' '.join(cmd)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
    except Exception as e:
        print(f"[初始化] 执行 SWF 立绘预处理脚本时出现异常: {e}")


def ensure_npc_state_db() -> bool:
    """
    检查 npc_state_db.json 是否存在，如果不存在则执行更新脚本。

    Returns:
        True 表示文件已存在或成功创建，False 表示创建失败
    """
    npc_state_path = get_npc_state_db_path()

    if npc_state_path.exists():
        print(f"[初始化] NPC 状态数据库已存在: {npc_state_path}")
        return True

    print(f"[初始化] NPC 状态数据库不存在，正在生成: {npc_state_path}")

    # 执行更新脚本
    try:
        # 动态导入并执行脚本
        from scripts.update_npc_state import main as update_npc_state_main
        update_npc_state_main()
        print("[初始化] NPC 状态数据库生成完成 ✓")
        return True
    except Exception as e:
        print(f"[初始化] NPC 状态数据库生成失败: {e}")
        return False


async def preload_embed_model_async() -> None:
    """
    异步预加载嵌入模型。

    这个函数会在后台线程中加载模型，避免阻塞主线程。
    """
    global _EMBED_MODEL_LOADED, _EMBED_MODEL_LOADING

    if _EMBED_MODEL_LOADED or _EMBED_MODEL_LOADING:
        return

    _EMBED_MODEL_LOADING = True
    print("[初始化] 正在后台加载嵌入模型...")

    def _load_model():
        from ai_engine.game_data_loader import ensure_embed_model
        ensure_embed_model(offline=True)
        # 触发模型权重加载（通过创建一个简单的嵌入）
        from llama_index.core import Settings
        if Settings.embed_model is not None:
            # 简单的预热：对空字符串进行嵌入
            Settings.embed_model.get_text_embedding("")

    try:
        await asyncio.to_thread(_load_model)
        _EMBED_MODEL_LOADED = True
        print("[初始化] 嵌入模型加载完成 ✓")
    except Exception as e:
        print(f"[初始化] 嵌入模型加载失败: {e}")
    finally:
        _EMBED_MODEL_LOADING = False


def start_embed_model_preload() -> asyncio.Task | None:
    """
    启动嵌入模型的异步预加载任务。

    Returns:
        返回创建的 asyncio Task，如果已经在加载或已加载完成则返回 None
    """
    global _EMBED_MODEL_LOAD_TASK, _EMBED_MODEL_LOADING, _EMBED_MODEL_LOADED

    if _EMBED_MODEL_LOADED or _EMBED_MODEL_LOADING:
        return None

    # 尝试获取现有的事件循环
    try:
        loop = asyncio.get_running_loop()
        _EMBED_MODEL_LOAD_TASK = loop.create_task(preload_embed_model_async())
        return _EMBED_MODEL_LOAD_TASK
    except RuntimeError:
        # 没有运行中的事件循环，将在后续调用时加载
        print("[初始化] 没有运行中的事件循环，模型将在首次请求时加载")
        return None


def is_embed_model_loaded() -> bool:
    """
    检查嵌入模型是否已加载完成
    """
    return _EMBED_MODEL_LOADED


def is_embed_model_loading() -> bool:
    """
    检查嵌入模型是否正在加载中
    """
    return _EMBED_MODEL_LOADING


async def ensure_embed_model_ready() -> None:
    """
    确保嵌入模型已加载完成。

    如果模型正在加载中，等待加载完成；
    如果模型未开始加载，立即开始加载并等待完成。

    注意：此方法会阻塞，适用于需要使用模型的接口（如 ask）。
    """
    global _EMBED_MODEL_LOAD_TASK

    if _EMBED_MODEL_LOADED:
        return

    if _EMBED_MODEL_LOADING and _EMBED_MODEL_LOAD_TASK is not None:
        # 等待现有加载任务完成
        await _EMBED_MODEL_LOAD_TASK
        return

    # 启动新的加载任务并等待
    await preload_embed_model_async()


async def trigger_embed_model_preload() -> None:
    """
    触发嵌入模型预加载，但不等待加载完成。

    适用于前端初始化接口（如 /sessions），只触发加载，不阻塞返回。
    """
    if _EMBED_MODEL_LOADED or _EMBED_MODEL_LOADING:
        return

    start_embed_model_preload()


async def _preload_index_async() -> None:
    """
    在嵌入模型加载完成后，异步构建知识库向量索引。
    """
    global _INDEX_BUILT, _INDEX_BUILDING

    if _INDEX_BUILT or _INDEX_BUILDING:
        return

    _INDEX_BUILDING = True
    print("[初始化] 正在后台加载或构建知识库索引（存在则从 resources/tools/vector_index 加载）...")

    def _build():
        from ai_engine.game_data_loader import get_cached_index
        get_cached_index()

    try:
        await asyncio.to_thread(_build)
        _INDEX_BUILT = True
        print("[初始化] 知识库索引构建完成 ✓")
    except Exception as e:
        print(f"[初始化] 知识库索引构建失败（将在首次请求时重试）: {e}")
    finally:
        _INDEX_BUILDING = False


def is_index_built() -> bool:
    return _INDEX_BUILT


async def run_startup_tasks() -> None:
    """
    执行所有启动初始化任务（全部非阻塞）。

    所有重型任务均在后台执行，不阻塞 uvicorn 接受连接。
    这样前端可以在后端初始化完成之前就正常访问轻量接口（如会话列表）。
    """
    print("\n" + "=" * 50)
    print("[初始化] 开始执行后端启动任务...")
    print("=" * 50)

    # 1. 在后台线程中检查/生成 NPC 状态数据库（不阻塞服务器启动）
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, ensure_npc_state_db)

    # 1-b. 若存在 illustration.zip 则后台解压到 resources/.../illustration（不检查 Java）
    loop.run_in_executor(None, _extract_illustration_zip_if_present)

    # 2. 初始化数据库（MemoryManager.create 会在首次请求时自动创建）
    print("[初始化] 数据库将在首次请求时自动初始化...")

    # 3. 链式后台预加载：嵌入模型 → 知识库索引
    async def _chained_preload():
        await preload_embed_model_async()
        if _EMBED_MODEL_LOADED:
            await _preload_index_async()

    loop.create_task(_chained_preload())

    print("=" * 50)
    print("[初始化] 所有启动任务已提交到后台，服务器即将就绪...")
    print("=" * 50 + "\n")

