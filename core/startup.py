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
from pathlib import Path

# 模型加载状态
_EMBED_MODEL_LOADING = False
_EMBED_MODEL_LOADED = False
_EMBED_MODEL_LOAD_TASK: asyncio.Task | None = None

# 索引构建状态
_INDEX_BUILT = False
_INDEX_BUILDING = False


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
    print("[初始化] 正在后台构建知识库索引...")

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

