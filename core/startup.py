"""后端启动初始化（core/startup.py 重写，对应 docs/v3-developer/06 §2）。

启动序列（重型任务全部并行 gather，print 改 logging，local 也输出到控制台 + cfn-rag.log）：
├─ OnnxEmbedder 预载 + 预热              (~0.5s)
├─ VectorStore.load（指纹匹配则 <1s）     │ 不匹配 → 后台构建 ≤25s，期间检索降级（空 bundle）
├─ GameDataRegistry 加载
├─ NPCManager 加载
├─ 任务文件 ensure（agent_tasks/agent_text/list.xml 注册，沿用现逻辑）
└─ 摘要 worker 启动

目标：索引已建时从进程启动到 /api/ask 可服务 ≤ 5s（不含 exe 自解压）。
已删除：立绘包解压任务、SWF/FFDec 预处理、npc_state_db 脚本生成链
（保留 backup 复制；缺失时由 NPCManager 从对话数据兜底生成）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from services.game_data.paths import find_resources_directory

logger = logging.getLogger(__name__)

_LOG_FILE_NAME = "cfn-rag.log"


def setup_logging() -> None:
    """本地路线日志：控制台 + 项目根 cfn-rag.log（幂等，重复调用不叠加 handler）。"""
    root = logging.getLogger()
    if getattr(root, "_cfn_logging_configured", False):
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    log_path = Path(__file__).resolve().parent.parent / _LOG_FILE_NAME
    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        root.warning("日志文件不可写，仅输出到控制台: %s", log_path)

    root._cfn_logging_configured = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 任务文件 ensure（agent_tasks / agent_text / list.xml 注册，沿用现逻辑）
# ---------------------------------------------------------------------------

def _get_backup_resources_dir() -> Path:
    """内置备份目录 backup_resources（PyInstaller / exe 同级 / 开发项目根）。"""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            meipass_backup = Path(sys._MEIPASS) / "backup_resources"
            if meipass_backup.is_dir():
                return meipass_backup
        return Path(sys.executable).parent / "backup_resources"
    return Path(__file__).resolve().parent.parent / "backup_resources"


def _copy_backup_file_if_missing(*, target_path: Path, backup_rel_path: Path) -> bool:
    """若 target 不存在则从 backup 复制（保留 backup 复制本身，逻辑简化）。"""
    if target_path.exists():
        return False
    backup_path = _get_backup_resources_dir() / backup_rel_path
    if not backup_path.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(backup_path), str(target_path))
    logger.info("已从备份复制: %s -> %s", backup_path, target_path)
    return True


def _ensure_json_file(*, target_path: Path, default_obj: Any, backup_rel_path: Path) -> None:
    """确保 JSON 文件存在：目标优先 → 备份 → 最小骨架。"""
    if target_path.exists():
        return
    if _copy_backup_file_if_missing(target_path=target_path, backup_rel_path=backup_rel_path):
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(default_obj, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("已创建默认 JSON 文件: %s", target_path)


def _ensure_xml_entry(*, list_xml_path: Path, tag: str, value: str) -> None:
    """若 list.xml 缺失 `<tag>value</tag>`，只在 `</root>` 前插入一行。"""
    if not list_xml_path.exists():
        logger.warning("未找到 XML 列表文件，跳过修复: %s", list_xml_path)
        return
    try:
        raw = list_xml_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("读取 XML 失败，跳过修复: %s, err=%s", list_xml_path, exc)
        return

    if re.search(rf"<{re.escape(tag)}>\s*{re.escape(value)}\s*</{re.escape(tag)}>", raw):
        return

    newline = "\r\n" if "\r\n" in raw else "\n"
    indent = "  "
    m = re.search(rf"^(?P<indent>\s*)<{re.escape(tag)}>\s*.+?\s*</{re.escape(tag)}>\s*$", raw, flags=re.M)
    if m:
        indent = m.group("indent") or indent
    insertion_line = f"{indent}<{tag}>{value}</{tag}>"
    close_idx = raw.rfind("</root>")
    if close_idx != -1:
        before = raw[:close_idx].rstrip("\r\n")
        after = raw[close_idx:]
        raw_new = before + newline + insertion_line + newline + after
    else:
        raw_new = raw.rstrip("\r\n") + newline + insertion_line + newline
    try:
        list_xml_path.write_text(raw_new, encoding="utf-8")
        logger.info("已修复 XML 列表条目: %s -> %s", list_xml_path, insertion_line)
    except OSError as exc:
        logger.warning("写入 XML 修复失败，跳过: %s, err=%s", list_xml_path, exc)


def ensure_task_agent_files_and_lists() -> None:
    """确保 Agent 生成类任务所需文件与 list.xml 注册项齐全。"""
    resources_dir = find_resources_directory()
    task_root = resources_dir / "data" / "task"
    task_text_root = task_root / "text"

    _ensure_json_file(
        target_path=task_root / "agent_tasks.json",
        default_obj={"tasks": []},
        backup_rel_path=Path("data") / "task" / "agent_tasks.json",
    )
    _ensure_json_file(
        target_path=task_text_root / "agent_text.json",
        default_obj={},
        backup_rel_path=Path("data") / "task" / "text" / "agent_text.json",
    )
    _ensure_xml_entry(
        list_xml_path=task_root / "list.xml", tag="task", value="agent_tasks.json"
    )
    _ensure_xml_entry(
        list_xml_path=task_text_root / "list.xml", tag="text", value="agent_text.json"
    )


def ensure_npc_state_db() -> bool:
    """npc_state_db.json：存在跳过 → backup 复制 → （缺失时 NPCManager 从对话数据兜底）。"""
    from services.npc.manager import get_npc_state_path

    path = get_npc_state_path()
    if path.exists():
        return True
    if _copy_backup_file_if_missing(
        target_path=path, backup_rel_path=Path("data") / "rag" / "npc_state_db.json"
    ):
        return True
    logger.warning("npc_state_db.json 不存在，将由 NPCManager 从对话数据初始化: %s", path)
    return False


# ---------------------------------------------------------------------------
# 并行初始化子任务
# ---------------------------------------------------------------------------

async def _init_retrieval() -> None:
    """embedder 预载 + 预热 → 向量库加载（指纹不匹配则后台构建，期间检索降级）。"""
    t0 = time.perf_counter()
    from services.retrieval.embedder import get_default_embedder
    from services.retrieval.hybrid import get_retrieval_engine
    from services.retrieval.loader import compute_corpus_fingerprint, load_corpus

    def _load_sync() -> tuple[bool, str]:
        embedder = get_default_embedder()
        embedder.warmup()
        engine = get_retrieval_engine()
        fingerprint = compute_corpus_fingerprint()
        return engine.try_load(fingerprint), fingerprint

    loaded, fingerprint = await asyncio.to_thread(_load_sync)
    logger.info("检索初始化: embedder 就绪，向量库%s（%.2fs）",
                "已加载" if loaded else "未命中（后台构建中）", time.perf_counter() - t0)

    if not loaded:
        engine = get_retrieval_engine()

        async def _build() -> None:
            try:
                nodes = await asyncio.to_thread(load_corpus)
                await asyncio.to_thread(engine.build_store, nodes, fingerprint)
                logger.info("向量库后台构建完成（%d 条）", len(nodes))
            except Exception as exc:
                logger.error("向量库后台构建失败（首次请求前可重试）: %s", exc)

        asyncio.create_task(_build())


async def _init_npc_manager() -> None:
    from services.npc.manager import NPCManager, set_npc_manager

    manager = await NPCManager.load()
    set_npc_manager(manager)
    state_count = len(await manager.all_states())
    logger.info("NPC 状态库加载完成（%d 个 NPC）", state_count)


async def _init_memory_and_summary() -> None:
    from services.memory.store import MemoryStore, set_memory_store
    from services.memory.summarize import SummaryWorker, set_summary_worker

    store = MemoryStore()
    set_memory_store(store)
    worker = SummaryWorker(store)
    worker.start()
    set_summary_worker(worker)
    logger.info("会话存储（SQLite/WAL）就绪，摘要 worker 已启动")


async def _init_game_data() -> None:
    def _load() -> None:
        from services.game_data.registry import init_game_data_registry

        init_game_data_registry()

    try:
        await asyncio.to_thread(_load)
        logger.info("GameDataRegistry 加载完成")
    except Exception as exc:
        logger.error("GameDataRegistry 加载失败（将在首次使用时重试）: %s", exc)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

async def run_startup_tasks() -> None:
    """执行所有启动初始化任务：快速文件任务同步做，重型任务并行 gather。"""
    setup_logging()
    t0 = time.perf_counter()
    logger.info("=" * 50)
    logger.info("开始执行后端启动任务...")

    # 快速任务：任务文件 ensure + npc_state backup 复制（主线程，毫秒级）
    ensure_task_agent_files_and_lists()
    ensure_npc_state_db()

    await asyncio.gather(
        _init_retrieval(),
        _init_npc_manager(),
        _init_memory_and_summary(),
        _init_game_data(),
    )

    logger.info("所有启动任务完成（%.2fs），服务就绪", time.perf_counter() - t0)
    logger.info("=" * 50)
