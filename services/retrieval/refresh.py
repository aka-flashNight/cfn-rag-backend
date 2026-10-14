"""索引惰性刷新（B2-7）：bind / unbind / 任务写盘 / 对话覆盖层写盘后按语料指纹检查，
失配则后台重建。

覆盖层（data/rag/agent_saves/）不列入 `_FINGERPRINT_SOURCE_DIRS`，不做 mtime 自动
监听；重建由上述语义事件显式触发（开发计划 §2.0.6 缓解口径①）。槽内四个文件本身
计入指纹，故写盘后一次检查即可发现失配。

安全口径：引擎未就绪（store is None）时直接返回 False——不建索引、不写盘、不起
线程（测试环境经写盘路径调用时天然 no-op）；检查与重建全程 best-effort，失败只记
日志，旧索引继续服务。重建在 daemon 线程内完成，期间检索仍用旧索引。
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_refresh_lock = threading.Lock()
_refreshing = False
#: 重建进行中收到的触发折叠为标记，由重建线程收尾复核（防「最后一次写盘恰好落在
#: 重建收尾窗口」时漏刷新）
_folded_trigger = False


def refresh_index_if_stale() -> bool:
    """语料指纹与已装载索引不一致时，起后台线程重建；永不抛异常。

    返回是否新启动了重建（未就绪 / 指纹一致 / 已有重建进行中 → False）。
    """
    global _refreshing, _folded_trigger
    started = False
    try:
        from services.retrieval.hybrid import get_retrieval_engine

        engine = get_retrieval_engine()
        if engine.store is None:
            return False
        from services.retrieval.loader import compute_corpus_fingerprint

        if engine.store.fingerprint == compute_corpus_fingerprint():
            return False
        with _refresh_lock:
            if _refreshing:
                _folded_trigger = True
                return False
            _refreshing = True
            started = True
        threading.Thread(
            target=_rebuild_until_current,
            args=(engine,),
            name="index-refresh",
            daemon=True,
        ).start()
        logger.info("语料指纹与索引不一致，已启动后台重建")
        return True
    except Exception as exc:
        if started:
            with _refresh_lock:
                _refreshing = False
        logger.warning("索引刷新检查失败（旧索引继续服务）: %s", exc)
        return False


def _rebuild_until_current(engine) -> None:
    """循环重建直至索引指纹追平当前语料（构建期间语料可能再次变化）。"""
    global _refreshing, _folded_trigger
    try:
        from services.retrieval.loader import compute_corpus_fingerprint, load_corpus

        while True:
            expected = compute_corpus_fingerprint()
            if engine.store is not None and engine.store.fingerprint == expected:
                break
            nodes = load_corpus()
            logger.info("索引后台重建: %d 条节点", len(nodes))
            engine.build_store(nodes, expected)
    except Exception as exc:
        logger.warning("索引后台重建失败（旧索引继续服务）: %s", exc)
    finally:
        with _refresh_lock:
            _refreshing = False
            folded = _folded_trigger
            _folded_trigger = False
        if folded:
            refresh_index_if_stale()
