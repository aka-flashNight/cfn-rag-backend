"""会话滚动摘要：有界队列 + 单 worker 协程（修 E2 fire-and-forget 失控）。

对应 docs/v3-developer/06-存储启动与打包瘦身.md §1.3：
- 保留「达到间隔触发滚动摘要」逻辑；改为有界队列 + 单 worker 串行执行
  （并发上限 1，带超时 60s，失败重试 1 次后放弃本轮）；
- LLM 调用走统一 LLMClient（purpose="summary"，不带图）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from services.llm import ChatRequest, LLMClient, LLMConfig
from services.memory.store import ChatMessage, MemoryStore, SUMMARIZE_INTERVAL

logger = logging.getLogger(__name__)

SUMMARIZE_TIMEOUT_S = 60
_QUEUE_MAX_SIZE = 16


def should_summarize(message_count: int, interval: int = SUMMARIZE_INTERVAL) -> bool:
    """消息数达到间隔整数倍时触发（沿用旧规则）。"""
    return message_count > 0 and interval > 0 and message_count % interval == 0


@dataclass
class SummaryRequest:
    session_id: str
    npc_name: str
    llm_config: LLMConfig


def _build_summary_prompt(
    npc_name: str, recent_messages: list[ChatMessage], existing_summary: str | None
) -> str:
    """摘要 prompt（沿用旧滚动整合规则：旧摘要 + 最近对话 → 一段完整摘要）。"""
    lines = [
        f"{'玩家' if m.role == 'user' else (npc_name or 'NPC')}: {m.content}"
        for m in recent_messages
    ]
    history_text = "\n".join(lines)
    base = (
        "保留关键信息：讨论的主要话题、双方态度与情感变化、"
        "玩家提出的重要问题或请求、涉及的剧情信息和承诺。\n"
        "请用第三人称客观视角撰写，直接输出摘要文本，不要添加标题或额外格式。\n\n"
    )
    if existing_summary:
        return (
            f"以下是玩家与游戏NPC「{npc_name}」之前对话的摘要：\n"
            f"{existing_summary}\n\n"
            "请在此摘要的基础上，补充以下最近对话的内容，"
            "将新旧信息整合成一段完整且连贯的摘要。\n"
            "注意：不要丢弃已有摘要中的重要信息，而是在其基础上追加新内容并统一整理。\n"
            + base
            + f"最近的对话记录：\n{history_text}"
        )
    return (
        f"请将以下玩家与游戏NPC「{npc_name}」的对话记录整理成一段简洁但信息完整的摘要。\n"
        + base
        + f"对话记录：\n{history_text}"
    )


class SummaryWorker:
    """摘要后台 worker：有界队列（满则丢弃并记日志）、串行执行、超时 + 重试 1 次。"""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._queue: asyncio.Queue[SummaryRequest] = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self.run())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    # 提交与消费
    # ------------------------------------------------------------------

    def submit(self, req: SummaryRequest) -> bool:
        """非阻塞入队；队列满（积压超限）则丢弃本轮摘要并记日志。"""
        try:
            self._queue.put_nowait(req)
            return True
        except asyncio.QueueFull:
            logger.warning("摘要队列已满，丢弃会话 %s 的本轮摘要", req.session_id)
            return False

    async def run(self) -> None:
        """单 worker 串行消费（并发上限 1）。"""
        while True:
            req = await self._queue.get()
            try:
                await self._summarize_with_retry(req)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "会话 %s 摘要生成失败（已放弃本轮）: %s", req.session_id, exc, exc_info=True
                )
            finally:
                self._queue.task_done()

    async def _summarize_with_retry(self, req: SummaryRequest) -> None:
        for attempt in (1, 2):
            try:
                await asyncio.wait_for(self._summarize(req), timeout=SUMMARIZE_TIMEOUT_S)
                return
            except asyncio.TimeoutError:
                logger.warning("会话 %s 摘要超时（第 %d 次）", req.session_id, attempt)
                if attempt == 2:
                    raise
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt == 2:
                    raise
                logger.warning("会话 %s 摘要失败（第 %d 次），重试", req.session_id, attempt)
                await asyncio.sleep(0.5)

    async def _summarize(self, req: SummaryRequest) -> None:
        """滚动摘要：已有摘要则整合新内容，否则生成新摘要。"""
        recent = await self._store.get_history(req.session_id, limit=SUMMARIZE_INTERVAL)
        if not recent:
            return
        existing = await self._store.get_summary(req.session_id)
        prompt = _build_summary_prompt(req.npc_name, recent, existing)

        client = LLMClient.for_config(req.llm_config)
        result = await client.chat(
            ChatRequest(
                messages=[{"role": "user", "content": prompt}],
                purpose="summary",
                send_image=False,
            )
        )
        summary = result.content.strip()
        if not summary:
            logger.warning("会话 %s 摘要生成返回空内容，跳过保存", req.session_id)
            return

        count = await self._store.count_messages(req.session_id)
        await self._store.save_summary(req.session_id, summary, count)
        logger.info("会话 %s 摘要已更新（消息数: %d）", req.session_id, count)


_SUMMARY_WORKER: SummaryWorker | None = None


def get_summary_worker() -> SummaryWorker:
    """全局单例（startup 用 get_memory_store 初始化；未初始化时懒加载兜底）。"""
    global _SUMMARY_WORKER
    if _SUMMARY_WORKER is None:
        from services.memory.store import get_memory_store

        _SUMMARY_WORKER = SummaryWorker(get_memory_store())
    return _SUMMARY_WORKER


def set_summary_worker(worker: SummaryWorker | None) -> None:
    global _SUMMARY_WORKER
    _SUMMARY_WORKER = worker
