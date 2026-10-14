"""对话修改子 Agent 的后台串行 worker（阶段四 B4-2，§4.0.1 独立链路）。

- asyncio 有界队列＋单 worker 串行执行（fire-and-forget），写法对齐
  ``services/memory/summarize.py`` 的 SummaryWorker；
- **绝不阻塞、绝不打断前台聊天**：submit 非阻塞入队，执行期间任何异常只记日志，
  不向主链路抛；
- 执行期间置全局 running 标志（``is_dialogue_agent_running()``）——下一批的注册闸
  用它实现「后台有子 Agent 在跑就完全不注册」（§4.0.4）。

NPC 资料与长期记忆在任务执行当刻装配（复用聊天侧的装配件：
``format_npc_role_tagline`` / ``build_appearance_block`` / ``long_term.load_snapshot``），
委派方只需给触发材料（原因、指示、任务详情、发布对话记录）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from services.llm import LLMClient, LLMConfig
from services.memory.long_term import load_snapshot
from services.orchestrator.prompts import build_appearance_block, format_npc_role_tagline
from services.subagents.dialogue_runner import DialogueRunner

logger = logging.getLogger(__name__)

#: 单个委派的总时限（三轮 LLM 调用，每轮都可能慢）
DIALOGUE_JOB_TIMEOUT_S = 300
_QUEUE_MAX_SIZE = 16


@dataclass
class DialogueJob:
    """一次对话修改委派（触发材料由调用方给全；NPC 资料执行当刻自装）。"""

    npc_name: str
    llm_config: LLMConfig
    #: 触发类型：关系变化 / 态度变化 / 关键信息 / 重要任务（§4.0.4 必传）
    source_trigger: str = ""
    #: 发生了什么（相关事实）
    reason: str = ""
    #: 重写指示（触发一发布任务时挂的指示；其余触发可空，由模型自行判断）
    instruction: str = ""
    #: 重要任务必传任务 id（只有首次传有效，后续静默跳过——跳过逻辑属下一批触发侧）
    source_task_id: int | None = None
    #: 任务详情与对话（如有）
    task_block: str = ""
    #: 发布那场对话的摘要＋摘要后精确对话（如有）
    publish_block: str = ""


def _build_profile_block(npc_name: str, state: Any) -> str:
    """NPC 资料块：身份一行＋形象描述（复用聊天侧装配件，口吻锚点）。"""
    parts = [format_npc_role_tagline(
        npc_name=npc_name,
        sex=getattr(state, "sex", None) or "",
        faction=getattr(state, "faction", None) or "",
        titles=getattr(state, "titles", None),
    )]
    appearance = build_appearance_block(state)
    if appearance:
        parts.append(appearance)
    return "\n".join(parts)


def _build_relationship_block(state: Any, snapshot: Any) -> str:
    """关系块：好感度/关系等级一行＋长期记忆（基础印象＋关键记录）。"""
    lines = [
        f"当前好感度 {getattr(state, 'favorability', 0)} / 100，"
        f"关系「{getattr(state, 'relationship_level', '') or '未知'}」。"
    ]
    if snapshot is not None and not snapshot.is_empty:
        if snapshot.base:
            lines.append(f"你对他的基础印象：{snapshot.base}")
        if snapshot.facts:
            lines.append("你记得的关于他的事：")
            lines.extend(f"{fact.entry_no}. {fact.content}" for fact in snapshot.facts)
    return "\n".join(lines)


class DialogueWorker:
    """对话修改后台 worker：有界队列（满则丢弃并记日志）、串行执行、总时限兜底。"""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[DialogueJob] = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._task: asyncio.Task | None = None
        self._running = False

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

    def submit(self, job: DialogueJob) -> bool:
        """非阻塞入队（fire-and-forget）；队列满则丢弃本次委派并记日志。"""
        self.start()
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            logger.warning("对话修改队列已满，丢弃对「%s」的本次委派", job.npc_name)
            return False

    @property
    def running(self) -> bool:
        """是否有委派正在执行（注册闸查询口径：在跑＝正在执行，排队不算）。"""
        return self._running

    async def run(self) -> None:
        """单 worker 串行消费（并发上限 1）；任何失败只记日志，绝不外抛。"""
        while True:
            job = await self._queue.get()
            self._running = True
            try:
                await asyncio.wait_for(self._execute(job), timeout=DIALOGUE_JOB_TIMEOUT_S)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "对话修改委派执行失败（已放弃本次）: npc=%s trigger=%s: %s",
                    job.npc_name, job.source_trigger, exc, exc_info=True,
                )
            finally:
                self._running = False
                self._queue.task_done()

    # ------------------------------------------------------------------

    async def _execute(self, job: DialogueJob) -> None:
        """装配材料 → 启动 DialogueRunner → 等到 final/failed（只记日志）。"""
        from services.integration.state import get_integration_state
        from services.npc.manager import get_npc_manager

        npc_manager = await get_npc_manager()
        state = await npc_manager.get(job.npc_name)
        store = None
        snapshot = None
        try:
            from services.memory.store import get_memory_store

            store = get_memory_store()
            snapshot = await load_snapshot(
                store, slot_key=get_integration_state().slot_key, npc=job.npc_name,
            )
        except Exception as exc:
            logger.warning("对话修改委派读取长期记忆失败（按无记忆继续）: %s", exc)

        try:
            from services.game_data.registry import get_game_data_registry

            game_data = get_game_data_registry()
        except Exception as exc:
            logger.warning("对话修改委派读取游戏数据失败（工具按降级口径继续）: %s", exc)
            game_data = None

        handle = DialogueRunner.launch(
            llm=LLMClient.for_config(job.llm_config),
            registry=_tool_registry(),
            npc_name=job.npc_name,
            profile_block=_build_profile_block(job.npc_name, state),
            relationship_block=_build_relationship_block(state, snapshot),
            source_trigger=job.source_trigger,
            reason=job.reason,
            instruction=job.instruction,
            source_task_id=job.source_task_id,
            task_block=job.task_block,
            publish_block=job.publish_block,
            game_data=game_data,
        )
        event = await handle.wait_final(timeout=DIALOGUE_JOB_TIMEOUT_S)
        if event.kind == "failed":
            logger.error(
                "对话修改子 Agent 失败: npc=%s reason=%s",
                job.npc_name, event.payload.get("reason", ""),
            )
            return
        logger.info(
            "对话修改子 Agent 完成: npc=%s changed=%s payload=%.200s",
            job.npc_name, event.payload.get("changed"), str(event.payload),
        )


def _tool_registry():
    from services.tools.base import get_tool_registry

    return get_tool_registry()


_DIALOGUE_WORKER: DialogueWorker | None = None


def get_dialogue_worker() -> DialogueWorker:
    """全局单例（startup 初始化；未初始化时懒加载兜底）。"""
    global _DIALOGUE_WORKER
    if _DIALOGUE_WORKER is None:
        _DIALOGUE_WORKER = DialogueWorker()
    return _DIALOGUE_WORKER


def set_dialogue_worker(worker: DialogueWorker | None) -> None:
    global _DIALOGUE_WORKER
    _DIALOGUE_WORKER = worker


def is_dialogue_agent_running() -> bool:
    """注册闸（下一批用）：后台是否有对话修改子 Agent 正在执行。"""
    worker = _DIALOGUE_WORKER
    return bool(worker is not None and worker.running)
