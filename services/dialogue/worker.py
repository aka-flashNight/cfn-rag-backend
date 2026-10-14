"""对话修改子 Agent 的后台串行 worker（阶段四 B4-2，§4.0.1 独立链路）。

- asyncio 有界队列＋单 worker 串行执行（fire-and-forget），写法对齐
  ``services/memory/summarize.py`` 的 SummaryWorker；
- **绝不阻塞、绝不打断前台聊天**：submit 非阻塞入队，执行期间任何异常只记日志，
  不向主链路抛；
- 执行期间置全局 running 标志（``is_dialogue_agent_running()``）——下一批的注册闸
  用它实现「后台有子 Agent 在跑就完全不注册」（§4.0.4）；
- **总时限到＝连子 Agent 一起收掉**（``_execute`` 的 finally）：子 Agent 是独立任务，
  只取消等待方会留下孤儿在「已放弃本次」之后继续落盘，且 running 已复位＝串行上限
  与注册闸一起被打穿（同存档两个写者后写抹掉先写，静默丢内容）。
- 状态回写接手受理侧：``dialogue_jobs`` 里受理侧只记到 delegated，之后由这里推
  generating → written / skipped / failed，排查时能看出卡在哪一步（回写失败只记日志）。

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
from services.memory.store import (
    DIALOGUE_JOB_FAILED,
    DIALOGUE_JOB_GENERATING,
    DIALOGUE_JOB_SKIPPED,
    DIALOGUE_JOB_WRITTEN,
)
from services.orchestrator.prompts import build_appearance_block, format_npc_role_tagline
from services.subagents.dialogue_runner import DialogueRunner

logger = logging.getLogger(__name__)

#: 单条委派的总时限：三轮 × 每轮 300 秒。口径依据＝LLM 层单次调用最坏合法耗时
#: ≈241 秒（`httpx` 读超时 120 秒 + 网络错误退避重试一次 120 秒），本流程的输出天然长
#: （查询一轮可回灌上万字符，修改一轮要把整组台词原样抄回），故 300 秒不是保险丝而是
#: 会被正常慢响应撞开的量级；900 秒内仍未拿到终态＝真出问题，此时才放弃。
DIALOGUE_JOB_TIMEOUT_S = 900
_QUEUE_MAX_SIZE = 16


@dataclass
class DialogueJob:
    """一次对话修改委派（触发材料由调用方给全；NPC 资料执行当刻自装）。"""

    npc_name: str
    llm_config: LLMConfig
    #: dialogue_jobs 里的记录 id（状态回写用；空＝不落库的直连委派，测试与手工调用）
    job_id: str = ""
    #: 受理当刻绑定的存档槽位：整条链路（长期记忆、覆盖层读写）都钉在它上面
    slot_key: str = ""
    #: 触发类型：关系变化 / 态度变化 / 关键信息 / 重要任务（§4.0.4 必传）
    source_trigger: str = ""
    #: 发生了什么（相关事实）
    reason: str = ""
    #: 重写指示（触发一发布任务时挂的指示；其余触发可空，由模型自行判断）
    instruction: str = ""
    #: 重要任务必传任务 id（只有首次传有效，后续静默跳过——由受理侧的硬幂等保证）
    source_task_id: int | None = None
    #: 任务详情与对话（如有）
    task_block: str = ""
    #: 发布那场对话的摘要＋摘要后精确对话（如有）
    publish_block: str = ""


async def _set_status(job_id: str, status: str, *, detail: str = "") -> None:
    """把委派状态推进到 dialogue_jobs（无 job_id＝不落库的直连委派，跳过）。

    回写失败只记日志：状态是给人事后排查用的，绝不能因为它把委派本身弄崩。
    """
    if not job_id:
        return
    try:
        from services.memory.store import get_memory_store

        await get_memory_store().dialogue_job_set_status(job_id, status, detail=detail)
    except Exception as exc:
        logger.warning("对话修改委派状态回写失败（不影响本次执行）: job=%s %s", job_id, exc)


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
            except asyncio.TimeoutError:
                # 总时限到：_execute 被取消，而它在 finally 里已经把子 Agent 任务收掉了
                logger.error(
                    "对话修改委派超过总上限 %d 秒，已放弃本次: npc=%s trigger=%s",
                    DIALOGUE_JOB_TIMEOUT_S, job.npc_name, job.source_trigger,
                )
                await _set_status(
                    job.job_id, DIALOGUE_JOB_FAILED,
                    detail=f"超过总上限 {DIALOGUE_JOB_TIMEOUT_S} 秒，已放弃",
                )
            except Exception as exc:
                logger.error(
                    "对话修改委派执行失败（已放弃本次）: npc=%s trigger=%s: %s",
                    job.npc_name, job.source_trigger, exc, exc_info=True,
                )
                await _set_status(job.job_id, DIALOGUE_JOB_FAILED, detail=f"执行异常：{exc}")
            finally:
                self._running = False
                self._queue.task_done()

    # ------------------------------------------------------------------

    async def _execute(self, job: DialogueJob) -> None:
        """装配材料 → 启动 DialogueRunner → 等到 final/failed，并把终态回写。"""
        from services.integration.state import get_integration_state
        from services.npc.manager import get_npc_manager

        bound = get_integration_state().slot_key
        if job.slot_key and bound != job.slot_key:
            # 排队期间玩家换了存档：材料（长期记忆、任务详情、发布对话）都是按原存档
            # 装配的，此时继续跑等于拿 A 档的印象改 A 档的对话却看着 B 档的进度。
            detail = f"执行时已换存档（委派 {job.slot_key}，当前 {bound}），放弃本次"
            logger.warning("对话修改委派放弃: npc=%s %s", job.npc_name, detail)
            await _set_status(job.job_id, DIALOGUE_JOB_FAILED, detail=detail)
            return
        slot_key = job.slot_key or bound
        await _set_status(job.job_id, DIALOGUE_JOB_GENERATING)

        npc_manager = await get_npc_manager()
        state = await npc_manager.get(job.npc_name)
        store = None
        snapshot = None
        try:
            from services.memory.store import get_memory_store

            store = get_memory_store()
            snapshot = await load_snapshot(store, slot_key=slot_key, npc=job.npc_name)
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
            slot_key=job.slot_key,
        )
        # 任何退出路径都必须收掉子 Agent 任务：总时限触发时本协程是被取消的一方，
        # finally 照样执行。否则孤儿会在「已放弃本次」之后继续跑完三轮并落盘，
        # 同时 running 标志已复位＝串行上限与注册闸两个口径一起被打穿。
        try:
            event = await handle.wait_final(timeout=DIALOGUE_JOB_TIMEOUT_S)
        finally:
            handle.cancel()
        if event.kind == "failed":
            reason = str(event.payload.get("reason", ""))
            logger.error("对话修改子 Agent 失败: npc=%s reason=%s", job.npc_name, reason)
            await _set_status(job.job_id, DIALOGUE_JOB_FAILED, detail=f"子 Agent 失败：{reason}")
            return
        if event.payload.get("changed"):
            applied = event.payload.get("applied") or []
            logger.info(
                "对话修改子 Agent 落盘: npc=%s applied=%d payload=%.200s",
                job.npc_name, len(applied), str(event.payload),
            )
            await _set_status(
                job.job_id, DIALOGUE_JOB_WRITTEN, detail=f"已修改 {len(applied)} 组",
            )
            return
        note = str(event.payload.get("note") or "").strip()
        if event.payload.get("exhausted"):
            note = note or "三轮用尽仍无落盘"
        logger.info("对话修改子 Agent 判断不动: npc=%s %.80s", job.npc_name, note)
        await _set_status(job.job_id, DIALOGUE_JOB_SKIPPED, detail=note or "子 Agent 判断不需要改")


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
