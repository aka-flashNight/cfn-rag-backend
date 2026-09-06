"""TurnOrchestrator：回合状态机（对应 docs/v3-developer/03 §2，SSE 契约见 events.py）。

一个玩家消息 = 一个 Turn（一个 orchestrator 实例）：

    IDLE → BUILD_CONTEXT → BURST_STREAMING（调用 #1：meta 行 → 正文流）
         → SYNC_ACTION（confirm/cancel）或 启动子 Agent
         → MERGE_WAIT（宽限/中间结果/汇合 #2）→ POST_PROCESS → DONE

硬约束：
- 情绪先于正文（meta 行解析瞬间发 meta 事件）；
- 中间结果只说一次、不报数字（merge.py）；
- confirm 失败丢弃已生成正文 → 1 次补救调用 → system_notice 真实原因；
- 同会话并发请求经会话级锁串行；新回合取消旧子 Agent（fire-and-steer）；
- LLM 预算：聊天主 Agent ≤3 次（#1 + #2a + #2b），TaskRunner ≤4 轮、SearchRunner ≤3 轮。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from services.agent_tools.draft_formatting import _detailed_draft_summary
from services.agent_tools.handlers import (
    execute_cancel_agent_task,
    execute_confirm_agent_task,
)
from services.llm import (
    ChatRequest,
    LLMClient,
    LLMConfig,
    Meta,
    split_meta_events,
)
from services.llm.errors import LLMError, is_tools_unsupported_error
from services.memory.store import MemoryStore, TaskDraftRow
from services.memory.summarize import SummaryRequest, get_summary_worker, should_summarize
from services.npc.manager import NPCManager
from services.orchestrator.context import TurnContext, assemble_context
from services.portraits import build_image_message_content
from services.orchestrator.events import (
    SSEEvent,
    accumulate_usage,
    agent_status_event,
    content_event,
    done_event,
    error_event,
    meta_event,
    system_notice_event,
    tool_status_event,
)
from services.orchestrator.merge import MergeCoordinator
from services.orchestrator.prompts import (
    CONFIRM_ARGS_SYSTEM,
    build_confirm_args_user_prompt,
    build_player_message,
    build_merge_user_prompt,
    build_static_system,
    build_user_shared_core,
    fallback_confirm_args,
    parse_confirm_args_json as parse_confirm_args,
)
from services.subagents import (
    SearchRunner,
    SessionSubagents,
    TaskRunner,
    get_session_subagents,
)
from services.tools.base import ToolContext, ToolRegistry, get_tool_registry
from services.agent_tools.schemas import TASK_TYPES

logger = logging.getLogger(__name__)

_PREPARE_STATUS_KEYS = ("collectable_items", "stage_list", "reward_item_candidates",
                        "holdable_items", "equipment_items", "special_items",
                        "npc_list", "challenge_targets", "stage_loot_list")


def _prepared_candidates_empty(prepared_json: str) -> bool:
    """prepare 结果是否没有任何候选（task_type 无专属字段时视为空）。"""
    try:
        data = json.loads(prepared_json) if prepared_json else None
    except json.JSONDecodeError:
        return True
    if not isinstance(data, dict):
        return True
    return not any(isinstance(data.get(k), list) and data.get(k) for k in _PREPARE_STATUS_KEYS)


# ---------------------------------------------------------------------------
# 依赖容器
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorDeps:
    """可注入依赖（生产用全局单例；测试注入 Fake）。"""

    memory: MemoryStore
    npc_manager: NPCManager
    registry: ToolRegistry
    game_data: Any
    engine: Any = None                       # RetrievalEngine（None 时检索降级）
    llm_factory: Callable[[LLMConfig], LLMClient] = LLMClient.for_config
    summary_llm_factory: Callable[[LLMConfig], LLMClient] = LLMClient.for_config
    merge_grace_ms: int = 1500
    subagent_timeout_s: int = 120
    task_max_rounds: int = 4
    search_max_rounds: int = 3
    draft_keep_turns: int = 3
    history_limit: int = 20
    enable_summary: bool = True


def default_deps() -> OrchestratorDeps:
    """生产依赖（全部来自 startup 初始化的全局单例）。"""
    from services.memory.store import get_memory_store
    from services.npc.manager import get_npc_manager

    from core.config import get_settings

    settings = get_settings()
    engine = None
    try:
        from services.retrieval import get_retrieval_engine

        engine = get_retrieval_engine()
    except Exception:  # pragma: no cover
        logger.warning("检索引擎不可用，Tier-1 检索降级")
    game_data = None
    try:
        from services.game_data.registry import get_game_data_registry

        game_data = get_game_data_registry()
    except Exception:  # pragma: no cover
        logger.warning("游戏数据不可用，任务功能将受限")
    return OrchestratorDeps(
        memory=get_memory_store(),
        npc_manager=None,  # type: ignore[arg-type]  # 运行时经 get_npc_manager()
        registry=get_tool_registry(),
        game_data=game_data,
        engine=engine,
        merge_grace_ms=settings.merge_grace_ms,
        subagent_timeout_s=settings.subagent_timeout_s,
        task_max_rounds=settings.subagent_task_max_rounds,
        search_max_rounds=settings.subagent_search_max_rounds,
        draft_keep_turns=settings.draft_keep_turns,
    )


# ---------------------------------------------------------------------------
# 会话级锁（同会话两请求不交叉，03 §9.6）
# ---------------------------------------------------------------------------

_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


async def get_session_lock(session_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _SESSION_LOCKS[session_id] = lock
        return lock


def reset_session_locks() -> None:
    """测试用。"""
    _SESSION_LOCKS.clear()


# ---------------------------------------------------------------------------
# TurnOrchestrator
# ---------------------------------------------------------------------------

class TurnOrchestrator:
    """一个实例跑一个回合；run() 产出 SSE 事件流。"""

    def __init__(
        self,
        *,
        session_id: str,
        npc_name: str,
        query: str,
        player_identity: str = "",
        progress_stage: Optional[int] = None,
        current_emotion: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        send_image: bool = True,
        deps: Optional[OrchestratorDeps] = None,
    ) -> None:
        """send_image 仅是「允许带图」开关（默认允许）；是否真带图由 07 §4 规则
        在上下文装配时判定（vision Profile + 未被标记不支持 + 立绘资产可用）。"""
        self.session_id = session_id
        self.npc_name = npc_name
        self.query = query
        self.player_identity = player_identity
        self.progress_stage = progress_stage
        self.current_emotion = current_emotion
        self.llm_config = llm_config or LLMConfig()
        self.send_image = send_image
        self.deps = deps

        self._ctx: Optional[TurnContext] = None
        self._usage: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[SSEEvent]:
        deps = self._resolve_deps()
        if deps.npc_manager is None:
            from services.npc.manager import get_npc_manager

            deps.npc_manager = await get_npc_manager()  # type: ignore[assignment]
        if deps.game_data is None:
            try:
                from services.game_data.registry import get_game_data_registry

                deps.game_data = get_game_data_registry()
            except Exception:  # pragma: no cover
                logger.warning("游戏数据不可用，任务功能将受限")
        lock = await get_session_lock(self.session_id)
        async with lock:
            async for ev in self._run_locked(deps):
                yield ev

    def _resolve_deps(self) -> OrchestratorDeps:
        return self.deps if self.deps is not None else default_deps()

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    async def _run_locked(self, deps: OrchestratorDeps) -> AsyncIterator[SSEEvent]:
        memory = deps.memory
        self._usage = None
        collected_text: list[str] = []
        handle = None
        merge: Optional[MergeCoordinator] = None
        act_kind: Optional[str] = None
        draft_touched = False  # 本回合任务工具被触碰（confirm/cancel 成功也算）

        try:
            # ---- steer：取消同会话未完成的旧子 Agent（01 §4）----
            sess = await get_session_subagents(self.session_id)
            sess.cancel_unfinished()

            # ---- BUILD_CONTEXT ----
            ctx = await assemble_context(
                session_id=self.session_id,
                npc_name=self.npc_name,
                player_query=self.query,
                player_identity=self.player_identity,
                progress_stage=self.progress_stage,
                current_emotion=self.current_emotion,
                llm_config=self.llm_config,
                memory=memory,
                npc_manager=deps.npc_manager,
                game_data=deps.game_data,
                engine=deps.engine,
                history_limit=deps.history_limit,
                send_image=self.send_image,
            )
            self._ctx = ctx

            llm = deps.llm_factory(self.llm_config)

            # ---- BURST_STREAMING（调用 #1）----
            # system 分层（从稳定到易变）：世界观 → 输出规则/prepare 指南（跨 NPC）
            # → 单 NPC 扮演块（含 meta 协议）；user1 = 历史 → 会话态 → RAG（同回合
            # 多次调用共享）；user2 = 玩家当轮话（+立绘图片 part 在尾）。
            base_messages = self._build_base_messages(ctx)
            player_message = build_player_message(self.query)
            # 立绘图片只进聊天调用 #1 的末条 user 消息（文本 part 前、图片 part 后）；
            # base_messages 保持纯文本，merge/补救/confirm 参数等调用天然无图
            messages = [
                *base_messages,
                {"role": "user", "content": build_image_message_content(player_message, ctx.image_data_url)},
            ]

            # 聊天 Agent 唯一工具：prepare_task_context（01 §3 D4 修订：
            # 情绪/委派意图仍走 meta 行，数据准备类工具按业务注册）
            prepare_tool = deps.registry.get("prepare_task_context")
            chat_tools = [prepare_tool.to_openai_tool()] if prepare_tool is not None else None

            stream = llm.chat_stream(ChatRequest(
                messages=messages,
                tools=chat_tools,
                purpose="chat",
                send_image=self.send_image,
            ))
            try:
                meta, events = await split_meta_events(
                    stream, list(ctx.npc_state.emotions or ["普通"]),
                )
            except LLMError as exc:
                if chat_tools and is_tools_unsupported_error(exc):
                    # 平台不支持流式 tools：剥工具重试一次（聊天功能降级可用）
                    logger.warning("模型不支持流式 tools，剥离后重试: %s", exc)
                    stream = llm.chat_stream(ChatRequest(
                        messages=messages, tools=None, purpose="chat",
                        send_image=self.send_image,
                    ))
                    meta, events = await split_meta_events(
                        stream, list(ctx.npc_state.emotions or ["普通"]),
                    )
                else:
                    raise
            act = meta.act
            act_kind = act.kind if act is not None else None
            if act is None and ctx.pending_draft_row is not None:
                # 观测点：有草案时玩家常在「接/不接」，模型漏发 act 会导致草案静默过期，
                # 玩家侧只看到「过期的委托草案已取消」，必须留痕便于排查
                logger.warning(
                    "会话 %s：存在待确认草案但模型未声明 act（玩家消息: %.60s）——"
                    "若玩家在确认/拒绝，本次会按纯聊天处理并累计草案过期计数",
                    self.session_id, self.query,
                )
            tool_sink: list[dict[str, Any]] = []  # 流内模型发出的 tool_calls（prepare 用）

            # 情绪/好感先于正文：解析到 meta 即更新并发事件
            yield await self._emit_meta(meta, deps)

            if act is None:
                # ---- 纯聊天路径 ----
                async for ev in self._forward_body(events, collected_text, tool_sink=tool_sink):
                    yield ev
                if tool_sink:
                    logger.info("纯聊天轮出现 prepare 调用（act=null），忽略: %s",
                                [tc.get("function", {}).get("name") for tc in tool_sink])

            elif act.kind in ("task_confirm", "task_cancel"):
                # ---- SYNC_ACTION：缓冲正文，先执行后端操作（prepare 调用忽略）----
                buffered: list[str] = []
                await self._collect_body(events, buffered, tool_sink=tool_sink)
                spoken_text = "".join(buffered)
                tool_name = "confirm_agent_task" if act.kind == "task_confirm" else "cancel_agent_task"
                ui_hint = "发布委托" if act.kind == "task_confirm" else "取消委托"

                draft_row = ctx.pending_draft_row
                if act.kind == "task_confirm":
                    yield tool_status_event(tool_name, "running", ui_hint=ui_hint)
                    if draft_row is None:
                        yield tool_status_event(tool_name, "failed", ui_hint=ui_hint)
                        fail_reason = "当前没有待确认的委托草案"
                        yield system_notice_event(f"委托确认失败：{fail_reason}")
                        async for ev in self._remediation_reply(
                            llm, base_messages, meta, spoken_text, fail_reason,
                            collected_text,
                        ):
                            yield ev
                    else:
                        confirm_args = await self._generate_confirm_args(
                            llm, ctx, draft_row, deps.game_data,
                            base_messages=base_messages, spoken_text=spoken_text,
                            emotion=meta.emotion,
                        )
                        outcome = execute_confirm_agent_task(
                            confirm_args,
                            pending_draft=draft_row.draft,
                            npc_name=self.npc_name,
                            player_progress=ctx.progress_stage or 1,
                            npc_affinity=ctx.favorability,
                            game_data=deps.game_data,
                            rag_context_text=ctx.rag_context_text,
                        )
                        if outcome.payload.get("status") == "confirmed":
                            draft_touched = True
                            await memory.delete_draft(self.session_id, draft_id=draft_row.draft_id)
                            await memory.reset_rounds_without_task(self.session_id)
                            yield tool_status_event(tool_name, "success", ui_hint=ui_hint)
                            for chunk in buffered:
                                yield content_event(chunk)
                            collected_text.extend(buffered)
                            yield system_notice_event("委托已发布")
                        else:
                            # confirm 失败：丢弃已生成正文 → 补救调用（03 §4.2）
                            yield tool_status_event(tool_name, "failed", ui_hint=ui_hint)
                            fail_reason = self._extract_fail_reason(outcome.payload)
                            yield system_notice_event(f"委托确认失败：{fail_reason}")
                            async for ev in self._remediation_reply(
                                llm, base_messages, meta, spoken_text, fail_reason,
                                collected_text,
                            ):
                                yield ev
                else:  # task_cancel
                    yield tool_status_event(tool_name, "running", ui_hint=ui_hint)
                    if draft_row is None:
                        # 无草案可取消：不视为失败，人设地放行正文
                        yield tool_status_event(tool_name, "success", ui_hint=ui_hint)
                        for chunk in buffered:
                            yield content_event(chunk)
                        collected_text.extend(buffered)
                    else:
                        outcome = execute_cancel_agent_task(
                            {"draft_id": draft_row.draft_id},
                            pending_draft=draft_row.draft,
                        )
                        draft_touched = True
                        await memory.delete_draft(self.session_id, draft_id=draft_row.draft_id)
                        await memory.reset_rounds_without_task(self.session_id)
                        yield tool_status_event(tool_name, "success", ui_hint=ui_hint)
                        for chunk in buffered:
                            yield content_event(chunk)
                        collected_text.extend(buffered)
                        yield system_notice_event("委托已取消")

            else:
                # ---- 后台子 Agent（fire-and-steer），正文直发 ----
                # task_draft：prepare 由聊天 Agent 在流内调用，流结束后执行并直通任务 Agent
                if act.kind == "task_draft":
                    async for ev in self._forward_body(events, collected_text, tool_sink=tool_sink):
                        yield ev
                    prepared = await self._execute_prepare(tool_sink, ctx, deps)
                else:  # search / task_update 不需要 prepare
                    prepared = (None, None, "")
                    async for ev in self._forward_body(events, collected_text):
                        yield ev

                handle = self._launch_subagent(act, ctx, deps, prepared, spoken_text="".join(collected_text))
                sess.register(handle)

                # ---- MERGE_WAIT ----
                merge = MergeCoordinator(
                    handle=handle,
                    llm=llm,
                    base_messages=base_messages,
                    npc_name=self.npc_name,
                    player_query=self.query,
                    spoken_text="".join(collected_text),
                    emotion=meta.emotion,
                    grace_s=deps.merge_grace_ms / 1000.0,
                    timeout_s=float(deps.subagent_timeout_s),
                )
                async for ev in merge.stream():
                    yield ev
                outcome = merge.outcome
                self._usage = accumulate_usage(self._usage, outcome.usage)
                if outcome.status == "final" and outcome.agent == "task" and outcome.payload.get("draft"):
                    await memory.upsert_draft(
                        self.session_id,
                        outcome.payload["draft"],
                        bargain_count=int(outcome.payload.get("bargain_count") or 0),
                    )
                    notice = (
                        "任务草案已更新，等待确认"
                        if handle.kind == "task_update"
                        else "任务草案已拟定，等待确认"
                    )
                    yield system_notice_event(notice)

            # ---- POST_PROCESS ----
            # 记忆分段：被 agent 等待（拟定委托/检索）中断过的连续说话各存一条
            # assistant 记录 —— 首段正文（collected_text）一段，子 Agent 汇合后的
            # 每次说话爆发（过渡语 #2a / 汇合 #2b / FAIL_REPLY）各一段。前端历史
            # 渲染按消息分气泡，落库分段即历史气泡分段。
            reply_segments: list[str] = []
            first_segment = "".join(collected_text)
            if first_segment.strip():
                reply_segments.append(first_segment)
            if merge is not None:
                reply_segments.extend(
                    seg for seg in merge.outcome.reply_segments if seg.strip()
                )
            await memory.add_message(self.session_id, "user", self.query)
            for segment_text in reply_segments:
                await memory.add_message(self.session_id, "assistant", segment_text)

            if not draft_touched:
                if act_kind is None:
                    # 岔开话题：草案触碰计数 +1，达到上限自动过期取消（03 §4.2）
                    counter = await memory.increment_rounds_without_task(self.session_id)
                    if (
                        ctx.pending_draft_row is not None
                        and counter >= deps.draft_keep_turns
                    ):
                        await memory.delete_draft(
                            self.session_id, draft_id=ctx.pending_draft_row.draft_id,
                        )
                        await memory.reset_rounds_without_task(self.session_id)
                        yield system_notice_event("过期的委托草案已取消")
                elif act_kind in ("task_draft", "task_update"):
                    await memory.reset_rounds_without_task(self.session_id)

            await self._maybe_summarize(deps)

            yield done_event(self.session_id, self._usage)

        except asyncio.CancelledError:
            raise
        except LLMError as exc:
            logger.warning("回合 LLM 错误: %s", exc)
            yield error_event(
                code=type(exc).__name__,
                message=str(exc),
                retryable=exc.retryable,
            )
            yield done_event(self.session_id, self._usage)
        except Exception as exc:
            logger.exception("回合处理异常")
            yield error_event("internal_error", f"{type(exc).__name__}: {exc}", retryable=False)
            yield done_event(self.session_id, self._usage)

    # ------------------------------------------------------------------
    # 组装与转发
    # ------------------------------------------------------------------

    def _build_base_messages(self, ctx: TurnContext) -> list[dict[str, Any]]:
        """[system 静态, user shared core]（前缀缓存对齐；tail 与玩家话在末条 user）。"""
        system = build_static_system(
            npc_name=self.npc_name,
            state=ctx.npc_state,
            same_faction_npcs=ctx.same_faction_block,
            has_shop=ctx.has_shop,
            shop_reward_types=ctx.shop_reward_types,
            player_can_challenge=ctx.player_can_challenge,
            has_pending_draft=ctx.pending_draft_row is not None,
        )
        shared = build_user_shared_core(
            retrieved_context=ctx.rag_context_text,
            mentioned_npcs_str=ctx.mentioned_npcs_str,
            summary_str=self._summary_text(ctx),
            history_str=self._recent_dialogue_text(ctx, spoken_text=""),
            player_identity=self.player_identity,
            progress_desc=ctx.progress_desc,
            favorability=ctx.favorability,
            relationship_level=ctx.relationship_level,
            pending_draft_summary=ctx.pending_draft_summary,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": shared},
        ]

    def _summary_text(self, ctx: TurnContext) -> str:
        """早期对话滚动摘要（比近期对话稳定，放对话块之前）。"""
        summary = getattr(ctx, "summary", None)
        return f"【早期对话摘要】\n{summary}" if summary else ""

    async def _forward_body(
        self, events: AsyncIterator[Any], sink: list[str],
        *, tool_sink: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[SSEEvent]:
        """转发调用 #1 的正文流（content/usage；finish 时提取 tool_calls 后终止）。"""
        async for ev in events:
            if ev.kind == "content" and ev.text:
                sink.append(ev.text)
                yield content_event(ev.text)
            elif ev.kind == "usage" and ev.usage:
                self._usage = accumulate_usage(self._usage, ev.usage)
            elif ev.kind == "finish":
                if tool_sink is not None and ev.tool_calls:
                    tool_sink.extend(ev.tool_calls)
                return

    async def _collect_body(
        self, events: AsyncIterator[Any], sink: list[str],
        *, tool_sink: list[dict[str, Any]] | None = None,
    ) -> None:
        """仅收集不转发（SYNC_ACTION 的正文缓冲，成功后才放行）。"""
        async for ev in events:
            if ev.kind == "content" and ev.text:
                sink.append(ev.text)
            elif ev.kind == "usage" and ev.usage:
                self._usage = accumulate_usage(self._usage, ev.usage)
            elif ev.kind == "finish":
                if tool_sink is not None and ev.tool_calls:
                    tool_sink.extend(ev.tool_calls)
                return

    async def _emit_meta(self, meta: Meta, deps: OrchestratorDeps) -> SSEEvent:
        if meta.favorability_change:
            state = await deps.npc_manager.update_favorability(
                self.npc_name, meta.favorability_change,
            )
            self._ctx.npc_state = state
        else:
            state = self._ctx.npc_state
        return meta_event(
            emotion=meta.emotion,
            favorability_change=meta.favorability_change,
            favorability=state.favorability,
            relationship_level=state.relationship_level,
        )

    # ------------------------------------------------------------------
    # prepare 执行（聊天 Agent 的 tool_calls → 候选池直通任务 Agent）
    # ------------------------------------------------------------------

    async def _execute_prepare(
        self,
        tool_calls: list[dict[str, Any]],
        ctx: TurnContext,
        deps: OrchestratorDeps,
    ) -> tuple[Optional[str], Optional[dict[str, Any]], str]:
        """执行聊天 Agent 的 prepare_task_context 调用。

        返回 (候选池 JSON | None, 参数 dict | None, 失败原因)。
        失败不回炉交流 Agent：任务 Agent 将以 prepare_then_draft 模式自行重试（继承方向）。
        """
        calls = [
            tc for tc in (tool_calls or [])
            if isinstance(tc, dict)
            and str(tc.get("function", {}).get("name") or "") == "prepare_task_context"
        ]
        if not calls:
            logger.info("TurnOrchestrator: task_draft 但交流阶段未调用 prepare，任务 Agent 将重试")
            return None, None, "交流阶段未调用 prepare_task_context"
        raw_args = calls[-1].get("function", {}).get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError):
            logger.warning("TurnOrchestrator: prepare 参数解析失败: %s", str(raw_args)[:200])
            return None, None, "prepare 参数解析失败"
        if not isinstance(args, dict):
            return None, None, "prepare 参数格式错误"
        task_type = str(args.get("task_type") or "").strip()
        if task_type not in TASK_TYPES:
            logger.warning("TurnOrchestrator: prepare 的 task_type 非法: %s", task_type)
            return None, args, f"prepare 的 task_type 非法：{task_type}"

        outcome = await deps.registry.dispatch("prepare_task_context", args, ToolContext(
            npc_name=self.npc_name,
            npc_faction=ctx.npc_state.faction or "",
            npc_challenge=ctx.npc_state.challenge,
            player_progress=ctx.progress_stage or 1,
            npc_affinity=ctx.favorability,
            npc_states=ctx.npc_states,
            game_data=deps.game_data,
        ))
        try:
            payload = json.loads(outcome.result_json)
            status = payload.get("status") if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            status = "unreadable"
        candidates_empty = _prepared_candidates_empty(outcome.result_json)
        logger.info(
            "TurnOrchestrator prepare 执行完成: task_type=%s status=%s 候选空=%s",
            task_type, status, candidates_empty,
        )
        if candidates_empty:
            return None, args, f"prepare 候选池为空（task_type={task_type}）"
        return outcome.result_json, args, ""

    def _recent_dialogue_text(self, ctx: TurnContext, spoken_text: str) -> str:
        """任务 Agent 的最近 5 轮对话（memory 近 4 轮 + 本轮已流出正文，含过渡话）。"""
        lines: list[str] = []
        for m in ctx.history[-8:]:
            lines.append(f"玩家：{m.content}" if m.role == "user" else f"{self.npc_name}：{m.content}")
        if spoken_text.strip():
            lines.append(f"{self.npc_name}：{spoken_text.strip()}")
        return "\n".join(lines) or "（暂无对话记录）"

    # ------------------------------------------------------------------
    # 同步动作
    # ------------------------------------------------------------------

    async def _generate_confirm_args(
        self,
        llm: LLMClient,
        ctx: TurnContext,
        draft_row: TaskDraftRow,
        game_data: Any,
        *,
        base_messages: list[dict[str, Any]],
        spoken_text: str = "",
        emotion: str = "",
    ) -> dict[str, Any]:
        """confirm 发布文本（title/description/接取与完成对话）生成。

        与聊天 Agent **同源前缀**（世界观/扮演/RAG/会话态/近期历史），追加本轮
        过渡话与草案详情——保证生成的对话贴合对话脉络。单次、无工具、宽松归一化；
        仅当 LLM 调用整体失败或解析不出 JSON 时回退模板（优先保证发布流程可用）。
        """
        draft = draft_row.draft
        summary = _detailed_draft_summary(
            draft, game_data, rag_context_text=ctx.rag_context_text,
        )
        user_prompt = (
            CONFIRM_ARGS_SYSTEM + "\n\n"
            + build_confirm_args_user_prompt(
                draft_summary=summary,
                npc_name=self.npc_name,
                spoken_text=spoken_text,
                emotion=emotion,
            )
        )
        args: dict[str, Any] | None = None
        try:
            result = await llm.chat(ChatRequest(
                messages=[*base_messages, {"role": "user", "content": user_prompt}],
                purpose="subagent",
                max_tokens=1600,
            ))
            self._usage = accumulate_usage(self._usage, result.usage)
            args = self._sanitize_confirm_args(parse_confirm_args(result.content))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("confirm 发布文本生成失败，使用模板兜底: %s", exc)
        if not args:
            args = fallback_confirm_args(draft, self.npc_name)
        args["draft_id"] = draft_row.draft_id
        return args

    def _sanitize_confirm_args(self, args: dict[str, Any] | None) -> dict[str, Any] | None:
        """宽松归一化（不硬打回）：字符串化 + 空对话补占位，优先保住 LLM 的产出。"""
        if not isinstance(args, dict):
            return None
        out: dict[str, Any] = {}
        title = args.get("title")
        desc = args.get("description")
        out["title"] = title.strip() if isinstance(title, str) else ""
        out["description"] = desc.strip() if isinstance(desc, str) else ""

        def _dlg(key: str, fallback_text: str) -> list[dict[str, Any]]:
            rows = args.get(key)
            cleaned: list[dict[str, Any]] = []
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    text = str(r.get("text") or "").strip()
                    if not text:
                        continue
                    cleaned.append({
                        "name": str(r.get("name") or self.npc_name).strip() or self.npc_name,
                        "title": str(r.get("title") or "").strip(),
                        "emotion": str(r.get("emotion") or "").strip(),
                        "text": text,
                    })
            if not cleaned:
                # 对话缺失不回炉重生成：补一条占位，保证发布流程可用
                cleaned = [{"name": self.npc_name, "title": "", "emotion": "", "text": fallback_text}]
            return cleaned

        out["get_dialogue"] = _dlg("get_dialogue", "这份委托就交给你了。")
        out["finish_dialogue"] = _dlg("finish_dialogue", "干得漂亮，报酬一分不少。")
        return out

    @staticmethod
    def _extract_fail_reason(payload: dict[str, Any]) -> str:
        """confirm 失败的结构化原因（system_notice 用真实原因）。"""
        issues = payload.get("issues") or []
        if issues:
            first = issues[0] if isinstance(issues[0], dict) else {}
            return str(first.get("message") or "草案校验未通过")
        message = payload.get("message")
        return str(message or "未知原因")

    async def _remediation_reply(
        self,
        llm: LLMClient,
        base_messages: list[dict[str, Any]],
        meta: Meta,
        discarded_text: str,
        reason: str,
        sink: list[str],
    ) -> AsyncIterator[SSEEvent]:
        """confirm 失败补救调用：让 NPC 解释（丢弃已生成正文，03 §4.2）。"""
        _ = discarded_text
        result_block = (
            f"【真实原因（对玩家只说人话，不要提系统/校验等词汇）】{reason}"
        )
        prompt = build_merge_user_prompt(
            npc_name=self.npc_name,
            player_query=self.query,
            spoken_text="",  # 已丢弃
            emotion=meta.emotion,
            result_block=result_block,
            instruction=(
                "刚才那份委托出了点问题，暂时给不了玩家。用 1~3 句话以你的口吻解释并自然收场"
                "（如「这份委托刚才出了点问题……改天再谈」），不要提任何系统词汇。"
            ),
        )
        before = len(sink)
        async for ev in self._stream_simple(llm, base_messages, prompt, sink):
            yield ev
        if len(sink) == before:
            text = "这份委托刚才出了点问题，改天再谈吧。"
            sink.append(text)
            yield content_event(text)

    async def _stream_simple(
        self,
        llm: LLMClient,
        base_messages: list[dict[str, Any]],
        user_prompt: str,
        sink: list[str],
    ) -> AsyncIterator[SSEEvent]:
        messages = [*base_messages, {"role": "user", "content": user_prompt}]
        try:
            stream = llm.chat_stream(ChatRequest(
                messages=messages, purpose="chat", send_image=False,
            ))
            _meta, events = await split_meta_events(stream, [])
            async for ev in events:
                if ev.kind == "content" and ev.text:
                    sink.append(ev.text)
                    yield content_event(ev.text)
                elif ev.kind == "usage" and ev.usage:
                    self._usage = accumulate_usage(self._usage, ev.usage)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("流式回复调用失败: %s", exc)

    # ------------------------------------------------------------------
    # 子 Agent 启动
    # ------------------------------------------------------------------

    @staticmethod
    def _direction_from_prepared(prepared_args: Optional[dict[str, Any]]) -> str:
        """任务方向 = 交流 Agent prepare 参数（结构化委派）的自然语言还原。"""
        if not prepared_args:
            return ""
        parts: list[str] = []
        task_type = str(prepared_args.get("task_type") or "").strip()
        if task_type:
            parts.append(f"任务类型：{task_type}")
        rt = prepared_args.get("reward_types")
        if isinstance(rt, dict):
            reg = "、".join(rt.get("regular") or [])
            opt = "、".join(rt.get("optional") or [])
            if reg or opt:
                parts.append(f"奖励方向：常规[{reg or '无'}] 可选[{opt or '无'}]")
        kws = prepared_args.get("requirement_keywords")
        if isinstance(kws, list) and kws:
            parts.append("目标关键词：" + "、".join(str(k) for k in kws))
        return "；".join(parts)

    def _launch_subagent(
        self,
        act: Any,
        ctx: TurnContext,
        deps: OrchestratorDeps,
        prepared: tuple[Optional[str], Optional[dict[str, Any]], str] = (None, None, ""),
        *,
        spoken_text: str = "",
    ) -> Any:
        progress = ctx.progress_stage or 1
        affinity = ctx.favorability
        npc_titles = list(ctx.npc_state.titles or [])
        npc_states = ctx.npc_states
        llm = deps.llm_factory(self.llm_config)
        prepared_context, prepared_args, prepare_error = prepared
        recent_dialogue = self._recent_dialogue_text(ctx, spoken_text)

        if act.kind == "search":
            return SearchRunner.launch(
                llm=llm,
                registry=deps.registry,
                query=act.query,
                player_query=self.query,
                npc_name=self.npc_name,
                npc_faction=ctx.npc_state.faction or "",
                npc_titles=npc_titles,
                npc_challenge=ctx.npc_state.challenge,
                player_progress=progress,
                npc_affinity=affinity,
                npc_states=npc_states,
                game_data=deps.game_data,
                retrieve_fn=ctx.retrieve_fn,
                max_rounds=deps.search_max_rounds,
            )

        if act.kind == "task_update":
            draft_row = ctx.pending_draft_row
            if draft_row is None or not isinstance(draft_row.draft, dict) or not draft_row.draft:
                # 无草案可更新：降级为按备注重拟（模型错误委派的容错）
                return TaskRunner.launch(
                    kind="task_draft",
                    llm=llm,
                    registry=deps.registry,
                    direction=act.note or self.query,
                    reward_hint="",
                    note="",
                    prepared_context=None,
                    prepare_error="以 task_update 委派但无待修改草案，按新任务重拟",
                    recent_dialogue=recent_dialogue,
                    npc_name=self.npc_name,
                    npc_faction=ctx.npc_state.faction or "",
                    npc_titles=npc_titles,
                    npc_challenge=ctx.npc_state.challenge,
                    player_progress=progress,
                    progress_desc=ctx.progress_desc,
                    npc_affinity=affinity,
                    npc_states=npc_states,
                    pending_draft=None,
                    bargain_count=0,
                    draft_commit_valid=False,
                    game_data=deps.game_data,
                    rag_context_text=ctx.rag_context_text,
                    retrieve_fn=ctx.retrieve_fn,
                    max_rounds=deps.task_max_rounds,
                )
            return TaskRunner.launch(
                kind="task_update",
                llm=llm,
                registry=deps.registry,
                direction="",
                reward_hint="",
                note=act.note,
                recent_dialogue=recent_dialogue,
                npc_name=self.npc_name,
                npc_faction=ctx.npc_state.faction or "",
                npc_titles=npc_titles,
                npc_challenge=ctx.npc_state.challenge,
                player_progress=progress,
                progress_desc=ctx.progress_desc,
                npc_affinity=affinity,
                npc_states=npc_states,
                pending_draft=draft_row.draft,
                bargain_count=draft_row.bargain_count,
                draft_commit_valid=True,
                game_data=deps.game_data,
                rag_context_text=ctx.rag_context_text,
                retrieve_fn=ctx.retrieve_fn,
                max_rounds=deps.task_max_rounds,
            )

        # task_draft：方向 = prepare 参数还原（交流 Agent 的结构化委派）；
        # prepare 失败 → 任务 Agent 以 prepare_then_draft 模式自行重试（继承方向）
        direction = self._direction_from_prepared(prepared_args) or self.query
        return TaskRunner.launch(
            kind="task_draft",
            llm=llm,
            registry=deps.registry,
            direction=direction,
            reward_hint="",
            note="",
            prepared_context=prepared_context,
            prepare_error=prepare_error,
            recent_dialogue=recent_dialogue,
            npc_name=self.npc_name,
            npc_faction=ctx.npc_state.faction or "",
            npc_titles=npc_titles,
            npc_challenge=ctx.npc_state.challenge,
            player_progress=progress,
            progress_desc=ctx.progress_desc,
            npc_affinity=affinity,
            npc_states=npc_states,
            pending_draft=None,
            bargain_count=0,
            draft_commit_valid=False,
            game_data=deps.game_data,
            rag_context_text=ctx.rag_context_text,
            retrieve_fn=ctx.retrieve_fn,
            max_rounds=deps.task_max_rounds,
        )

    # ------------------------------------------------------------------
    # 摘要触发
    # ------------------------------------------------------------------

    async def _maybe_summarize(self, deps: OrchestratorDeps) -> None:
        if not deps.enable_summary:
            return
        try:
            count = await deps.memory.count_messages(self.session_id)
            if not should_summarize(count):
                return
            worker = get_summary_worker()
            if worker is None:
                return
            worker.submit(SummaryRequest(
                session_id=self.session_id,
                npc_name=self.npc_name,
                llm_config=self.llm_config,
            ))
        except Exception as exc:  # 摘要不阻断回合
            logger.warning("摘要提交失败（忽略）: %s", exc)
