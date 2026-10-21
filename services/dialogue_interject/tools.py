"""日常插话的工具接线（B2）。

复用终端的路由工具、任务子 Agent、检索子 Agent 与长期记忆工具；只把正文增量
交给 JSONL 解析器，工具调用参数与结果绝不进入台词/session。

一次回合最多一个路由工具（与终端一致）：工具轮不输出台词；工具结果回来后
再做一次无工具的 LLM 调用，只输出最终 JSONL 对话与结束语。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from services.agent_tools.draft_formatting import _detailed_draft_summary
from services.agent_tools.handlers import (
    execute_cancel_agent_task,
    execute_confirm_agent_task,
)
from services.dialogue.delegation import (
    DELEGATE_TOOL_NAME,
    apply_delegate_calls,
    delegate_tool_or_none,
)
from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.prompts import build_daily_messages
from services.dialogue_interject.schemas import InterjectRequest
from services.llm import ChatRequest, LLMClient, split_meta_events
from services.llm.meta import DEFAULT_EMOTION
from services.memory.long_term import (
    CHAT_TOOL_NAME,
    apply_chat_tool,
    build_chat_tool,
    tool_call_name,
)
from services.orchestrator.prompts import (
    build_interim_prompt_parts,
    build_merge_user_prompt,
    fallback_confirm_args,
)
from services.orchestrator.search_prefetch import (
    build_prefetch_result_block,
    run_search_prefetch,
)
from services.orchestrator.turn import Route, _parse_route, generate_confirm_args
from services.subagents.search_runner import SearchRunner
from services.subagents.task_runner import TaskRunner
from services.tools.base import ROUTE_TOOLS_DRAFT, ROUTE_TOOLS_NEW
from services.tools.route.route_task_draft import build_description

logger = logging.getLogger(__name__)

#: 工具执行超时（秒）；任务 Agent 自身还有轮数上限
TOOL_TIMEOUT_S = 120.0
#: 过渡话单独限时：生成失败/超时只是少一句填充，不得拖住草案终态
_TRANSITION_TIMEOUT_S = 30.0


class DailyToolError(RuntimeError):
    """工具执行失败/超时；带 code 供 SSE error 使用。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ToolExecutionResult:
    """一次工具轮的执行结果。"""

    requires_followup: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    result_block: str = ""
    instruction: str = ""
    route_kind: str = ""


@dataclass
class ToolProgress:
    """工具执行期间的旁路进度（不进入工具结果、不落库）。

    `ui_hint` 刷新等待占位；`line` 是已写好的过渡话，由流循环立刻作为一条
    add 行推给 Host，填住草案拟定期间的空窗。
    """

    ui_hint: str = ""
    line: str = ""
    emotion: str = ""


class DailyToolExecutor(Protocol):
    """`stream_daily_llm` 的工具执行接口（测试可注入假实现）。"""

    def tools(self) -> list[dict[str, Any]] | None: ...

    async def execute(
        self, tool_calls: list[dict[str, Any]], *, round_text: str = "",
    ) -> ToolExecutionResult: ...


class DailyToolRuntime:
    """B2 日常工具运行时：工具表 + 路由执行（复用终端实现）。"""

    def __init__(
        self,
        request: InterjectRequest,
        blocks: DailyBlocks,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        self.request = request
        self.blocks = blocks
        self.memory = blocks.memory
        self.registry = blocks.registry
        self.llm = llm
        #: 执行期间的旁路进度；由流循环并发消费（测试注入的执行器没有该队列）
        self.progress_queue: asyncio.Queue[ToolProgress] = asyncio.Queue()

    def _push_progress(self, progress: ToolProgress) -> None:
        try:
            self.progress_queue.put_nowait(progress)
        except Exception:  # pragma: no cover - 队列无界，仅防御已关闭的事件循环
            pass

    # ------------------------------------------------------------------
    # 工具表
    # ------------------------------------------------------------------

    def _route_names(self) -> list[str]:
        if self.blocks.task_publish_disabled:
            return ["route_search"]
        if self.blocks.pending_draft_row is not None:
            return list(ROUTE_TOOLS_DRAFT)
        return list(ROUTE_TOOLS_NEW)

    def tools(self) -> list[dict[str, Any]] | None:
        if self.registry is None:
            return None
        schemas = self.registry.get_openai_tools(names=self._route_names())
        # 可切磋时给 draft 路由补上「切磋/对战」判据：与聊天主 Agent 同一份描述，
        # 不可切磋时保持静态描述（不出现切磋字样）。
        if self.blocks.player_can_challenge:
            for schema in schemas:
                if schema.get("function", {}).get("name") == "route_task_draft":
                    schema["function"]["description"] = build_description(
                        player_can_challenge=True,
                    )
        if self.blocks.long_term is not None:
            schemas = [*schemas, build_chat_tool(self.blocks.long_term)]
        if self.blocks.delegate_enabled:
            # 注册闸每轮现查：后台 worker 在跑/未绑定存档时不出现（提示词也不提）
            try:
                delegate_tool = delegate_tool_or_none(include_task_fallback=False)
            except Exception as exc:  # 判定失败按不注册处理，不影响主线
                logger.warning("[interject] 委派工具注册判定失败: %s", exc)
                delegate_tool = None
            if delegate_tool is not None:
                schemas = [*schemas, delegate_tool]
        return schemas or None

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    async def execute(
        self, tool_calls: list[dict[str, Any]], *, round_text: str = "",
    ) -> ToolExecutionResult:
        calls = list(tool_calls or [])
        if not calls:
            return ToolExecutionResult()

        tool_messages: list[dict[str, Any]] = []
        memory_results: dict[int, str] = {}
        delegate_results: dict[int, str] = {}
        for index, call in enumerate(calls):
            name = tool_call_name(call)
            if name == CHAT_TOOL_NAME:
                try:
                    result = await apply_chat_tool(
                        self.memory,
                        call,
                        snapshot=self.blocks.long_term,
                        session_id=self.blocks.session_id,
                    )
                except Exception as exc:  # 记忆失败不阻断对话
                    logger.warning("[interject] 长期记忆工具失败: %s", exc)
                    result = None
                memory_results[index] = result or "未记录"
            elif name == DELEGATE_TOOL_NAME:
                if self.blocks.delegate_enabled:
                    delegate_results[index] = await self._run_delegate(call)
                else:
                    # 任务模式/注册闸关闭：委派不可用，按未知工具忽略（协议 4.1）
                    logger.warning(
                        "[interject] 未注册的委派工具调用，已忽略 mode=%s",
                        self.request.mode,
                    )

        route = _parse_route(calls)
        if route is None:
            if not delegate_results:
                # 只有记忆/未知工具：正文已随流推送，无需后续调用
                return ToolExecutionResult()
            # 只发了委派：仍回到无工具轮，让玩家听到正常台词（worker 在后台跑）
            for index, call in enumerate(calls):
                content = (
                    delegate_results.get(index)
                    or memory_results.get(index)
                    or "已忽略"
                )
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": content,
                })
            instruction = "继续用 JSONL 回应当前对话；不要提工具，也不要解释你做了什么处理。"
            return ToolExecutionResult(
                requires_followup=True,
                messages=[
                    {"role": "assistant", "content": round_text or "", "tool_calls": calls},
                    *tool_messages,
                    {"role": "user", "content": instruction},
                ],
                instruction=instruction,
                route_kind="delegate",
            )

        route_index = next(
            (
                index for index, call in enumerate(calls)
                if tool_call_name(call).startswith("route_")
            ),
            None,
        )
        if route.kind.startswith("task_") and self.blocks.task_publish_disabled:
            result_block = "【办理结果】你当前不能发布委托，这件事没有被受理。"
            instruction = "以角色口吻婉拒，不要编造任务内容，不要提工具，也不要解释办理过程。"
        else:
            try:
                result_block, instruction = await asyncio.wait_for(
                    self._execute_route(route), timeout=TOOL_TIMEOUT_S,
                )
            except Exception as exc:  # 工具失败/超时：记录后继续，不拖垮整轮对话
                logger.exception("[interject] 工具执行失败 route=%s", route.kind)
                result_block = "【办理结果】这件事办理失败或超时，这次未能完成。"
                instruction = (
                    "以角色口吻说明这次没办成，不要编造任务内容或数字，不要提工具，也不要解释办理过程。"
                )

        route_result = result_block
        for index, call in enumerate(calls):
            call_id = str(call.get("id") or "")
            if index == route_index:
                content = route_result
            elif index in memory_results:
                content = memory_results[index]
            elif index in delegate_results:
                content = delegate_results[index]
            else:
                content = "已忽略（一轮只处理一个路由工具）"
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })

        messages: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": round_text or "",
                "tool_calls": calls,
            },
            *tool_messages,
            {
                "role": "user",
                "content": f"{result_block}\n\n{instruction}".strip(),
            },
        ]
        return ToolExecutionResult(
            requires_followup=True,
            messages=messages,
            result_block=result_block,
            instruction=instruction,
            route_kind=route.kind,
        )

    async def _run_delegate(self, call: dict[str, Any]) -> str:
        """受理对话改写委派：只提交后台任务，绝不等 worker 跑完。"""
        try:
            results = await apply_delegate_calls(
                [call],
                npc_name=self.request.npc,
                session_id=self.blocks.session_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 委派失败只记日志，不阻断本轮对话
            logger.warning("[interject] 委派受理失败（忽略）: %s", exc)
            return "未受理"
        if not results:
            return "未受理"
        logger.info(
            "[interject] 委派已受理 session=%s detail=%s",
            self.blocks.session_id, results[0],
        )
        return "已记下"

    async def _execute_route(self, route: Route) -> tuple[str, str]:
        if route.kind == "search":
            return await self._run_search(route)
        if route.kind in ("task_draft", "task_update"):
            return await self._run_task_runner(route)
        if route.kind == "task_confirm":
            return await self._run_confirm(route)
        if route.kind == "task_cancel":
            return await self._run_cancel(route)
        return (
            "【办理结果】未能识别的工具调用，已忽略。",
            "用 JSONL 正常回应当前对话，不要提工具，也不要解释办理过程。",
        )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    async def _run_search(self, route: Route) -> tuple[str, str]:
        keywords = (route.keywords or route.query or self.request.player_input).strip()
        from services.orchestrator.context import forbidden_other_chars

        forbidden = forbidden_other_chars(self.blocks.npc_states)

        def _prefetch() -> Any:
            return run_search_prefetch(
                engine=self.blocks.engine,
                game_data=self.blocks.game_data,
                keywords=keywords,
                sources=route.sources,
                npc_name=self.request.npc,
                npc_titles=list(getattr(self.blocks.npc_state, "titles", []) or []),
                npc_faction=getattr(self.blocks.npc_state, "faction", None),
                forbidden=forbidden,
                existing_bundle=self.blocks.retrieval_bundle,
            )

        prefetch = None
        try:
            prefetch = await asyncio.to_thread(_prefetch)
        except Exception as exc:
            logger.warning("[interject] 检索预取失败，降级为空: %s", exc)

        text = prefetch.text if prefetch is not None else ""
        if route.deep_search and route.query.strip():
            try:
                text = await self._run_search_runner(route, prefetch)
            except DailyToolError:
                raise
            except Exception as exc:
                logger.warning("[interject] 深度检索失败，回落到预取结果: %s", exc)

        instruction = (
            "基于上面的资料，按任务 JSONL 格式（add/line/end 等）输出最终结果；"
            "不要提工具，也不要解释办理过程，不要输出动作描写。"
            if self.request.mode == "task"
            else "基于上面的资料，用 JSONL 输出最终台词；"
                 "不要提工具，也不要解释办理过程，不要输出动作描写。"
        )
        return build_prefetch_result_block(text), instruction

    async def _run_search_runner(self, route: Route, prefetch: Any) -> str:
        handle = SearchRunner.launch(
            llm=self.llm,
            registry=self.registry,
            query=route.query or route.keywords or self.request.player_input,
            player_query=self.request.player_input,
            npc_name=self.request.npc,
            npc_faction=getattr(self.blocks.npc_state, "faction", "") or "",
            npc_titles=list(getattr(self.blocks.npc_state, "titles", []) or []),
            npc_challenge=getattr(self.blocks.npc_state, "challenge", None),
            player_progress=self.blocks.progress_stage or 1,
            npc_affinity=getattr(self.blocks.npc_state, "favorability", 0),
            npc_states=self.blocks.npc_states,
            game_data=self.blocks.game_data,
            retrieve_fn=self.blocks.retrieve_fn,
            existing_material=self.blocks.rag_context_text,
            first_round_material=prefetch.text if prefetch is not None else "",
            max_rounds=_search_max_rounds(),
        )
        event = await self._consume_handle(
            handle,
            lambda ev: self._push_progress(ToolProgress(ui_hint=ev.ui_hint)),
        )
        if event.kind == "failed":
            return prefetch.text if prefetch is not None else ""
        payload = event.payload or {}
        return str(payload.get("findings") or payload.get("text") or "")

    # ------------------------------------------------------------------
    # 任务草案 / 修改
    # ------------------------------------------------------------------

    async def _run_task_runner(self, route: Route) -> tuple[str, str]:
        draft_row = self.blocks.pending_draft_row
        kind = "task_update" if route.kind == "task_update" and draft_row is not None else "task_draft"
        if route.kind == "task_update" and draft_row is None:
            logger.info("[interject] 无待修改草案，task_update 降级为新草案")

        handle = TaskRunner.launch(
            kind=kind,
            llm=self.llm,
            registry=self.registry,
            direction=route.note or self.request.player_input,
            player_query=self.request.player_input,
            note=route.note if kind == "task_update" else "",
            recent_dialogue=self.blocks.recent_dialogue,
            memory_summary=self.blocks.long_term_block,
            npc_name=self.request.npc,
            npc_faction=getattr(self.blocks.npc_state, "faction", "") or "",
            npc_titles=list(getattr(self.blocks.npc_state, "titles", []) or []),
            npc_challenge=getattr(self.blocks.npc_state, "challenge", None),
            player_progress=self.blocks.progress_stage or 1,
            player_level=self.blocks.player_level,
            progress_desc=self.blocks.progress_desc,
            npc_affinity=getattr(self.blocks.npc_state, "favorability", 0),
            npc_states=self.blocks.npc_states,
            pending_draft=draft_row.draft if kind == "task_update" and draft_row else None,
            bargain_count=draft_row.bargain_count if kind == "task_update" and draft_row else 0,
            draft_commit_valid=bool(draft_row is not None and kind == "task_update"),
            has_shop=self.blocks.has_shop,
            shop_reward_types=list(self.blocks.shop_reward_types),
            player_can_challenge=self.blocks.player_can_challenge,
            game_data=self.blocks.game_data,
            rag_context_text=self.blocks.rag_context_text,
            retrieve_fn=self.blocks.retrieve_fn,
            max_rounds=_task_max_rounds(),
        )
        transition_task: asyncio.Task | None = None
        transition_text = ""
        final_seen = False

        async def _transition_after(ev: Any) -> None:
            nonlocal transition_text
            try:
                text = await asyncio.wait_for(
                    self._build_transition_line(ev.vague_note, ev.pool_note),
                    timeout=_TRANSITION_TIMEOUT_S,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[interject] 过渡话生成失败/超时: %s", exc)
                return
            # 草案终态已到：这句话已经错过填窗时机，不再追加。
            if final_seen or not text:
                return
            transition_text = text
            self._push_progress(ToolProgress(line=text, emotion=DEFAULT_EMOTION))

        def _on_progress(ev: Any) -> None:
            nonlocal transition_task
            if ev.ui_hint:
                self._push_progress(ToolProgress(ui_hint=ev.ui_hint))
            if ev.kind == "intermediate" and transition_task is None:
                transition_task = asyncio.create_task(_transition_after(ev))

        try:
            event = await self._consume_handle(handle, _on_progress)
        finally:
            final_seen = True
            if transition_task is not None:
                if not transition_task.done():
                    transition_task.cancel()
                await asyncio.gather(transition_task, return_exceptions=True)

        if event.kind == "failed":
            reason = str((event.payload or {}).get("reason") or "未能拟定委托")
            instruction = (
                "以角色口吻说明这次没法准备委托，不要编造任务内容或数字，"
                "不要提工具，也不要解释办理过程。"
            )
            if transition_text:
                instruction += (
                    f"你已经对玩家说过过渡话「{transition_text}」，顺着它收场，不要重复这句。"
                )
            return f"【办理结果】{reason}", instruction

        payload = event.payload or {}
        draft = payload.get("draft")
        if isinstance(draft, dict) and draft:
            try:
                await self.memory.upsert_draft(
                    self.blocks.session_id,
                    draft,
                    bargain_count=int(payload.get("bargain_count") or 0),
                )
            except Exception as exc:
                logger.warning("[interject] 草案写库失败: %s", exc)
        summary = str(payload.get("draft_summary") or "").strip()
        if not summary and isinstance(draft, dict):
            try:
                summary = _detailed_draft_summary(
                    draft,
                    self.blocks.game_data,
                    rag_context_text=self.blocks.rag_context_text,
                    npc_states=self.blocks.npc_states,
                )
            except Exception:
                summary = ""
        title = "草案已按玩家条件调整" if kind == "task_update" else "委托草案已拟定"
        instruction = (
            "现在只输出最终确认对话（add）和结束语（closing）：讲清委托内容/要求/报酬并征询是否接取；"
            "不要输出“我去准备一下”之类过渡语；结束语表达“不接就算了/这活就搁着”，不要求固定措辞；"
            "玩家点头前不要说已发布，任务内容/数字以草案结果为准。"
        )
        if transition_text:
            instruction += (
                f"你已经对玩家说过过渡话「{transition_text}」，最终确认对话顺着它说，不要重复这句。"
            )
        return f"【办理结果：{title}，尚未发布】\n{summary or '（草案摘要不可用）'}", instruction

    async def _build_transition_line(self, vague_note: str, pool_note: str) -> str:
        """草案拟定期间的过渡话（终端 #2a 同源）：只点方向，不报具体内容。"""
        if self.llm is None:
            return ""
        result_block, instruction = build_interim_prompt_parts(vague_note, pool_note)
        prompt = build_merge_user_prompt(
            npc_name=self.request.npc,
            player_query=self.request.player_input,
            spoken_text="",
            emotion=DEFAULT_EMOTION,
            result_block=result_block,
            instruction=instruction,
        )
        # 插话 system 要求 JSONL；这一轮是单独的填充话，必须明确豁免格式。
        prompt += (
            "\n\n【过渡话专用】这一轮不要套用 JSONL 输出格式，只输出过渡话正文本身，"
            "不要输出任何 JSON、字段名、前缀或引号。"
        )
        messages = [
            *build_daily_messages(self.request, self.blocks),
            {"role": "user", "content": prompt},
        ]
        stream = self.llm.chat_stream(ChatRequest(
            messages=messages,
            purpose="chat",
            send_image=False,
        ))
        _meta, events = await split_meta_events(stream, self.blocks.npc_emotions)
        parts: list[str] = []
        async for ev in events:
            if ev.kind == "content" and ev.text:
                parts.append(ev.text)
        return _unwrap_transition_text("".join(parts))

    async def _consume_handle(self, handle: Any, on_event: Any) -> Any:
        """持续消费子 Agent 事件直到 final/failed；期间把 progress/intermediate 交给回调。

        超时/取消时确保后台任务被取消，与旧的 wait_final 语义一致。
        """
        from core.config import get_settings

        timeout = float(get_settings().subagent_timeout_s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                try:
                    ev = await handle.next_event(remaining)
                except asyncio.TimeoutError as exc:
                    raise DailyToolError(
                        "tool_timeout", f"子 Agent 超时（>{timeout:.0f}s）",
                    ) from exc
                if ev.kind in ("final", "failed"):
                    return ev
                on_event(ev)
        finally:
            if not handle.done:
                handle.cancel()

    # ------------------------------------------------------------------
    # 确认 / 取消
    # ------------------------------------------------------------------

    async def _run_confirm(self, route: Any = None) -> tuple[str, str]:
        draft_row = self.blocks.pending_draft_row
        if draft_row is None:
            return (
                "【办理结果】当前没有待确认的委托草案，无法发布。",
                "以角色口吻说明没有可确认的委托，不要编造任务内容。",
            )
        # 确认不跑子 Agent，没有子 Agent 的 ui_hint 可透传：优先用模型在调用里自带的
        # 进度提示（≤12 字），没给才回落到固定文案。游戏侧据此把等待文案从
        # 「拟定委托中」换成「派发委托中」这类说法。
        self._push_progress(ToolProgress(ui_hint=_route_hint(route, "派发委托中")))
        args, warnings, source = await self._build_confirm_args(draft_row)
        logger.info(
            "[interject] confirm 文案 dialogue=%s draft=%s source=%s warnings=%d",
            self.blocks.session_id, draft_row.draft_id, source, len(warnings),
        )
        outcome = execute_confirm_agent_task(
            args,
            pending_draft=draft_row.draft,
            npc_name=self.request.npc,
            player_progress=self.blocks.progress_stage or 1,
            npc_affinity=getattr(self.blocks.npc_state, "favorability", 0),
            game_data=self.blocks.game_data,
            rag_context_text=self.blocks.rag_context_text,
            npc_states=self.blocks.npc_states,
            publish_session_id=self.blocks.session_id,
            generation_warnings=warnings or None,
        )
        payload = outcome.payload or {}
        if payload.get("status") == "confirmed":
            try:
                await self.memory.delete_draft(
                    self.blocks.session_id, draft_id=draft_row.draft_id,
                )
            except Exception as exc:
                logger.warning("[interject] 草案删除失败: %s", exc)
            logger.info(
                "[interject] confirm applied session=%s draft=%s task=%s",
                self.blocks.session_id, draft_row.draft_id,
                payload.get("task_id") or payload.get("id") or "",
            )
            return (
                "【办理结果】这份委托已经正式派给玩家了。",
                "以角色口吻确认委托已经交给玩家，给一句结束语（去办吧/别让我等太久），不要重复任务细节。",
            )
        reason = str(payload.get("message") or payload.get("status") or "发布失败")
        logger.info(
            "[interject] confirm rejected session=%s draft=%s reason=%s",
            self.blocks.session_id, draft_row.draft_id, reason,
        )
        return (
            f"【办理结果】委托发布失败：{reason}",
            "以角色口吻说明暂时没法发布，不要编造任务内容。",
        )

    async def _build_confirm_args(
        self, draft_row: Any,
    ) -> tuple[dict[str, Any], list[str], str]:
        """生成 confirm 发布文本（与终端同源）；失败时回模板并标明 source=fallback。"""
        draft = draft_row.draft if isinstance(draft_row.draft, dict) else {}
        summary = self.blocks.pending_draft_summary.strip()
        if not summary:
            try:
                summary = _detailed_draft_summary(
                    draft,
                    self.blocks.game_data,
                    rag_context_text=self.blocks.rag_context_text,
                    npc_states=self.blocks.npc_states,
                )
            except Exception as exc:
                logger.warning("[interject] 草案摘要回算失败，降级为标题: %s", exc)
                summary = str(draft.get("title") or "委托草案")

        if self.llm is None:
            args = fallback_confirm_args(draft, self.request.npc)
            args["draft_id"] = draft_row.draft_id
            return args, [], "fallback"

        finish_reference = await asyncio.to_thread(self._finish_npc_reference, draft)
        try:
            outcome = await generate_confirm_args(
                self.llm,
                draft=draft,
                npc_name=self.request.npc,
                base_messages=build_daily_messages(self.request, self.blocks),
                draft_summary=summary,
                finish_npc_reference=finish_reference,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[interject] confirm 文案生成失败，用模板兜底: %s", exc)
            args = fallback_confirm_args(draft, self.request.npc)
            args["draft_id"] = draft_row.draft_id
            return args, [], "fallback"
        args = dict(outcome.args)
        args["draft_id"] = draft_row.draft_id
        return args, list(outcome.warnings), outcome.source

    def _finish_npc_reference(self, draft: dict[str, Any]) -> str:
        """完成 NPC ≠ 当前 NPC 时的说话风格参考（读失败降级为空）。"""
        try:
            from services.agent_tools.task_tools import resolve_task_npcs
            from services.dialogue.view import build_view
            from services.npc.scene_context import npc_state_of
            from services.orchestrator.prompts import build_finish_npc_reference

            _, finish_npc = resolve_task_npcs(draft, self.request.npc)
            name = (finish_npc or "").strip()
            if not name or name == self.request.npc:
                return ""
            state = npc_state_of(self.blocks.npc_states, name)
            save_info = self.blocks.save_info if isinstance(self.blocks.save_info, dict) else {}
            slot_key = str(save_info.get("slot_key") or "")
            try:
                lines = build_view(
                    game_data=self.blocks.game_data, slot_key=slot_key,
                ).npc_line_samples(name)
            except Exception as exc:
                logger.warning("[interject] 完成 NPC 台词样本读取失败: %s", exc)
                lines = []
            return build_finish_npc_reference(
                npc_name=name,
                lines=lines,
                emotions=(
                    list(getattr(state, "emotions", None) or []) if state is not None else []
                ),
                favorability=state.favorability if state is not None else None,
                relationship_level=state.relationship_level if state is not None else "",
            )
        except Exception as exc:
            logger.warning("[interject] 完成 NPC 参考装配失败，降级为空: %s", exc)
            return ""

    async def _run_cancel(self, route: Any = None) -> tuple[str, str]:
        draft_row = self.blocks.pending_draft_row
        if draft_row is None:
            return (
                "【办理结果】当前没有待取消的委托草案。",
                "以角色口吻回应即可，不要提工具，也不要解释办理过程。",
            )
        # 同 confirm：模型自带提示优先，免得 publishing 阶段回落成「派发委托中」
        self._push_progress(ToolProgress(ui_hint=_route_hint(route, "撤销委托中")))
        execute_cancel_agent_task(
            {"draft_id": draft_row.draft_id}, pending_draft=draft_row.draft,
        )
        try:
            await self.memory.delete_draft(
                self.blocks.session_id, draft_id=draft_row.draft_id,
            )
        except Exception as exc:
            logger.warning("[interject] 草案删除失败: %s", exc)
        return (
            "【办理结果】这份委托已经撤销。",
            "以角色口吻确认这事算了，给一句结束语。",
        )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _route_hint(route: Any, default: str) -> str:
    """路由工具自带的进度提示；模型没给（或给了空白）时回落固定文案。

    解析阶段已截到 12 字，这里再兜一次，防止测试或旁路构造绕过解析。
    """
    hint = str(getattr(route, "ui_hint", "") or "").strip()
    return hint[:12] or default


def _task_max_rounds() -> int:
    from core.config import get_settings

    return max(1, int(get_settings().subagent_task_max_rounds))


def _search_max_rounds() -> int:
    from core.config import get_settings

    return max(1, int(get_settings().subagent_search_max_rounds))


def _unwrap_transition_text(raw: str) -> str:
    """模型偶尔无视豁免仍输出一行 JSONL：取出正文，避免把 JSON 当成台词。"""
    text = (raw or "").strip()
    if not text.startswith("{"):
        return text
    candidates = [text]
    first_line = text.splitlines()[0].strip()
    if first_line and first_line != text:
        candidates.insert(0, first_line)
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            for key in ("add", "closing", "line"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return text
