"""日常插话的工具接线（B2）。

复用终端的路由工具、任务子 Agent、检索子 Agent 与长期记忆工具；只把正文增量
交给 JSONL 解析器，工具调用参数与结果绝不进入台词/session。

一次回合最多一个路由工具（与终端一致）：工具轮不输出台词；工具结果回来后
再做一次无工具的 LLM 调用，只输出最终 JSONL 对话与结束语。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from services.agent_tools.draft_formatting import _detailed_draft_summary
from services.agent_tools.handlers import (
    execute_cancel_agent_task,
    execute_confirm_agent_task,
)
from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.schemas import InterjectRequest
from services.llm import LLMClient
from services.memory.long_term import (
    CHAT_TOOL_NAME,
    apply_chat_tool,
    build_chat_tool,
    tool_call_name,
)
from services.orchestrator.prompts import fallback_confirm_args
from services.orchestrator.search_prefetch import (
    build_prefetch_result_block,
    run_search_prefetch,
)
from services.orchestrator.turn import Route, _parse_route
from services.subagents.search_runner import SearchRunner
from services.subagents.task_runner import TaskRunner
from services.tools.base import ROUTE_TOOLS_DRAFT, ROUTE_TOOLS_NEW

logger = logging.getLogger(__name__)

#: 工具执行超时（秒）；任务 Agent 自身还有轮数上限
TOOL_TIMEOUT_S = 120.0


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
        if self.blocks.long_term is not None:
            schemas = [*schemas, build_chat_tool(self.blocks.long_term)]
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
        for index, call in enumerate(calls):
            if tool_call_name(call) != CHAT_TOOL_NAME:
                continue
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

        route = _parse_route(calls)
        if route is None:
            # 只有记忆/未知工具：正文已随流推送，无需后续调用
            return ToolExecutionResult()

        route_index = next(
            (
                index for index, call in enumerate(calls)
                if tool_call_name(call).startswith("route_")
            ),
            None,
        )
        if route.kind.startswith("task_") and self.blocks.task_publish_disabled:
            result_block = "【后台结果】你当前不能发布委托，后台没有受理。"
            instruction = "以角色口吻婉拒，不要编造任务内容，不要提工具或后台。"
        else:
            try:
                result_block, instruction = await asyncio.wait_for(
                    self._execute_route(route), timeout=TOOL_TIMEOUT_S,
                )
            except Exception as exc:  # 工具失败/超时：记录后继续，不拖垮整轮对话
                logger.exception("[interject] 工具执行失败 route=%s", route.kind)
                result_block = "【后台结果】后台处理失败或超时，这次未能完成。"
                instruction = (
                    "以角色口吻说明这次没办成，不要编造任务内容或数字，不要提工具或后台。"
                )

        route_result = result_block
        for index, call in enumerate(calls):
            call_id = str(call.get("id") or "")
            if index == route_index:
                content = route_result
            elif index in memory_results:
                content = memory_results[index]
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

    async def _execute_route(self, route: Route) -> tuple[str, str]:
        if route.kind == "search":
            return await self._run_search(route)
        if route.kind in ("task_draft", "task_update"):
            return await self._run_task_runner(route)
        if route.kind == "task_confirm":
            return await self._run_confirm()
        if route.kind == "task_cancel":
            return await self._run_cancel()
        return (
            "【后台结果】未能识别的工具调用，已忽略。",
            "用 JSONL 正常回应当前对话，不要提工具或后台。",
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

        return (
            build_prefetch_result_block(text),
            "基于上面的资料，用 JSONL 输出最终台词；不要提工具或后台，不要输出动作描写。",
        )

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
        event = await _wait_handle(handle)
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
        event = await _wait_handle(handle)
        if event.kind == "failed":
            reason = str((event.payload or {}).get("reason") or "后台未能拟定委托")
            return (
                f"【后台结果】{reason}",
                "以角色口吻说明这次没法准备委托，不要编造任务内容或数字，不要提工具或后台。",
            )

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
        return (
            f"【后台结果：{title}，尚未发布】\n{summary or '（草案摘要不可用）'}",
            "现在只输出最终确认对话（add）和结束语（closing）：讲清委托内容/要求/报酬并征询是否接取；"
            "不要输出“我去准备一下”之类过渡语；结束语表达“不接就算了/这活就搁着”，不要求固定措辞；"
            "玩家点头前不要说已发布，任务内容/数字以草案结果为准。",
        )

    # ------------------------------------------------------------------
    # 确认 / 取消
    # ------------------------------------------------------------------

    async def _run_confirm(self) -> tuple[str, str]:
        draft_row = self.blocks.pending_draft_row
        if draft_row is None:
            return (
                "【后台结果】当前没有待确认的委托草案，无法发布。",
                "以角色口吻说明没有可确认的委托，不要编造任务内容。",
            )
        args = fallback_confirm_args(draft_row.draft, self.request.npc)
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
        )
        payload = outcome.payload or {}
        if payload.get("status") == "confirmed":
            try:
                await self.memory.delete_draft(
                    self.blocks.session_id, draft_id=draft_row.draft_id,
                )
            except Exception as exc:
                logger.warning("[interject] 草案删除失败: %s", exc)
            return (
                "【后台结果】这份委托已经正式派给玩家了。",
                "以角色口吻确认委托已经交给玩家，给一句结束语（去办吧/别让我等太久），不要重复任务细节。",
            )
        reason = str(payload.get("message") or payload.get("status") or "发布失败")
        return (
            f"【后台结果】委托发布失败：{reason}",
            "以角色口吻说明暂时没法发布，不要编造任务内容。",
        )

    async def _run_cancel(self) -> tuple[str, str]:
        draft_row = self.blocks.pending_draft_row
        if draft_row is None:
            return (
                "【后台结果】当前没有待取消的委托草案。",
                "以角色口吻回应即可，不要提工具或后台。",
            )
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
            "【后台结果】这份委托已经撤销。",
            "以角色口吻确认这事算了，给一句结束语。",
        )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _task_max_rounds() -> int:
    from core.config import get_settings

    return max(1, int(get_settings().subagent_task_max_rounds))


def _search_max_rounds() -> int:
    from core.config import get_settings

    return max(1, int(get_settings().subagent_search_max_rounds))


async def _wait_handle(handle: Any) -> Any:
    """等待子 Agent 终态；超时/取消时确保后台任务被取消。"""
    from core.config import get_settings

    timeout = float(get_settings().subagent_timeout_s)
    try:
        return await asyncio.wait_for(handle.wait_final(timeout), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise DailyToolError("tool_timeout", f"子 Agent 超时（>{timeout:.0f}s）") from exc
    except asyncio.CancelledError:
        raise
    finally:
        if not handle.done:
            handle.cancel()
