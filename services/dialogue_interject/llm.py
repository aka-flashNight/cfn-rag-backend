"""日常插话的真 LLM 流 + 工具轮（B2）。

- 复用 `services/llm/` 的客户端、配置与降级链；不新增 provider/依赖。
- 工具轮：只把模型正文增量交给 JSONL 解析器；工具调用参数与结果不进入台词，
  也不落 session。工具结果回来后做一次无工具 LLM 调用，只输出最终对话。
- 日常工具：route_search / route_task_draft / route_task_update /
  route_task_confirm / route_task_cancel / update_long_term_memory /
  delegate_dialogue_rewrite（委派只提交后台任务，不阻塞最终对话）。
- 任务（P4）：只有 route_search + update_long_term_memory；共用流循环与工具轮，
  status 只发 searching，工具结果后不再给工具。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from services.dialogue_interject.context import (
    DailyBlocks,
    assemble_daily_blocks,
    assemble_task_blocks,
)
from services.dialogue_interject.prompts import (
    build_daily_messages,
    build_task_messages,
)
from services.dialogue_interject.schemas import InterjectRequest, StreamStatus
from services.dialogue_interject.tools import (
    TOOL_TIMEOUT_S,
    DailyToolError,
    DailyToolExecutor,
    DailyToolRuntime,
    ToolExecutionResult,
    ToolProgress,
)
from services.llm import ChatRequest, LLMClient, LLMConfig
from services.llm.meta import DEFAULT_EMOTION
from services.memory.long_term import tool_call_name


logger = logging.getLogger(__name__)

#: 工具轮状态映射（只有这两类需要让玩家等；记忆/委派不显式提示）
_DRAFTING_TOOLS = {
    "route_task_draft", "route_task_update", "route_task_confirm", "route_task_cancel",
}

#: 最多两轮：工具轮 + 无工具的最终对话轮；与终端“一轮最多路由一件事”一致
MAX_CHAT_ROUNDS = 2
#: 工具执行外层兜底超时（工具运行时内部还有子 Agent 超时）
_TOOL_WAIT_S = TOOL_TIMEOUT_S + 10.0


class InterjectLLMUnavailable(RuntimeError):
    """真实 LLM 未配置（无 api_key）。"""

    code = "llm_not_configured"


def resolve_llm_config(llm_config: LLMConfig | None = None) -> LLMConfig:
    """请求不带模型配置：沿用全局 settings（与终端默认一致）。"""
    return (llm_config or LLMConfig()).merged_with_settings()


def llm_is_configured(llm_config: LLMConfig | None = None) -> bool:
    return bool(resolve_llm_config(llm_config).api_key.strip())


def _require_llm_config(llm_config: LLMConfig | None) -> LLMConfig:
    cfg = resolve_llm_config(llm_config)
    if not cfg.api_key.strip():
        raise InterjectLLMUnavailable("未配置 llm_api_key，插话真模型路径不可用")
    return cfg


async def stream_daily_llm(
    request: InterjectRequest,
    *,
    llm_config: LLMConfig | None = None,
    client: Any = None,
    blocks: DailyBlocks | None = None,
    tool_executor: DailyToolExecutor | None = None,
    npc_manager: Any = None,
    game_data: Any = None,
    engine: Any = None,
    memory: Any = None,
    registry: Any = None,
) -> AsyncIterator[str | StreamStatus]:
    """日常真 LLM 的正文增量流（含一次工具轮）。

    `client` / `blocks` / `tool_executor` 仅用于离线测试注入；生产路径用全局
    配置与真实工具运行时。
    """
    cfg = _require_llm_config(llm_config)
    if blocks is None:
        blocks = await assemble_daily_blocks(
            request,
            npc_manager=npc_manager,
            game_data=game_data,
            engine=engine,
            memory=memory,
            registry=registry,
            prompt_max=request.limits.line_prompt_max,
        )
    async for chunk in _stream_with_blocks(
        request,
        blocks=blocks,
        messages=build_daily_messages(request, blocks),
        cfg=cfg,
        client=client,
        tool_executor=tool_executor,
    ):
        yield chunk


async def stream_task_llm(
    request: InterjectRequest,
    *,
    llm_config: LLMConfig | None = None,
    client: Any = None,
    blocks: DailyBlocks | None = None,
    tool_executor: DailyToolExecutor | None = None,
    npc_manager: Any = None,
    game_data: Any = None,
    engine: Any = None,
    memory: Any = None,
    registry: Any = None,
) -> AsyncIterator[str | StreamStatus]:
    """任务真 LLM 的正文增量流（含一次工具轮；工具只有检索与长期记忆）。

    与日常共用流循环、工具轮与降级链；任务块由 `assemble_task_blocks` 装配
    （不读草案、不开任务工具/委派）。工具结果之后不再给工具，只输出最终 JSONL。
    """
    cfg = _require_llm_config(llm_config)
    if blocks is None:
        blocks = await assemble_task_blocks(
            request,
            npc_manager=npc_manager,
            game_data=game_data,
            engine=engine,
            memory=memory,
            registry=registry,
            prompt_max=request.limits.line_prompt_max,
        )
    async for chunk in _stream_with_blocks(
        request,
        blocks=blocks,
        messages=build_task_messages(request, blocks),
        cfg=cfg,
        client=client,
        tool_executor=tool_executor,
    ):
        yield chunk


async def _stream_with_blocks(
    request: InterjectRequest,
    *,
    blocks: DailyBlocks,
    messages: list[dict[str, str]],
    cfg: LLMConfig,
    client: Any = None,
    tool_executor: DailyToolExecutor | None = None,
) -> AsyncIterator[str | StreamStatus]:
    """日常/任务共用的 LLM 流循环：正文增量 → 工具轮 → 无工具最终轮。"""
    llm = client or LLMClient.for_config(cfg)
    executor = tool_executor or DailyToolRuntime(request, blocks, llm=llm)
    tools = executor.tools()

    started = time.perf_counter()
    first_content_ms: float | None = None
    content_chars = 0
    tool_rounds = 0

    logger.info(
        "[interject] LLM 请求 mode=%s dialogue=%s turn=%s model=%s tools=%s "
        "system_chars=%d user_chars=%d",
        request.mode, request.dialogue_instance_id, request.turn_id, cfg.model_name,
        len(tools or []), len(messages[0]["content"]), len(messages[1]["content"]),
    )

    for _round_index in range(MAX_CHAT_ROUNDS):
        tool_calls: list[dict[str, Any]] = []
        round_text_parts: list[str] = []
        chat_request = ChatRequest(
            messages=messages,
            tools=tools,
            purpose="chat",
            send_image=False,
        )
        async for event in llm.chat_stream(chat_request):
            if event.kind == "content" and event.text:
                if first_content_ms is None:
                    first_content_ms = (time.perf_counter() - started) * 1000
                content_chars += len(event.text)
                round_text_parts.append(event.text)
                yield event.text
            elif event.kind == "finish":
                tool_calls = list(event.tool_calls or [])

        if not tool_calls:
            break
        if tools is None:
            logger.warning("[interject] 无工具轮出现 tool_calls，忽略")
            break

        round_text = "".join(round_text_parts)
        stage = _status_stage_for(tool_calls, mode=request.mode)
        if stage:
            # 工具轮前的旁路状态（协议 5.2 status）：Host 可显示“正在准备…”
            yield StreamStatus(stage)
        outcome: ToolExecutionResult | None = None
        async for item in _execute_with_progress(executor, tool_calls, round_text):
            if isinstance(item, ToolProgress):
                if item.ui_hint:
                    yield StreamStatus(stage or "thinking", item.ui_hint)
                if item.line:
                    # 过渡话当作一条 add 行走现有 JSONL 解析（拆分/情绪回落同源）；
                    # 前置换行结束路由轮可能残留的半行，保证它是独立一行。
                    yield "\n" + json.dumps(
                        {
                            "add": item.line,
                            "emo": item.emotion or DEFAULT_EMOTION,
                            "interim": True,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                continue
            outcome = item
        if outcome is None:  # pragma: no cover - 内部契约：必然产出终态结果
            raise DailyToolError("tool_failed", "插话工具没有返回结果")

        if round_text.strip() and outcome.requires_followup:
            # 协议要求路由轮不说话；模型违规时已推送的正文无法撤回，留痕供走查
            logger.warning(
                "[interject] 工具轮仍输出正文（已按流推送）mode=%s dialogue=%s turn=%s "
                "route=%s text=%.80s",
                request.mode, request.dialogue_instance_id, request.turn_id,
                outcome.route_kind, round_text,
            )
        if not outcome.requires_followup:
            break

        messages = [*messages, *outcome.messages]
        tools = None  # 工具结果之后只输出最终对话，不再给工具
        tool_rounds += 1
    else:
        logger.warning(
            "[interject] 达到工具轮上限 %d mode=%s dialogue=%s turn=%s",
            MAX_CHAT_ROUNDS, request.mode,
            request.dialogue_instance_id, request.turn_id,
        )

    logger.info(
        "[interject] LLM 完成 mode=%s dialogue=%s turn=%s first_content_ms=%s "
        "chars=%d tools=%d elapsed_ms=%.0f",
        request.mode, request.dialogue_instance_id, request.turn_id,
        f"{first_content_ms:.0f}" if first_content_ms is not None else "n/a",
        content_chars, tool_rounds, (time.perf_counter() - started) * 1000,
    )


async def _execute_with_progress(
    executor: DailyToolExecutor,
    tool_calls: list[dict[str, Any]],
    round_text: str,
) -> AsyncIterator[ToolProgress | ToolExecutionResult]:
    """执行工具轮并并发吐出旁路进度；最后一条必为 ToolExecutionResult。

    生产执行器暴露 `progress_queue`（子 Agent 进度与过渡话）；测试注入的假执行器
    没有该队列时退化为原来的单次 await。
    """
    queue = getattr(executor, "progress_queue", None)
    if queue is None:
        try:
            outcome = await asyncio.wait_for(
                executor.execute(tool_calls, round_text=round_text),
                timeout=_TOOL_WAIT_S,
            )
        except asyncio.TimeoutError as exc:
            raise DailyToolError("tool_timeout", "插话工具执行超时") from exc
        yield outcome
        return

    exec_task = asyncio.create_task(executor.execute(tool_calls, round_text=round_text))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TOOL_WAIT_S
    try:
        while not exec_task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            try:
                progress = await asyncio.wait_for(queue.get(), timeout=min(0.1, remaining))
            except asyncio.TimeoutError:
                continue
            yield progress
        outcome = await exec_task
        while not queue.empty():
            yield queue.get_nowait()
        yield outcome
    except asyncio.TimeoutError as exc:
        raise DailyToolError("tool_timeout", "插话工具执行超时") from exc
    finally:
        if not exec_task.done():
            exec_task.cancel()


def _status_stage_for(tool_calls: list[dict[str, Any]], *, mode: str) -> str:
    """工具轮前的 status 阶段；未识别/纯记忆/委派返回空串（不发 status）。

    任务模式只可能检索，绝不发 drafting（任务工具与委派本就不注册）。
    """
    names = {tool_call_name(call) for call in tool_calls or []}
    if mode == "task":
        return "searching" if "route_search" in names else ""
    if names & _DRAFTING_TOOLS:
        return "drafting"
    if "route_search" in names:
        return "searching"
    return ""
