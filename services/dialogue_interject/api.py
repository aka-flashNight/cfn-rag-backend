"""对话内嵌插话端点（B4：日常 + 任务真模型）。

- `POST /api/game/dialogue-interject`（SSE）：
  - 日常：`assemble_daily_blocks` + `stream_daily_llm`；
  - 任务：`assemble_task_blocks` + `stream_task_llm`（工具只有检索/记忆）；
  - 测试/联调可注入假流，注入时保留 P0 fixture 路径。
- `POST /api/game/dialogue-interject/finalize`：幂等对账；跨进程并发冲突回 409。

与终端 `/api/game/ask` 完全独立：不改其路由、不共享 TurnOrchestrator。
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from services.dialogue_interject.context import (
    assemble_daily_blocks,
    assemble_task_blocks,
)
from services.dialogue_interject.llm import (
    llm_is_configured,
    stream_daily_llm,
    stream_task_llm,
)
from services.dialogue_interject.orchestrator import (
    InterjectOrchestrator,
    StreamFactory,
)
from services.dialogue_interject.schemas import (
    FinalizeRequest,
    FinalizeResponse,
    InterjectRequest,
    error_event,
)
from services.dialogue_interject.sessions import (
    ConcurrentFinalizeError,
    ensure_session,
    finalize,
)
from services.llm.meta import DEFAULT_EMOTION
from services.llm.recent import get_recent_llm_config

logger = logging.getLogger(__name__)
router: APIRouter = APIRouter()

#: 测试/联调注入点；生产默认走真实 LLM。测试 teardown 必须复位。
_PRIMARY_FACTORY: StreamFactory | None = None
_REPAIR_FACTORY: StreamFactory | None = None

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def set_stream_factories(
    primary: StreamFactory | None, repair: StreamFactory | None = None,
) -> None:
    """注入主/修复流工厂（None＝未接线，端点回 error）。测试 teardown 需复位。"""
    global _PRIMARY_FACTORY, _REPAIR_FACTORY
    _PRIMARY_FACTORY = primary
    _REPAIR_FACTORY = repair


def get_stream_factories() -> tuple[StreamFactory | None, StreamFactory | None]:
    return _PRIMARY_FACTORY, _REPAIR_FACTORY


async def _sse_stream(orchestrator: InterjectOrchestrator):
    try:
        async for event in orchestrator.run():
            yield event.encode()
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        logger.info("[interject] 客户端断开，取消在途编排")
        raise
    except Exception as exc:  # noqa: BLE001 - 兜底，避免半截流无终止事件
        logger.exception("[interject] SSE 流意外中断")
        yield error_event("internal_error", str(exc)[:200]).encode()


async def _error_stream(code: str, message: str):
    yield error_event(code, message).encode()


@router.post(
    "/dialogue-interject",
    summary="对话内嵌插话（SSE）",
)
async def dialogue_interject(payload: InterjectRequest):
    stream_factory: StreamFactory | None = _PRIMARY_FACTORY
    npc_emotions: tuple[str, ...] = (DEFAULT_EMOTION,)
    binding = None

    if stream_factory is None:
        # 打包版没有 .env：复用终端 ask 最近一次由页面设置传入的配置；
        # 没有时再回落全局 settings（源码开发/测试）。
        recent_llm = get_recent_llm_config()
        if not llm_is_configured(recent_llm):
            return StreamingResponse(
                _error_stream("llm_not_configured", "未配置 llm_api_key，插话真模型路径不可用"),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        # 先建/取 session：草案与工具写盘用同一把钥匙，终端继续聊才接得上
        binding = await ensure_session(payload)
        try:
            if payload.mode == "task":
                blocks = await assemble_task_blocks(
                    payload,
                    prompt_max=payload.limits.line_prompt_max,
                    session_id=binding.session_id,
                )
            else:
                blocks = await assemble_daily_blocks(
                    payload,
                    prompt_max=payload.limits.line_prompt_max,
                    session_id=binding.session_id,
                )
        except Exception as exc:  # noqa: BLE001 - 上下文失败按 D-17 回原对话
            logger.exception("[interject] 插话稳定块装配失败 mode=%s", payload.mode)
            return StreamingResponse(
                _error_stream("context_failed", str(exc)[:200]),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        npc_emotions = blocks.npc_emotions
        stream_factory = partial(
            stream_task_llm if payload.mode == "task" else stream_daily_llm,
            blocks=blocks,
            llm_config=recent_llm,
        )

    if binding is None:
        binding = await ensure_session(payload)
    orchestrator = InterjectOrchestrator(
        payload,
        session_id=binding.session_id,
        stream_factory=stream_factory,
        repair_factory=_REPAIR_FACTORY,
        npc_emotions=npc_emotions,
    )
    return StreamingResponse(
        _sse_stream(orchestrator),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post(
    "/dialogue-interject/finalize",
    response_model=FinalizeResponse,
    summary="对话内嵌插话 · 已播对账（幂等）",
)
async def dialogue_interject_finalize(payload: FinalizeRequest) -> FinalizeResponse:
    try:
        return await finalize(payload)
    except ConcurrentFinalizeError as exc:
        # 跨进程并发撞同一 seq：不静默吞错，让 Host 稍后重发 finalize
        raise HTTPException(status_code=409, detail="concurrent_finalize_conflict") from exc
