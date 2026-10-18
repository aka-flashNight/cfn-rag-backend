"""对话内嵌插话端点（P0 骨架）。

- `POST /api/game/dialogue-interject`：SSE；真 LLM 编排在 P1，P0 通过
  `set_stream_factories()` 注入假流做离线契约测试。
- `POST /api/game/dialogue-interject/finalize`：幂等对账。

与终端 `/api/game/ask` 完全独立：不改其路由、不共享 TurnOrchestrator。
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

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
from services.dialogue_interject.sessions import ensure_session, finalize

logger = logging.getLogger(__name__)
router: APIRouter = APIRouter()

#: P0 测试注入点；P1 接线真实 LLM 后由启动期设置（或保留供回归假流使用）
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
    summary="对话内嵌插话（SSE；P0 假流骨架）",
)
async def dialogue_interject(payload: InterjectRequest):
    if _PRIMARY_FACTORY is None:
        # P0：不接真模型；未注入假流时明确回 error，绝不静默补台词
        return StreamingResponse(
            _error_stream("llm_not_configured", "P0 骨架：未接入真实模型，也未注入假流"),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    binding = await ensure_session(payload)
    orchestrator = InterjectOrchestrator(
        payload,
        session_id=binding.session_id,
        stream_factory=_PRIMARY_FACTORY,
        repair_factory=_REPAIR_FACTORY,
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
    return await finalize(payload)
