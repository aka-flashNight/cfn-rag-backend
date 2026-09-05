"""游戏对话 API（v3 重写）。

- POST /api/game/ask：**唯一**对话路径，内部全部走 TurnOrchestrator（单一路径，决策 D1）。
  stream=true → SSE（事件契约见 services/orchestrator/events.py，前端适配见 09）；
  stream=false → 消费同一事件流聚合为 NPCChatResponse。
- /ask/confirm 已删除：HITL 走正常对话轮（草案确认 = 玩家下一条消息，03 §4.2）。
- 代理改为客户端级：proxy_url 进 LLMConfig（httpx 客户端级），不再改进程环境变量（修 E3）。
- 前端可按请求覆盖 api_key/base/model：仅在内存中按会话保存最近一次配置（01 §8），不再按消息存库。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from core.config import get_settings
from schemas.knowledge_schema import (
    ChatMessage,
    NPCCandidate,
    NPCChatRequest,
    NPCChatResponse,
    NPCFavorabilityResponse,
    ResetKnowledgeBaseResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionInfo,
    SessionListResponse,
    SessionTitleUpdateRequest,
    SessionTitleUpdateResponse,
)
from services.llm import LLMConfig
from services.memory.store import get_memory_store
from services.npc.manager import NPCManager, get_npc_manager
from services.orchestrator import TurnOrchestrator

logger = logging.getLogger(__name__)
router: APIRouter = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "X-CFN-Version": "3.0.0",  # 09 §4：前端可据此提示版本配套
}


# ---------------------------------------------------------------------------
# 会话级 LLM 配置（内存；01 §8：不再按消息存库）
# ---------------------------------------------------------------------------

_SESSION_LLM_CONFIGS: dict[str, LLMConfig] = {}
_CONFIG_LOCK = threading.Lock()


def resolve_llm_config(payload: NPCChatRequest) -> LLMConfig:
    """请求覆盖 > 会话记忆（最近一次） > 全局默认；并把结果记回会话内存。"""
    with _CONFIG_LOCK:
        remembered = _SESSION_LLM_CONFIGS.get(payload.session_id)
    merged = LLMConfig(
        api_key=payload.api_key or (remembered.api_key if remembered else ""),
        api_base=payload.api_base or (remembered.api_base if remembered else ""),
        model_name=payload.model_name or (remembered.model_name if remembered else ""),
        proxy_url=payload.proxy_url or (remembered.proxy_url if remembered else ""),
    )
    with _CONFIG_LOCK:
        _SESSION_LLM_CONFIGS[payload.session_id] = merged
    return merged


# ---------------------------------------------------------------------------
# 对话（单一路径）
# ---------------------------------------------------------------------------

@router.post(
    "/ask",
    summary="游戏 NPC 对话（v3 单一路径：聊天 + 任务 + 检索全部经 TurnOrchestrator）",
)
async def ask(
    payload: NPCChatRequest,
    stream: bool = Query(False, description="为 true 时返回 SSE 流式响应（v3 事件契约）"),
    npc_manager: NPCManager = Depends(get_npc_manager),
):
    llm_config = resolve_llm_config(payload)
    orchestrator = TurnOrchestrator(
        session_id=payload.session_id,
        npc_name=payload.npc_name,
        query=payload.query,
        player_identity=payload.player_identity or "",
        progress_stage=payload.progress_stage,
        current_emotion=payload.current_emotion,
        llm_config=llm_config,
        send_image=False,  # P7 立绘/多模态接入前恒不发图
    )

    if stream:
        return StreamingResponse(
            _sse_generate(orchestrator),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return await _collect_response(orchestrator, payload.npc_name, npc_manager)


async def _sse_generate(orchestrator: TurnOrchestrator):
    # 首条注释帧促使代理立即刷新缓冲
    yield b":\n\n"
    try:
        async for ev in orchestrator.run():
            yield ev.encode()
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("SSE 流意外中断")
        import json as _json

        err = _json.dumps(
            {"code": "internal_error", "message": str(exc), "retryable": False},
            ensure_ascii=False,
        )
        yield f"event: error\ndata: {err}\n\n".encode("utf-8")


async def _collect_response(
    orchestrator: TurnOrchestrator,
    npc_name: str,
    npc_manager: NPCManager,
) -> NPCChatResponse:
    """stream=false：消费同一事件流聚合为 JSON（保持单一路径）。"""
    reply_parts: list[str] = []
    emotion = ""
    favorability_change = 0
    favorability: Optional[int] = None
    relationship_level = ""
    error_message: str | None = None

    async for ev in orchestrator.run():
        if ev.event == "content":
            reply_parts.append(str(ev.data.get("delta") or ""))
        elif ev.event == "meta":
            emotion = str(ev.data.get("emotion") or "")
            favorability_change = int(ev.data.get("favorability_change") or 0)
            favorability = int(ev.data.get("favorability") or 0)
            relationship_level = str(ev.data.get("relationship_level") or "")
        elif ev.event == "error":
            error_message = str(ev.data.get("message") or "未知错误")

    if error_message is not None:
        raise HTTPException(status_code=502, detail=error_message)

    if favorability is None:
        state = await npc_manager.get(npc_name)
        favorability = state.favorability
        relationship_level = relationship_level or state.relationship_level

    return NPCChatResponse(
        reply="".join(reply_parts),
        npc_name=npc_name,
        favorability=favorability,
        relationship_level=relationship_level or "陌生",
        favorability_change=favorability_change,
        emotion=emotion or "普通",
    )


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------

def _memory():
    return get_memory_store()


@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="获取指定会话的历史对话记录（分页，倒序）",
)
async def get_session_history(
    session_id: str,
    limit: int = Query(50, ge=1, description="单页条数"),
    offset: int = Query(0, ge=0, description="跳过条数，0=最新一页"),
) -> SessionHistoryResponse:
    memory = _memory()
    records = await memory.get_history(session_id, limit=limit, offset=offset, order="desc")
    messages = [
        ChatMessage(id=rec.id, role=rec.role, content=rec.content, timestamp=rec.timestamp)
        for rec in records
    ]
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="获取所有会话列表及可选 NPC 列表",
)
async def list_sessions(npc_manager: NPCManager = Depends(get_npc_manager)) -> SessionListResponse:
    memory = _memory()
    sessions_raw = await memory.list_sessions()
    sessions = [
        SessionInfo(
            session_id=item.session_id,
            npc_name=item.npc_name,
            title=item.title,
            created_at=item.created_at,
        )
        for item in sessions_raw
    ]
    states = await npc_manager.all_states()
    npc_candidates = [
        NPCCandidate(npc_name=name, faction=state.faction, challenge=state.challenge)
        for name, state in sorted(states.items())
    ]
    return SessionListResponse(sessions=sessions, npc_candidates=npc_candidates)


@router.post(
    "/sessions",
    response_model=SessionCreateResponse,
    summary="创建新的 NPC 会话并返回 session_id",
)
async def create_session(payload: SessionCreateRequest) -> SessionCreateResponse:
    memory = _memory()
    info = await memory.create_session(npc_name=payload.npc_name, title=payload.title)
    return SessionCreateResponse(
        session_id=info.session_id,
        npc_name=info.npc_name,
        title=info.title,
        created_at=info.created_at,
    )


@router.get(
    "/npc/{npc_name}/favorability",
    response_model=NPCFavorabilityResponse,
    summary="获取 NPC 好感度信息",
)
async def get_npc_favorability(
    npc_name: str,
    npc_manager: NPCManager = Depends(get_npc_manager),
) -> NPCFavorabilityResponse:
    state = await npc_manager.get(npc_name)
    return NPCFavorabilityResponse(
        npc_name=npc_name,
        favorability=state.favorability,
        relationship_level=state.relationship_level,
    )


@router.put(
    "/sessions/{session_id}/title",
    response_model=SessionTitleUpdateResponse,
    summary="更新会话标题",
)
async def update_session_title(
    session_id: str,
    payload: SessionTitleUpdateRequest,
) -> SessionTitleUpdateResponse:
    memory = _memory()
    try:
        result = await memory.update_session_title(session_id, payload.title)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return SessionTitleUpdateResponse(session_id=result.session_id, title=result.title)


@router.delete(
    "/sessions/{session_id}",
    status_code=204,
    summary="删除会话",
)
async def delete_session(session_id: str) -> None:
    memory = _memory()
    try:
        await memory.delete_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ---------------------------------------------------------------------------
# 知识库重建（新检索栈：load_corpus → build_store）
# ---------------------------------------------------------------------------

@router.post(
    "/knowledge-base/reset",
    response_model=ResetKnowledgeBaseResponse,
    summary="重置/重建向量知识库",
)
async def reset_knowledge_base_endpoint() -> ResetKnowledgeBaseResponse:
    try:
        from services.retrieval import (
            compute_corpus_fingerprint,
            get_retrieval_engine,
            load_corpus,
        )

        def _rebuild() -> int:
            engine = get_retrieval_engine()
            fingerprint = compute_corpus_fingerprint()
            nodes = load_corpus()
            engine.build_store(nodes, fingerprint)
            return len(nodes)

        count = await asyncio.to_thread(_rebuild)
        return ResetKnowledgeBaseResponse(success=True, message=f"向量库已重建（{count} 条）")
    except Exception as exc:
        logger.exception("知识库重建失败")
        return ResetKnowledgeBaseResponse(success=False, message=str(exc))
