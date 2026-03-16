"""
游戏知识库 RAG API，供 Web 端调用。
"""

import json
import os

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from core.startup import ensure_embed_model_ready, trigger_embed_model_preload
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
    SessionListResponse,
    SessionInfo,
    SessionTitleUpdateRequest,
    SessionTitleUpdateResponse,
)
from ai_engine.game_data_loader import reset_knowledge_base
from services.game_rag_service import GameRAGService
from services.memory_manager import MemoryManager
from services.npc_manager import NPCManager

router: APIRouter = APIRouter()


def apply_proxy_config(proxy_url: str | None) -> None:
    """
    设置或清除代理环境变量。
    - proxy_url 非空：设置代理
    - proxy_url 为空或空字符串：清除代理
    """
    if not proxy_url or not proxy_url.strip():
        # 清除代理配置
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        print("[代理] 已清除代理配置")
        return

    proxy = proxy_url.strip()
    # 确保代理地址有 http:// 前缀
    if not proxy.startswith("http://") and not proxy.startswith("https://"):
        proxy = "http://" + proxy

    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    print(f"[代理] 已设置代理: {proxy}")

_game_rag_service: GameRAGService | None = None
_memory_manager: MemoryManager | None = None


def get_game_rag_service() -> GameRAGService:
    """GameRAGService 单例依赖注入。"""
    global _game_rag_service
    if _game_rag_service is None:
        _game_rag_service = GameRAGService()
    return _game_rag_service


async def get_memory_manager() -> MemoryManager:
    """MemoryManager 单例依赖注入。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = await MemoryManager.create()
    return _memory_manager


async def get_npc_manager() -> NPCManager:
    """NPCManager 依赖注入。"""
    return await NPCManager.load()


@router.post(
    "/ask",
    summary="游戏 NPC RAG + 好感度对话（支持流式）",
)
async def ask_game_knowledge(
    payload: NPCChatRequest,
    stream: bool = Query(False, description="为 true 时返回 SSE 流式响应，前端可做打字机效果"),
    service: GameRAGService = Depends(get_game_rag_service),
    memory: MemoryManager = Depends(get_memory_manager),
    npc_manager: NPCManager = Depends(get_npc_manager),
):
    """
    基于游戏资料（剧情、人物、世界观设定等）进行 RAG 问答，
    扮演指定 NPC 与玩家对话，并驱动好感度变化。
    - stream=false（默认）：返回 JSON 体 NPCChatResponse。
    - stream=true：返回 text/event-stream，事件类型为 content（正文片段）与 done（结尾携带 reply/emotion/favorability 等）。
    """
    await ensure_embed_model_ready()
    apply_proxy_config(payload.proxy_url)

    if not stream:
        return await service.ask(payload, npc_manager=npc_manager, memory=memory)

    async def sse_generate():
        # 先发一条 SSE 注释，促使代理/服务器立即刷新缓冲，避免整段响应被缓冲后再返回
        yield b":\n\n"
        try:
            async for event_type, data in service.ask_stream(
                payload, npc_manager=npc_manager, memory=memory
            ):
                if event_type == "content":
                    line = json.dumps({"delta": data}, ensure_ascii=False)
                    yield f"event: content\ndata: {line}\n\n".encode("utf-8")
                elif event_type == "done":
                    line = json.dumps(data, ensure_ascii=False)
                    yield f"event: done\ndata: {line}\n\n".encode("utf-8")
        except Exception as e:
            err_line = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {err_line}\n\n".encode("utf-8")

    return StreamingResponse(
        sse_generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="获取指定会话的历史对话记录（分页，倒序）",
)
async def get_session_history(
    session_id: str,
    limit: int = Query(50, ge=1, description="单页条数，前端固定 50；分页后不设上限，可逐页查看全部记录"),
    offset: int = Query(0, ge=0, description="跳过条数，0=最新一页，50=更早一页"),
    memory: MemoryManager = Depends(get_memory_manager),
) -> SessionHistoryResponse:
    records = await memory.get_history(session_id, limit=limit, offset=offset, order="desc")
    messages: list[ChatMessage] = [
        ChatMessage(
            id=rec["id"],
            role=rec["role"],
            content=rec["content"],
            timestamp=rec["timestamp"],
        )
        for rec in records
    ]
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="获取所有会话列表及可选 NPC 列表",
)
async def list_sessions(
    memory: MemoryManager = Depends(get_memory_manager),
    npc_manager: NPCManager = Depends(get_npc_manager),
) -> SessionListResponse:
    # 触发嵌入模型预加载，但不阻塞返回
    await trigger_embed_model_preload()

    sessions_raw = await memory.list_sessions()
    sessions: list[SessionInfo] = [
        SessionInfo(
            session_id=item["session_id"],
            npc_name=item["npc_name"],
            title=item["title"],
            created_at=item["created_at"],
        )
        for item in sessions_raw
    ]

    npc_candidates: list[NPCCandidate] = [
        NPCCandidate(npc_name=name, faction=npc_manager.state[name].faction)
        for name in sorted(npc_manager.state.keys())
    ]

    return SessionListResponse(
        sessions=sessions,
        npc_candidates=npc_candidates,
    )


@router.post(
    "/sessions",
    response_model=SessionCreateResponse,
    summary="创建新的 NPC 会话并返回 session_id",
)
async def create_session(
    payload: SessionCreateRequest,
    memory: MemoryManager = Depends(get_memory_manager),
) -> SessionCreateResponse:
    # 设置代理（如果前端传入了 proxy_url）
    apply_proxy_config(payload.proxy_url)

    info = await memory.create_session(npc_name=payload.npc_name, title=payload.title)
    return SessionCreateResponse(
        session_id=info["session_id"],
        npc_name=info["npc_name"],
        title=info["title"],
        created_at=info["created_at"],
    )


@router.get(
    "/npc/{npc_name}/favorability",
    response_model=NPCFavorabilityResponse,
    summary="获取 NPC 好感度信息",
)
async def get_npc_favorability(
    npc_name: str,
    service: GameRAGService = Depends(get_game_rag_service),
    npc_manager: NPCManager = Depends(get_npc_manager),
) -> NPCFavorabilityResponse:
    """
    获取指定 NPC 对玩家的好感度、关系等级和当前情绪状态。
    """
    from fastapi import HTTPException

    try:
        npc_name_decoded = npc_name  # FastAPI 会自动进行 URL 解码
        name, favorability, relationship_level = await service.get_npc_favorability(
            npc_name_decoded, npc_manager=npc_manager
        )
        return NPCFavorabilityResponse(
            npc_name=name,
            favorability=favorability,
            relationship_level=relationship_level,
        )
    except ValueError as e:
        if "不存在" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{str(e)}")


@router.put(
    "/sessions/{session_id}/title",
    response_model=SessionTitleUpdateResponse,
    summary="更新会话标题",
)
async def update_session_title(
    session_id: str,
    payload: SessionTitleUpdateRequest,
    memory: MemoryManager = Depends(get_memory_manager),
) -> SessionTitleUpdateResponse:
    """
    修改指定会话的标题。
    """
    from fastapi import HTTPException

    try:
        result = await memory.update_session_title(session_id, payload.title)
        return SessionTitleUpdateResponse(
            session_id=result["session_id"],
            title=result["title"],
        )
    except ValueError as e:
        if "不存在" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{str(e)}")


@router.delete(
    "/sessions/{session_id}",
    status_code=204,
    summary="删除会话",
)
async def delete_session(
    session_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
) -> None:
    """
    删除指定的会话及其所有聊天记录。
    """
    from fastapi import HTTPException

    try:
        await memory.delete_session(session_id)
    except ValueError as e:
        if "不存在" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{str(e)}")


@router.post(
    "/knowledge-base/reset",
    response_model=ResetKnowledgeBaseResponse,
    summary="重置知识库",
)
async def reset_knowledge_base_endpoint(
    service: GameRAGService = Depends(get_game_rag_service),
) -> ResetKnowledgeBaseResponse:
    """
    供用户手动重置/重新生成向量库（单 exe 无法在打包时重置时使用）。
    若 docs 中存在「核心设定与世界合理性补足」文档则强制覆盖重建；
    否则仅当尚无向量库时生成，已有则返回数据文档不全错误。
    """
    await ensure_embed_model_ready()
    success, message = reset_knowledge_base()
    if success:
        service.invalidate_index()
    return ResetKnowledgeBaseResponse(success=success, message=message)
