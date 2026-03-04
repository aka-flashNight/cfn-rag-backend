"""
游戏知识库 RAG API，供 Web 端调用。
"""

from fastapi import APIRouter, Depends, Query

from schemas.knowledge_schema import (
    ChatMessage,
    NPCChatRequest,
    NPCChatResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionInfo,
)
from services.game_rag_service import GameRAGService
from services.memory_manager import MemoryManager
from services.npc_manager import NPCManager

router: APIRouter = APIRouter()

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
    response_model=NPCChatResponse,
    summary="游戏 NPC RAG + 好感度对话",
)
async def ask_game_knowledge(
    payload: NPCChatRequest,
    service: GameRAGService = Depends(get_game_rag_service),
    memory: MemoryManager = Depends(get_memory_manager),
    npc_manager: NPCManager = Depends(get_npc_manager),
) -> NPCChatResponse:
    """
    基于游戏资料（剧情、人物、世界观设定等）进行 RAG 问答，
    扮演指定 NPC 与玩家对话，并驱动好感度变化。
    """
    return await service.ask(payload, npc_manager=npc_manager, memory=memory)


@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="获取指定会话的历史对话记录",
)
async def get_session_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=200, description="返回的最大历史条数"),
    memory: MemoryManager = Depends(get_memory_manager),
) -> SessionHistoryResponse:
    records = await memory.get_history(session_id, limit=limit)
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

    npc_names = sorted(npc_manager.state.keys())

    return SessionListResponse(
        sessions=sessions,
        npc_candidates=npc_names,
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
    info = await memory.create_session(npc_name=payload.npc_name, title=payload.title)
    return SessionCreateResponse(
        session_id=info["session_id"],
        npc_name=info["npc_name"],
        title=info["title"],
        created_at=info["created_at"],
    )
