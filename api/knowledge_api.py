from fastapi import APIRouter, Depends

from core.config import Settings, get_settings
from schemas.knowledge_schema import AskRequest, AskResponse
from services.knowledge_service import KnowledgeService


router: APIRouter = APIRouter()


def get_knowledge_service(
    settings: Settings = Depends(get_settings),
) -> KnowledgeService:
    """
    KnowledgeService 依赖注入工厂。
    """

    return KnowledgeService(settings=settings)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="知识库问答",
)
async def ask_knowledge(
    payload: AskRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> AskResponse:
    """
    接收用户 Query，调用知识库/LLM 获取回答。
    """

    return await service.ask_question(payload)

