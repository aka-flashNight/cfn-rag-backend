from typing import List, Optional

from pydantic import BaseModel, Field


class NPCChatRequest(BaseModel):
    """
    游戏 NPC 对话请求体。
    """

    query: str = Field(..., description="玩家对 NPC 说的话")
    npc_name: str = Field(..., description="目标 NPC 名称，例如 'Andy Law'")
    player_identity: Optional[str] = Field(
        default=None,
        description="玩家当前身份描述，为空则使用默认设定",
    )
    session_id: str = Field(
        ...,
        description="会话 ID，由后端创建并返回，后续对话必须携带",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="可选的大模型 API Key，优先级高于后端 .env 中的默认配置",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="可选的大模型 API Base，优先级高于后端 .env 中的默认配置",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="可选的大模型名称，优先级高于后端 .env 中的默认配置",
    )


class NPCChatResponse(BaseModel):
    """
    游戏 NPC 对话响应体，包含好感度变更信息。
    """

    reply: str = Field(..., description="NPC 给玩家的回复内容")
    npc_name: str = Field(..., description="NPC 名称")
    favorability: int = Field(..., description="当前 NPC 对玩家的好感度（0-100）")
    relationship_level: str = Field(..., description="关系等级：陌生/熟悉/朋友/生死之交")
    favorability_change: int = Field(
        ...,
        description="本次对话好感度变化（-5 到 +5）",
    )
    emotion: str = Field(..., description="本次对话后 NPC 的情绪，用于选择立绘")


class ChatMessage(BaseModel):
    """
    单条会话消息。
    """

    id: int = Field(..., description="消息主键 ID")
    role: str = Field(..., description="角色：user 或 assistant")
    content: str = Field(..., description="消息内容")
    timestamp: float = Field(..., description="UNIX 时间戳")


class SessionCreateRequest(BaseModel):
    """
    创建新会话的请求。
    """

    npc_name: str = Field(..., description="本会话绑定的 NPC 名称")
    title: str = Field(..., description="会话标题 / 对话名称，供前端展示")


class SessionCreateResponse(BaseModel):
    """
    创建新会话的响应。
    """

    session_id: str = Field(..., description="后端生成的会话 ID")
    npc_name: str = Field(..., description="会话绑定的 NPC 名称")
    title: str = Field(..., description="会话标题")
    created_at: float = Field(..., description="会话创建时间戳")


class SessionInfo(BaseModel):
    """
    已存在会话的概要信息。
    """

    session_id: str = Field(..., description="会话 ID")
    npc_name: str = Field(..., description="NPC 名称")
    title: str = Field(..., description="会话标题")
    created_at: float = Field(..., description="创建时间戳")


class SessionHistoryResponse(BaseModel):
    """
    指定会话的历史消息列表。
    """

    session_id: str = Field(..., description="会话 ID")
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="按时间正序排列的消息列表",
    )


class SessionListResponse(BaseModel):
    """
    会话列表 + NPC 候选列表。
    """

    sessions: List[SessionInfo] = Field(
        default_factory=list,
        description="已有的会话列表",
    )
    npc_candidates: List[str] = Field(
        default_factory=list,
        description="可供选择的 NPC 名称候选，来自 npc_state_db.json",
    )

