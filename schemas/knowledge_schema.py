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
    progress_stage: Optional[int] = Field(
        default=None,
        ge=1,
        le=6,
        description="玩家当前主线进度阶段，1-6 的整数；不传时表示未知或未开始，用于后续 LLM 与后端工具根据进度做差异化响应",
    )
    session_id: str = Field(
        ...,
        description="会话 ID，由后端创建并返回，后续对话必须携带",
    )
    current_emotion: Optional[str] = Field(
        default=None,
        description="上一轮 ask 返回的 NPC 情绪，供后端/AI 做连贯回复与情绪过渡；可选、可空",
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
    proxy_url: Optional[str] = Field(
        default=None,
        description="可选的 HTTP 代理地址，例如 http://127.0.0.1:10809",
    )
    summarize_interval: Optional[int] = Field(
        default=None,
        description="精确短期记忆的总结间隔/历史长度档位。取值 10/30/100/500，对应前端短/中/长/几乎无限。不传时后端使用默认值 30。",
    )
    progress_stage: Optional[int] = Field(
        default=None,
        ge=1,
        le=6,
        description="玩家当前所在的游戏进度阶段。取值为 1-6 的整数；不传时后端视为未知阶段，按旧逻辑兼容处理。",
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
    proxy_url: Optional[str] = Field(
        default=None,
        description="可选的 HTTP 代理地址，例如 http://127.0.0.1:10809",
    )


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


class NPCCandidate(BaseModel):
    """
    NPC 候选项，用于会话列表接口中的可选 NPC。
    """

    npc_name: str = Field(..., description="NPC 名称")
    faction: Optional[str] = Field(None, description="阵营，来自 npc_state_db.json，无则为空")
    challenge: Optional[str] = Field(
        None,
        description="可选：当前 NPC 的切磋关卡名（npc_state_db.json 的 challenge 字段）",
    )


class SessionListResponse(BaseModel):
    """
    会话列表 + NPC 候选列表。
    """

    sessions: List[SessionInfo] = Field(
        default_factory=list,
        description="已有的会话列表",
    )
    npc_candidates: List[NPCCandidate] = Field(
        default_factory=list,
        description="可供选择的 NPC 候选（含名称与阵营），来自 npc_state_db.json",
    )


class NPCFavorabilityResponse(BaseModel):
    """
    获取 NPC 好感度信息的响应。
    """

    npc_name: str = Field(..., description="NPC 名称")
    favorability: int = Field(..., description="好感度数值（0-100）")
    relationship_level: str = Field(..., description="关系等级（如：陌生、熟悉、朋友、生死之交）")


class SessionTitleUpdateRequest(BaseModel):
    """
    更新会话标题的请求体。
    """

    title: str = Field(..., description="新的会话标题（1-100 字符）", min_length=1, max_length=100)


class SessionTitleUpdateResponse(BaseModel):
    """
    更新会话标题的响应。
    """

    session_id: str = Field(..., description="会话 ID")
    title: str = Field(..., description="更新后的标题")


class ResetKnowledgeBaseResponse(BaseModel):
    """
    重置知识库接口的响应。前端根据 success 显示成功/失败，并展示 message（若有）。
    """

    success: bool = Field(..., description="是否成功")
    message: str | None = Field(None, description="说明信息，如失败原因或成功提示")

