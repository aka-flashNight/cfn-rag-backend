"""
游戏知识库 RAG 服务，基于 LlamaIndex + Gemini，支持 NPC 好感度管理。
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from openai import AsyncOpenAI

from ai_engine.game_data_loader import get_cached_index
from core.config import Settings, get_settings
from schemas.knowledge_schema import NPCChatRequest, NPCChatResponse
from services.npc_manager import NPCManager, NPCState
from services.memory_manager import MemoryManager


class GameRAGService:
    """
    游戏世界观 / NPC 知识库 RAG 服务 + 好感度 Agent。
    """

    def __init__(self) -> None:
        self._index: VectorStoreIndex | None = None

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            self._index = get_cached_index()
        return self._index

    async def ask(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> NPCChatResponse:
        """
        基于游戏知识库进行 RAG + Agent 对话，并更新 NPC 好感度。
        """

        npc_name: str = payload.npc_name.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")

        settings: Settings = get_settings()
        # 支持前端传入大模型配置，未提供则回退到 .env；Key 必须存在
        request_api_key: str | None = (
            payload.api_key.strip() if payload.api_key and payload.api_key.strip() else None
        )
        request_api_base: str | None = (
            payload.api_base.strip() if payload.api_base and payload.api_base.strip() else None
        )
        request_model: str | None = (
            payload.model_name.strip()
            if payload.model_name and payload.model_name.strip()
            else None
        )

        effective_api_key: str | None = request_api_key or settings.gemini_api_key
        effective_api_base: str = request_api_base or settings.llm_api_base
        effective_model: str = request_model or settings.llm_model_name

        if not effective_api_key:
            raise ValueError(
                "未提供可用的大模型 API Key，请在请求中传入 api_key 或在 .env 中配置 GEMINI_API_KEY。"
            )

        # 1. 读取 / 初始化 NPC 当前好感度
        current_state: NPCState | None = npc_manager.state.get(npc_name)
        if current_state is None:
            current_state = NPCState(
                favorability=0,
                relationship_level="陌生",
                emotions=["普通"],
            )

        favorability: int = current_state.favorability
        relationship_level: str = current_state.relationship_level
        sex: str | None = current_state.sex
        emotions: list[str] = current_state.emotions or ["普通"]

        # 2. 使用 LlamaIndex 检索 NPC 过往台词 + 世界观设定
        retrieved_context: str = await asyncio.to_thread(
            self._retrieve_context, npc_name, payload.query
        )

        # 3. 构造 Prompt，并调用 Gemini
        player_identity: str = (
            payload.player_identity.strip()
            if payload.player_identity and payload.player_identity.strip()
            else "一个末日后加入A兵团成为佣兵的幸存者"
        )

        sex_desc = f"（性别：{sex}）" if sex else ""

        # 3. 从记忆库中加载该会话最近的历史消息
        history_records = await memory.get_history(payload.session_id, limit=10)
        history_lines: List[str] = []
        for msg in history_records:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prefix = "玩家"
            else:
                prefix = npc_name
            history_lines.append(f"{prefix}: {content}")

        history_str = ""
        if history_lines:
            joined_history = "\n".join(history_lines)
            history_str = (
                "下面是你与玩家之间的对话历史（按时间从早到晚排列），"
                "请在保持人物性格与情节连贯的前提下继续对话：\n"
                f"{joined_history}\n\n"
            )

        emotions_str = "、".join(emotions)

        system_prompt = (
            f"你现在扮演游戏角色「{npc_name}」{sex_desc}。\n"
            f"玩家的身份是：{player_identity}。\n\n"
            "下面是与你相关的世界观设定和你的过往台词片段"
            "（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
            f"{retrieved_context or '（当前没有检索到任何上下文，你可以根据自己的设定自由发挥，但要保持合理。）'}\n\n"
            f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。\n"
            f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。请选择其中最合适的一种作为你当前的情绪立绘。\n"
            "请以符合你身份、当前好感度和所选情绪的语气，用简体中文回答玩家本次的发言。\n\n"
            f"{history_str}"
            "输出格式必须严格为两行：\n"
            "第一行：你的回复内容（只包含对话文本，不要包含 JSON，不要带前缀）。\n"
            "第二行：一个 JSON 对象，必须且仅包含两个字段：\n"
            "  - \"favorability_change\"：一个整数字段，取值范围 -5 到 5，例如："
            "{\"favorability_change\": 1, \"emotion\": \"普通\"}\n"
            "  - \"emotion\"：一个字符串字段，值必须从上文提供的情绪标签中选择；"
            "如果没有特别合适的情绪，请使用 \"普通\"。\n"
            "常规对话无需调整好感度。如果本次对话不应影响好感度，请输出 {\"favorability_change\": 0}。\n"
            "不要输出第三行及更多内容，不要添加多余的空行或注释。"
        )

        full_prompt = f"{system_prompt}\n\n玩家：{payload.query}\n"

        reply_text: str = await self._call_llm(
            settings,
            api_key=effective_api_key,
            api_base=effective_api_base,
            model_name=effective_model,
            prompt=full_prompt,
        )

        # 4. 解析回复与好感度变化与情绪
        reply, delta, emotion = self._parse_reply_and_delta(
            reply_text, allowed_emotions=emotions
        )

        # 5. 写入对话记忆（玩家+NPC）
        await memory.add_message(payload.session_id, "user", payload.query)
        await memory.add_message(payload.session_id, "assistant", reply)

        # 6. 更新好感度并落盘
        updated_state: NPCState = npc_manager.update_favorability(npc_name, delta)
        await npc_manager.save()

        return NPCChatResponse(
            reply=reply,
            npc_name=npc_name,
            favorability=updated_state.favorability,
            relationship_level=updated_state.relationship_level,
            favorability_change=delta,
            emotion=emotion,
        )

    def _retrieve_context(self, npc_name: str, query: str) -> str:
        """
        在同步线程中使用 LlamaIndex 检索 NPC 台词 + 世界观设定。
        """

        index: VectorStoreIndex = self._get_index()

        npc_filters = MetadataFilters(
            filters=[MetadataFilter(key="character", value=npc_name)]
        )
        lore_filters = MetadataFilters(
            filters=[MetadataFilter(key="type", value="world_lore")]
        )

        npc_retriever = index.as_retriever(
            similarity_top_k=6,
            filters=npc_filters,
        )
        lore_retriever = index.as_retriever(
            similarity_top_k=4,
            filters=lore_filters,
        )

        npc_nodes = npc_retriever.retrieve(query)
        lore_nodes = lore_retriever.retrieve(query)

        def _nodes_to_text(nodes) -> str:
            chunks: List[str] = []
            for node in nodes:
                text = getattr(node, "text", None)
                if text is None and hasattr(node, "get_content"):
                    text = node.get_content()
                if not text:
                    continue
                chunks.append(str(text))
            return "\n\n".join(chunks)

        parts: List[str] = []
        if npc_nodes:
            parts.append("【NPC 过往台词示例】\n" + _nodes_to_text(npc_nodes))
        if lore_nodes:
            parts.append("【世界观设定节选】\n" + _nodes_to_text(lore_nodes))

        return "\n\n".join(parts)

    async def _call_llm(
        self,
        settings: Settings,
        api_key: str,
        api_base: str,
        model_name: str,
        prompt: str,
    ) -> str:
        """
        调用任意 OpenAI 兼容的大模型生成回复。
        """

        client = AsyncOpenAI(api_key=api_key, base_url=api_base)

        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        message = completion.choices[0].message
        content = message.content or ""
        if isinstance(content, str):
            return content

        # OpenAI 兼容协议中 content 通常为 str；此处兜底处理为字符串拼接
        try:
            return "".join(part.get("text", "") for part in content)  # type: ignore[arg-type]
        except Exception:
            return str(content)

    @staticmethod
    def _parse_reply_and_delta(
        full_text: str,
        allowed_emotions: List[str],
    ) -> Tuple[str, int, str]:
        """
        将大模型输出解析为 (回复内容, 好感度变化值, 情绪)。
        """

        if not full_text.strip():
            return "（当前未能生成有效回复，请稍后再试。）", 0, "普通"

        lines = [line for line in full_text.splitlines() if line.strip()]
        if not lines:
            return "（当前未能生成有效回复，请稍后再试。）", 0, "普通"

        reply_line = lines[0].strip()
        json_line = lines[-1].strip()

        delta: int = 0
        emotion: str = "普通"
        try:
            obj = json.loads(json_line)
            value = int(obj.get("favorability_change", 0))
            # 限制在 -5 到 5 之间
            delta = max(-5, min(5, value))
            emo_raw = obj.get("emotion")
            if isinstance(emo_raw, str) and emo_raw.strip():
                emo_candidate = emo_raw.strip()
                if emo_candidate in allowed_emotions:
                    emotion = emo_candidate
        except Exception:
            delta = 0
            emotion = "普通"

        if emotion not in allowed_emotions:
            emotion = "普通"

        return reply_line, delta, emotion
