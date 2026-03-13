"""
游戏知识库 RAG 服务，基于 LlamaIndex + Gemini，支持 NPC 好感度管理。
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from openai import AsyncOpenAI

from ai_engine.game_data_loader import get_cached_index
from core.config import Settings, get_settings
from schemas.knowledge_schema import NPCChatRequest, NPCChatResponse
from services.npc_manager import NPCManager, NPCState
from services.memory_manager import MemoryManager

# ---------------------------------------------------------------------------
# 固定世界观背景（始终注入 prompt，不依赖 RAG 检索）
# ---------------------------------------------------------------------------
WORLD_BACKGROUND = (
    "近未来末世废土世界。曾经历科技爆发的\u201c统合时代\u201d，"
    "因基因编辑实用化和生化技术发展引发阶级大洗牌，社会矛盾激化。"
    "统合政府推行\u201c天网计划\u201d（全球监控网络）与"
    "\u201c诺亚方舟计划\u201d（超级战士研发）以维持统治，"
    "却因系统被病毒入侵、激光武器失控而在2012年触发\u201c审判日\u201d灾难："
    "卫星轰击、地层武器、天网机器人暴走、僵尸病毒、天灾，"
    "导致文明崩溃、环境荒漠化。\n"
    "灾后世界以\u201c尘都\u201d为剧情主要聚焦的区域（地理原型为成都），"
    "分裂为废城（丧尸肆虐）、堕落城（人类势力割据）、荒漠（军阀控制）等废墟地带。"
    "贵金属、纳米机器人、强化石、加密货币成为新世界的资源与货币基础。\n"
    "主要势力包括：佣兵组织A兵团（前期在废城活动）、"
    "军政府军阀（荒漠活动并对抗失控天网）、"
    "政教合一黑铁会（堕落城实控者，冷兵器、肉体强化）、"
    "商业偶像团体摇滚公园（堕落城二号组织，全女、高科技武器，名义上效忠黑铁会）、"
    "科技至上诺亚组织（在尘都之外，神秘重重）、"
    "宪政残余军警派（失势）、联合大学（堕落城组织），"
    "彼此在资源、理念与生存空间上博弈。\n"
    "核心主题：末世生存、科技伦理、阶级重构、秩序重建。"
    "角色多为基因/芯片改造者、佣兵、幸存者，在废墟中追寻力量、真相或新世界的可能。"
)


# ---------------------------------------------------------------------------
# 阵营别名配置（用于模糊匹配用户输入）
# ---------------------------------------------------------------------------
FACTION_ALIASES: Dict[str, List[str]] = {
    # 联合大学 -> 别名包含"大学"
    "联合大学": ["大学"],
    # 摇滚公园 -> 别名包含"摇滚"
    "摇滚公园": ["摇滚"],
}


def _normalize_text(text: str) -> str:
    """
    标准化文本：转为小写并移除所有空格（包括全角/半角空格、制表符等）。

    用于 NPC 名称、阵营、头衔的模糊匹配，提高检索鲁棒性。
    """
    if not text:
        return ""
    # 转为小写
    text = text.lower()
    # 移除所有空白字符（空格、制表符、换行等）
    text = "".join(text.split())
    return text


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

        effective_api_key: str | None = request_api_key or settings.llm_api_key
        effective_api_base: str = request_api_base or settings.llm_api_base
        effective_model: str = request_model or settings.llm_model_name

        if not effective_api_key:
            raise ValueError(
                "未提供可用的大模型 API Key，请在请求中传入 api_key 或在 .env 中配置 LLM_API_KEY。"
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
        faction: str | None = current_state.faction
        titles: list[str] = current_state.titles or []

        # 2. 使用 LlamaIndex 检索 NPC 过往台词 + 世界观设定
        retrieved_context: str = await asyncio.to_thread(
            self._retrieve_context, npc_name, payload.query
        )

        # 3. 构造 Prompt
        player_identity: str = (
            payload.player_identity.strip()
            if payload.player_identity and payload.player_identity.strip()
            else "一个末日后加入A兵团成为佣兵的幸存者"
        )

        sex_desc = f"（性别：{sex}）" if sex else ""
        faction_desc = f"（阵营：{faction}）" if faction else ""
        titles_desc = f"（身份或称呼：{'、'.join(titles)}）" if titles else ""

        # 3-a. NPC 交叉引用：检测玩家输入中提及的其他角色
        all_npc_states = npc_manager.state
        mentioned_npcs: List[str] = self._find_mentioned_npcs(
            payload.query, npc_name, all_npc_states, faction
        )
        mentioned_npcs_str = ""
        if mentioned_npcs:
            mentioned_npcs_str = (
                "可能涉及到的其他角色的设定（注意，和你不是一个阵营的角色你可能了解不多）：\n"
                + "\n".join(mentioned_npcs)
                + "\n\n"
            )

        # 3-b. 从记忆库中加载该会话最近的历史消息
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
            f"你现在扮演游戏角色「{npc_name}」{sex_desc}{faction_desc}{titles_desc}。\n"
            f"玩家的身份是：{player_identity}\n\n"
            "【世界观背景概要】\n"
            f"{WORLD_BACKGROUND}\n\n"
            f"{mentioned_npcs_str}"
            "下面是与你相关的检索设定和你的过往台词片段"
            "（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
            f"{retrieved_context or '（当前没有检索到任何上下文，你可以根据自己的设定自由发挥，但要保持合理。）'}\n\n"
            f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。\n"
            f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。请选择其中最合适的一种作为你当前的情绪立绘。\n"
            "请以符合你身份、当前好感度和所选情绪的语气，用简体中文回答玩家本次的发言。\n\n"
            f"{history_str}"
            "输出格式必须严格为两行：\n"
            "第一行：你的回复内容（只包含对话文本，不要包含 JSON，不要带前缀，不要有第一行：的字样）。\n"
            "第二行：一个 JSON 对象，必须且仅包含两个字段：\n"
            "  - \"favorability_change\"：一个整数字段，取值范围 -5 到 5，例如："
            "{\"favorability_change\": 1, \"emotion\": \"普通\"}\n"
            "  - \"emotion\"：一个字符串字段，值必须从上文提供的情绪标签中选择；"
            "如果没有特别合适的情绪，请使用 \"普通\"。\n"
            "常规对话无需调整好感度，小幅度的情绪起伏可以只+或-1点好感度。如果本次对话不应影响好感度，请输出 {\"favorability_change\": 0}。\n"
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
        print(full_prompt)
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

    # ------------------------------------------------------------------
    # NPC 交叉引用：从用户输入中发现提及的其他角色
    # ------------------------------------------------------------------
    @staticmethod
    def _find_mentioned_npcs(
        query: str,
        current_npc: str,
        all_npc_states: Dict[str, NPCState],
        current_faction: str | None,
    ) -> str:
        """
        扫描用户输入，如果提到了其他 NPC 的名称 / 阵营 / 头衔，
        返回其基本信息字符串供 prompt 使用。

        跳过阵营为 "成员" 或 "彩蛋" 的 NPC（除非当前 NPC 自己也属于这两个阵营）。

        匹配时会统一大小写并去空格，以处理用户输入中的大小写和空格差异。
        同时支持阵营别名匹配（如"大学"匹配"联合大学"，"摇滚"匹配"摇滚公园"）。
        """
        skip_factions = {"成员", "彩蛋"}
        if current_faction in skip_factions:
            skip_factions = set()

        mentioned: List[str] = []

        # 标准化玩家输入
        normalized_query = _normalize_text(query)

        for name, state in all_npc_states.items():
            if name == current_npc:
                continue
            if state.faction in skip_factions:
                continue

            # 收集所有匹配项：名称、阵营、头衔
            terms: List[str] = [name]
            if state.faction:
                terms.append(state.faction)
                # 添加阵营别名
                for faction_name, aliases in FACTION_ALIASES.items():
                    if state.faction == faction_name:
                        terms.extend(aliases)
            terms.extend(state.titles or [])

            # 标准化所有匹配项并进行匹配
            normalized_terms = [_normalize_text(t) for t in terms]

            if any(term in normalized_query for term in normalized_terms):
                parts = [f"「{name}」"]
                if state.sex:
                    parts.append(f"（性别：{state.sex}）")
                if state.faction:
                    parts.append(f"（阵营：{state.faction}）")
                if state.titles:
                    parts.append(f"（身份或称呼：{'、'.join(state.titles)}）")
                mentioned.append("".join(parts))

        return mentioned

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def _retrieve_context(self, npc_name: str, query: str) -> str:
        """
        在同步线程中使用 LlamaIndex 检索：
          1. NPC 过往台词 (character=npc_name)
          2. 核心世界观 (type=world_lore)
          3. 任务对话 (type=task)
          4. 补充设定 + 情报（pool: supplementary_lore & intelligence）
        """

        index: VectorStoreIndex = self._get_index()

        # --- 构建各类检索器 ---
        npc_retriever = index.as_retriever(
            similarity_top_k=5,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="character", value=npc_name)]
            ),
        )
        world_lore_retriever = index.as_retriever(
            similarity_top_k=2,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="world_lore")]
            ),
        )
        task_retriever = index.as_retriever(
            similarity_top_k=2,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="task")]
            ),
        )
        supp_retriever = index.as_retriever(
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="supplementary_lore")]
            ),
        )
        intel_retriever = index.as_retriever(
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="intelligence")]
            ),
        )

        # --- 执行检索 ---
        npc_nodes = npc_retriever.retrieve(query)
        world_lore_nodes = world_lore_retriever.retrieve(query)
        task_nodes = task_retriever.retrieve(query)
        supp_nodes = supp_retriever.retrieve(query)
        intel_nodes = intel_retriever.retrieve(query)

        # 补充设定 + 情报合并到同一池子，按相似度排序取最佳 2 条
        SUPP_SCORE_THRESHOLD = 0.30
        pooled = sorted(
            [
                n for n in (supp_nodes + intel_nodes)
                if (getattr(n, "score", None) or 0) >= SUPP_SCORE_THRESHOLD
            ],
            key=lambda n: getattr(n, "score", 0),
            reverse=True,
        )[:2]

        # --- 拼装上下文 ---
        def _nodes_to_text(nodes, max_chars: int | None = None) -> str:
            chunks: List[str] = []
            for node in nodes:
                text = getattr(node, "text", None)
                if text is None and hasattr(node, "get_content"):
                    text = node.get_content()
                if not text:
                    continue
                text = str(text).strip()
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars] + "…"
                chunks.append(text)
            return "\n\n".join(chunks)

        parts: List[str] = []
        if npc_nodes:
            parts.append("【NPC 过往台词示例】\n" + _nodes_to_text(npc_nodes))
        if world_lore_nodes:
            parts.append("【世界观设定摘取片段】\n" + _nodes_to_text(world_lore_nodes, max_chars=350))
        if task_nodes:
            parts.append("【参考任务对话(任务可能超过玩家当前进度，仅参考语气)】\n" + _nodes_to_text(task_nodes[:2], max_chars=250))
        if pooled:
            parts.append("【补充设定与情报参考】\n" + _nodes_to_text(pooled, max_chars=200))

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

    async def get_npc_favorability(
        self,
        npc_name: str,
        npc_manager: NPCManager,
    ) -> Tuple[str, int, str]:
        """
        获取指定 NPC 的好感度信息。

        Args:
            npc_name: NPC 名称
            npc_manager: NPCManager 实例

        Returns:
            (npc_name, favorability, relationship_level) 元组

        Raises:
            ValueError: 当 NPC 不存在时
        """
        npc_name = npc_name.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")

        current_state: NPCState | None = npc_manager.state.get(npc_name)
        if current_state is None:
            raise ValueError(f"NPC '{npc_name}' 不存在。")

        return (
            npc_name,
            current_state.favorability,
            current_state.relationship_level,
        )
