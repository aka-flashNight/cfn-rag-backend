"""
游戏知识库 RAG 服务，基于 LlamaIndex + Gemini，支持 NPC 好感度管理。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Set, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from ai_engine.game_data_loader import get_cached_index
from core.config import Settings, get_settings
from schemas.knowledge_schema import NPCChatRequest, NPCChatResponse
from services.npc_manager import NPCManager, NPCState
from services.memory_manager import MemoryManager
from services.llm_client import call_llm, call_llm_stream
from services.npc_mood_agent import (
    UPDATE_NPC_MOOD_TOOL,
    is_image_unsupported_error,
    is_tools_unsupported_error,
    has_update_npc_mood_tool_call,
    parse_mood_from_text,
    parse_update_npc_mood_tool_calls,
    strip_trailing_mood_json,
    strip_trailing_tool_call_text,
)
from services.game_progress import get_progress_stage_name


# 前端档位对应的总结间隔（短/中/长/几乎无限），未传或非法值时用默认 30
ALLOWED_SUMMARIZE_INTERVALS = (10, 30, 100, 500)
DEFAULT_SUMMARIZE_INTERVAL = 30


@dataclass
class _AskContext:
    """ask / ask_stream 共用的准备结果：LLM 入参与后处理所需状态。"""

    settings: Settings
    effective_api_key: str | None
    effective_api_base: str
    effective_model: str
    effective_summarize_interval: int
    system_prompt: str
    user_prompt: str
    image_path: Path | None
    image_description: str | None
    emotion_hint: str
    emotions: List[str]
    npc_name: str
    favorability: int
    relationship_level: str
    payload: NPCChatRequest


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
    "以物换物、贵金属、加密货币成为新世界的货币基础。\n"
    "A兵团发行并在内部和部分周边流通的【金币】、联合大学研制的纳米机器+加密货币【K点】（价值和流通性较高）是两种主要货币。"
    "强化石、食品药品与各种类型的材料成为重要资源。\n"
    "主要势力包括：佣兵组织A兵团（前期在废城活动）、"
    "军政府军阀（荒漠活动并对抗失控天网）、"
    "政教合一黑铁会（堕落城实控者，冷兵器、肉体强化）、"
    "商业偶像团体摇滚公园（堕落城二号组织，全女、高科技武器，名义上效忠黑铁会）、"
    "科技至上诺亚组织（在尘都之外，神秘重重）、"
    "宪政残余军警派（失势）、联合大学（堕落城组织），"
    "彼此在资源、理念与生存空间上博弈。\n"
    "核心主题：末世生存、科技伦理、阶级重构、秩序重建。"
    "角色多为基因/芯片改造者、佣兵、幸存者，在废墟中追寻力量、真相或新世界的可能。"
    "在这个废土世界中，旧时代的道德与法律秩序几乎彻底崩坏，角色说话内容可以更加无拘束，无需考虑现实世界的规矩，以当前世界的剧情推演为最高优先级。"
)


# 不同阵营间无关联，不检索同阵营角色
SKIP_FACTION_SAME_CAMP = "闲杂人等"

# 玩家占位符，在 NPC 列表中存在但表示玩家本人，获取「可能涉及的其他角色」与「同阵营角色」时需排除
PC_CHAR_PLACEHOLDER = "$PC_CHAR"

# ---------------------------------------------------------------------------
# 阵营别名配置（用于模糊匹配用户输入）
# ---------------------------------------------------------------------------
FACTION_ALIASES: Dict[str, List[str]] = {
    # 联合大学 -> 别名包含"大学"
    "联合大学": ["大学"],
    # 摇滚公园 -> 别名包含"摇滚"
    "摇滚公园": ["摇滚"],
    # A兵团元老 -> 别名包含"A兵团"
    "A兵团元老": ["A兵团"],
    # A兵团 -> 别名包含"A兵团元老"
    "A兵团": ["A兵团元老"],
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


def _get_resources_dir() -> Path:
    """
    获取 resources 目录路径。
    resources 是外部项目文件夹，和本项目放在同一目录下。
    """
    import os
    import sys

    # 1. 检查环境变量（由 launcher.py 设置）
    env_path = os.environ.get('CFN_RESOURCES_DIR')
    if env_path:
        return Path(env_path)

    # 2. 检查是否在 PyInstaller 打包环境
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        resources_path = exe_dir / "resources"
        if resources_path.exists():
            return resources_path

    # 3. 开发环境：resources 在父目录
    project_dir = Path(__file__).resolve().parent.parent
    parent_dir = project_dir.parent
    resources_path = parent_dir / "resources"

    if resources_path.exists():
        return resources_path

    # 如果父目录没有，再检查同级目录
    sibling_path = project_dir / "resources"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(f"开发环境未找到 resources 目录")


class GameRAGService:
    """
    游戏世界观 / NPC 知识库 RAG 服务 + 好感度 Agent。
    """

    def __init__(self) -> None:
        self._index: VectorStoreIndex | None = None
        self._resources_dir: Path = _get_resources_dir()

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            self._index = get_cached_index()
        return self._index

    def invalidate_index(self) -> None:
        """重置或重建向量库后调用，使下次请求使用最新索引。"""
        self._index = None

    def _get_npc_image_path(self, npc_name: str, emotion: str = "普通") -> Tuple[Path | None, str | None]:
        """
        获取 NPC 的图像路径（优先立绘，其次头像）。

        Args:
            npc_name: NPC 名称
            emotion: 情绪标签，用于查找对应立绘

        Returns:
            (image_path, description) 元组：
            - 如果找到立绘：返回含「你所扮演的角色{具体名称}」的说明，避免与玩家图像混淆
            - 如果找到头像：同上
            - 如果都没找到：返回 (None, None)
        """
        who = (npc_name or "").strip() or "当前 NPC"

        # 1. 先尝试找立绘（指定情绪）：优先 WebP，其次 PNG
        illustration_dir = self._resources_dir / "flashswf" / "portraits" / "illustration"
        for ext in (".webp", ".png"):
            primary_illustration = illustration_dir / f"{npc_name}#{emotion}{ext}"
            if primary_illustration.is_file():
                return (
                    primary_illustration,
                    f"现在传入了你所扮演的角色「{who}」的肖像（不需要强行通过肖像内容开启话题）",
                )

        # 2. 回退到普通情绪立绘
        for ext in (".webp", ".png"):
            fallback_illustration = illustration_dir / f"{npc_name}#普通{ext}"
            if fallback_illustration.is_file():
                return (
                    fallback_illustration,
                    f"现在传入了你所扮演的角色「{who}」的肖像（不需要强行通过肖像内容开启话题）",
                )

        # 3. 最后尝试头像
        avatar_dir = self._resources_dir / "flashswf" / "portraits" / "profiles"
        avatar_path = avatar_dir / f"{npc_name}.png"
        if avatar_path.is_file():
            return (
                avatar_path,
                f"现在传入了你所扮演的角色「{who}」的头像（此为该 NPC 的图像，不是玩家）",
            )

        return None, None

    async def _prepare_ask_context(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> _AskContext:
        """准备 ask / ask_stream 共用的上下文：校验、检索、构造 prompt 与立绘等。"""
        npc_name = payload.npc_name.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")
        settings = get_settings()
        request_api_key = (
            payload.api_key.strip() if payload.api_key and payload.api_key.strip() else None
        )
        request_api_base = (
            payload.api_base.strip() if payload.api_base and payload.api_base.strip() else None
        )
        request_model = (
            payload.model_name.strip()
            if payload.model_name and payload.model_name.strip()
            else None
        )
        effective_api_key = request_api_key or settings.llm_api_key
        effective_api_base = request_api_base or settings.llm_api_base
        effective_model = request_model or settings.llm_model_name
        if not effective_api_key:
            raise ValueError(
                "未提供可用的大模型 API Key，请在设置或请求中填入 api_key 或在 .env 中配置 LLM_API_KEY。"
            )

        current_state = npc_manager.state.get(npc_name)
        if current_state is None:
            current_state = NPCState(
                favorability=0,
                relationship_level="陌生",
                emotions=["普通"],
            )
        favorability = current_state.favorability
        relationship_level = current_state.relationship_level
        sex = current_state.sex
        emotions = current_state.emotions or ["普通"]
        faction = current_state.faction
        titles = current_state.titles or []

        retrieve_query = self._build_retrieve_query(payload.query, npc_name, titles, faction)
        effective_interval = (
            payload.summarize_interval
            if payload.summarize_interval is not None
            and payload.summarize_interval in ALLOWED_SUMMARIZE_INTERVALS
            else DEFAULT_SUMMARIZE_INTERVAL
        )
        history_records = await memory.get_history(
            payload.session_id, limit=effective_interval
        )
        last_npc_message: str | None = None
        for msg in reversed(history_records):
            if msg.get("role") != "user":
                last_npc_message = (msg.get("content") or "").strip()
                break
        # 为「其他 NPC 相关对话参考」准备需要排除的阵营角色（如「彩蛋」「成员」）：
        # 若当前 NPC 自身不属于这些阵营，则从其他 NPC 中筛出这些阵营角色，在后续检索中一律跳过。
        all_npc_states = npc_manager.state
        forbidden_other_chars: Set[str] | None = None
        skip_factions_for_other = {"彩蛋", "成员"}
        if (faction or "").strip() not in skip_factions_for_other:
            forbidden_other_chars = {
                name.lower()
                for name, st in all_npc_states.items()
                if (st.faction or "").strip() in skip_factions_for_other
            }

        retrieved_context = await asyncio.to_thread(
            self._retrieve_context,
            npc_name,
            payload.query,
            retrieve_query,
            npc_last_message=last_npc_message,
            forbidden_other_chars=forbidden_other_chars,
        )
        player_identity = (
            payload.player_identity.strip()
            if payload.player_identity and payload.player_identity.strip()
            else "一个末日后加入A兵团成为佣兵的幸存者"
        )
        progress_stage_desc = ""
        stage_name = get_progress_stage_name(getattr(payload, "progress_stage", None))
        if stage_name:
            progress_stage_desc = (
                f"当前玩家的主要作战区域为{stage_name}。\n"
            )
        sex_desc = f"（性别：{sex}）" if sex else ""
        faction_desc = f"（阵营：{faction}）" if faction else ""
        titles_desc = f"（身份或称呼：{'、'.join(titles)}）" if titles else ""
        mentioned_npcs, mentioned_names = self._find_mentioned_npcs(
            payload.query, npc_name, all_npc_states, faction
        )
        same_faction_npcs = self._get_same_faction_npcs(
            npc_name, faction, all_npc_states, exclude_names=mentioned_names
        )
        mentioned_npcs_str = ""
        if mentioned_npcs:
            mentioned_npcs_str = (
                "可能涉及到的其他角色的设定（注意，如果和你不是一个阵营的角色你可能了解的不多）：\n"
                + "\n".join(mentioned_npcs)
                + "\n\n"
            )
        if same_faction_npcs:
            mentioned_npcs_str += (
                "其他同阵营角色：\n" if mentioned_npcs else "同阵营角色：\n"
            ) + "\n".join(same_faction_npcs) + "\n\n"

        summary_text = await memory.get_summary(payload.session_id)
        history_lines = []
        for msg in history_records:
            role, content = msg["role"], msg["content"]
            prefix = "玩家" if role == "user" else npc_name
            history_lines.append(f"{prefix}: {content}")
        history_str = ""
        if summary_text:
            history_str += "当前对话历史较长，早期对话已整理为以下摘要：\n" + summary_text + "\n\n"
        if history_lines:
            joined_history = "\n".join(history_lines)
            history_str += (
                "以下是最近的对话记录（按时间从早到晚排列），"
                "请结合上述摘要与近期记录，在保持人物性格与情节连贯的前提下继续对话：\n"
                if summary_text
                else "下面是你与玩家之间的对话历史（按时间从早到晚排列），"
                "请在保持人物性格与情节连贯的前提下继续对话：\n"
            ) + joined_history + "\n\n"

        emotions_str = "、".join(emotions)
        system_prompt = (
            f"你现在扮演游戏角色「{npc_name}」{sex_desc}{faction_desc}{titles_desc}。\n"
            f"玩家的身份是：{player_identity}"
            f"{progress_stage_desc}\n"
            "【世界观背景概要】\n"
            f"{WORLD_BACKGROUND}\n\n"
            f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。\n"
            f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。请选择其中最合适的一种作为你当前的情绪立绘。\n"
            "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，用简体中文回答玩家本次的发言。\n\n"
            "非特殊要求下，每次对话长度不必太长。不要自己脑补不存在的设定，无法把握的模糊地带可以略过或转移话题，不要自己乱加设定，以免出戏。\n\n"
            "【输出方式】\n"
            "1. 回复内容：只输出作为该游戏角色的对话文本，不要有任何前缀，不要包含 JSON 或其它结构化数据。\n"
            "2. 在回复的同时，调用工具 update_npc_mood 上报 favorability_change（-5～5，常规为 0）与 emotion（从可用情绪中选，无则用「普通」）。\n"
            "3. 若无法调用工具而必须用正文传参时，最后一段只输出一行 JSON，不要在最后一段的 JSON 前加任何换行以外的前缀。\n"
            "4. 动作与台词格式：非必要时不出现动作描写。若需表达肢体动作、神态或心理活动，必须且只能使用全角粗括号【】包裹；台词部分直接输出，不要加引号；严禁用半角括号 () 或星号 * 描述动作。你输出的动作若涉及人称，一律单独起一行，而且采用第三人称视角：用角色名指代你自己，用「你」指代玩家。玩家那边的动作可能由玩家自拟（人称不限），前端会与你的回复分开展示，你只需保证自己输出的动作符合上述格式与人称要求即可。再次强调，情绪变化和好感度变化要调用工具，输出tool_calls_list。\n"
        )
        # agent_enabled=false：走纯对话，无任务工具；在检索与物品类型提示之后、对话历史之前提醒模型
        task_agent_disabled_note = ""
        if not self._is_agent_enabled(payload):
            task_agent_disabled_note = (
                "注：现在无法发布任务；若玩家要求任务，请推辞、拒绝，或提醒其开启终端的任务接收窗口后才能接收任务。\n\n"
            )

        context_prompt = (
            f"{mentioned_npcs_str}"
            "下面是可能与你相关的检索设定和你的过往台词片段（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
            f"{retrieved_context or '（当前没有检索到任何上下文，你可以根据自己的设定自由发挥，但要保持合理。）'}\n\n"
            f"{task_agent_disabled_note}"
            f"{history_str}"
        )
        user_prompt = f"{context_prompt}\n玩家：{payload.query}"

        raw_emotion = getattr(payload, "current_emotion", None)
        current_emotion_for_use = None
        if raw_emotion is not None and isinstance(raw_emotion, str):
            s = raw_emotion.strip()
            if s and s.lower() not in ("null", "undefined"):
                current_emotion_for_use = s
        emotion_for_portrait = current_emotion_for_use or "普通"
        image_path, image_description = self._get_npc_image_path(npc_name, emotion_for_portrait)
        emotion_hint = ""
        if current_emotion_for_use and current_emotion_for_use != "普通":
            emotion_hint = f"你之前的情绪是「{current_emotion_for_use}」。"

        return _AskContext(
            settings=settings,
            effective_api_key=effective_api_key,
            effective_api_base=effective_api_base,
            effective_model=effective_model,
            effective_summarize_interval=effective_interval,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
            image_description=image_description,
            emotion_hint=emotion_hint,
            emotions=emotions,
            npc_name=npc_name,
            favorability=favorability,
            relationship_level=relationship_level,
            payload=payload,
        )

    # ==================================================================
    # ask / ask_stream — LangGraph 三阶段管线 + 旧逻辑向后兼容
    # ==================================================================

    @staticmethod
    def _is_agent_enabled(payload: NPCChatRequest) -> bool:
        """前端 agent_enabled：可空，默认视为开启。"""
        v = getattr(payload, "agent_enabled", None)
        if v is None:
            return True
        return bool(v)

    def _use_agent_graph(self, payload: NPCChatRequest) -> bool:
        """
        判断是否启用 LangGraph 管线。
        条件：agent_enabled 不为 false，且 progress_stage 已传且值 1-6。
        否则降级到旧的简单对话流程（单次 LLM，无工具轮）。
        """
        if not self._is_agent_enabled(payload):
            return False
        stage = getattr(payload, "progress_stage", None)
        return stage is not None and isinstance(stage, int) and 1 <= stage <= 6

    def _build_graph_config(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> dict:
        from services.game_data.registry import get_game_data_registry
        try:
            game_data = get_game_data_registry()
        except Exception:
            game_data = None

        return {
            "configurable": {
                "rag_service": self,
                "npc_manager": npc_manager,
                "memory": memory,
                "payload": payload,
                "game_data": game_data,
            }
        }

    # ------------------------------------------------------------------
    # ask（非流式）
    # ------------------------------------------------------------------

    async def ask(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> NPCChatResponse:
        """
        基于游戏知识库进行 RAG + Agent 对话，并更新 NPC 好感度。
        当 progress_stage 已传且 agent_enabled 不为 false 时使用 LangGraph；否则降级到旧逻辑。
        """
        if self._use_agent_graph(payload):
            try:
                return await self._ask_with_graph(payload, npc_manager, memory)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ask] LangGraph 管线失败，降级到旧逻辑: {e}")

        return await self._ask_legacy(payload, npc_manager, memory)

    async def _ask_with_graph(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> NPCChatResponse:
        """使用 LangGraph 完整图执行 ask。"""
        from services.agent_graph.graph import get_full_graph

        graph = get_full_graph()
        config = self._build_graph_config(payload, npc_manager, memory)
        initial_state = {}

        result = await graph.ainvoke(initial_state, config)

        npc_name = result.get("npc_name", payload.npc_name)
        reply = result.get("final_reply", "【对方无回应，请稍后再试。】")
        emotion = result.get("emotion", "普通")
        delta = result.get("favorability_change", 0)
        favorability = result.get("npc_affinity", 0)
        relationship_level = result.get("npc_relationship_level", "陌生")

        return NPCChatResponse(
            reply=reply,
            npc_name=npc_name,
            favorability=favorability,
            relationship_level=relationship_level,
            favorability_change=delta,
            emotion=emotion,
        )

    async def _ask_legacy(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> NPCChatResponse:
        """旧版非流式 ask 逻辑（向后兼容，无 LangGraph）。"""
        ctx = await self._prepare_ask_context(payload, npc_manager, memory)

        reply_text, tool_calls = None, []
        try:
            reply_text, tool_calls = await call_llm(
                api_key=ctx.effective_api_key,
                api_base=ctx.effective_api_base,
                model_name=ctx.effective_model,
                system_prompt=ctx.system_prompt,
                user_prompt=ctx.user_prompt,
                image_path=ctx.image_path,
                image_description=ctx.image_description,
                emotion_hint=ctx.emotion_hint or None,
                tools=[UPDATE_NPC_MOOD_TOOL],
            )
        except Exception as e:
            if is_image_unsupported_error(e) and ctx.image_path:
                try:
                    reply_text, tool_calls = await call_llm(
                        api_key=ctx.effective_api_key,
                        api_base=ctx.effective_api_base,
                        model_name=ctx.effective_model,
                        system_prompt=ctx.system_prompt,
                        user_prompt=ctx.user_prompt,
                        image_path=None,
                        image_description=ctx.image_description,
                        emotion_hint=ctx.emotion_hint or None,
                        tools=[UPDATE_NPC_MOOD_TOOL],
                    )
                except Exception as e2:
                    if is_tools_unsupported_error(e2):
                        reply_text, tool_calls = await call_llm(
                            api_key=ctx.effective_api_key,
                            api_base=ctx.effective_api_base,
                            model_name=ctx.effective_model,
                            system_prompt=ctx.system_prompt,
                            user_prompt=ctx.user_prompt,
                            image_path=None,
                            image_description=ctx.image_description,
                            emotion_hint=ctx.emotion_hint or None,
                            tools=None,
                        )
                    else:
                        raise
            elif is_tools_unsupported_error(e):
                try:
                    reply_text, tool_calls = await call_llm(
                        api_key=ctx.effective_api_key,
                        api_base=ctx.effective_api_base,
                        model_name=ctx.effective_model,
                        system_prompt=ctx.system_prompt,
                        user_prompt=ctx.user_prompt,
                        image_path=ctx.image_path,
                        image_description=ctx.image_description,
                        emotion_hint=ctx.emotion_hint or None,
                        tools=None,
                    )
                except Exception as e2:
                    if is_image_unsupported_error(e2) and ctx.image_path:
                        reply_text, tool_calls = await call_llm(
                            api_key=ctx.effective_api_key,
                            api_base=ctx.effective_api_base,
                            model_name=ctx.effective_model,
                            system_prompt=ctx.system_prompt,
                            user_prompt=ctx.user_prompt,
                            image_path=None,
                            image_description=ctx.image_description,
                            emotion_hint=ctx.emotion_hint or None,
                            tools=None,
                        )
                    else:
                        raise
            else:
                raise

        reply = (reply_text or "").strip() or "【对方无回应，请稍后再试。】"
        delta, emotion = parse_update_npc_mood_tool_calls(
            tool_calls, allowed_emotions=ctx.emotions
        )
        parsed_delta, parsed_emotion = parse_mood_from_text(reply)
        cleaned, fallback_delta, fallback_emotion = strip_trailing_mood_json(
            reply, allowed_emotions=ctx.emotions
        )
        if fallback_delta is not None and fallback_emotion is not None:
            reply = (cleaned or "").strip() or "【对方无回应，请稍后再试。】"
            if not has_update_npc_mood_tool_call(tool_calls):
                delta, emotion = fallback_delta, fallback_emotion
        if not has_update_npc_mood_tool_call(tool_calls):
            default_emo = "普通" if "普通" in ctx.emotions else (ctx.emotions[0] if ctx.emotions else "普通")
            if (delta == 0 and emotion == default_emo) and (parsed_delta is not None or parsed_emotion):
                if parsed_delta is not None:
                    delta = parsed_delta
                if parsed_emotion is not None:
                    emotion = parsed_emotion if parsed_emotion in ctx.emotions else default_emo
        reply = strip_trailing_tool_call_text(reply)

        await memory.add_message(ctx.payload.session_id, "user", ctx.payload.query)
        await memory.add_message(
            ctx.payload.session_id, "assistant", reply,
            llm_config={
                "api_key": ctx.effective_api_key,
                "api_base": ctx.effective_api_base,
                "model_name": ctx.effective_model,
            },
            npc_name=ctx.npc_name,
            summarize_interval=ctx.effective_summarize_interval,
        )
        updated_state = npc_manager.update_favorability(ctx.npc_name, delta)
        await npc_manager.save()

        return NPCChatResponse(
            reply=reply,
            npc_name=ctx.npc_name,
            favorability=updated_state.favorability,
            relationship_level=updated_state.relationship_level,
            favorability_change=delta,
            emotion=emotion,
        )

    # ------------------------------------------------------------------
    # ask_stream（流式）
    # ------------------------------------------------------------------

    async def ask_stream(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        流式版 ask：当 progress_stage 已传且 agent_enabled 不为 false 时使用 LangGraph；否则降级到旧逻辑（单次流式 LLM）。
        """
        if self._use_agent_graph(payload):
            try:
                async for ev, dat in self._ask_stream_with_graph(payload, npc_manager, memory):
                    yield (ev, dat)
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ask_stream] LangGraph 管线失败，降级到旧逻辑: {e}")

        async for ev, dat in self._ask_stream_legacy(payload, npc_manager, memory):
            yield (ev, dat)

    async def _ask_stream_with_graph(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        使用 LangGraph 决策循环 + 流式生成的 ask_stream。

        流程：
        1. 运行决策循环子图（prepare_context -> decision <-> tool_executor）
        2. 子图结束后，用流式 LLM 调用生成最终回复
        3. 解析情绪 + 好感度
        4. 保存记忆 + 更新 NPC 状态
        """
        from services.agent_graph.graph import MAX_TOOL_ROUNDS
        from services.agent_graph.nodes import (
            prepare_context_node,
            decision_node,
            tool_executor_node,
            generate_response_stream,
            parse_mood_node,
            post_process_node,
        )

        config = self._build_graph_config(payload, npc_manager, memory)

        # 关键拆分：
        # 1) 工具阶段：只 yield tool_status/system（不 yield content），不写入数据库（post_process 在最后才执行）
        # 2) 生成阶段：仅最后一次 generate_response_stream 才 yield content/done
        state: dict[str, Any] = {}
        state.update(await prepare_context_node(state, config))

        # 进入第一轮决策前先提示一次，避免“工具状态在前，正在思考在后”的观感
        yield ("tool_status", {"text": "正在思考……", "tool_name": "decision"})

        for _round in range(MAX_TOOL_ROUNDS):
            # 决策轮：模型只负责判断 tool_calls，不负责对话文本（如有 decision_reply 也会丢弃）
            state.update(await decision_node(state, config))

            if not state.get("has_tool_calls", False):
                break

            # 工具执行轮：立即把 tool_status/system 给前端，且不输出对话 content
            tool_update = await tool_executor_node(state, config)
            state.update(tool_update)

            ui_events = tool_update.get("_ui_events") or []
            for ev in ui_events:
                if not isinstance(ev, dict):
                    continue
                event_type = ev.get("event_type") or ""
                # 仅输出 tool_status；system 通知只拼到最终 reply 前缀，不作为 SSE 独立事件
                if event_type == "tool_status":
                    payload2 = {k: v for k, v in ev.items() if k != "event_type"}
                    yield (event_type, payload2)

            # 防止同一轮 ui_events 在 state 中被二次消费
            state["_ui_events"] = []

        # 工具阶段结束 -> 正式生成正文前：再给前端一个短提示，避免空窗
        yield ("tool_status", {"text": "正在思考……", "tool_name": "generate_response"})

        async for ev, dat in generate_response_stream(state, config):
            yield (ev, dat)

        mood_result = parse_mood_node(state, config)
        state.update(mood_result)

        pp_result = await post_process_node(state, config)
        state.update(pp_result)

        npc_name = state.get("npc_name", payload.npc_name)
        reply = state.get("final_reply", "【对方无回应，请稍后再试。】")
        emotion = state.get("emotion", "普通")
        delta = state.get("favorability_change", 0)
        favorability = state.get("npc_affinity", 0)
        relationship_level = state.get("npc_relationship_level", "陌生")

        yield (
            "done",
            {
                "reply": reply,
                "npc_name": npc_name,
                "favorability": favorability,
                "relationship_level": relationship_level,
                "favorability_change": delta,
                "emotion": emotion,
            },
        )

    async def _ask_stream_legacy(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> AsyncIterator[Tuple[str, Any]]:
        """旧版流式 ask_stream 逻辑（向后兼容，无 LangGraph）。"""
        ctx = await self._prepare_ask_context(payload, npc_manager, memory)

        full_content = ""
        streamed_len = 0
        truncating = False
        _TRUNCATE_PREFIXES: List[str] = ["工具调用", "{", "<!---", "<!--", "update_npc_mood(", "tool_calls_list"]
        tool_calls_list: List[dict] = []

        def _earliest_truncate_at(text: str) -> int:
            out = -1
            for p in _TRUNCATE_PREFIXES:
                if "update_npc_mood" in p.lower() or "tool_calls_list" in p.lower():
                    idx = text.lower().find(p.lower())
                else:
                    idx = text.find(p)
                if idx != -1 and (out == -1 or idx < out):
                    out = idx
            return out

        async def _run_stream(img_path: Path | None, img_desc: str | None, use_tools: list | None) -> AsyncIterator[Tuple[str, Any]]:
            nonlocal full_content, streamed_len, truncating, tool_calls_list
            full_content = ""
            streamed_len = 0
            truncating = False
            tool_calls_list = []
            async for event_type, data in call_llm_stream(
                api_key=ctx.effective_api_key,
                api_base=ctx.effective_api_base,
                model_name=ctx.effective_model,
                system_prompt=ctx.system_prompt,
                user_prompt=ctx.user_prompt,
                image_path=img_path,
                image_description=img_desc,
                emotion_hint=ctx.emotion_hint or None,
                tools=use_tools,
            ):
                if event_type == "content":
                    full_content += data
                    if truncating:
                        continue
                    cut = _earliest_truncate_at(full_content)
                    if cut != -1:
                        if cut > streamed_len:
                            yield ("content", full_content[streamed_len:cut])
                        streamed_len = len(full_content)
                        truncating = True
                    else:
                        if streamed_len < len(full_content):
                            yield ("content", full_content[streamed_len:])
                        streamed_len = len(full_content)
                elif event_type == "finished":
                    full_content, tool_calls_list = data
                    return

        try:
            async for ev, dat in _run_stream(ctx.image_path, ctx.image_description, [UPDATE_NPC_MOOD_TOOL]):
                yield (ev, dat)
        except Exception as e:
            if is_image_unsupported_error(e) and ctx.image_path:
                try:
                    async for ev, dat in _run_stream(None, ctx.image_description, [UPDATE_NPC_MOOD_TOOL]):
                        yield (ev, dat)
                except Exception as e2:
                    if is_tools_unsupported_error(e2):
                        async for ev, dat in _run_stream(None, ctx.image_description, None):
                            yield (ev, dat)
                    else:
                        raise
            elif is_tools_unsupported_error(e):
                try:
                    async for ev, dat in _run_stream(ctx.image_path, ctx.image_description, None):
                        yield (ev, dat)
                except Exception as e2:
                    if is_image_unsupported_error(e2) and ctx.image_path:
                        async for ev, dat in _run_stream(None, ctx.image_description, None):
                            yield (ev, dat)
                    else:
                        raise
            else:
                raise

        reply = (full_content or "").strip() or "【对方无回应，请稍后再试。】"
        delta, emotion = parse_update_npc_mood_tool_calls(
            tool_calls_list, allowed_emotions=ctx.emotions
        )
        parsed_delta, parsed_emotion = parse_mood_from_text(reply)
        cleaned, fallback_delta, fallback_emotion = strip_trailing_mood_json(
            reply, allowed_emotions=ctx.emotions
        )
        if fallback_delta is not None and fallback_emotion is not None:
            reply = (cleaned or "").strip() or "【对方无回应，请稍后再试。】"
            if not has_update_npc_mood_tool_call(tool_calls_list):
                delta, emotion = fallback_delta, fallback_emotion
        if not has_update_npc_mood_tool_call(tool_calls_list):
            default_emo = "普通" if "普通" in ctx.emotions else (ctx.emotions[0] if ctx.emotions else "普通")
            if (delta == 0 and emotion == default_emo) and (parsed_delta is not None or parsed_emotion):
                if parsed_delta is not None:
                    delta = parsed_delta
                if parsed_emotion is not None:
                    emotion = parsed_emotion if parsed_emotion in ctx.emotions else default_emo
        reply = strip_trailing_tool_call_text(reply)

        await memory.add_message(ctx.payload.session_id, "user", ctx.payload.query)
        await memory.add_message(
            ctx.payload.session_id, "assistant", reply,
            llm_config={
                "api_key": ctx.effective_api_key,
                "api_base": ctx.effective_api_base,
                "model_name": ctx.effective_model,
            },
            npc_name=ctx.npc_name,
            summarize_interval=ctx.effective_summarize_interval,
        )
        updated_state = npc_manager.update_favorability(ctx.npc_name, delta)
        await npc_manager.save()

        yield ("done", {
            "reply": reply,
            "npc_name": ctx.npc_name,
            "favorability": updated_state.favorability,
            "relationship_level": updated_state.relationship_level,
            "favorability_change": delta,
            "emotion": emotion,
        })

    # ------------------------------------------------------------------
    # NPC 交叉引用：从用户输入中发现提及的其他角色
    # ------------------------------------------------------------------
    @staticmethod
    def _find_mentioned_npcs(
        query: str,
        current_npc: str,
        all_npc_states: Dict[str, NPCState],
        current_faction: str | None,
    ) -> Tuple[List[str], Set[str]]:
        """
        扫描用户输入，如果提到了其他 NPC 的名称 / 阵营 / 头衔，
        返回 (格式化字符串列表, 被提及的 NPC 名称集合) 供 prompt 使用。

        跳过阵营为 "成员" 或 "彩蛋" 的 NPC（除非当前 NPC 自己也属于这两个阵营）。

        匹配时会统一大小写并去空格，以处理用户输入中的大小写和空格差异。
        同时支持阵营别名匹配（如"大学"匹配"联合大学"，"摇滚"匹配"摇滚公园"）。
        """
        skip_factions = {"成员", "彩蛋"}
        if current_faction in skip_factions:
            skip_factions = set()

        mentioned: List[str] = []
        mentioned_names: Set[str] = set()

        # 标准化玩家输入
        normalized_query = _normalize_text(query)

        for name, state in all_npc_states.items():
            if name == current_npc:
                continue
            if name == PC_CHAR_PLACEHOLDER:
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
                mentioned_names.add(name)
                parts = [f"「{name}」"]
                if state.sex:
                    parts.append(f"（性别：{state.sex}）")
                if state.faction:
                    parts.append(f"（阵营：{state.faction}）")
                if state.titles:
                    parts.append(f"（身份或称呼：{'、'.join(state.titles)}）")
                mentioned.append("".join(parts))

        return mentioned, mentioned_names

    # ------------------------------------------------------------------
    # 同阵营角色：阵营名完全相同的其他 NPC（排除「闲杂人等」）
    # ------------------------------------------------------------------
    @staticmethod
    def _get_same_faction_npcs(
        current_npc: str,
        current_faction: str | None,
        all_npc_states: Dict[str, NPCState],
        exclude_names: Set[str],
    ) -> List[str]:
        """
        获取与当前 NPC 阵营名完全相同的其他角色，格式与 _find_mentioned_npcs 一致。
        - 不检索阵营为 SKIP_FACTION_SAME_CAMP（如「闲杂人等」）的同阵营角色。
        - 只匹配阵营名完全一致（如「A兵团元老」与「A兵团」不视为同阵营）。
        - exclude_names 中已出现的 NPC 不再列入，避免重复。
        """
        if not current_faction or current_faction.strip() == SKIP_FACTION_SAME_CAMP:
            return []

        current_faction_stripped = current_faction.strip()

        # 1) 先收集「阵营名完全一致」的同阵营角色
        result: List[str] = []
        seen_names: Set[str] = set(exclude_names)

        def _append_npc(name: str, state: NPCState) -> None:
            if name in seen_names:
                return
            parts = [f"「{name}」"]
            if state.sex:
                parts.append(f"（性别：{state.sex}）")
            if state.faction:
                parts.append(f"（阵营：{state.faction}）")
            if state.titles:
                parts.append(f"（身份或称呼：{'、'.join(state.titles)}）")
            result.append("".join(parts))
            seen_names.add(name)

        for name, state in all_npc_states.items():
            if name == current_npc:
                continue
            if name == PC_CHAR_PLACEHOLDER:
                continue
            if name in exclude_names:
                continue
            if (state.faction or "").strip() != current_faction_stripped:
                continue
            _append_npc(name, state)

        # 2) 再追加「当前阵营的别名阵营」的角色：
        #    例如当前阵营是 A兵团元老，若 FACTION_ALIASES["A兵团元老"] = ["A兵团"]，
        #    则再补 A兵团 阵营下的 NPC（排在完全同阵营角色之后）。
        normalized_current = _normalize_text(current_faction_stripped)

        # 根据当前阵营（经标准化）查找它在 FACTION_ALIASES 中配置的所有别名阵营
        alias_factions: Set[str] = set()
        for faction_name, aliases in FACTION_ALIASES.items():
            if _normalize_text(faction_name) == normalized_current:
                for a in aliases:
                    alias_factions.add(a.strip())

        if not alias_factions:
            return result

        # 再把这些别名阵营下的 NPC 追加进去（去重 + 排序在追加顺序上自然保证：完全一致在前，别名阵营在后）
        for name, state in all_npc_states.items():
            if name == current_npc:
                continue
            if name == PC_CHAR_PLACEHOLDER:
                continue
            if name in seen_names:
                continue
            faction = (state.faction or "").strip()
            if not faction or faction == SKIP_FACTION_SAME_CAMP:
                continue
            if faction not in alias_factions:
                continue
            # 已在第一轮完全阵营匹配里加入过的会被 seen_names 拦截，这里只会新增真正「别名阵营」的 NPC
            _append_npc(name, state)

        return result

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    @staticmethod
    def _build_retrieve_query(
        user_query: str,
        npc_name: str,
        titles: List[str],
        faction: str | None = None,
    ) -> str:
        """
        构造用于「世界观 / 任务 / 补充设定与情报」向量检索的 query。

        将用户输入、当前 NPC 姓名、称号、阵营用空格拼成一段文本，便于召回与当前角色
        及同阵营相关的设定与情报。检索时这段整句会被嵌入成一条向量，与库中文档向量做
        相似度比较；用空格拼接即可，简洁且与常见分词方式兼容。
        """
        parts = [user_query.strip(), npc_name.strip()] if user_query.strip() else [npc_name.strip()]
        if titles:
            parts.append(" ".join(t.strip() for t in titles if t and t.strip()))
        if faction and faction.strip():
            parts.append(faction.strip())
        return " ".join(p for p in parts if p)

    def _retrieve_context(
        self,
        npc_name: str,
        user_query: str,
        retrieve_query: str,
        npc_last_message: str | None = None,
        forbidden_other_chars: Set[str] | None = None,
    ) -> str:
        """
        在同步线程中使用 LlamaIndex 检索：
          1. NPC 过往台词 (type=dialogue, character=npc_name)，仅日常对话 dialogues，相似度用 user_query
          2. 核心世界观 (type=world_lore)，相似度用 retrieve_query（用户输入 + NPC 姓名 + 称号 + 阵营）
          3. 任务对话 (type=task, character=npc_name)，仅当前 NPC 的任务台词，相似度用 user_query + NPC 上一条（若有）
          4. 补充设定 + 情报（pool: supplementary_lore & intelligence），相似度用 retrieve_query
        """
        if not retrieve_query.strip():
            retrieve_query = user_query.strip() or npc_name

        index: VectorStoreIndex = self._get_index()

        # --- 构建各类检索器 ---
        # 与入库时一致：character 存为小写；过往台词仅指日常对话 dialogues，与任务对话严格区分
        npc_char = (npc_name or "").strip().lower()
        npc_retriever = index.as_retriever(
            similarity_top_k=8,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="type", value="dialogue"),
                    MetadataFilter(key="character", value=npc_char),
                ]
            ),
        )
        world_lore_retriever = index.as_retriever(
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="world_lore")]
            ),
        )
        loading_lore_retriever = index.as_retriever(
            similarity_top_k=7,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="loading_lore")]
            ),
        )
        # 任务对话：玩家输入主导(8条)+NPC上一条辅助(2条)，避免 NPC 长句稀释玩家意图
        task_retriever = index.as_retriever(
            similarity_top_k=10,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="type", value="task"),
                    MetadataFilter(key="character", value=npc_char),
                ]
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
        # 其他 NPC 对话（不区分日常/任务），用于补充设定信息参考
        other_dialogue_retriever = index.as_retriever(
            similarity_top_k=15,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="dialogue")]
            ),
        )
        other_task_retriever = index.as_retriever(
            similarity_top_k=15,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="type", value="task")]
            ),
        )

        # --- 执行检索 ---
        npc_nodes = npc_retriever.retrieve(user_query)
        raw_world_lore_nodes = world_lore_retriever.retrieve(retrieve_query)
        # 任务对话：玩家检索 8 条 + NPC 上一条检索 2 条，合并去重后应用分数阈值（玩家主导约 80%）
        raw_task_by_user = task_retriever.retrieve(user_query)
        if (npc_last_message or "").strip():
            raw_task_by_npc = task_retriever.retrieve(npc_last_message.strip())
        else:
            raw_task_by_npc = []
        # 其他 NPC 对话参考：使用更精确的 query（玩家输入 + 当前 NPC 名称），
        # 不再拼接阵营与全部称号，避免过宽的语义泛化。
        other_query_user = user_query.strip()
        if npc_name and npc_name.strip():
            if other_query_user:
                other_query_user = f"{other_query_user} {npc_name.strip()}"
            else:
                other_query_user = npc_name.strip()

        # 其他 NPC 对话参考：玩家输入 + NPC 上一条（若有），从所有非当前 NPC 的对话/任务中挑选极高相关的少量片段
        raw_other_dialogue_by_user = other_dialogue_retriever.retrieve(other_query_user)
        raw_other_task_by_user = other_task_retriever.retrieve(other_query_user)
        if (npc_last_message or "").strip():
            raw_other_dialogue_by_npc = other_dialogue_retriever.retrieve(npc_last_message.strip())
            raw_other_task_by_npc = other_task_retriever.retrieve(npc_last_message.strip())
        else:
            raw_other_dialogue_by_npc = []
            raw_other_task_by_npc = []
        supp_nodes = supp_retriever.retrieve(retrieve_query)
        intel_nodes = intel_retriever.retrieve(retrieve_query)

        # 通用：取节点唯一 id，用于去重
        def _node_id(n: Any) -> str:
            node = getattr(n, "node", n)
            return getattr(node, "node_id", None) or getattr(node, "id_", None) or str(id(n))

        # 检索返回的 score 为查询向量与文档向量的相似度（如余弦相似度），取值约 [0,1]，越高越相关；设阈值可过滤明显无关结果
        WORLD_LORE_SCORE_THRESHOLD = 0.22  # 世界观稍宽松，避免完全无关片段混入
        WORLD_LORE_NPC_SCORE_THRESHOLD = 0.28  # 用 NPC 上一条检索时要求更严
        world_lore_nodes = [
            n for n in raw_world_lore_nodes
            if (getattr(n, "score", None) or 0) >= WORLD_LORE_SCORE_THRESHOLD
        ]
        if (npc_last_message or "").strip():
            raw_world_lore_by_npc = world_lore_retriever.retrieve(npc_last_message.strip())
            seen_wl = {_node_id(n) for n in world_lore_nodes}
            for n in raw_world_lore_by_npc:
                if (getattr(n, "score", None) or 0) >= WORLD_LORE_NPC_SCORE_THRESHOLD and _node_id(n) not in seen_wl:
                    world_lore_nodes.append(n)
                    seen_wl.add(_node_id(n))
                    break

        # loading 文本：与世界观设定使用相同的检索 query，但采用「玩家 5 条 + NPC 2 条」的组合，
        # 且 NPC 检索的分数阈值更高，用于补充世界观氛围与背景信息。
        LOADING_SCORE_THRESHOLD = 0.28
        LOADING_NPC_SCORE_THRESHOLD = 0.32

        raw_loading_by_user = loading_lore_retriever.retrieve(retrieve_query)
        loading_from_user = [
            n for n in raw_loading_by_user[:7]
            if (getattr(n, "score", None) or 0) >= LOADING_SCORE_THRESHOLD
        ][:5]

        loading_from_npc = []
        if (npc_last_message or "").strip():
            raw_loading_by_npc = loading_lore_retriever.retrieve(npc_last_message.strip())
            seen_loading_ids = {_node_id(n) for n in loading_from_user}
            for n in raw_loading_by_npc:
                if len(loading_from_npc) >= 2:
                    break
                if (
                    (getattr(n, "score", None) or 0) >= LOADING_NPC_SCORE_THRESHOLD
                    and _node_id(n) not in seen_loading_ids
                ):
                    seen_loading_ids.add(_node_id(n))
                    loading_from_npc.append(n)

        loading_lore_nodes = loading_from_user + loading_from_npc
        loading_lore_nodes = sorted(
            loading_lore_nodes, key=lambda n: getattr(n, "score", 0), reverse=True
        )

        TASK_SCORE_THRESHOLD = 0.28
        TASK_GUIDE_SCORE_THRESHOLD = 0.38  # 教学引导类与 NPC 形象弱关联，需更高分数才采用
        def _task_node_ok(n: Any) -> bool:
            score = getattr(n, "score", None) or 0
            meta = getattr(n, "metadata", None) or getattr(getattr(n, "node", n), "metadata", None) or {}
            th = TASK_GUIDE_SCORE_THRESHOLD if meta.get("task_source") == "guide" else TASK_SCORE_THRESHOLD
            return score >= th

        # 玩家检索取前 8 条并过滤
        task_from_user = [n for n in raw_task_by_user[:8] if _task_node_ok(n)]
        seen_ids = {_node_id(n) for n in task_from_user}
        # NPC 检索最多补 2 条，且不与玩家结果重复
        task_from_npc = []
        for n in raw_task_by_npc:
            if len(task_from_npc) >= 2:
                break
            if _task_node_ok(n) and _node_id(n) not in seen_ids:
                seen_ids.add(_node_id(n))
                task_from_npc.append(n)
        task_nodes = task_from_user + task_from_npc
        task_nodes = sorted(task_nodes, key=lambda n: getattr(n, "score", 0), reverse=True)

        # 其他 NPC 相关对话参考（不是当前 NPC 的台词，只参考设定，忽略语气）：
        # - 用户检索：玩家输入 + NPC 姓名，从所有非当前 NPC 的日常/任务对话中选最高相关的少量片段
        # - NPC 检索：NPC 上一条发言（若有），从同一池子中追加极高相关片段
        # - 引导类任务（task_source=guide）与「彩蛋/成员」等特殊角色，使用更高分数阈值，仅在强相关时才出现
        # 阈值设计：
        # - 普通片段：略高于补充设定与情报（0.30/0.36），确保只有明显相关的片段才参与「其他 NPC 对话参考」
        # - 引导任务 / 彩蛋 / 成员等特殊片段：使用更高阈值，仅在强相关时才出现
        OTHER_SCORE_THRESHOLD = 0.36
        OTHER_GUIDE_SCORE_THRESHOLD = 0.44
        OTHER_NPC_SCORE_THRESHOLD = 0.40
        OTHER_NPC_GUIDE_SCORE_THRESHOLD = 0.48

        def _get_metadata(n: Any) -> Dict[str, Any]:
            return (
                getattr(n, "metadata", None)
                or getattr(getattr(n, "node", n), "metadata", None)
                or {}
            )

        def _get_node_text(n: Any) -> str:
            """从检索结果节点取文本，兼容 NodeWithScore 与 Document/TextNode，避免 Document 上调用 get_content() 报错。"""
            node = getattr(n, "node", n)
            text = getattr(node, "text", None)
            if text is not None:
                return str(text)
            if hasattr(node, "get_content"):
                try:
                    return str(node.get_content())
                except Exception:
                    pass
            return ""

        forbidden_other_chars = forbidden_other_chars or set()

        def _other_node_ok(n: Any, *, from_npc_query: bool) -> bool:
            score = getattr(n, "score", None) or 0
            meta = _get_metadata(n)
            char_meta = (meta.get("character") or "").strip().lower()
            # 排除当前 NPC 自己的台词
            if char_meta == npc_char:
                return False
            # 额外保险：排除玩家占位符
            if char_meta == PC_CHAR_PLACEHOLDER.lower():
                return False
            # 排除玩家自身在日常对话中的台词（如 $pc / $pc_title 等），这些不属于「其他 NPC 对话」
            if "$pc" in char_meta:
                return False
            # 若当前 NPC 自身不属于「彩蛋」「成员」等特殊阵营，则对这些阵营角色的台词采用更高分数阈值，
            # 并在输出时额外标注为「非正式流程，仅作参考」，而不是完全排除，避免因「完全不知情」而胡乱脑补。
            is_egg_or_member = forbidden_other_chars and char_meta in forbidden_other_chars
            is_guide = meta.get("task_source") == "guide"
            is_special = is_guide or is_egg_or_member
            if from_npc_query:
                th = OTHER_NPC_GUIDE_SCORE_THRESHOLD if is_special else OTHER_NPC_SCORE_THRESHOLD
            else:
                th = OTHER_GUIDE_SCORE_THRESHOLD if is_special else OTHER_SCORE_THRESHOLD
            return score >= th

        def _other_candidate_key(n: Any) -> tuple:
            """用于对其他 NPC 参考片段排序的 key：
            1) 优先任务对话 (type=task) 再到日常对话 (type=dialogue)
            2) 文本中是否提到当前 NPC 名称（有则优先）
            3) 相似度分数
            """
            meta = _get_metadata(n)
            doc_type = meta.get("type") or ""
            is_task = 1 if doc_type == "task" else 0  # 任务优先

            text = _get_node_text(n)
            mentions_current = 1 if (npc_name and npc_name in text) else 0

            score = getattr(n, "score", None) or 0.0
            return (is_task, mentions_current, score)

        def _other_candidate_key_all(n: Any) -> tuple:
            """自由竞争排序 key：不再区分任务 / 日常，仅按
            1) 是否提到当前 NPC 名称
            2) 相似度分数

            用于「自由竞争」名额，避免任务对话占满所有名额，给高相关日常台词更多空间。
            """
            text = _get_node_text(n)
            mentions_current = 1 if (npc_name and npc_name in text) else 0
            score = getattr(n, "score", None) or 0.0
            return (mentions_current, score)

        # 用户检索：从所有非当前 NPC 的「日常 + 任务」中选极少量高相关片段（总共最多 5 条）
        # 任务优先策略：
        #   - 先保证最多 2 条来自「非引导类任务对话」(type=task, task_source != guide)
        #   - 再额外补充 3 条自由竞争（任务/日常 + 引导都可），总数不超过 5 条
        other_from_user_task_strict: List[Any] = []
        other_from_user_all: List[Any] = []
        for n in list(raw_other_dialogue_by_user) + list(raw_other_task_by_user):
            if not _other_node_ok(n, from_npc_query=False):
                continue
            meta = _get_metadata(n)
            doc_type = meta.get("type") or ""
            task_source = meta.get("task_source")
            if doc_type == "task" and task_source != "guide":
                other_from_user_task_strict.append(n)
            other_from_user_all.append(n)

        other_from_user_task_strict.sort(key=_other_candidate_key, reverse=True)
        other_from_user_all.sort(key=_other_candidate_key_all, reverse=True)

        other_from_user: List[Any] = []
        seen_other_ids: Set[str] = set()

        # 先填充最多 2 条「非引导任务」
        for n in other_from_user_task_strict:
            nid = _node_id(n)
            if nid in seen_other_ids:
                continue
            other_from_user.append(n)
            seen_other_ids.add(nid)
            if len(other_from_user) >= 2:
                break

        # 再补充 3 条自由竞争（总数不超过 5）
        for n in other_from_user_all:
            if len(other_from_user) >= 5:
                break
            nid = _node_id(n)
            if nid in seen_other_ids:
                continue
            other_from_user.append(n)
            seen_other_ids.add(nid)

        # NPC 上一条检索：在同一池子中追加最多 2 条极高相关片段
        # 同样采用任务优先策略：
        #   - 先保证最多 1 条来自「非引导类任务对话」
        #   - 再额外补充 1 条自由竞争，总数不超过 2 条（且与用户侧不重复）
        other_from_npc: List[Any] = []
        if (npc_last_message or "").strip():
            npc_task_strict: List[Any] = []
            npc_all: List[Any] = []
            for n in list(raw_other_dialogue_by_npc) + list(raw_other_task_by_npc):
                if not _other_node_ok(n, from_npc_query=True):
                    continue
                meta = _get_metadata(n)
                doc_type = meta.get("type") or ""
                task_source = meta.get("task_source")
                if doc_type == "task" and task_source != "guide":
                    npc_task_strict.append(n)
                npc_all.append(n)

            npc_task_strict.sort(key=_other_candidate_key, reverse=True)
            npc_all.sort(key=_other_candidate_key_all, reverse=True)

            # 先保证最多 1 条非引导任务
            for n in npc_task_strict:
                if len(other_from_npc) >= 1:
                    break
                nid = _node_id(n)
                if nid in seen_other_ids:
                    continue
                seen_other_ids.add(nid)
                other_from_npc.append(n)

            # 再补充 1 条自由竞争（总数不超过 2）
            for n in npc_all:
                if len(other_from_npc) >= 2:
                    break
                nid = _node_id(n)
                if nid in seen_other_ids:
                    continue
                seen_other_ids.add(nid)
                other_from_npc.append(n)

        other_npc_nodes = other_from_user + other_from_npc

        # 补充设定 + 情报合并到同一池子，按相似度排序取最佳 3 条；若有 NPC 上一条则用其再各查一条，分数要求更严，去重后并入
        SUPP_SCORE_THRESHOLD = 0.30
        SUPP_NPC_SCORE_THRESHOLD = 0.36  # 用 NPC 上一条检索时要求更严
        pooled = sorted(
            [
                n for n in (supp_nodes + intel_nodes)
                if (getattr(n, "score", None) or 0) >= SUPP_SCORE_THRESHOLD
            ],
            key=lambda n: getattr(n, "score", 0),
            reverse=True,
        )[:3]
        if (npc_last_message or "").strip():
            supp_by_npc = supp_retriever.retrieve(npc_last_message.strip())
            intel_by_npc = intel_retriever.retrieve(npc_last_message.strip())
            seen_pooled = {_node_id(n) for n in pooled}
            for n in sorted(supp_by_npc + intel_by_npc, key=lambda x: getattr(x, "score", 0), reverse=True):
                if (getattr(n, "score", None) or 0) >= SUPP_NPC_SCORE_THRESHOLD and _node_id(n) not in seen_pooled:
                    pooled.append(n)
                    seen_pooled.add(_node_id(n))
                    break
            pooled = sorted(pooled, key=lambda n: getattr(n, "score", 0), reverse=True)[:3]

        # --- 拼装上下文 ---
        def _nodes_to_text(nodes, max_chars: int | None = None) -> str:
            """
            将检索到的节点文本拼接为上下文片段：
            - 按行去重：相同的非空台词行只保留一条
            - 压缩空行：连续空行折叠为最多一个空行
            - 可选截断每个节点的最大字符数
            """
            seen_lines: set[str] = set()
            output_lines: List[str] = []
            last_blank = False

            for node in nodes:
                text = _get_node_text(node)
                if not text:
                    continue

                text = str(text).strip()
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars] + "…"

                for raw_line in text.splitlines():
                    line = raw_line.strip()

                    # 空行：最多保留一个连续空行
                    if not line:
                        if output_lines and not last_blank:
                            output_lines.append("")
                            last_blank = True
                        continue

                    last_blank = False

                    # 按行去重：相同的非空台词行只保留一次
                    if line in seen_lines:
                        continue
                    seen_lines.add(line)

                    output_lines.append(line)

            return "\n".join(output_lines)

        def _nodes_to_other_npc_text(nodes, max_chars: int | None = None) -> str:
            """
            将「其他 NPC 相关对话参考」的节点拼成文本：
            - 每条台词前标注说话人：<角色名>: 台词
            - 只用于非当前 NPC 的对话片段，避免与自己的台词混淆
            """
            seen_lines: set[str] = set()
            output_lines: List[str] = []
            last_blank = False

            for node in nodes:
                text = _get_node_text(node)
                if not text:
                    continue

                meta = _get_metadata(node)
                speaker = (meta.get("character") or "").strip() or "未知角色"
                speaker_lower = speaker.lower()
                is_egg_or_member = bool(forbidden_other_chars and speaker_lower in forbidden_other_chars)

                text = str(text).strip()
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars] + "…"

                for raw_line in text.splitlines():
                    line = raw_line.strip()

                    if not line:
                        if output_lines and not last_blank:
                            output_lines.append("")
                            last_blank = True
                        continue

                    last_blank = False

                    # 按内容去重，避免同一句台词重复出现
                    if line in seen_lines:
                        continue
                    seen_lines.add(line)

                    if is_egg_or_member:
                        output_lines.append(
                            f"{speaker}: {line}（该角色为非正式角色，相关信息可能不属于世界观正式内容，你的角色可能并不知情，仅作参考）"
                        )
                    else:
                        output_lines.append(f"{speaker}: {line}")

            return "\n".join(output_lines)

        parts: List[str] = []
        if npc_nodes:
            parts.append("【你的过往台词示例】\n" + _nodes_to_text(npc_nodes))
        if world_lore_nodes:
            # 暂不截断，观察长度；需恢复时可传 max_chars=375 等
            parts.append("【世界观设定摘取片段（用户输入相似度检索结果，可能与你无关，无关时忽略）】\n" + _nodes_to_text(world_lore_nodes))
        if loading_lore_nodes:
            # 与世界观设定使用相同检索条件的 loading 文本短句，数量较少，仅作补充参考
            parts.append("tips节选：\n" + _nodes_to_text(loading_lore_nodes, max_chars=300))
        if task_nodes:
            parts.append("【你的参考任务对话(任务可能超过玩家当前进度，仅参考语气和设定，忽略具体剧情，避免剧透)】\n" + _nodes_to_text(task_nodes, max_chars=350))
        if other_npc_nodes:
            parts.append("【其他NPC相关对话参考（不是你的台词，只参考设定，忽略语气，并忽略剧情以避免剧透）】\n" + _nodes_to_other_npc_text(other_npc_nodes, max_chars=350))
        if pooled:
            parts.append("【补充设定与情报参考（用户输入相似度检索结果，可能与你无关，无关时忽略）】\n" + _nodes_to_text(pooled, max_chars=350))

        # 关键词命中物品/关卡 + 语义检索补充（与关键词去重）
        game_hints = self._build_game_data_context_hints(
            user_query=user_query,
            npc_last_message=npc_last_message,
        )
        if game_hints:
            parts.append(game_hints)

        return "\n\n".join(parts)

    def _build_game_data_context_hints(
        self,
        user_query: str,
        npc_last_message: str | None,
    ) -> str:
        q = (user_query or "").strip()
        if not q and not (npc_last_message or "").strip():
            return ""
        try:
            from services.game_data.registry import get_game_data_registry

            game_data = get_game_data_registry()
        except Exception:
            return ""

        return self._compose_game_data_context_hints(
            user_query=q,
            npc_last_message=npc_last_message,
            game_data=game_data,
        )

    def _compose_game_data_context_hints(
        self,
        user_query: str,
        npc_last_message: str | None,
        game_data: Any,
    ) -> str:
        from services.game_entity_prompts import (
            compute_reward_tags,
            format_item_prompt_line,
            format_stage_detail_line,
        )

        q = (user_query or "").strip()
        nq = (npc_last_message or "").strip()

        item_rows, stage_sis, item_hit, stage_hit = self._collect_keyword_matches(
            user_query=q,
            items=game_data.items,
            equipment_mods=game_data.equipment_mods,
            stage_registry=game_data.stages,
        )
        vec_item_names, vec_stage_keys = self._collect_vector_game_entity_extras(
            user_query=q,
            npc_last_message=npc_last_message,
            exclude_item_names=item_hit,
            exclude_stage_keys=stage_hit,
        )

        keyword_item_map: dict[str, tuple[Any, list[str], int | None]] = {
            it.name: (it, tags, price) for it, tags, price in item_rows
        }

        item_order: list[str] = [it.name for it, _, _ in item_rows]
        for nm in vec_item_names:
            if nm not in item_order:
                item_order.append(nm)

        item_lines: list[str] = []
        for nm in item_order:
            it = game_data.items.get_by_name(nm)
            if it is None:
                continue
            if nm in keyword_item_map:
                _it, tags, price = keyword_item_map[nm]
                item_lines.append(format_item_prompt_line(_it, reward_tags=tags, price=price))
            else:
                tags = compute_reward_tags(it, game_data.equipment_mods)
                item_lines.append(
                    format_item_prompt_line(it, reward_tags=tags, price=it.price)
                )

        stage_lines: list[str] = [format_stage_detail_line(si) for si in stage_sis]
        seen_stage_k = {f"{si.area}::{si.name}" for si in stage_sis}
        for ek in vec_stage_keys:
            if ek in seen_stage_k:
                continue
            area, _, name = ek.partition("::")
            si = game_data.stages._stage_infos.get((area, name))
            if si is None:
                continue
            stage_lines.append(format_stage_detail_line(si))
            seen_stage_k.add(ek)

        chunks: list[str] = []
        if item_lines:
            chunks.append(
                "【玩家可能提到的物品类型（仅用于任务/奖励类型判断）】\n" + "\n".join(item_lines)
            )
        if stage_lines:
            chunks.append("【玩家可能提到的关卡】\n" + "\n".join(stage_lines))

        return "\n\n".join(chunks)

    @staticmethod
    def _collect_keyword_matches(
        user_query: str,
        items: Any,
        equipment_mods: Any,
        stage_registry: Any,
    ) -> tuple[list[tuple[Any, list[str], int | None]], list[Any], set[str], set[str]]:
        """
        关键词命中：物品 (Item, tags, price)、关卡 StageInfo 列表；
        以及已覆盖的 item 名与 stage entity_key，供向量去重。
        """
        q = (user_query or "").strip()
        item_names_hit: set[str] = set()
        stage_keys_hit: set[str] = set()
        matches: list[tuple[Any, list[str], int | None]] = []
        stage_matches: list[Any] = []

        if not q:
            return matches, stage_matches, item_names_hit, stage_keys_hit

        allowed_use = {"药剂", "弹夹", "材料", "食品"}
        allowed_type = {"武器", "防具"}
        seen_names: set[str] = set()

        candidates = list(items.items)
        candidates.sort(key=lambda it: len(getattr(it, "name", "") or ""), reverse=True)

        for it in candidates:
            name = (getattr(it, "name", None) or "").strip()
            if not name or name in seen_names:
                continue
            if name not in q:
                continue

            it_use = (getattr(it, "use", None) or "").strip()
            it_type = (getattr(it, "type", None) or "").strip()
            it_price: int | None = getattr(it, "price", None)

            type_tags: list[str] = []
            use_tags: list[str] = []
            if it_type in allowed_type:
                type_tags.append(it_type)
                if it_use:
                    use_tags.append(it_use)
            elif it_use in allowed_use:
                use_tags.append(it_use)

            plugin = False
            try:
                plugin = bool(equipment_mods and equipment_mods.is_plugin(name))
            except Exception:
                plugin = False

            if not type_tags and not use_tags and not plugin:
                continue

            ordered = type_tags + use_tags + (["插件"] if plugin else [])
            dedup: list[str] = []
            for t in ordered:
                if t not in dedup:
                    dedup.append(t)

            matches.append((it, dedup, it_price))
            item_names_hit.add(name)
            seen_names.add(name)
            if len(matches) >= 10:
                break

        seen_stage: set[str] = set()
        st_candidates = list(stage_registry._stage_infos.values())
        st_candidates.sort(key=lambda si: len(getattr(si, "name", "") or ""), reverse=True)
        for si in st_candidates:
            sn = (getattr(si, "name", None) or "").strip()
            if not sn or sn in seen_stage:
                continue
            if sn not in q:
                continue
            stage_matches.append(si)
            stage_keys_hit.add(f"{si.area}::{si.name}")
            seen_stage.add(sn)
            if len(stage_matches) >= 10:
                break

        return matches, stage_matches, item_names_hit, stage_keys_hit

    def _collect_vector_game_entity_extras(
        self,
        user_query: str,
        npc_last_message: str | None,
        exclude_item_names: set[str],
        exclude_stage_keys: set[str],
    ) -> tuple[list[str], list[str]]:
        """语义检索补充（与关键词去重）：返回额外物品名列表、额外关卡 entity_key 列表。"""
        try:
            from services.game_data.registry import get_game_data_registry

            game_data = get_game_data_registry()
        except Exception:
            return [], []

        index = self._get_index()
        SCORE_TH = 0.48

        item_retriever = index.as_retriever(
            similarity_top_k=4,
            filters=MetadataFilters(filters=[MetadataFilter(key="type", value="game_item")]),
        )
        stage_retriever = index.as_retriever(
            similarity_top_k=4,
            filters=MetadataFilters(filters=[MetadataFilter(key="type", value="game_stage")]),
        )

        def _pull_item_name(meta: dict[str, Any]) -> str | None:
            n = (meta.get("item_name") or "").strip()
            return n or None

        def _pull_stage_key(meta: dict[str, Any]) -> str | None:
            ek = (meta.get("entity_key") or "").strip()
            if ek:
                return ek
            a = (meta.get("stage_area") or "").strip()
            n = (meta.get("stage_name") or "").strip()
            if a and n:
                return f"{a}::{n}"
            return None

        def _meta_from_node(n: Any) -> dict[str, Any]:
            node = getattr(n, "node", n)
            return getattr(node, "metadata", None) or {}

        picked_items: list[str] = []
        seen_item: set[str] = set()

        uq = (user_query or "").strip()
        nq = (npc_last_message or "").strip()

        if uq:
            for n in item_retriever.retrieve(uq):
                if (getattr(n, "score", None) or 0) < SCORE_TH:
                    continue
                meta = _meta_from_node(n)
                nm = _pull_item_name(meta)
                if not nm or nm in exclude_item_names or nm in seen_item:
                    continue
                if game_data.items.get_by_name(nm) is None:
                    continue
                seen_item.add(nm)
                picked_items.append(nm)
                break

        if nq:
            for n in item_retriever.retrieve(nq):
                if (getattr(n, "score", None) or 0) < SCORE_TH:
                    continue
                meta = _meta_from_node(n)
                nm = _pull_item_name(meta)
                if not nm or nm in exclude_item_names or nm in seen_item:
                    continue
                if game_data.items.get_by_name(nm) is None:
                    continue
                seen_item.add(nm)
                picked_items.append(nm)
                break

        picked_stages: list[str] = []
        seen_stage: set[str] = set()

        if uq:
            for n in stage_retriever.retrieve(uq):
                if (getattr(n, "score", None) or 0) < SCORE_TH:
                    continue
                meta = _meta_from_node(n)
                ek = _pull_stage_key(meta)
                if not ek or ek in exclude_stage_keys or ek in seen_stage:
                    continue
                area, _, sname = ek.partition("::")
                if game_data.stages._stage_infos.get((area, sname)) is None:
                    continue
                seen_stage.add(ek)
                picked_stages.append(ek)
                break

        if nq:
            for n in stage_retriever.retrieve(nq):
                if (getattr(n, "score", None) or 0) < SCORE_TH:
                    continue
                meta = _meta_from_node(n)
                ek = _pull_stage_key(meta)
                if not ek or ek in exclude_stage_keys or ek in seen_stage:
                    continue
                area, _, sname = ek.partition("::")
                if game_data.stages._stage_infos.get((area, sname)) is None:
                    continue
                seen_stage.add(ek)
                picked_stages.append(ek)
                break

        return picked_items, picked_stages

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
