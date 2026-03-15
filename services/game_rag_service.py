"""
游戏知识库 RAG 服务，基于 LlamaIndex + Gemini，支持 NPC 好感度管理。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Set, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from openai import AsyncOpenAI

from ai_engine.game_data_loader import get_cached_index
from core.config import Settings, get_settings
from schemas.knowledge_schema import NPCChatRequest, NPCChatResponse
from services.npc_manager import NPCManager, NPCState
from services.memory_manager import MemoryManager
from services.portrait_utils import prepare_portrait_for_ai


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
    "在这个废土世界中，旧时代的道德与法律秩序几乎彻底崩坏，角色说话内容可以更加无拘束，无需考虑现实世界的规矩，以当前世界的剧情推演为最高优先级。"
)


# 不同阵营间无关联，不检索同阵营角色
SKIP_FACTION_SAME_CAMP = "闲杂人等"

# ---------------------------------------------------------------------------
# Function Calling：update_npc_mood 工具定义（为流式输出解耦 JSON 拼装，后续可扩展 agent）
# ---------------------------------------------------------------------------
UPDATE_NPC_MOOD_TOOL = {
    "type": "function",
    "function": {
        "name": "update_npc_mood",
        "description": (
            "在每次以 NPC 身份回复玩家后调用，用于上报本次对话的好感度变化与当前情绪。"
            "好感度变化取值范围为 -5 到 5，常规对话可传 0；情绪必须从当前 NPC 的可用情绪标签中选择。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "favorability_change": {
                    "type": "integer",
                    "description": "本次对话对玩家的好感度变化，范围 -5 到 5。0 表示不变，正数增加好感，负数减少。",
                },
                "emotion": {
                    "type": "string",
                    "description": "当前回复对应的情绪标签，用于立绘展示，必须从系统提供的可用情绪列表中选择。",
                },
            },
            "required": ["favorability_change", "emotion"],
        },
    },
}

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
}


def _is_tools_unsupported_error(exc: BaseException) -> bool:
    """
    判断是否为「API 不支持 tools/function calling」类错误。
    用于降级：带 tools 请求失败时重试不带 tools，保证远古模型也能正常返回对话。
    """
    raw = getattr(exc, "message", None) or getattr(exc, "body", None) or str(exc)
    if isinstance(raw, dict):
        msg = str(raw.get("error", raw)).lower()
    else:
        msg = str(raw).lower()
    if getattr(exc, "status_code", None) in (400, 422):
        return True
    for kw in ("tool", "function_call", "function call", "not support"):
        if kw in msg:
            return True
    return False


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
            - 如果找到立绘：返回 (立绘路径，"现在传入了你所扮演的角色的立绘")
            - 如果找到头像：返回 (头像路径，"现在传入了你所扮演的角色的头像")
            - 如果都没找到：返回 (None, None)
        """
        # 1. 先尝试找立绘（指定情绪）：优先 WebP，其次 PNG
        illustration_dir = self._resources_dir / "flashswf" / "portraits" / "illustration"
        for ext in (".webp", ".png"):
            primary_illustration = illustration_dir / f"{npc_name}#{emotion}{ext}"
            if primary_illustration.is_file():
                return primary_illustration, "现在传入了你所扮演的角色的肖像（但不需要强行通过你的肖像内容开启话题）"

        # 2. 回退到普通情绪立绘
        for ext in (".webp", ".png"):
            fallback_illustration = illustration_dir / f"{npc_name}#普通{ext}"
            if fallback_illustration.is_file():
                return fallback_illustration, "现在传入了你所扮演的角色的肖像（但不需要强行通过你的肖像内容开启话题）"

        # 3. 最后尝试头像
        avatar_dir = self._resources_dir / "flashswf" / "portraits" / "profiles"
        avatar_path = avatar_dir / f"{npc_name}.png"
        if avatar_path.is_file():
            return avatar_path, "现在传入了你所扮演的角色的头像"

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
                "未提供可用的大模型 API Key，请在请求中传入 api_key 或在 .env 中配置 LLM_API_KEY。"
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
        retrieved_context = await asyncio.to_thread(
            self._retrieve_context, npc_name, payload.query, retrieve_query
        )
        player_identity = (
            payload.player_identity.strip()
            if payload.player_identity and payload.player_identity.strip()
            else "一个末日后加入A兵团成为佣兵的幸存者"
        )
        sex_desc = f"（性别：{sex}）" if sex else ""
        faction_desc = f"（阵营：{faction}）" if faction else ""
        titles_desc = f"（身份或称呼：{'、'.join(titles)}）" if titles else ""
        all_npc_states = npc_manager.state
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

        effective_interval = (
            payload.summarize_interval
            if payload.summarize_interval is not None
            and payload.summarize_interval in ALLOWED_SUMMARIZE_INTERVALS
            else DEFAULT_SUMMARIZE_INTERVAL
        )
        history_records = await memory.get_history(
            payload.session_id, limit=effective_interval
        )
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
            f"玩家的身份是：{player_identity}\n\n"
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
        context_prompt = (
            f"{mentioned_npcs_str}"
            "下面是可能与你相关的检索设定和你的过往台词片段（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
            f"{retrieved_context or '（当前没有检索到任何上下文，你可以根据自己的设定自由发挥，但要保持合理。）'}\n\n"
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

    async def ask(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> NPCChatResponse:
        """
        基于游戏知识库进行 RAG + Agent 对话，并更新 NPC 好感度。
        """
        ctx = await self._prepare_ask_context(payload, npc_manager, memory)

        try:
            reply_text, tool_calls = await self._call_llm(
                ctx.settings,
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
            if _is_tools_unsupported_error(e):
                reply_text, tool_calls = await self._call_llm(
                    ctx.settings,
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
            else:
                raise


        reply = (reply_text or "").strip() or "（当前未能生成有效回复，请稍后再试。）"
        delta, emotion = self._parse_update_npc_mood_tool_calls(
            tool_calls, allowed_emotions=ctx.emotions
        )
        parsed_delta, parsed_emotion = self._parse_mood_from_text(reply)
        cleaned, fallback_delta, fallback_emotion = self._strip_trailing_mood_json(
            reply, allowed_emotions=ctx.emotions
        )
        if fallback_delta is not None and fallback_emotion is not None:
            reply = (cleaned or "").strip() or "（当前未能生成有效回复，请稍后再试。）"
            if not self._has_update_npc_mood_tool_call(tool_calls):
                delta, emotion = fallback_delta, fallback_emotion
        if not self._has_update_npc_mood_tool_call(tool_calls):
            default_emo = "普通" if "普通" in ctx.emotions else (ctx.emotions[0] if ctx.emotions else "普通")
            if (delta == 0 and emotion == default_emo) and (parsed_delta is not None or parsed_emotion):
                if parsed_delta is not None:
                    delta = parsed_delta
                if parsed_emotion is not None:
                    emotion = parsed_emotion if parsed_emotion in ctx.emotions else default_emo
        reply = self._strip_trailing_tool_call_text(reply)

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

    async def ask_stream(
        self,
        payload: NPCChatRequest,
        npc_manager: NPCManager,
        memory: "MemoryManager",
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        流式版 ask：前面内容正常流式 yield；一旦检测到截断前缀（工具调用：、\\n{、update_npc_mood(）
        则停止向客户端输出、只缓冲，避免首字延迟。结束后再做完整截断得到 reply，done 时带上 reply。
        """
        ctx = await self._prepare_ask_context(payload, npc_manager, memory)

        full_content = ""
        streamed_len = 0
        truncating = False
        # 截断前缀：任一出现即停发（含 {、HTML 注释头、工具名，不考虑误伤）
        _TRUNCATE_PREFIXES: List[str] = ["工具调用：", "{", "<!---", "<!--", "update_npc_mood("]
        tool_calls_list: List[dict] = []

        def _earliest_truncate_at(text: str) -> int:
            out = -1
            for p in _TRUNCATE_PREFIXES:
                if "update_npc_mood" in p.lower():
                    idx = text.lower().find(p)
                else:
                    idx = text.find(p)
                if idx != -1 and (out == -1 or idx < out):
                    out = idx
            return out

        try:
            async for event_type, data in self._call_llm_stream(
                ctx.settings,
                api_key=ctx.effective_api_key,
                api_base=ctx.effective_api_base,
                model_name=ctx.effective_model,
                system_prompt=ctx.system_prompt,
                user_prompt=ctx.user_prompt,
                image_path=ctx.image_path,
                image_description=ctx.image_description,
                emotion_hint=ctx.emotion_hint or None,
                tools=[UPDATE_NPC_MOOD_TOOL],
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
                    break
        except Exception as e:
            if _is_tools_unsupported_error(e):
                full_content = ""
                streamed_len = 0
                truncating = False
                tool_calls_list = []
                async for event_type, data in self._call_llm_stream(
                    ctx.settings,
                    api_key=ctx.effective_api_key,
                    api_base=ctx.effective_api_base,
                    model_name=ctx.effective_model,
                    system_prompt=ctx.system_prompt,
                    user_prompt=ctx.user_prompt,
                    image_path=ctx.image_path,
                    image_description=ctx.image_description,
                    emotion_hint=ctx.emotion_hint or None,
                    tools=None,
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
                        break
            else:
                raise

        reply = (full_content or "").strip() or "（当前未能生成有效回复，请稍后再试。）"
        delta, emotion = self._parse_update_npc_mood_tool_calls(
            tool_calls_list, allowed_emotions=ctx.emotions
        )
        parsed_delta, parsed_emotion = self._parse_mood_from_text(reply)
        cleaned, fallback_delta, fallback_emotion = self._strip_trailing_mood_json(
            reply, allowed_emotions=ctx.emotions
        )
        if fallback_delta is not None and fallback_emotion is not None:
            reply = (cleaned or "").strip() or "（当前未能生成有效回复，请稍后再试。）"
            if not self._has_update_npc_mood_tool_call(tool_calls_list):
                delta, emotion = fallback_delta, fallback_emotion
        if not self._has_update_npc_mood_tool_call(tool_calls_list):
            default_emo = "普通" if "普通" in ctx.emotions else (ctx.emotions[0] if ctx.emotions else "普通")
            if (delta == 0 and emotion == default_emo) and (parsed_delta is not None or parsed_emotion):
                if parsed_delta is not None:
                    delta = parsed_delta
                if parsed_emotion is not None:
                    emotion = parsed_emotion if parsed_emotion in ctx.emotions else default_emo
        reply = self._strip_trailing_tool_call_text(reply)

        # print("\n" + "=" * 60 + " [ask_stream] 大模型原始输出与工具调用 " + "=" * 60)
        # print("[ask_stream] content 原始 (前 500 字):", (full_content or "")[:500])
        # print("[ask_stream] tool_calls_list:", tool_calls_list)
        # if not tool_calls_list:
        #     print("[ask_stream] 提示: 流式下 tool_calls 为空...")
        # print("[ask_stream] 解析结果: delta=%s emotion=%s" % (delta, emotion))
        # print("[ask_stream] 截断后 reply (前 300 字):", (reply or "")[:300])
        # print("=" * 60 + "\n")

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

        result: List[str] = []
        for name, state in all_npc_states.items():
            if name == current_npc:
                continue
            if name == PC_CHAR_PLACEHOLDER:
                continue
            if name in exclude_names:
                continue
            if (state.faction or "").strip() != current_faction.strip():
                continue
            parts = [f"「{name}」"]
            if state.sex:
                parts.append(f"（性别：{state.sex}）")
            if state.faction:
                parts.append(f"（阵营：{state.faction}）")
            if state.titles:
                parts.append(f"（身份或称呼：{'、'.join(state.titles)}）")
            result.append("".join(parts))
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
        self, npc_name: str, user_query: str, retrieve_query: str
    ) -> str:
        """
        在同步线程中使用 LlamaIndex 检索：
          1. NPC 过往台词 (character=npc_name)，已限定为该角色，相似度仅用 user_query
          2. 核心世界观 (type=world_lore)，相似度用 retrieve_query（用户输入 + NPC 姓名 + 称号 + 阵营）
          3. 任务对话 (type=task)，相似度用 retrieve_query
          4. 补充设定 + 情报（pool: supplementary_lore & intelligence），相似度用 retrieve_query
        """
        if not retrieve_query.strip():
            retrieve_query = user_query.strip() or npc_name

        index: VectorStoreIndex = self._get_index()

        # --- 构建各类检索器 ---
        npc_retriever = index.as_retriever(
            similarity_top_k=5,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="character", value=npc_name)]
            ),
        )
        world_lore_retriever = index.as_retriever(
            similarity_top_k=3,
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

        # --- 执行检索：NPC 台词已按 character 限定，仅用用户输入做相似度；其余用 retrieve_query（含姓名、称号与阵营）---
        npc_nodes = npc_retriever.retrieve(user_query)
        world_lore_nodes = world_lore_retriever.retrieve(retrieve_query)
        task_nodes = task_retriever.retrieve(retrieve_query)
        supp_nodes = supp_retriever.retrieve(retrieve_query)
        intel_nodes = intel_retriever.retrieve(retrieve_query)

        # 补充设定 + 情报合并到同一池子，按相似度排序取最佳 3 条
        SUPP_SCORE_THRESHOLD = 0.30
        pooled = sorted(
            [
                n for n in (supp_nodes + intel_nodes)
                if (getattr(n, "score", None) or 0) >= SUPP_SCORE_THRESHOLD
            ],
            key=lambda n: getattr(n, "score", 0),
            reverse=True,
        )[:3]

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
                text = getattr(node, "text", None)
                if text is None and hasattr(node, "get_content"):
                    text = node.get_content()
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

        parts: List[str] = []
        if npc_nodes:
            parts.append("【NPC 过往台词示例】\n" + _nodes_to_text(npc_nodes))
        if world_lore_nodes:
            # 暂不截断，观察长度；需恢复时可传 max_chars=375 等
            parts.append("【世界观设定摘取片段（用户输入相似度检索结果，可能与你无关，无关时忽略）】\n" + _nodes_to_text(world_lore_nodes))
        if task_nodes:
            parts.append("【参考任务对话(任务可能超过玩家当前进度，仅参考语气，忽略具体内容。)】\n" + _nodes_to_text(task_nodes[:2], max_chars=350))
        if pooled:
            parts.append("【补充设定与情报参考（用户输入相似度检索结果，可能与你无关，无关时忽略）】\n" + _nodes_to_text(pooled, max_chars=350))

        return "\n\n".join(parts)

    async def _call_llm_stream(
        self,
        settings: Settings,
        api_key: str,
        api_base: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        image_description: str | None = None,
        emotion_hint: str | None = None,
        tools: List[dict] | None = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        流式调用 LLM：yield ("content", delta_str) 逐片输出正文；
        结束时 yield ("finished", (full_content, tool_calls_list)) 供调用方做后处理。
        """
        client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        prefix_parts: List[str] = []
        if image_description and image_description.strip():
            prefix_parts.append(image_description.strip() + "。")
        if emotion_hint and emotion_hint.strip():
            prefix_parts.append(emotion_hint.strip())
        prompt_prefix = ("".join(prefix_parts) + "\n\n") if prefix_parts else ""

        if image_path and image_path.is_file():
            portrait_bytes, media_type = prepare_portrait_for_ai(image_path)
            image_data = base64.b64encode(portrait_bytes).decode("utf-8")
            prompt_with_desc = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
            messages = [
                {"role": "system", "content": prompt_with_desc},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            # print(prompt_with_desc)
            # print("——————")
            # print(user_prompt)
        else:
            system_content = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ]

        kwargs: dict = {"model": model_name, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools

        stream = await client.chat.completions.create(**kwargs)
        content_parts: List[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None) and delta.content:
                content_parts.append(delta.content)
                yield ("content", delta.content)
            tool_calls_delta = getattr(delta, "tool_calls", None) or []
            for tc in tool_calls_delta:
                idx = getattr(tc, "index", None)
                if idx is None:
                    continue
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": getattr(tc, "id", "") or "", "name": "", "arguments": ""}
                fn = getattr(tc, "function", None)
                if fn:
                    if getattr(fn, "name", None):
                        tool_calls_acc[idx]["name"] = (tool_calls_acc[idx].get("name") or "") + (fn.name or "")
                    if getattr(fn, "arguments", None):
                        tool_calls_acc[idx]["arguments"] = (tool_calls_acc[idx].get("arguments") or "") + (fn.arguments or "")

        full_content = "".join(content_parts)
        tool_calls_list = []
        for i in sorted(tool_calls_acc.keys()):
            acc = tool_calls_acc[i]
            tool_calls_list.append({
                "type": "function",
                "id": acc.get("id", ""),
                "function": {
                    "name": acc.get("name", ""),
                    "arguments": acc.get("arguments", ""),
                },
            })
        yield ("finished", (full_content, tool_calls_list))

    async def _call_llm(
        self,
        settings: Settings,
        api_key: str,
        api_base: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        image_description: str | None = None,
        emotion_hint: str | None = None,
        tools: List[dict] | None = None,
    ) -> Tuple[str, List[dict]]:
        """
        调用任意 OpenAI 兼容的大模型生成回复。
        支持 Function Calling：传入 tools 时，返回 (回复正文, tool_calls 列表)；未传时 tool_calls 为空列表。
        为后续流式输出预留：流式时由调用方收集 content delta 与 tool_calls 再解析。

        Args:
            image_path: 可选的图像文件路径，如果提供则会将图像作为多模态输入传给模型
            image_description: 图像描述，用于在提示中说明传入了什么图像
            emotion_hint: 可选的上一轮情绪说明（如「你之前的情绪是「开心」。」），用于连贯与情绪过渡
            tools: 可选的工具定义列表（OpenAI tools 格式），用于 Function Calling
        """

        client = AsyncOpenAI(api_key=api_key, base_url=api_base)

        # 在 system 前拼接成一行：立绘说明与上一轮情绪同一行，便于 AI 关联情绪与立绘；缺其一时提示词也正常
        prefix_parts: List[str] = []
        if image_description and image_description.strip():
            prefix_parts.append(image_description.strip() + "。")
        if emotion_hint and emotion_hint.strip():
            prefix_parts.append(emotion_hint.strip())
        prompt_prefix = ("".join(prefix_parts) + "\n\n") if prefix_parts else ""

        # 构建消息内容
        if image_path and image_path.is_file():
            # 多模态输入：立绘先裁剪+压缩再传 AI，统一输出 WebP；Base64 格式 data:image/webp;base64,...
            portrait_bytes, media_type = prepare_portrait_for_ai(image_path)
            image_data = base64.b64encode(portrait_bytes).decode("utf-8")
            # 在 system 前添加图像说明与上一轮情绪
            prompt_with_desc = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
            messages = [
                {
                    "role": "system",
                    "content": prompt_with_desc,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]
        else:
            # 纯文本输入（无立绘时若有 emotion_hint 仍拼在 system 前）
            system_content = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
            messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

        kwargs: dict = {"model": model_name, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        completion = await client.chat.completions.create(**kwargs)

        message = completion.choices[0].message
        content = message.content or ""
        if not isinstance(content, str):
            try:
                content = "".join(part.get("text", "") for part in content)  # type: ignore[arg-type]
            except Exception:
                content = str(content)

        tool_calls_raw = getattr(message, "tool_calls", None) or []
        # 转为可 JSON 序列化的 dict 列表，便于统一处理（含流式后续收集）
        tool_calls_list: List[dict] = []
        for tc in tool_calls_raw:
            if hasattr(tc, "model_dump"):
                tool_calls_list.append(tc.model_dump())
            elif hasattr(tc, "dict"):
                tool_calls_list.append(tc.dict())
            elif isinstance(tc, dict):
                tool_calls_list.append(tc)
            else:
                fn = getattr(tc, "function", None)
                tool_calls_list.append({
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "") if fn else "",
                    },
                })

        return content, tool_calls_list

    @staticmethod
    def _has_update_npc_mood_tool_call(tool_calls: List[dict]) -> bool:
        """判断 tool_calls 中是否包含 update_npc_mood 调用。"""
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            f = tc.get("function") if tc.get("type") == "function" else tc.get("function")
            if isinstance(f, dict) and f.get("name") == "update_npc_mood":
                return True
        return False

    @staticmethod
    def _strip_trailing_tool_call_text(reply_text: str) -> str:
        """
        按双换行分段；若某段内出现任一截断条件（工具调用：、含关键词的 HTML 注释、含关键词的 {}、
        或直接出现 update_npc_mood/emotion/favorability_change），则从该段起删掉该段及之后全部内容，
        只保留该段之前的段落，避免各种变体漏网。
        """
        if not reply_text or not reply_text.strip():
            return reply_text
        _TOOL_KEYWORDS = ("update_npc_mood", "emotion", "favorability_change")

        def _segment_has_trigger(seg: str) -> bool:
            if "工具调用：" in seg:
                return True
            seg_lower = seg.lower()
            if any(kw in seg_lower for kw in _TOOL_KEYWORDS):
                return True
            for pat in [r"<!---.*?--->", r"<!--.*?-->"]:
                for m in re.finditer(pat, seg, re.IGNORECASE | re.DOTALL):
                    if any(kw in m.group(0).lower() for kw in _TOOL_KEYWORDS):
                        return True
            pos = 0
            while True:
                i = seg.find("{", pos)
                if i == -1:
                    break
                depth = 0
                j = i
                while j < len(seg):
                    if seg[j] == "{":
                        depth += 1
                    elif seg[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if depth != 0:
                    break
                if any(kw in seg[i : j + 1].lower() for kw in _TOOL_KEYWORDS):
                    return True
                pos = j + 1
            return False

        paragraphs = re.split(r"\n\s*\n", reply_text)
        cut = len(paragraphs)
        for i, p in enumerate(paragraphs):
            if _segment_has_trigger(p):
                cut = i
                break
        return "\n\n".join(paragraphs[:cut]).strip()

    @staticmethod
    def _parse_mood_from_text(text: str) -> Tuple[int | None, str | None]:
        """
        在删除前从正文中用正则解析 favorability_change 与 emotion，兼容 =/:、有引号/无引号等。
        返回 (delta, emotion_raw)，未找到则对应为 None。调用方需用 allowed_emotions 校验 emotion。
        """
        if not text or not text.strip():
            return None, None
        delta: int | None = None
        emotion_raw: str | None = None
        # favorability_change: = 或 : ，可选引号，数字（含负号）
        for m in re.finditer(
            r"favorability_change\s*[=:]\s*[\"']*(-?\d+)[\"']*",
            text,
            re.IGNORECASE,
        ):
            try:
                n = int(m.group(1))
                delta = max(-5, min(5, n))
            except (TypeError, ValueError):
                pass
        # emotion: = 或 : ，双引号/单引号内任意内容，或无引号时到逗号/} / ) / 空白 止
        for m in re.finditer(
            r"emotion\s*[=:]\s*[\"']([^\"']*)[\"']|emotion\s*[=:]\s*([^,}\s\)]+)",
            text,
            re.IGNORECASE,
        ):
            em = (m.group(1) or m.group(2) or "").strip()
            if em:
                emotion_raw = em
        return (delta, emotion_raw)

    @staticmethod
    def _strip_trailing_mood_json(
        reply_text: str, allowed_emotions: List[str]
    ) -> Tuple[str, int | None, str | None]:
        """
        若回复末尾是类似 update_npc_mood 的 JSON（模型未走工具而把参数写在内容里），
        则剥离并解析为 (delta, emotion)，返回 (剥离后的回复, delta, emotion)；
        否则返回 (原回复, None, None)。
        支持两种格式：
        1) 平铺：{"favorability_change": 2, "emotion": "微笑"} 或换行缩进形式；
        2) 与调用一致之嵌套：{"name": "update_npc_mood", "arguments": "{\"favorability_change\": 2, \"emotion\": \"微笑\"}"}。
        流式输出时可在最终拼接的完整文本上同样调用此方法。
        """
        if not reply_text or not reply_text.strip():
            return reply_text, None, None
        text = reply_text
        default_emotion = (
            "普通" if "普通" in allowed_emotions else (allowed_emotions[0] if allowed_emotions else "普通")
        )

        def _delta_emotion_from_obj(obj: dict) -> Tuple[int, str] | None:
            if not isinstance(obj, dict) or "emotion" not in obj:
                return None
            emo_raw = obj.get("emotion")
            if not isinstance(emo_raw, str) or not emo_raw.strip():
                return None
            try:
                delta = max(-5, min(5, int(obj.get("favorability_change", 0))))
            except (TypeError, ValueError):
                delta = 0
            emotion = (
                emo_raw.strip()
                if emo_raw.strip() in allowed_emotions
                else default_emotion
            )
            return delta, emotion

        brace_indices = [i for i, c in enumerate(text) if c == "{"]
        for idx in reversed(brace_indices):
            suffix = text[idx:]
            try:
                obj = json.loads(suffix)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            # 1) 平铺：直接含 emotion（及可选 favorability_change）
            mood = _delta_emotion_from_obj(obj)
            if mood is not None:
                cleaned = text[:idx].rstrip()
                return cleaned, mood[0], mood[1]
            # 2) 嵌套：含 arguments 字符串，内层为 mood JSON
            args_str = obj.get("arguments")
            if isinstance(args_str, str) and args_str.strip():
                try:
                    inner = json.loads(args_str)
                    mood = _delta_emotion_from_obj(inner) if isinstance(inner, dict) else None
                    if mood is not None:
                        cleaned = text[:idx].rstrip()
                        return cleaned, mood[0], mood[1]
                except Exception:
                    pass
        return reply_text, None, None

    @staticmethod
    def _parse_update_npc_mood_tool_calls(
        tool_calls: List[dict],
        allowed_emotions: List[str],
    ) -> Tuple[int, str]:
        """
        从 Function Calling 的 tool_calls 中解析并校验 update_npc_mood 的 (好感度变化, 情绪)。
        校验规则：favorability_change 非数字或超出 [-5,5] 则钳位/置 0；emotion 不在允许列表则用「普通」。
        未找到有效调用时返回 (0, "普通")。为后续流式输出预留：流式时由后端收集完整 tool_calls 再调用本方法。
        """
        if not allowed_emotions:
            allowed_emotions = ["普通"]
        default_emotion = "普通" if "普通" in allowed_emotions else allowed_emotions[0]

        delta = 0
        emotion = default_emotion

        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            f = tc.get("function") if tc.get("type") == "function" else None
            if not f or f.get("name") != "update_npc_mood":
                continue
            args_str = f.get("arguments")
            if not args_str:
                break
            try:
                obj = json.loads(args_str)
            except Exception:
                break
            # 好感度：非数字或超出范围则钳位或置 0
            raw_delta = obj.get("favorability_change", 0)
            try:
                delta = int(raw_delta)
            except (TypeError, ValueError):
                delta = 0
            delta = max(-5, min(5, delta))
            # 情绪：必须在允许列表中，否则用默认
            emo_raw = obj.get("emotion")
            if isinstance(emo_raw, str) and emo_raw.strip():
                emo_candidate = emo_raw.strip()
                if emo_candidate in allowed_emotions:
                    emotion = emo_candidate
            break

        if emotion not in allowed_emotions:
            emotion = default_emotion
        return delta, emotion

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
