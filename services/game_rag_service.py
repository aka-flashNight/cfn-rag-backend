"""
游戏知识库 RAG 服务，基于 LlamaIndex + Gemini，支持 NPC 好感度管理。
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from openai import AsyncOpenAI

from ai_engine.game_data_loader import get_cached_index
from core.config import Settings, get_settings
from schemas.knowledge_schema import NPCChatRequest, NPCChatResponse
from services.npc_manager import NPCManager, NPCState
from services.memory_manager import MemoryManager, SUMMARIZE_INTERVAL
from services.portrait_utils import prepare_portrait_for_ai

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

    def _get_npc_image_path(self, npc_name: str, emotion: str = "普通") -> Tuple[Path | None, str | None]:
        """
        获取 NPC 的图像路径（优先立绘，其次头像）。

        Args:
            npc_name: NPC 名称
            emotion: 情绪标签，用于查找对应立绘

        Returns:
            (image_path, description) 元组：
            - 如果找到立绘：返回 (立绘路径，"现在传入了你所扮演的 npc 的立绘")
            - 如果找到头像：返回 (头像路径，"现在传入了你所扮演的 npc 的头像")
            - 如果都没找到：返回 (None, None)
        """
        # 1. 先尝试找立绘（指定情绪）：优先 WebP，其次 PNG
        illustration_dir = self._resources_dir / "flashswf" / "portraits" / "illustration"
        for ext in (".webp", ".png"):
            primary_illustration = illustration_dir / f"{npc_name}#{emotion}{ext}"
            if primary_illustration.is_file():
                return primary_illustration, "现在传入了你所扮演的 npc 的肖像（但不需要强行通过你的肖像内容开启话题）"

        # 2. 回退到普通情绪立绘
        for ext in (".webp", ".png"):
            fallback_illustration = illustration_dir / f"{npc_name}#普通{ext}"
            if fallback_illustration.is_file():
                return fallback_illustration, "现在传入了你所扮演的 npc 的肖像（但不需要强行通过你的肖像内容开启话题）"

        # 3. 最后尝试头像
        avatar_dir = self._resources_dir / "flashswf" / "portraits" / "profiles"
        avatar_path = avatar_dir / f"{npc_name}.png"
        if avatar_path.is_file():
            return avatar_path, "现在传入了你所扮演的 npc 的头像"

        return None, None

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

        # 2. 使用 LlamaIndex 检索 NPC 过往台词 + 世界观设定（检索时加入 NPC 姓名与称号，提高与当前角色相关内容的召回）
        retrieve_query = self._build_retrieve_query(payload.query, npc_name, titles)
        retrieved_context: str = await asyncio.to_thread(
            self._retrieve_context, npc_name, payload.query, retrieve_query
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

        # 3-a. NPC 交叉引用：检测玩家输入中提及的其他角色 + 同阵营角色（阵营名完全一致，排除「闲杂人等」）
        all_npc_states = npc_manager.state
        mentioned_npcs: List[str]
        mentioned_names: Set[str]
        mentioned_npcs, mentioned_names = self._find_mentioned_npcs(
            payload.query, npc_name, all_npc_states, faction
        )
        same_faction_npcs: List[str] = self._get_same_faction_npcs(
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
            if mentioned_npcs:
                mentioned_npcs_str += (
                    "其他同阵营角色：\n"
                    + "\n".join(same_faction_npcs)
                    + "\n\n"
                )
            else:
                mentioned_npcs_str += (
                    "同阵营角色：\n"
                    + "\n".join(same_faction_npcs)
                    + "\n\n"
                )

        # 3-b. 从记忆库中加载该会话最近的历史消息与摘要
        history_records = await memory.get_history(
            payload.session_id, limit=SUMMARIZE_INTERVAL
        )
        summary_text = await memory.get_summary(payload.session_id)

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
        if summary_text:
            history_str += (
                "当前对话历史较长，早期对话已整理为以下摘要：\n"
                f"{summary_text}\n\n"
            )
        if history_lines:
            joined_history = "\n".join(history_lines)
            if summary_text:
                history_str += (
                    "以下是最近的对话记录（按时间从早到晚排列），"
                    "请结合上述摘要与近期记录，在保持人物性格与情节连贯的前提下继续对话：\n"
                    f"{joined_history}\n\n"
                )
            else:
                history_str += (
                    "下面是你与玩家之间的对话历史（按时间从早到晚排列），"
                    "请在保持人物性格与情节连贯的前提下继续对话：\n"
                    f"{joined_history}\n\n"
                )

        emotions_str = "、".join(emotions)

        # system 消息：身份设定 + 世界观 + 输出格式与硬性规则
        system_prompt = (
            f"你现在扮演游戏角色「{npc_name}」{sex_desc}{faction_desc}{titles_desc}。\n"
            f"玩家的身份是：{player_identity}\n\n"
            "【世界观背景概要】\n"
            f"{WORLD_BACKGROUND}\n\n"
            f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。\n"
            f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。请选择其中最合适的一种作为你当前的情绪立绘。\n"
            "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，用简体中文回答玩家本次的发言。\n\n"
            "非特殊要求下，每次对话长度不必太长。不要自己脑补不存在的设定，无法把握的模糊地带可以略过或转移话题，不要自己乱加设定，以免出戏。\n\n"
            "输出格式必须严格为两行：\n"
            "第一行：你的回复内容（只包含对话文本，不要包含 JSON，不要带前缀，不要有“第一行：”的字样）。\n"
            "第二行：一个 JSON 对象，必须且仅包含两个字段：\n"
            "  - \"favorability_change\"：一个整数字段，取值范围 -5 到 5，例如："
            "{\"favorability_change\": 1, \"emotion\": \"普通\"}\n"
            "  - \"emotion\"：一个字符串字段，值必须从上文提供的情绪标签中选择；"
            "如果没有特别合适的情绪，请使用 \"普通\"。\n"
            "常规对话无需调整好感度，小幅度的情绪起伏可以只 +1 或 -1 点好感度。"
            "如果本次对话不应影响好感度，请输出 {\"favorability_change\": 0}。\n"
            "禁止输出第三行及更多内容，禁止添加多余的空行或注释。"
        )

        # user 消息：检索上下文 + 其他 NPC 提示 + 对话历史 + 玩家本次发言
        context_prompt = (
            f"{mentioned_npcs_str}"
            "下面是可能与你相关的检索设定和你的过往台词片段（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
            f"{retrieved_context or '（当前没有检索到任何上下文，你可以根据自己的设定自由发挥，但要保持合理。）'}\n\n"
            f"{history_str}"
        )

        user_prompt = f"{context_prompt}\n玩家：{payload.query}"

        # 获取 NPC 图像（优先立绘，其次头像）
        image_path, image_description = self._get_npc_image_path(npc_name, "普通")

        reply_text: str = await self._call_llm(
            settings,
            api_key=effective_api_key,
            api_base=effective_api_base,
            model_name=effective_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
            image_description=image_description,
        )
        # 4. 解析回复与好感度变化与情绪
        reply, delta, emotion = self._parse_reply_and_delta(
            reply_text, allowed_emotions=emotions
        )

        # 5. 写入对话记忆（玩家+NPC）
        await memory.add_message(payload.session_id, "user", payload.query)
        await memory.add_message(
            payload.session_id, "assistant", reply,
            llm_config={
                "api_key": effective_api_key,
                "api_base": effective_api_base,
                "model_name": effective_model,
            },
            npc_name=npc_name,
        )

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
    def _build_retrieve_query(user_query: str, npc_name: str, titles: List[str]) -> str:
        """
        构造用于「世界观 / 任务 / 补充设定与情报」向量检索的 query。

        将用户输入、当前 NPC 姓名、称号用空格拼成一段文本。检索时这段整句会被
        嵌入成一条向量，与库中文档向量做相似度比较，**没有**对某一段落（如姓名、
        称号）单独设权重或分隔符语义；逗号、分号等只相当于多几个字符，一般不改变
        召回逻辑。用空格拼接即可，简洁且与常见分词方式兼容。
        """
        parts = [user_query.strip(), npc_name.strip()] if user_query.strip() else [npc_name.strip()]
        if titles:
            parts.append(" ".join(t.strip() for t in titles if t and t.strip()))
        return " ".join(p for p in parts if p)

    def _retrieve_context(
        self, npc_name: str, user_query: str, retrieve_query: str
    ) -> str:
        """
        在同步线程中使用 LlamaIndex 检索：
          1. NPC 过往台词 (character=npc_name)，已限定为该角色，相似度仅用 user_query
          2. 核心世界观 (type=world_lore)，相似度用 retrieve_query（用户输入 + NPC 姓名 + 称号）
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

        # --- 执行检索：NPC 台词已按 character 限定，仅用用户输入做相似度；其余用 retrieve_query（含姓名与称号）---
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
    ) -> str:
        """
        调用任意 OpenAI 兼容的大模型生成回复。

        Args:
            image_path: 可选的图像文件路径，如果提供则会将图像作为多模态输入传给模型
            image_description: 图像描述，用于在提示中说明传入了什么图像
        """

        client = AsyncOpenAI(api_key=api_key, base_url=api_base)

        # 构建消息内容
        if image_path and image_path.is_file():
            # 多模态输入：立绘先裁剪+压缩再传 AI，统一输出 WebP；Base64 格式 data:image/webp;base64,...
            portrait_bytes, media_type = prepare_portrait_for_ai(image_path)
            image_data = base64.b64encode(portrait_bytes).decode("utf-8")
            # 在用户提示前添加图像说明
            prompt_with_desc = (
                f"{image_description}。\n\n{user_prompt}"
                if image_description
                else user_prompt
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
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
                            "text": prompt_with_desc,
                        },
                    ],
                },
            ]
            print(system_prompt)
            print("——————")
            print(prompt_with_desc)
        else:
            # 纯文本输入
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
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
