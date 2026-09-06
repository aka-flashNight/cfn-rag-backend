"""回合上下文装配（对应 docs/v3-developer/03 §7）。

并行 asyncio.gather：NPC 状态 / 会话记忆（近 N 条 + 滚动摘要 + 待确认草案）/
Tier-1 检索（≤3 变体单次嵌入，经 to_thread）/ 提及 NPC 子串匹配 / appearance。
立绘（07 §4）：仅 purpose=chat 且模型 Profile 允许视觉（vision 非 False）且该模型
未被标记图像不支持时，按 current_emotion（上回合 meta emo；空则「普通」）取图，
结果放 TurnContext.image_data_url；立绘不可用/任何失败 → None，纯文本不报错。
save_info（gamebridge）恒 None，字段预留（01 §9）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from services.game_progress import StageConfig, get_progress_stage_config
from services.llm import LLMConfig
from services.memory.store import ChatMessage, MemoryStore, TaskDraftRow
from services.npc.manager import NPCManager, NPCState
from services.orchestrator.prompts import (
    FACTION_ALIASES,
    PC_CHAR_PLACEHOLDER,
    SKIP_FACTION_SAME_CAMP,
    progress_stage_desc,
)

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """小写 + 去空白（NPC 名称/阵营/头衔模糊匹配，保留资产）。"""
    if not text:
        return ""
    return "".join(str(text).lower().split())


def find_mentioned_npcs(
    query: str,
    current_npc: str,
    all_states: dict[str, NPCState],
) -> tuple[list[str], set[str]]:
    """从玩家输入中发现提及的其他角色（名称/阵营/头衔子串匹配，保留资产）。

    跳过「成员/彩蛋」阵营（除非当前 NPC 自己属于它们）。
    返回 (格式化块列表, 被提及 NPC 名集合)。
    """
    skip_factions = {"成员", "彩蛋"}
    current_faction = (all_states.get(current_npc).faction if all_states.get(current_npc) else None)
    if current_faction in skip_factions:
        skip_factions = set()

    mentioned: list[str] = []
    mentioned_names: set[str] = set()
    normalized_query = _normalize_text(query)

    for name, state in all_states.items():
        if name == current_npc or name == PC_CHAR_PLACEHOLDER:
            continue
        if state.faction in skip_factions:
            continue
        terms: list[str] = [name]
        if state.faction:
            terms.append(state.faction)
            for faction_name, aliases in FACTION_ALIASES.items():
                if state.faction == faction_name:
                    terms.extend(aliases)
        terms.extend(state.titles or [])
        if any(_normalize_text(t) in normalized_query for t in terms if t):
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


def same_faction_npcs_block(
    current_npc: str,
    current_faction: Optional[str],
    all_states: dict[str, NPCState],
    exclude_names: set[str],
) -> str:
    """同阵营角色表（system 静态区，保留资产；阵营名完全一致才视为同阵营）。"""
    if not current_faction or current_faction.strip() == SKIP_FACTION_SAME_CAMP:
        return ""
    result: list[str] = []
    seen: set[str] = set(exclude_names)

    def _fmt(name: str, state: NPCState) -> str:
        parts = [f"「{name}」"]
        if state.sex:
            parts.append(f"（性别：{state.sex}）")
        if state.faction:
            parts.append(f"（阵营：{state.faction}）")
        if state.titles:
            parts.append(f"（身份或称呼：{'、'.join(state.titles)}）")
        return "".join(parts)

    for name, state in all_states.items():
        if name == current_npc or name in seen:
            continue
        if (state.faction or "").strip() != current_faction.strip():
            continue
        result.append(_fmt(name, state))
        seen.add(name)
    if not result:
        return ""
    return "同阵营角色（设定供参考，不要替他们发言）：\n" + "\n".join(result)


def forbidden_other_chars(all_states: dict[str, NPCState]) -> set[str]:
    """彩蛋/成员阵营角色名（小写），检索时从「其他 NPC」池排除。"""
    out: set[str] = set()
    for name, state in all_states.items():
        if (state.faction or "").strip() in {"成员", "彩蛋"}:
            out.add(name.lower())
    return out


# ---------------------------------------------------------------------------
# Tier-1 检索
# ---------------------------------------------------------------------------

def retrieve_tier1(
    *,
    engine: Any,
    user_query: str,
    npc_name: str,
    npc_titles: list[str],
    npc_faction: Optional[str],
    npc_last_message: Optional[str],
    forbidden: set[str],
) -> Any:
    """单轮 Tier-1 检索（同步实现；调用方用 to_thread 包装）。engine.ready False 时返回 None。"""
    from services.retrieval import RetrievalInput

    if engine is None or not getattr(engine, "ready", False):
        return None
    try:
        return engine.retrieve(RetrievalInput(
            user_query=user_query,
            npc_name=npc_name,
            npc_titles=npc_titles,
            npc_faction=npc_faction,
            npc_last_message=npc_last_message,
            forbidden_other_chars=forbidden,
        ))
    except Exception as exc:  # 检索降级不阻断回合（04）
        logger.warning("Tier-1 检索失败，降级为空上下文: %s", exc)
        return None


def entity_hints_block(bundle: Any, game_data: Any) -> str:
    """从 bundle.entity_items / entity_stages 拼「玩家可能提到的物品类型/关卡」提示。"""
    if bundle is None or game_data is None:
        return ""
    from services.game_entity_prompts import (
        compute_reward_tags,
        format_item_prompt_line,
        format_stage_detail_line,
    )

    lines: list[str] = []
    items = [sn.node for sn in (bundle.entity_items or [])][:6]
    if items:
        lines.append("【玩家可能提到的物品类型】")
        for node in items:
            item = game_data.items.get_by_name(node.item_name or "")
            if item is None:
                continue
            tags = compute_reward_tags(item, game_data.equipment_mods)
            lines.append(format_item_prompt_line(item, reward_tags=tags, price=item.price))
    stages = [sn.node for sn in (bundle.entity_stages or [])][:6]
    if stages:
        lines.append("【玩家可能提到的关卡】")
        for node in stages:
            si = None
            for (_area, name), info in getattr(game_data.stages, "_stage_infos", {}).items():
                if name == (node.stage_name or ""):
                    si = info
                    break
            if si is None:
                continue
            lines.append(format_stage_detail_line(si))
    return "\n".join(lines)


def build_retrieve_fn(
    *,
    engine: Any,
    npc_name: str,
    npc_titles: list[str],
    npc_faction: Optional[str],
    forbidden: set[str],
) -> Callable[[str], str]:
    """search_knowledge 工具用的检索函数（子 Agent 工具循环内同步执行）。"""
    from services.retrieval import format_retrieval_context

    def _retrieve(keyword: str) -> str:
        bundle = retrieve_tier1(
            engine=engine,
            user_query=keyword,
            npc_name=npc_name,
            npc_titles=npc_titles,
            npc_faction=npc_faction,
            npc_last_message=None,
            forbidden=forbidden,
        )
        if bundle is None or bundle.is_empty:
            return ""
        return format_retrieval_context(bundle, forbidden)

    return _retrieve


# ---------------------------------------------------------------------------
# 立绘装配（07 §4）
# ---------------------------------------------------------------------------

def _resolve_effective_model(llm_config: LLMConfig) -> str:
    """生效模型名（请求覆盖 > 会话记忆由 API 层完成 > 全局默认）。"""
    try:
        return llm_config.merged_with_settings().model_name
    except Exception:
        return llm_config.model_name or ""


def should_send_image(llm_config: LLMConfig) -> bool:
    """07 §4 带图前置判定（不含「是否有立绘资产」，资产缺失由 provider 兜底 None）：

    - Profile vision=False → 不带（纯文本模型）；
    - 模型曾被图像不支持错误标记 → 不带（client 降级链写入，02 §3.2.c）；
    - vision=True 与探测型（None）→ 带图（400 由 client 降级链去图重试并标记）。
    """
    from services.llm.client import is_image_unsupported_model
    from services.llm.profiles import get_profile

    model_name = _resolve_effective_model(llm_config)
    if not model_name:
        return False
    if get_profile(model_name).vision is False:
        return False
    return not is_image_unsupported_model(model_name)


def load_portrait_image_url(npc_name: str, emotion: Optional[str]) -> Optional[str]:
    """同步取当前情绪立绘 data URL（调用方经 to_thread 包装）；失败/缺失返回 None。"""
    from services.portraits import get_portrait_data_url

    try:
        return get_portrait_data_url(npc_name, emotion or "普通")
    except Exception as exc:  # 立绘任何异常不阻断回合（07 §1）
        logger.warning("立绘取图失败，本回合无图: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 装配结果
# ---------------------------------------------------------------------------

@dataclass
class TurnContext:
    """一轮对话装配好的上下文（prompt 消息由 turn.py 组装）。"""

    npc_name: str
    npc_state: NPCState
    npc_states: dict[str, NPCState]
    player_query: str
    player_identity: str = ""
    progress_stage: Optional[int] = None
    stage_cfg: Optional[StageConfig] = None
    rag_context_text: str = ""
    mentioned_npcs_str: str = ""
    history: list[ChatMessage] = field(default_factory=list)
    summary: Optional[str] = None
    pending_draft_row: Optional[TaskDraftRow] = None
    pending_draft_summary: str = ""
    same_faction_block: str = ""
    has_shop: bool = False
    shop_reward_types: list[str] = field(default_factory=list)
    forbidden: set[str] = field(default_factory=set)
    save_info: Any = None  # gamebridge 预留（01 §9），恒 None
    retrieve_fn: Optional[Callable[[str], str]] = None
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    # 当前情绪立绘的 data URL（07 §4；None = 不带图：非 vision / 已标记不支持 / 无立绘）
    image_data_url: Optional[str] = None

    @property
    def favorability(self) -> int:
        return self.npc_state.favorability

    @property
    def relationship_level(self) -> str:
        return self.npc_state.relationship_level

    @property
    def progress_desc(self) -> str:
        return progress_stage_desc(self.progress_stage, self.stage_cfg)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

async def assemble_context(
    *,
    session_id: str,
    npc_name: str,
    player_query: str,
    player_identity: str = "",
    progress_stage: Optional[int] = None,
    current_emotion: Optional[str] = None,
    llm_config: LLMConfig,
    memory: MemoryStore,
    npc_manager: NPCManager,
    game_data: Any,
    engine: Any = None,
    history_limit: int = 20,
    with_summary: bool = True,
    send_image: bool = True,
) -> TurnContext:
    """并行装配一轮上下文（03 §7；检索/取图经 to_thread，目标 ≤300ms）。"""
    stage_cfg = get_progress_stage_config(progress_stage) if progress_stage else None

    npc_state_task = npc_manager.get(npc_name)
    states_task = npc_manager.all_states()
    history_task = memory.get_history(session_id, limit=history_limit)
    summary_task = memory.get_summary(session_id) if with_summary else _none()
    draft_task = memory.get_draft(session_id)
    npc_state, npc_states, history, summary, draft_row = await asyncio.gather(
        npc_state_task, states_task, history_task, summary_task, draft_task,
    )

    mentioned, mentioned_names = find_mentioned_npcs(player_query, npc_name, npc_states)
    mentioned_npcs_str = ""
    if mentioned:
        mentioned_npcs_str = "【对话中提到的其他角色（设定供参考）】\n" + "\n".join(mentioned)
    forbidden = forbidden_other_chars(npc_states)
    same_faction_block = same_faction_npcs_block(
        npc_name, npc_state.faction, npc_states, mentioned_names,
    )

    titles = list(npc_state.titles or [])
    last_npc_message = next(
        (m.content for m in reversed(history) if m.role == "assistant"), None,
    )

    # Tier-1 检索 + 实体提示（同步实现，to_thread）
    def _retrieve_and_format() -> tuple[str, Any]:
        bundle = retrieve_tier1(
            engine=engine,
            user_query=player_query,
            npc_name=npc_name,
            npc_titles=titles,
            npc_faction=npc_state.faction,
            npc_last_message=last_npc_message,
            forbidden=forbidden,
        )
        text = ""
        if bundle is not None and not bundle.is_empty:
            from services.retrieval import format_retrieval_context

            text = format_retrieval_context(bundle, forbidden)
        hints = entity_hints_block(bundle, game_data)
        if hints:
            text = f"{text}\n\n{hints}" if text else hints
        return text, bundle

    try:
        rag_context_text, _bundle = await asyncio.to_thread(_retrieve_and_format)
    except Exception as exc:
        logger.warning("上下文检索装配失败，降级: %s", exc)
        rag_context_text = ""

    # 待确认草案摘要
    pending_draft_summary = ""
    if draft_row is not None and isinstance(draft_row.draft, dict) and draft_row.draft:
        try:
            from services.agent_tools.draft_formatting import _detailed_draft_summary

            pending_draft_summary = _detailed_draft_summary(
                draft_row.draft, game_data, rag_context_text=rag_context_text,
            )
        except Exception:
            pending_draft_summary = ""

    has_shop = False
    shop_reward_types: list[str] = []
    try:
        shops = getattr(game_data, "shops", None)
        if shops is not None and shops.has_shop(npc_name):
            has_shop = True
            seen: set[str] = set()
            for item_name in shops.get_npc_shop(npc_name):
                item = game_data.items.get_by_name(item_name)
                if item and item.type and item.type not in seen:
                    seen.add(item.type)
                    shop_reward_types.append(item.type)
    except Exception:
        pass

    engine_obj = engine
    retrieve_fn = build_retrieve_fn(
        engine=engine_obj,
        npc_name=npc_name,
        npc_titles=titles,
        npc_faction=npc_state.faction,
        forbidden=forbidden,
    ) if engine_obj is not None else None

    # 立绘（07 §4）：情绪跟随 current_emotion（上回合 meta emo；首轮为空用「普通」）
    image_data_url: Optional[str] = None
    if send_image and should_send_image(llm_config):
        image_data_url = await asyncio.to_thread(load_portrait_image_url, npc_name, current_emotion)

    return TurnContext(
        npc_name=npc_name,
        npc_state=npc_state,
        npc_states=npc_states,
        player_query=player_query,
        player_identity=player_identity,
        progress_stage=progress_stage,
        stage_cfg=stage_cfg,
        rag_context_text=rag_context_text,
        mentioned_npcs_str=mentioned_npcs_str,
        history=history,
        summary=summary,
        pending_draft_row=draft_row,
        pending_draft_summary=pending_draft_summary,
        same_faction_block=same_faction_block,
        has_shop=has_shop,
        shop_reward_types=shop_reward_types,
        forbidden=forbidden,
        retrieve_fn=retrieve_fn,
        llm_config=llm_config,
        image_data_url=image_data_url,
    )


async def _none() -> None:
    return None
