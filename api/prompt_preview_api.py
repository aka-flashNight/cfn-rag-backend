"""外部提示词预览接口：只读汇编「NPC 资料 + 角色扮演要求」，不调模型、不写盘。

用途：外部功能需要扮演某个 NPC，或查询某段输入相关的设定时，直接取本接口返回的
`messages`，复用后端已有的 RAG 混合检索与提示词资产。

与插话/终端主流程的区别（按需求逐条裁剪）：
- 不含任何输出格式要求（JSONL / add-closing / 字段 / 长度 / 立即输出 / 说话人前缀）；
  只保留一条「只输出角色台词，不写旁白、动作、心理或括号」的扮演纪律。
- 不含任务发布、委托草案、工具注册与工具说明。
- 不含玩家侧信息（名字 / 身份 / 等级 / 进度 / 好感度 / 已播历史 / 剩余台词 / 物品账本）。
- 保留 NPC 侧资料：身份、情绪枚举、形象、场景、装备、同阵营、切磋、商店货品类型。
- 保留六类混合检索结果与「玩家可能提到的物品类型」（含获取途径）。

本模块只读；不建 session、不写 memory、不写覆盖层、不产生任何 LLM 调用。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from services.npc.npc_equipment import build_npc_equipment_block
from services.npc.scene_context import build_scene_block
from services.orchestrator.context import (
    entity_hints_block,
    forbidden_other_chars,
    retrieve_tier1,
    same_faction_npcs_block,
)
from services.orchestrator.prompts import (
    ANTI_HALLUCINATION_RULE,
    SPEECH_STYLE_RULES,
    build_challenge_hint,
    format_npc_role_tagline,
    worldview_block,
)
from services.retrieval.hybrid import format_sections

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

#: 玩家占位符：全量 NPC 状态库里会作为一条记录存在，公共接口必须排除
_PLAYER_PLACEHOLDERS = {"$PC_CHAR", "$PC_TITLE"}

#: 角色扮演约束（roleplay=true 且 NPC 存在时注入）
_ROLEPLAY_TONE = (
    "请始终以符合该角色身份、口吻、记忆和立场的语气，用简体中文说话。"
)
#: 台词纪律：只保留「不写旁白/动作/心理」这一条，其余格式要求全部舍弃
_ACTING_RULE = "只输出角色台词，不写旁白、动作、心理或任何括号内容。"

#: 检索块的标题沿用插话口径，保证与现有提示词同源
_RAG_HEADER = "【检索到的设定资料（仅用于保持设定与说话风格，不要逐字复读原文）】"


class PromptPreviewRequest(BaseModel):
    """输入：NPC 名（可空）+ 一段字符串（说话内容或查询关键词）。"""

    model_config = ConfigDict(extra="ignore")

    npc: str = Field(default="", description="NPC 名称；为空或不存在时只返回通用资料")
    input: str = Field(default="", description="玩家说的话，或查询用关键词")
    roleplay: bool = Field(
        default=True,
        description="true=附带角色扮演要求（说话方式/扮演约束/台词纪律）；false=只给资料",
    )


class PromptPreviewResponse(BaseModel):
    npc: str
    npc_exists: bool
    roleplay: bool
    messages: list[dict[str, str]]
    blocks: dict[str, list[dict[str, Any]]]
    retrieval: dict[str, Any]
    meta: dict[str, Any]
    elapsed_ms: int


def _plain_tagline(npc_name: str, state: Any) -> str:
    """roleplay=false 时的纯资料身份行（不出现「你现在扮演…」）。"""
    parts = [f"角色：{npc_name}"]
    sex = str(getattr(state, "sex", "") or "").strip()
    faction = str(getattr(state, "faction", "") or "").strip()
    titles = [str(t).strip() for t in (getattr(state, "titles", None) or []) if str(t).strip()]
    if sex:
        parts.append(f"（性别：{sex}）")
    if faction:
        parts.append(f"（阵营：{faction}）")
    if titles:
        parts.append(f"（身份或称呼：{'、'.join(titles)}）")
    return "".join(parts) + "。"


def _shop_line(game_data: Any, npc_name: str) -> str:
    """有商店时的纯资料说明；去掉「当作委托报酬」等任务功能描述。"""
    shops = getattr(game_data, "shops", None)
    if shops is None:
        return ""
    try:
        if not shops.has_shop(npc_name):
            return ""
        types: list[str] = []
        for item_name in shops.get_npc_shop(npc_name):
            item = game_data.items.get_by_name(item_name)
            item_type = str(getattr(item, "type", "") or "").strip()
            if item_type and item_type not in types:
                types.append(item_type)
    except Exception:  # noqa: BLE001 - 商店数据异常只影响这一块
        return ""
    return f"你经营商店，货品大致覆盖这些类型：[{'、'.join(types) or '未知'}]。"


def _build_system(
    *,
    npc_name: str,
    state: Any,
    npc_exists: bool,
    roleplay: bool,
    game_data: Any,
    scene_block: str,
    equipment_block: str,
    same_faction_block: str,
) -> tuple[str, list[dict[str, Any]]]:
    """按定稿清单顺序拼 system，并记录每块字数。"""
    parts: list[str] = []
    meta: list[dict[str, Any]] = []

    def add(name: str, text: str) -> None:
        body = (text or "").strip()
        if not body:
            return
        parts.append(body)
        meta.append({"name": name, "chars": len(body)})

    add("世界观背景概要", worldview_block())
    if roleplay:
        add("说话方式", SPEECH_STYLE_RULES)
    if npc_exists:
        add(
            "角色扮演" if roleplay else "角色资料",
            format_npc_role_tagline(
                npc_name=npc_name,
                sex=str(getattr(state, "sex", "") or ""),
                faction=str(getattr(state, "faction", "") or ""),
                titles=list(getattr(state, "titles", None) or []),
            ) if roleplay else _plain_tagline(npc_name, state),
        )
        emotions = "、".join(getattr(state, "emotions", None) or ["普通"])
        add("情绪标签", f"你的可用情绪标签：[{emotions}]。")
        if roleplay:
            add("扮演约束", _ROLEPLAY_TONE)
            add("台词纪律", _ACTING_RULE)
    add("脑补禁令", ANTI_HALLUCINATION_RULE)
    if npc_exists:
        add("商店资料", _shop_line(game_data, npc_name))
        # 公共接口没有玩家信息：切磋按「默认可用」的 NPC 侧事实给出
        add("切磋", build_challenge_hint(bool(getattr(state, "challenge", "")), True))
        add("场景", scene_block)
        add("形象", _appearance(state))
        add("装备", equipment_block)
        add("同阵营角色", same_faction_block)
    return "\n\n".join(parts), meta


def _appearance(state: Any) -> str:
    appearance = str(getattr(state, "appearance", "") or "").strip()
    return f"【你的形象】\n{appearance}" if appearance else ""


def _build_user(*, rag_text: str, input_text: str) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    meta: list[dict[str, Any]] = []
    if rag_text.strip():
        body = f"{_RAG_HEADER}\n{rag_text.strip()}"
        parts.append(body)
        meta.append({"name": "检索到的设定资料", "chars": len(body)})
    if input_text.strip():
        body = f"【最新输入】\n{input_text.strip()}"
        parts.append(body)
        meta.append({"name": "最新输入", "chars": len(body)})
    return "\n\n".join(parts), meta


@router.post(
    "/prompt-preview",
    response_model=PromptPreviewResponse,
    summary="外部提示词预览（只读，不调模型）",
)
async def prompt_preview(payload: PromptPreviewRequest) -> PromptPreviewResponse:
    started = time.perf_counter()
    npc_name = (payload.npc or "").strip()
    input_text = (payload.input or "").strip()

    # ---- NPC 设定（同步只读；不存在不报错，走通用分支）----
    from services.npc.manager import get_npc_manager

    manager = await get_npc_manager()
    state = manager.peek_static(npc_name) if npc_name else None
    npc_exists = state is not None
    npc_states = await manager.all_states()

    # ---- 游戏数据（缺失只影响 NPC 资料与物品提示，不影响接口可用）----
    game_data = None
    try:
        from services.game_data.registry import get_game_data_registry

        game_data = get_game_data_registry()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[prompt-preview] 游戏数据不可用，按无场景处理: %s", exc)
    map_reg = getattr(game_data, "map_registry", None)

    # ---- RAG：与插话/终端同源的 Tier-1 混合检索（同步实现放线程池）----
    rag_text = ""
    sections_meta: list[dict[str, Any]] = []
    bundle = None
    engine = None
    try:
        from services.retrieval import get_retrieval_engine

        engine = get_retrieval_engine()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[prompt-preview] 检索引擎不可用，按无 RAG 处理: %s", exc)

    if engine is not None and getattr(engine, "ready", False):
        forbidden = forbidden_other_chars(npc_states)
        titles = list(getattr(state, "titles", None) or []) if npc_exists else []
        npc_faction = str(getattr(state, "faction", "") or "") if npc_exists else None
        # NPC 不存在时把它并进检索词，符合「没有 NPC 设定就当成查询词的一部分」
        query = f"{npc_name} {input_text}".strip() if not npc_exists else input_text

        def _retrieve() -> Any:
            return retrieve_tier1(
                engine=engine,
                user_query=query,
                npc_name=npc_name if npc_exists else "",
                npc_titles=titles,
                npc_faction=npc_faction,
                npc_last_message=None,
                forbidden=forbidden,
            )

        try:
            bundle = await asyncio.to_thread(_retrieve)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[prompt-preview] 检索失败，按无 RAG 处理: %s", exc)

        if bundle is not None and not bundle.is_empty:
            for header, body in format_sections(bundle, forbidden):
                sections_meta.append({
                    "header": header,
                    "count": len([ln for ln in body.splitlines() if ln.strip()]),
                    "chars": len(body),
                })
                rag_text = f"{rag_text}\n\n{header}\n{body}" if rag_text else f"{header}\n{body}"
            hints = entity_hints_block(bundle, game_data)
            if hints:
                rag_text = f"{rag_text}\n\n{hints}" if rag_text else hints

    # ---- NPC 侧资料块 ----
    scene_block = ""
    equipment_block = ""
    same_faction_block = ""
    if npc_exists:
        scene_block = build_scene_block(
            map_reg=map_reg, npc_states=npc_states, npc_name=npc_name,
        )
        equipment_block = build_npc_equipment_block(
            npc_name=npc_name, game_data=game_data,
        )
        faction = str(getattr(state, "faction", "") or "")
        if faction:
            same_faction_block = same_faction_npcs_block(
                npc_name, faction, npc_states, {npc_name, *_PLAYER_PLACEHOLDERS}, map_reg,
            )

    system_text, system_meta = _build_system(
        npc_name=npc_name,
        state=state,
        npc_exists=npc_exists,
        roleplay=bool(payload.roleplay),
        game_data=game_data,
        scene_block=scene_block,
        equipment_block=equipment_block,
        same_faction_block=same_faction_block,
    )
    user_text, user_meta = _build_user(rag_text=rag_text, input_text=input_text)

    return PromptPreviewResponse(
        npc=npc_name,
        npc_exists=npc_exists,
        roleplay=bool(payload.roleplay),
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        blocks={"system": system_meta, "user": user_meta},
        retrieval={
            "chars": len(rag_text),
            "sections": sections_meta,
            "empty_fallback": not bool(rag_text.strip()),
        },
        meta={"llm_calls": 0, "read_only": True, "model_free": True},
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
