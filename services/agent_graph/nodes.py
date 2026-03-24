"""
LangGraph 图节点实现（对应文档 6.2.3 – 6.2.7）。

节点列表：
  - prepare_context   — 构建 prompt、检索 RAG、注入 NPC 状态
  - decision          — 非流式 LLM 调用（带全部工具）判断是否使用工具
  - tool_executor     — 执行工具调用
  - generate_response — LLM 调用，生成 NPC 对话回复
  - parse_mood        — 从回复 / tool_calls 中解析情绪与好感度
  - post_process      — 保存记忆、更新好感度
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from services.llm_client import call_llm, call_llm_stream
from services.npc_mood_agent import (
    has_update_npc_mood_tool_call,
    is_image_unsupported_error,
    is_tools_unsupported_error,
    parse_mood_from_text,
    parse_update_npc_mood_tool_calls,
    strip_trailing_mood_json,
    strip_trailing_tool_call_text,
)
from services.agent_tools.tool_executor import dispatch_tool_call
from services.agent_tools import (
    CONFIRM_AGENT_TASK_TOOL,
    PREPARE_TASK_CONTEXT_TOOL,
    SEARCH_KNOWLEDGE_TOOL,
    DRAFT_AGENT_TASK_TOOL,
    UPDATE_TASK_DRAFT_TOOL,
    UPDATE_NPC_MOOD_TOOL,
)

from .prompts import (
    DECISION_ROUND_SUFFIX,
    GENERATION_ROUND_SUFFIX,
    build_system_prompt,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

# 同一批 tool_calls 内保证任务流水线顺序，避免模型先 confirm 后 draft 导致「无草案」失败。
_TASK_TOOL_PIPELINE_ORDER: dict[str, int] = {
    "prepare_task_context": 10,
    "draft_agent_task": 20,
    "update_task_draft": 30,
    "confirm_agent_task": 40,
    "cancel_agent_task": 50,
}


def _sort_pending_tool_calls_for_task_pipeline(
    pending_calls: list[dict],
) -> list[dict]:
    if len(pending_calls) <= 1:
        return pending_calls

    def _name(tc: dict) -> str:
        func_info = tc.get("function", tc)
        return str(func_info.get("name", "") or "")

    return [
        tc
        for _, tc in sorted(
            enumerate(pending_calls),
            key=lambda pair: (_TASK_TOOL_PIPELINE_ORDER.get(_name(pair[1]), 1000), pair[0]),
        )
    ]


ALL_TOOLS = [
    PREPARE_TASK_CONTEXT_TOOL,
    SEARCH_KNOWLEDGE_TOOL,
    DRAFT_AGENT_TASK_TOOL,
    UPDATE_TASK_DRAFT_TOOL,
    UPDATE_NPC_MOOD_TOOL,
]

CANCEL_AGENT_TASK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "cancel_agent_task",
        "description": "取消当前待确认的任务草案。",
        "parameters": {
            "type": "object",
            "properties": {
                "draft_id": {"type": "string"},
                # 用于 SSE/前端显示：非常短的“正在进行中”提示
                "ui_hint": {"type": "string", "maxLength": 12},
            },
            "required": ["draft_id"],
            "additionalProperties": False,
        },
    },
}


def _get_full_tools() -> list[dict[str, Any]]:
    return ALL_TOOLS + [CONFIRM_AGENT_TASK_TOOL, CANCEL_AGENT_TASK_TOOL]


_DRAFT_SUCCESS_STATUSES = {"draft_created", "draft_updated", "confirmed"}

# ---------------------------------------------------------------------------
# SSE: tool 调度状态提示
# ---------------------------------------------------------------------------
MAX_UI_HINT_LEN = 12
DEFAULT_UI_HINTS: dict[str, str] = {
    "prepare_task_context": "正在构思任务……",
    "draft_agent_task": "正在拟派清单……",
    "update_task_draft": "正在调整草案……",
    "confirm_agent_task": "正在提交任务……",
    "cancel_agent_task": "正在取消任务……",
}

SYSTEM_MESSAGES: dict[tuple[str, str], str] = {
    ("draft_agent_task", "draft_created"): "{任务草案拟定完成}",
    ("draft_agent_task", "draft_updated"): "{任务草案拟定更新}",
    ("update_task_draft", "draft_updated"): "{任务草案已更新}",
    ("confirm_agent_task", "confirmed"): "{任务发布成功}",
    ("cancel_agent_task", "cancelled"): "{任务已取消}",
}


def _sanitize_ui_hint(value: Any, default_hint: str) -> str:
    """
    前端显示用：空/过长 -> 默认提示。
    这里用字符长度做“严格控制”，保证提示足够短。
    """
    if not isinstance(value, str):
        return default_hint
    s = value.strip()
    if not s:
        return default_hint
    if len(s) > MAX_UI_HINT_LEN:
        return default_hint
    return s


def _strip_instruction_for_assistant_from_confirm_json(result: str) -> str:
    """
    confirm_agent_task 成功时的 instruction_for_assistant 只给「紧随其后的决策轮」用；
    拼正式对话 prompt 时去掉，避免模型在台词里被长期约束，也避免与「下一条玩家消息可再议任务」冲突。
    """
    try:
        parsed = json.loads(result or "{}")
    except Exception:
        return result
    if not isinstance(parsed, dict) or parsed.get("status") != "confirmed":
        return result
    if "instruction_for_assistant" not in parsed:
        return result
    trimmed = {k: v for k, v in parsed.items() if k != "instruction_for_assistant"}
    return json.dumps(trimmed, ensure_ascii=False)


def _build_gen_tool_messages(
    tool_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    为 generate_response 阶段精简工具结果：
    如果 draft_agent_task / update_task_draft / confirm_agent_task 已成功，
    其返回的 draft_summary 已包含完整信息，就不再保留冗长的 prepare_task_context 输出。
    """
    tool_messages = [
        {
            "tool_name": tm.get("tool_name") or "",
            "result": (
                _strip_instruction_for_assistant_from_confirm_json(tm.get("result") or "")
                if (tm.get("tool_name") == "confirm_agent_task")
                else (tm.get("result") or "")
            ),
        }
        for tm in tool_messages
    ]

    def _get_status(tm: dict[str, str]) -> Optional[str]:
        try:
            parsed = json.loads(tm.get("result") or "{}")
            st = parsed.get("status")
            return st if isinstance(st, str) else None
        except Exception:
            return None

    has_draft_success = any(
        (tm.get("tool_name") in ("draft_agent_task", "update_task_draft", "confirm_agent_task")
         and (_get_status(tm) in _DRAFT_SUCCESS_STATUSES))
        for tm in tool_messages
    )
    if not has_draft_success:
        return tool_messages

    has_update_success = any(
        (tm.get("tool_name") == "update_task_draft" and _get_status(tm) in _DRAFT_SUCCESS_STATUSES)
        for tm in tool_messages
    )
    has_confirm_success = any(
        (tm.get("tool_name") == "confirm_agent_task" and _get_status(tm) in _DRAFT_SUCCESS_STATUSES)
        for tm in tool_messages
    )

    # 若后续已经拿到了 update/confirm 成功结果，则 draft_agent_task 的失败明细对 LLM 没有必要。
    drop_draft_agent_task = has_update_success or has_confirm_success
    drop_update_task_draft = has_confirm_success

    condensed: list[dict[str, str]] = []
    for tm in tool_messages:
        tool_name = tm.get("tool_name") or ""
        if drop_draft_agent_task and tool_name == "draft_agent_task":
            continue
        if drop_update_task_draft and tool_name == "update_task_draft":
            continue

        if tool_name == "prepare_task_context":
            condensed.append({
                "tool_name": tool_name,
                "result": '{"status":"ok","message":"上下文已被后续草案工具使用，详见 draft_summary。"}',
            })
        else:
            condensed.append(tm)
    return condensed


def _format_tool_result_for_prompt(tool_name: str, result: str, *, limit: int = 1000) -> str:
    """
    拼 prompt 时的 tool 输出展示策略。

    - `prepare_task_context` 结果包含结构化候选列表，不能被完整 JSON 截断
      （否则会导致 LLM 看不到 `reward_item_candidates` 的后半段）。
    - 其他工具结果保持字符上限，避免 prompt 过长。
    """
    if tool_name == "prepare_task_context":
        return result
    if not result:
        return ""
    if len(result) <= limit:
        return result
    return result[:limit] + "…"


# ---------------------------------------------------------------------------
# Node: prepare_context
# ---------------------------------------------------------------------------

async def prepare_context_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    构建 system prompt 和 user prompt，设置初始状态。
    该节点需要外部注入的配置（通过 config["configurable"]）：
    - rag_service: GameRAGService 实例（检索用）
    - npc_manager: NPCManager 实例
    - memory: MemoryManager 实例
    - payload: NPCChatRequest 对象
    """
    cfgable = config.get("configurable", {})
    rag_service = cfgable["rag_service"]
    npc_manager = cfgable["npc_manager"]
    memory = cfgable["memory"]
    payload = cfgable["payload"]
    game_data = cfgable.get("game_data")

    from services.npc_manager import NPCState
    from services.game_rag_service import (
        ALLOWED_SUMMARIZE_INTERVALS,
        DEFAULT_SUMMARIZE_INTERVAL,
    )
    from services.game_progress import (
        get_progress_stage_name,
        get_progress_stage_level_range,
        get_progress_stage_main_task_range,
    )

    npc_name = payload.npc_name.strip()
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
    npc_challenge = getattr(current_state, "challenge", None)
    player_can_challenge: Optional[bool] = None
    has_shop = False
    shop_reward_types: list[str] = []
    if game_data:
        has_shop = game_data.shops.has_shop(npc_name)
        if has_shop:
            # 推导：当前 NPC 商店“实际覆盖”的 reward_types.optional 集合
            #（用于 prompt 精确约束模型只选择商店确实能卖的类型。）
            from services.agent_tools.context_builder import _matches_reward_type
            from services.agent_tools.schemas import REWARD_OPTIONAL

            equip_mods = game_data.equipment_mods
            items = game_data.items
            shop_item_names = game_data.shops.get_npc_shop(npc_name)

            supported: set[str] = set()
            for shop_item_name in shop_item_names:
                item = items.get_by_name(shop_item_name)
                if item is None:
                    continue
                for rt in REWARD_OPTIONAL:
                    if _matches_reward_type(item, rt, equip_mods):
                        supported.add(rt)

            shop_reward_types = [rt for rt in REWARD_OPTIONAL if rt in supported]

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

    all_npc_states = npc_manager.state

    retrieve_query = rag_service._build_retrieve_query(
        payload.query, npc_name, titles, faction
    )

    forbidden_other_chars = None
    skip_factions_for_other = {"彩蛋", "成员"}
    if (faction or "").strip() not in skip_factions_for_other:
        forbidden_other_chars = {
            name.lower()
            for name, st in all_npc_states.items()
            if (st.faction or "").strip() in skip_factions_for_other
        }

    retrieved_context = await asyncio.to_thread(
        rag_service._retrieve_context,
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

    progress_stage = getattr(payload, "progress_stage", None)
    progress_stage_desc = ""
    stage_name = get_progress_stage_name(progress_stage)
    if stage_name:
        progress_stage_desc = f"当前玩家的主要作战区域为{stage_name}。"

    # 切磋提示三态（副本/关卡混合逻辑）：
    # - NPC确实拥有切磋关卡：has_challenge=True
    # - 若条目有 recommended_min_level，则按推荐等级判定（副本/关卡都适用）
    # - 若条目没有 recommended_min_level：
    #   - 若为副本（大区为“副本任务”）则排除
    #   - 若为关卡，则按 stages.unlock_condition（主线id上限）判定
    if game_data and npc_challenge:
        try:
            stage_num = int(progress_stage or 1)
        except Exception:
            stage_num = 1
        stage_num = max(1, min(7, stage_num))

        level_range = get_progress_stage_level_range(stage_num) or (1, 50)
        player_max_level = (
            int(level_range[1]) if level_range and len(level_range) > 1 else 50
        )

        main_task_range = get_progress_stage_main_task_range(stage_num) or (0, 77)
        main_task_max_id = (
            int(main_task_range[1]) if main_task_range and len(main_task_range) > 1 else 77
        )

        # 同名关卡可能出现在多个 area；不能只用“副本优先”的单一归属判断。
        # 规则：无推荐等级时，若存在任一非副本 area 且其解锁id满足进度，则按关卡可用。
        areas = {
            area
            for (area, name), _si in game_data.stages._stage_infos.items()
            if name == npc_challenge
        }
        has_dungeon_area = "副本任务" in areas
        non_dungeon_areas = [a for a in areas if a != "副本任务"]

        matched_merc_tasks = [
            m
            for m in game_data.mercenary_tasks.list_all()
            if m.stage_name == npc_challenge
        ]
        eligible: list[Any] = []
        unlock_ok = False
        for area in non_dungeon_areas:
            unlock_id = game_data.stages.get_unlock_condition(area, npc_challenge)
            if unlock_id > 0 and unlock_id <= int(main_task_max_id):
                unlock_ok = True
                break

        for m in matched_merc_tasks:
            rec = getattr(m, "recommended_min_level", None)
            if rec is not None:
                if int(rec) <= int(player_max_level):
                    eligible.append(m)
                continue

            if unlock_ok:
                eligible.append(m)
                continue

            if has_dungeon_area:
                continue

        # 若没有任何 mercenary 条目，但该关卡存在可解锁的非副本 area，
        # 仍应按关卡解锁条件判定为可挑战。
        if not matched_merc_tasks and unlock_ok:
            player_can_challenge = True
        else:
            player_can_challenge = bool(eligible)

    mentioned_npcs, mentioned_names = rag_service._find_mentioned_npcs(
        payload.query, npc_name, all_npc_states, faction
    )
    same_faction_npc_strs = rag_service._get_same_faction_npcs(
        npc_name, faction, all_npc_states, exclude_names=mentioned_names
    )

    mentioned_npcs_str = ""
    if mentioned_npcs:
        mentioned_npcs_str += (
            "可能涉及到的其他角色的设定（注意，如果和你不是一个阵营的角色你可能了解的不多）：\n"
            + "\n".join(mentioned_npcs)
            + "\n\n"
        )

    same_faction_str = ""
    if same_faction_npc_strs:
        same_faction_str = (
            ("其他同阵营角色：\n" if mentioned_npcs else "同阵营角色：\n")
            + "\n".join(same_faction_npc_strs)
            + "\n\n"
        )

    pending_draft = state.get("pending_task_draft")
    if pending_draft is None:
        from services.task_draft_store import get_session_task_draft_store
        try:
            db_draft = await get_session_task_draft_store().get_draft_json_by_session_id(
                payload.session_id
            )
            if db_draft and isinstance(db_draft, dict):
                pending_draft = db_draft
        except Exception:
            logger.warning("加载 DB 草案失败", exc_info=True)

    pending_draft_summary = ""
    if pending_draft:
        from services.agent_tools.tool_executor import _detailed_draft_summary
        pending_draft_summary = _detailed_draft_summary(
            pending_draft,
            game_data,
            rag_context_text=retrieved_context or "",
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
        joined = "\n".join(history_lines)
        history_str += (
            "以下是最近的对话记录（按时间从早到晚排列），"
            "请结合上述摘要与近期记录，在保持人物性格与情节连贯的前提下继续对话：\n"
            if summary_text
            else "下面是你与玩家之间的对话历史（按时间从早到晚排列），"
            "请在保持人物性格与情节连贯的前提下继续对话：\n"
        ) + joined + "\n\n"

    raw_emotion = getattr(payload, "current_emotion", None)
    current_emotion_for_use = None
    if raw_emotion is not None and isinstance(raw_emotion, str):
        s = raw_emotion.strip()
        if s and s.lower() not in ("null", "undefined"):
            current_emotion_for_use = s
    emotion_for_portrait = current_emotion_for_use or "普通"
    image_path, image_description = rag_service._get_npc_image_path(
        npc_name, emotion_for_portrait
    )
    emotion_hint = ""
    if current_emotion_for_use and current_emotion_for_use != "普通":
        emotion_hint = f"你之前的情绪是「{current_emotion_for_use}」。"

    system_prompt = build_system_prompt(
        npc_name=npc_name,
        sex=sex or "",
        faction=faction or "",
        titles=titles,
        emotions=emotions,
        has_shop=has_shop,
        shop_reward_types=shop_reward_types,
        has_challenge=bool(npc_challenge),
        player_can_challenge=player_can_challenge,
        same_faction_npcs=same_faction_str,
        player_identity=player_identity,
        progress_stage_desc=progress_stage_desc,
        favorability=favorability,
        relationship_level=relationship_level,
        mentioned_npcs_str=mentioned_npcs_str,
        pending_draft_summary=pending_draft_summary,
    )

    # 立绘描述不写入共享 user_prompt，避免缓存变动；仅在流式/生成阶段按需拼接到 user 内容末尾
    user_prompt = build_user_prompt(
        retrieved_context=retrieved_context or "",
        history_str=history_str,
        user_query=payload.query,
        # emotion_hint 不要写入共享 user_prompt，避免与 llm_client 末尾拼接重复。
        # 由 llm_client 统一决定：是否与 image_description 同行、或单独一行。
        emotion_hint="",
        image_description="",
    )

    from core.config import get_settings
    settings = get_settings()
    effective_api_key = (
        (payload.api_key.strip() if payload.api_key and payload.api_key.strip() else None)
        or settings.llm_api_key
    )
    effective_api_base = (
        (payload.api_base.strip() if payload.api_base and payload.api_base.strip() else None)
        or settings.llm_api_base
    )
    effective_model = (
        (payload.model_name.strip() if payload.model_name and payload.model_name.strip() else None)
        or settings.llm_model_name
    )

    return {
        "npc_name": npc_name,
        "player_progress": progress_stage or 0,
        "npc_affinity": favorability,
        "npc_relationship_level": relationship_level,
        "npc_faction": faction or "",
        "npc_titles": titles,
        "npc_sex": sex or "",
        "npc_challenge": npc_challenge,
        "npc_emotions": emotions,
        "session_id": payload.session_id,
        "retrieved_context": retrieved_context or "",
        "api_key": effective_api_key or "",
        "api_base": effective_api_base or "",
        "model_name": effective_model or "",
        "image_path": str(image_path) if image_path else None,
        "image_description": image_description,
        "emotion_hint": emotion_hint,
        "tool_call_round": 0,
        "has_tool_calls": False,
        "pending_task_draft": pending_draft,
        "task_confirmed": False,
        "task_write_result": None,
        "payload_dict": {
            "query": payload.query,
            "npc_name": npc_name,
            "session_id": payload.session_id,
        },
        "effective_summarize_interval": effective_interval,
        # 存储构建好的 prompt（供 decision / generate 节点复用）
        "_system_prompt": system_prompt,
        "_user_prompt": user_prompt,
    }


# ---------------------------------------------------------------------------
# Node: decision
# ---------------------------------------------------------------------------

async def decision_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    非流式 LLM 调用（带全部工具），判断是否需要使用工具。
    """
    system_prompt = state.get("_system_prompt", "")
    user_prompt = state.get("_user_prompt", "")
    round_num = state.get("tool_call_round", 0)

    decision_suffix = f"\n\n{DECISION_ROUND_SUFFIX}"
    full_user_prompt = user_prompt + decision_suffix

    tool_messages = state.get("_tool_messages", [])
    if tool_messages:
        full_user_prompt += "\n\n【工具执行结果】\n"
        for tm in tool_messages:
            full_user_prompt += (
                f"[{tm['tool_name']}]: "
                f"{_format_tool_result_for_prompt(tm['tool_name'], tm['result'])}\n"
            )
            # Debug
            # print('——————【工具执行结果】decision_node——————')
            # print(
            #     f"[{tm['tool_name']}]: "
            #     f"{_format_tool_result_for_prompt(tm['tool_name'], tm['result'])}\n"
            # )

    # 非流式决策轮不传立绘，仅流式/生成阶段再传，减少 token 与缓存变动
    reply_text = ""
    tool_calls: list[dict] = []

    try:
        reply_text, tool_calls = await call_llm(
            api_key=state.get("api_key"),
            api_base=state.get("api_base"),
            model_name=state.get("model_name"),
            system_prompt=system_prompt,
            user_prompt=full_user_prompt,
            image_path=None,
            image_description=None,
            emotion_hint=state.get("emotion_hint") if round_num == 0 else None,
            tools=_get_full_tools(),
        )
    except Exception as e:
        if is_tools_unsupported_error(e):
            reply_text, tool_calls = await call_llm(
                api_key=state.get("api_key"),
                api_base=state.get("api_base"),
                model_name=state.get("model_name"),
                system_prompt=system_prompt,
                user_prompt=full_user_prompt,
                image_path=None,
                image_description=None,
                emotion_hint=state.get("emotion_hint") if round_num == 0 else None,
                tools=None,
            )
        else:
            raise

    mood_tool_calls = [
        tc for tc in tool_calls
        if (tc.get("function", {}).get("name") or tc.get("name", "")) == "update_npc_mood"
    ]
    other_tool_calls = [
        tc for tc in tool_calls
        if (tc.get("function", {}).get("name") or tc.get("name", "")) != "update_npc_mood"
    ]

    has_tools = bool(other_tool_calls)

    return {
        "tool_call_round": round_num + 1,
        "has_tool_calls": has_tools,
        "_pending_tool_calls": other_tool_calls,
        "_mood_tool_calls": state.get("_mood_tool_calls", []) + mood_tool_calls,
        "_decision_reply": reply_text or "",
    }


# ---------------------------------------------------------------------------
# Node: tool_executor
# ---------------------------------------------------------------------------

async def tool_executor_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    执行 decision 节点产出的 tool_calls。
    """
    cfgable = config.get("configurable", {})
    game_data = cfgable.get("game_data")
    npc_manager = cfgable.get("npc_manager")
    rag_service = cfgable.get("rag_service")

    pending_calls = _sort_pending_tool_calls_for_task_pipeline(
        list(state.get("_pending_tool_calls", [])),
    )
    pending_draft = state.get("pending_task_draft")
    npc_name = state.get("npc_name", "")
    npc_faction = state.get("npc_faction", "")
    npc_challenge = state.get("npc_challenge")
    player_progress = state.get("player_progress", 1)
    npc_affinity = state.get("npc_affinity", 0)

    npc_states = npc_manager.state if npc_manager else None

    def _make_retrieve_fn():
        if rag_service is None:
            return None

        def _fn(keyword: str) -> str:
            titles = state.get("npc_titles", [])
            faction = state.get("npc_faction", "")
            retrieve_query = rag_service._build_retrieve_query(
                keyword, npc_name, titles, faction
            )
            return rag_service._retrieve_context(
                npc_name, keyword, retrieve_query
            )
        return _fn

    tool_messages: list[dict[str, str]] = list(state.get("_tool_messages", []))
    task_write_result = state.get("task_write_result")
    ui_events: list[dict[str, Any]] = list(state.get("_ui_events", []))
    system_prefixes: list[str] = []

    for tc in pending_calls:
        func_info = tc.get("function", tc)
        tool_name = func_info.get("name", "")
        raw_args = func_info.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                tool_args = json.loads(raw_args)
            except Exception:
                tool_args = {}
        else:
            tool_args = raw_args or {}

        # 1) 工具开始：推送一个极短提示给前端
        start_hint = _sanitize_ui_hint(
            tool_args.get("ui_hint"),
            DEFAULT_UI_HINTS.get(tool_name, "正在思考……"),
        )
        ui_events.append({
            "event_type": "tool_status",
            "text": start_hint,
            "tool_name": tool_name,
        })

        result_str, updated_draft, write_result = dispatch_tool_call(
            tool_name,
            tool_args,
            npc_name=npc_name,
            npc_faction=npc_faction,
            npc_challenge=npc_challenge,
            player_progress=player_progress,
            npc_affinity=npc_affinity,
            npc_states=npc_states,
            game_data=game_data,
            pending_draft=pending_draft,
            retrieve_fn=_make_retrieve_fn(),
            rag_context_text=state.get("retrieved_context") or "",
        )

        if updated_draft is not None:
            pending_draft = updated_draft
        elif tool_name in ("confirm_agent_task", "cancel_agent_task"):
            pending_draft = None

        if write_result:
            task_write_result = write_result

        # 2) 工具完成：根据结果 status 推送关键系统消息（成功时）
        try:
            parsed = json.loads(result_str or "{}")
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            st = parsed.get("status")
            sys_text = SYSTEM_MESSAGES.get((tool_name, st))
            if sys_text:
                payload: dict[str, Any] = {"event_type": "system", "text": sys_text}
                if tool_name == "draft_agent_task":
                    if isinstance(parsed.get("draft_id"), str):
                        payload["draft_id"] = parsed.get("draft_id")
                if tool_name == "confirm_agent_task":
                    if isinstance(parsed.get("task_id"), (int, str)):
                        payload["task_id"] = parsed.get("task_id")
                ui_events.append(payload)
                system_prefixes.append(sys_text)

        tool_messages.append({
            "tool_name": tool_name,
            "result": result_str,
        })

    return {
        "_tool_messages": tool_messages,
        "_pending_tool_calls": [],
        "pending_task_draft": pending_draft,
        "task_write_result": task_write_result,
        "_ui_events": ui_events,
        "_system_prefix_text": (
            (state.get("_system_prefix_text") or "")
            + ("".join(system_prefixes) + "\n\n" if system_prefixes else "")
        ),
    }


# ---------------------------------------------------------------------------
# Node: generate_response (non-streaming for `ask`)
# ---------------------------------------------------------------------------

async def generate_response_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    生成最终 NPC 对话回复（非流式，用于 ask 接口）。
    """
    system_prompt = state.get("_system_prompt", "")
    user_prompt = state.get("_user_prompt", "")

    gen_suffix = f"\n\n{GENERATION_ROUND_SUFFIX}"
    full_user_prompt = user_prompt + gen_suffix

    raw_tool_messages = state.get("_tool_messages", [])
    gen_messages = _build_gen_tool_messages(raw_tool_messages)
    if gen_messages:
        full_user_prompt += "\n\n【工具执行结果】\n"
        for tm in gen_messages:
            full_user_prompt += (
                f"[{tm['tool_name']}]: "
                f"{_format_tool_result_for_prompt(tm['tool_name'], tm['result'])}\n"
            )
            # Debug
            # print('——————【工具执行结果】generate_response_node——————')
            # print(
            #     f"[{tm['tool_name']}]: "
            #     f"{_format_tool_result_for_prompt(tm['tool_name'], tm['result'])}\n"
            # )

    # 为了与决策轮在缓存结构上保持一致，本轮依然传入完整工具 definitions，
    # 但通过额外说明严格约束：本轮只允许（可选）调用 update_npc_mood，不得再次调用任务/检索相关工具。
    full_user_prompt += (
        "\n\n【本轮工具调用约束】本轮只允许（可选）调用 update_npc_mood，用于上报情绪和好感度变化；"
        "严禁调用任何与任务或检索相关的工具（如 prepare_task_context、draft_agent_task、update_task_draft、"
        "confirm_agent_task、cancel_agent_task、search_knowledge 等）。"
        "请根据前文说明和工具执行结果，生成对话回复。"
    )

    decision_reply = state.get("_decision_reply", "")
    # 决策阶段返回的文本（如果有）不参与后续生成，避免“决策轮自述”污染正文。

    image_path_str = state.get("image_path")
    image_path = None
    if image_path_str:
        from pathlib import Path
        image_path = Path(image_path_str)

    try:
        reply_text, tool_calls = await call_llm(
            api_key=state.get("api_key"),
            api_base=state.get("api_base"),
            model_name=state.get("model_name"),
            system_prompt=system_prompt,
            user_prompt=full_user_prompt,
            image_path=image_path,
            image_description=state.get("image_description"),
            emotion_hint=state.get("emotion_hint") or None,
            tools=_get_full_tools(),
        )
    except Exception as e:
        if is_image_unsupported_error(e) and image_path:
            try:
                reply_text, tool_calls = await call_llm(
                    api_key=state.get("api_key"),
                    api_base=state.get("api_base"),
                    model_name=state.get("model_name"),
                    system_prompt=system_prompt,
                    user_prompt=full_user_prompt,
                    image_path=None,
                    image_description=state.get("image_description"),
                    emotion_hint=state.get("emotion_hint") or None,
                    tools=_get_full_tools(),
                )
            except Exception as e2:
                if is_tools_unsupported_error(e2):
                    reply_text, tool_calls = await call_llm(
                        api_key=state.get("api_key"),
                        api_base=state.get("api_base"),
                        model_name=state.get("model_name"),
                        system_prompt=system_prompt,
                        user_prompt=full_user_prompt,
                        image_path=None,
                        image_description=state.get("image_description"),
                        emotion_hint=state.get("emotion_hint") or None,
                        tools=None,
                    )
                else:
                    raise
        elif is_tools_unsupported_error(e):
            try:
                    reply_text, tool_calls = await call_llm(
                    api_key=state.get("api_key"),
                    api_base=state.get("api_base"),
                    model_name=state.get("model_name"),
                    system_prompt=system_prompt,
                    user_prompt=full_user_prompt,
                    image_path=image_path,
                    image_description=state.get("image_description"),
                    emotion_hint=state.get("emotion_hint") or None,
                    tools=None,
                )
            except Exception as e2:
                if is_image_unsupported_error(e2) and image_path:
                    reply_text, tool_calls = await call_llm(
                        api_key=state.get("api_key"),
                        api_base=state.get("api_base"),
                        model_name=state.get("model_name"),
                        system_prompt=system_prompt,
                        user_prompt=full_user_prompt,
                        image_path=None,
                        image_description=state.get("image_description"),
                        emotion_hint=state.get("emotion_hint") or None,
                        tools=None,
                    )
                else:
                    raise
        else:
            raise

    # 仅在本轮解析情绪工具；其他工具调用（若有）全部忽略，不执行任何副作用。
    mood_calls = state.get("_mood_tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", tc)
        name = func.get("name") or ""
        if name == "update_npc_mood":
            mood_calls.append(tc)

    system_prefix = state.get("_system_prefix_text") or ""
    if system_prefix:
        reply_text = f"{system_prefix}{reply_text or ''}"

    return {
        "final_reply": reply_text or "",
        "_mood_tool_calls": mood_calls,
    }


# ---------------------------------------------------------------------------
# Streaming generate (standalone coroutine, not a graph node)
# ---------------------------------------------------------------------------

async def generate_response_stream(
    state: dict[str, Any],
    config: RunnableConfig,
):
    """
    流式生成 NPC 对话回复（用于 ask_stream 接口）。
    这是一个独立协程，不是 LangGraph 节点——因为它需要 yield 流式 chunk。

    Yields: ("content", delta_text)
    Returns: (full_reply, mood_tool_calls) 更新到 state
    """
    system_prompt = state.get("_system_prompt", "")
    user_prompt = state.get("_user_prompt", "")

    gen_suffix = f"\n\n{GENERATION_ROUND_SUFFIX}"
    full_user_prompt = user_prompt + gen_suffix

    raw_tool_messages = state.get("_tool_messages", [])
    gen_messages = _build_gen_tool_messages(raw_tool_messages)
    if gen_messages:
        full_user_prompt += "\n\n【工具执行结果】\n"
        for tm in gen_messages:
            full_user_prompt += (
                f"[{tm['tool_name']}]: "
                f"{_format_tool_result_for_prompt(tm['tool_name'], tm['result'])}\n"
            )
            # Debug
            # print('——————【工具执行结果】generate_response_stream——————')
            # print(
            #     f"[{tm['tool_name']}]: "
            #     f"{_format_tool_result_for_prompt(tm['tool_name'], tm['result'])}\n"
            # )

    # 与非流式生成保持一致：传入完整工具 definitions，但通过 prompt 约束本轮仅允许（可选）调用 update_npc_mood。
    full_user_prompt += (
        "\n\n【本轮工具调用约束】本轮只允许（可选）调用 update_npc_mood，用于上报情绪和好感度变化；"
        "严禁调用任何与任务或检索相关的工具（如 prepare_task_context、draft_agent_task、update_task_draft、"
        "confirm_agent_task、cancel_agent_task、search_knowledge 等）。"
        "请根据前文说明和工具执行结果，生成对话回复。"
    )

    decision_reply = state.get("_decision_reply", "")
    # 决策阶段返回的文本（如果有）不参与后续生成，避免“决策轮自述”污染正文。

    image_path_str = state.get("image_path")
    image_path = None
    if image_path_str:
        from pathlib import Path
        image_path = Path(image_path_str)

    _TRUNCATE_PREFIXES = ["工具调用", "{", "<!---", "<!--", "update_npc_mood(", "tool_calls_list"]
    system_prefix = state.get("_system_prefix_text") or ""
    full_content = system_prefix
    streamed_len = len(system_prefix)
    truncating = False
    tool_calls_list: list[dict] = []

    def _earliest_truncate_at(text: str) -> int:
        out = -1
        for p in _TRUNCATE_PREFIXES:
            if "update_npc_mood" in p.lower() or "tool_calls_list" in p.lower():
                idx = text.lower().find(p.lower())
            else:
                idx = text.find(p)
            # 避免系统前缀（以 "{" 开头）触发截断逻辑
            if system_prefix and p == "{" and idx != -1 and idx < streamed_len:
                continue
            if idx != -1 and (out == -1 or idx < out):
                out = idx
        return out

    # 若有系统前缀，则先把它作为“正文增量”推给前端，保证与 done 一致
    if system_prefix:
        yield ("content", system_prefix)

    async def _run_stream(img_path, img_desc, use_tools):
        nonlocal full_content, streamed_len, truncating, tool_calls_list
        # 继续向后追加模型输出（system_prefix 已在外层 yield 并写入 full_content）
        truncating = False
        tool_calls_list = []
        async for event_type, data in call_llm_stream(
            api_key=state.get("api_key"),
            api_base=state.get("api_base"),
            model_name=state.get("model_name"),
            system_prompt=system_prompt,
            user_prompt=full_user_prompt,
            image_path=img_path,
            image_description=img_desc,
            emotion_hint=state.get("emotion_hint") or None,
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
                model_full_content, tool_calls_list = data
                full_content = system_prefix + (model_full_content or "")
                return

    try:
        async for ev, dat in _run_stream(image_path, state.get("image_description"), _get_full_tools()):
            yield ev, dat
    except Exception as e:
        if is_image_unsupported_error(e) and image_path:
            try:
                async for ev, dat in _run_stream(None, state.get("image_description"), _get_full_tools()):
                    yield ev, dat
            except Exception as e2:
                if is_tools_unsupported_error(e2):
                    async for ev, dat in _run_stream(None, state.get("image_description"), None):
                        yield ev, dat
                else:
                    raise
        elif is_tools_unsupported_error(e):
            try:
                async for ev, dat in _run_stream(image_path, state.get("image_description"), None):
                    yield ev, dat
            except Exception as e2:
                if is_image_unsupported_error(e2) and image_path:
                    async for ev, dat in _run_stream(None, state.get("image_description"), None):
                        yield ev, dat
                else:
                    raise
        else:
            raise

    # 仅在本轮解析情绪工具；其他工具调用（若有）全部忽略，不执行任何副作用。
    mood_calls = list(state.get("_mood_tool_calls", []))
    for tc in tool_calls_list:
        func = tc.get("function", tc)
        name = func.get("name") or ""
        if name == "update_npc_mood":
            mood_calls.append(tc)

    state["final_reply"] = full_content or ""
    state["_mood_tool_calls"] = mood_calls


# ---------------------------------------------------------------------------
# Node: parse_mood
# ---------------------------------------------------------------------------

def parse_mood_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    从回复文本 / tool_calls 中解析情绪与好感度变化。
    """
    reply = state.get("final_reply", "")
    emotions = state.get("npc_emotions", ["普通"])
    mood_tool_calls = state.get("_mood_tool_calls", [])

    delta, emotion = parse_update_npc_mood_tool_calls(
        mood_tool_calls, allowed_emotions=emotions
    )
    parsed_delta, parsed_emotion = parse_mood_from_text(reply)
    cleaned, fallback_delta, fallback_emotion = strip_trailing_mood_json(
        reply, allowed_emotions=emotions
    )

    if fallback_delta is not None and fallback_emotion is not None:
        reply = (cleaned or "").strip() or "【对方无回应，请稍后再试。】"
        if not has_update_npc_mood_tool_call(mood_tool_calls):
            delta, emotion = fallback_delta, fallback_emotion

    if not has_update_npc_mood_tool_call(mood_tool_calls):
        default_emo = "普通" if "普通" in emotions else (emotions[0] if emotions else "普通")
        if (delta == 0 and emotion == default_emo) and (
            parsed_delta is not None or parsed_emotion
        ):
            if parsed_delta is not None:
                delta = parsed_delta
            if parsed_emotion is not None:
                emotion = (
                    parsed_emotion
                    if parsed_emotion in emotions
                    else default_emo
                )

    reply = strip_trailing_tool_call_text(reply)

    return {
        "final_reply": reply.strip() or "【对方无回应，请稍后再试。】",
        "emotion": emotion,
        "favorability_change": delta,
    }


# ---------------------------------------------------------------------------
# Node: post_process
# ---------------------------------------------------------------------------

async def post_process_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    保存对话记录、更新好感度、持久化任务草案状态。
    """
    cfgable = config.get("configurable", {})
    memory = cfgable["memory"]
    npc_manager = cfgable["npc_manager"]
    payload = cfgable["payload"]

    npc_name = state.get("npc_name", "")
    reply = state.get("final_reply", "")
    delta = state.get("favorability_change", 0)
    effective_interval = state.get("effective_summarize_interval", 30)

    await memory.add_message(payload.session_id, "user", payload.query)
    await memory.add_message(
        payload.session_id,
        "assistant",
        reply,
        llm_config={
            "api_key": state.get("api_key"),
            "api_base": state.get("api_base"),
            "model_name": state.get("model_name"),
        },
        npc_name=npc_name,
        summarize_interval=effective_interval,
    )

    updated_state = npc_manager.update_favorability(npc_name, delta)
    await npc_manager.save()

    # 持久化任务草案状态 + 4.5 草案自动过期（连续 3 次 ask 未调用任务相关工具则清除）
    pending_draft = state.get("pending_task_draft")
    try:
        from services.task_draft_store import get_session_task_draft_store
        store = get_session_task_draft_store()
        tool_messages = state.get("_tool_messages") or []
        had_task_tool = any(
            (tm.get("tool_name") or "") in store._TASK_RELATED_TOOL_NAMES
            for tm in tool_messages
        )
        if had_task_tool:
            await store.reset_rounds_without_task(payload.session_id)
            should_persist_draft = True
        else:
            rounds = await store.increment_rounds_without_task(payload.session_id)
            if rounds >= 3:
                await store.delete_by_session_id(payload.session_id)
                await store.reset_rounds_without_task(payload.session_id)
                should_persist_draft = False
            else:
                should_persist_draft = True
        if pending_draft and isinstance(pending_draft, dict) and should_persist_draft:
            await store.upsert_draft(
                session_id=payload.session_id,
                draft=pending_draft,
            )
        elif not pending_draft or not should_persist_draft:
            await store.delete_by_session_id(payload.session_id)
    except Exception:
        logger.warning("持久化任务草案失败", exc_info=True)

    return {
        "npc_affinity": updated_state.favorability,
        "npc_relationship_level": updated_state.relationship_level,
    }
