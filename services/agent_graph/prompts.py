"""
分层 Prompt 模板（对应文档 6.5.3）。

Layer 1 — 固定层（跨所有 NPC 和请求）
Layer 2 — NPC 层（同一 NPC 的多次请求相同）
Layer 3 — 会话层（按变化频率从低到高排列）
Layer 4 — 检索层（按请求变化）

前缀越靠前越稳定，最大化 LLM API 的 prompt prefix cache 命中率。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.game_rag_service import WORLD_BACKGROUND


# ---------------------------------------------------------------------------
# Layer 1: 固定层
# ---------------------------------------------------------------------------

TOOL_USAGE_GUIDE = """\
【Agent 工具使用指南】
你可以在合适时机调用以下工具：
- prepare_task_context：当你想为玩家发布任务时，先调用此工具获取可选数据和规则。
- draft_agent_task：根据 prepare_task_context 返回的数据，生成任务草案。
- update_task_draft：局部修改已有的待确认草案（如调整奖励、换关卡）。
- confirm_agent_task：玩家接受任务后确认并写入。
- cancel_agent_task：取消当前待确认的任务草案。
- update_npc_mood：每次回复后上报好感度变化与情绪。
- search_knowledge：检索游戏设定或情报。

【任务发布原则】
- 只在对话氛围合适时考虑发布任务，不是每次都需要。
- 任务必须符合你的角色定位和能力范围。
- 发布动机应自然融入对话，不要生硬地突然提出。
- 同一时间只处理一个任务草案。
"""

DIALOGUE_FORMAT_RULES = """\
【输出方式】
1. 回复内容：只输出角色对话文本，不要有 JSON 或结构化数据。
2. 在回复的同时，调用工具 update_npc_mood 上报 favorability_change（-5～5，常规为 0）与 emotion（从可用情绪中选）。
3. 若无法调用工具而必须用正文传参时，最后一段只输出一行 JSON。
4. 动作与台词格式：非必要时不出现动作描写。若需表达肢体动作、神态或心理活动，\
必须且只能使用全角粗括号【】包裹；台词直接输出，不加引号；\
严禁用半角括号 () 或星号 * 描述动作。\
你输出的动作若涉及人称，一律单独起一行，采用第三人称视角：用角色名指代你自己，用「你」指代玩家。
"""

DECISION_ROUND_SUFFIX = (
    "本轮请仅判断是否需要调用工具并生成 tool_calls。"
    "若不需要任何工具，返回空内容即可，不要输出对话文本。"
)

GENERATION_ROUND_SUFFIX = "请根据以上信息，以 NPC 身份生成对话回复。"


def build_layer1() -> str:
    return (
        "【世界观背景概要】\n"
        f"{WORLD_BACKGROUND}\n\n"
        f"{DIALOGUE_FORMAT_RULES}\n"
        f"{TOOL_USAGE_GUIDE}"
    )


# ---------------------------------------------------------------------------
# Layer 2: NPC 层
# ---------------------------------------------------------------------------

def build_layer2(
    *,
    npc_name: str,
    sex: str = "",
    faction: str = "",
    titles: list[str] | None = None,
    emotions: list[str] | None = None,
    has_shop: bool = False,
    has_challenge: bool = False,
) -> str:
    sex_desc = f"（性别：{sex}）" if sex else ""
    faction_desc = f"（阵营：{faction}）" if faction else ""
    titles_desc = f"（身份或称呼：{'、'.join(titles)}）" if titles else ""
    emotions_list = emotions or ["普通"]
    emotions_str = "、".join(emotions_list)

    shop_constraint = (
        "你可以将自己商店的物品作为任务奖励（物品等级需匹配玩家进度）。"
        if has_shop
        else "你不经营商店，奖励以金币、经验、药剂等通用物资为主。"
    )

    return (
        f"你现在扮演游戏角色「{npc_name}」{sex_desc}{faction_desc}{titles_desc}。\n"
        f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。\n"
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，"
        "用简体中文回答玩家本次的发言。\n"
        "非特殊要求下，每次对话长度不必太长。不要自己脑补不存在的设定。\n\n"
        f"【任务发布约束】\n"
        f"你作为「{npc_name}」（{faction or '未知阵营'}），在决定是否发布任务时：\n"
        "- 只发布符合你身份和能力范围的任务；\n"
        "- 即使玩家进度较高，你也不应发布超出你角色定位的高难度任务；\n"
        "- 在你的身份允许的情况下，可以优先发布最符合玩家当前进度的任务；\n"
        "- 发布任务的动机应自然融入对话，不要生硬地突然提出任务；\n"
        "- 只有在对话氛围合适时才考虑发布任务，不是每次对话都需要任务；\n"
        f"- {shop_constraint}\n"
    )


# ---------------------------------------------------------------------------
# Layer 3: 会话层（按变化频率从低到高排列）
# ---------------------------------------------------------------------------

def build_layer3(
    *,
    same_faction_npcs: str = "",
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    mentioned_npcs_str: str = "",
    pending_draft_summary: str = "",
) -> str:
    parts: list[str] = []

    if same_faction_npcs:
        parts.append(same_faction_npcs)

    if player_identity:
        parts.append(f"玩家的身份是：{player_identity}")

    if progress_stage_desc:
        parts.append(progress_stage_desc)

    parts.append(
        f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。"
    )

    if mentioned_npcs_str:
        parts.append(mentioned_npcs_str)

    if pending_draft_summary:
        parts.append(
            "【待确认的任务草案】\n"
            f"{pending_draft_summary}\n"
            "玩家可能会接受、拒绝或讨价还价。"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 4: 检索层
# ---------------------------------------------------------------------------

def build_layer4(
    *,
    retrieved_context: str = "",
    history_str: str = "",
    user_query: str = "",
) -> str:
    parts: list[str] = []

    if retrieved_context:
        parts.append(
            "下面是可能与你相关的检索设定和你的过往台词片段"
            "（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
            f"{retrieved_context}"
        )

    if history_str:
        parts.append(history_str)

    if user_query:
        parts.append(f"玩家：{user_query}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 组装完整 system prompt
# ---------------------------------------------------------------------------

def build_system_prompt(
    *,
    npc_name: str,
    sex: str = "",
    faction: str = "",
    titles: list[str] | None = None,
    emotions: list[str] | None = None,
    has_shop: bool = False,
    has_challenge: bool = False,
    same_faction_npcs: str = "",
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    mentioned_npcs_str: str = "",
    pending_draft_summary: str = "",
) -> str:
    """组装 Layer 1 + Layer 2 + Layer 3 为 system prompt。"""
    layer1 = build_layer1()
    layer2 = build_layer2(
        npc_name=npc_name,
        sex=sex,
        faction=faction,
        titles=titles,
        emotions=emotions,
        has_shop=has_shop,
        has_challenge=has_challenge,
    )
    layer3 = build_layer3(
        same_faction_npcs=same_faction_npcs,
        player_identity=player_identity,
        progress_stage_desc=progress_stage_desc,
        favorability=favorability,
        relationship_level=relationship_level,
        mentioned_npcs_str=mentioned_npcs_str,
        pending_draft_summary=pending_draft_summary,
    )

    return f"{layer2}\n{layer1}\n\n{layer3}"


def build_user_prompt(
    *,
    retrieved_context: str = "",
    history_str: str = "",
    user_query: str = "",
    emotion_hint: str = "",
    image_description: str = "",
) -> str:
    """组装 Layer 4 为 user prompt。"""
    prefix_parts: list[str] = []
    if image_description:
        prefix_parts.append(image_description + "。")
    if emotion_hint:
        prefix_parts.append(emotion_hint)
    prefix = "".join(prefix_parts)

    layer4 = build_layer4(
        retrieved_context=retrieved_context,
        history_str=history_str,
        user_query=user_query,
    )
    if prefix:
        return f"{prefix}\n\n{layer4}"
    return layer4
