"""
分层 Prompt 模板（对应文档 6.5.3）。

Layer 1 — 固定层（跨所有 NPC 和请求）
Layer 2 — NPC 层（同一 NPC 的多次请求相同）
Layer 3 — 会话层（按变化频率从低到高排列）
Layer 4 — 检索层（按请求变化）

拼接 system 时固定层（Layer 1）置于最前，使跨 NPC 请求共享最长公共前缀，最大化 LLM API 的 prompt prefix cache 命中率。

路线二以来的重构要点：
- 工具使用指南不再夹带任务发布/协商等长段**流程**描述——这些迁到 ``services/skills/<name>/SKILL.md``（Anthropic 2026 规范），
  Agent 仅在需要时通过 ``list_skills`` / ``read_skill`` / ``read_skill_file`` 三个元工具按需加载。
- 本文件只保留 **跨 Agent 公用的薄骨架** + **skills 索引**；各 Agent 的专属指引在 Route 3 的 services/agents 里各自拼装。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.game_rag_service import WORLD_BACKGROUND
from services.skills import get_skill_registry


# ---------------------------------------------------------------------------
# Layer 1: 固定层（只放与具体流程无关的**通用骨架**）
# ---------------------------------------------------------------------------

DIALOGUE_FORMAT_RULES = """\
【输出方式】
1. 正式对话轮：只输出角色对话文本，不要有 JSON 或结构化数据；该轮不提供工具。
2. 情绪与好感度：在决策轮通过 update_npc_mood 上报 favorability_change（-5～5，常规为 0）与 emotion（从可用情绪中选）；正式台词应与所报情绪一致。详见 skill: mood-tracking。
3. 动作与台词格式：非必要时不出现动作描写；若需要肢体动作/神态/环境描写，必须且只能使用全角粗括号【】包裹；台词直接输出，不加引号；严禁用半角括号 () 或星号 * 描述动作。\
你输出的动作若涉及人称，一律单独起一行，采用第三人称视角：用角色名指代你自己，用「你」指代玩家。
4. 全角粗括号【】仅可用于动作/神态/环境描写，其它内容不得放入。
5. 若输出对话时发现此前工具调用失败，要以执行失败为前提继续对话（例如任务发布失败时，取消/拒绝发布任务而不是继续发布）。
"""


def _skills_index_block() -> str:
    """把 SKILL.md frontmatter 的 name + description 汇成一段**索引**拼进 system prompt。

    只给 Level 1 简表；Agent 在需要时再用 read_skill / read_skill_file 拉取 body / references。
    """
    rows = get_skill_registry().index()
    if not rows:
        return ""
    lines = ["【可用 Skills 索引（需要时用 read_skill 拉取完整指引）】"]
    for r in rows:
        lines.append(f"- {r['name']}: {r['description']}")
    return "\n".join(lines)


TOOL_USAGE_GUIDE_TEMPLATE = """\
【Agent 工具使用指南】

工具分为**原子工具**（本 Agent 可直接调用的函数）与**Skills**（流程/领域知识文档，按需 read_skill 拉取正文）。

- 原子工具按类别划分：task / query / mood / system；本 Agent 在 tools 参数中会暴露其**需要**的工具子集。
- 你不需要记住每个工具的详细用法——所有详细流程、边界条件、参数约束都写在 Skills 里。
- 处理陌生场景或不确定边界时：
  1. 先调用 list_skills 查看当前可用的 skills 索引；
  2. 对感兴趣的 skill，调用 read_skill(skill_name) 拉取完整正文；
  3. 如正文里提到 `references/xxx.md`，调用 read_skill_file(skill_name, file) 进一步阅读。
- 在调用需要 `ui_hint` 的工具时，填一段 ≤12 字符的极短"正在……"提示，用于前端展示；不确定则留空。

{skills_index}
"""


def build_tool_usage_guide() -> str:
    block = _skills_index_block()
    return TOOL_USAGE_GUIDE_TEMPLATE.format(skills_index=block or "（当前无已注册 skills，使用原子工具的默认 description 即可。）")


DECISION_ROUND_SUFFIX = (
    "本轮请仅判断是否需要调用工具并生成 tool_calls；可调用工具，不要输出对话文本。"
    "若不需要任何工具，返回空内容即可。"
    "进入正式对话前需调用 update_npc_mood（可与本轮其它工具并列）；同一用户消息内重复调用时仅最后一次有效。"
)

GENERATION_ROUND_SUFFIX = "请根据以上信息，以 NPC 身份生成对话回复。必须生成对话内容，不得输出空值。"


def build_layer1() -> str:
    return (
        "【世界观背景概要】\n"
        f"{WORLD_BACKGROUND}\n\n"
        f"{DIALOGUE_FORMAT_RULES}\n"
        f"{build_tool_usage_guide()}"
    )


def format_npc_role_tagline(
    *,
    npc_name: str,
    sex: str = "",
    faction: str = "",
    titles: list[str] | None = None,
) -> str:
    """与 Layer 2 开头一致的角色扮演首句，供 Layer 2 与 system 末尾重申共用。"""
    sex_desc = f"（性别：{sex}）" if sex else ""
    faction_desc = f"（阵营：{faction}）" if faction else ""
    titles_desc = f"（身份或称呼：{'、'.join(titles)}）" if titles else ""
    return (
        f"你现在扮演游戏角色「{npc_name}」{sex_desc}{faction_desc}{titles_desc}。"
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
    shop_reward_types: Optional[list[str]] = None,
    has_challenge: bool = False,
    player_can_challenge: Optional[bool] = None,
) -> str:
    emotions_list = emotions or ["普通"]
    emotions_str = "、".join(emotions_list)
    shop_reward_types = shop_reward_types or []

    shop_constraint = (
        "你可以将自己商店的物品作为任务奖励（物品等级需匹配玩家进度）。"
        f"当前商店可覆盖的奖励类型包括：[{('、'.join(shop_reward_types) if shop_reward_types else '未知')}]。"
        "在 reward_types 中可选这些类型；如果玩家索要的物品类型在候选列表中没有（表示你店里没有/不适配），你应回绝玩家的索要请求或推荐其向对应的商人索取，而非同意，因为你没有这类物品。"
        if has_shop
        else "你不经营商店，奖励以金币、经验值、K点、技能点、药剂、弹夹、材料等通用物资为主。"
        "在 reward_types 中不可选武器/防具/插件。如果玩家向你索要装备，请推荐其向售卖物品的角色提出请求，而非同意，因为你无法满足玩家的需求。"
    )

    if not has_challenge:
        challenge_hint = "- 你没有切磋关卡，不可发布'切磋'类型的任务。\n"
    else:
        # 三态提示：
        # - NPC确实有切磋关卡，但玩家暂时实力不足：不可发布'切磋'任务
        # - NPC有切磋关卡且玩家满足挑战：可以发布'切磋'任务
        # - player_can_challenge 未提供时：回退为“可以发布”的默认语义
        if player_can_challenge is False:
            challenge_hint = (
                "- 你拥有可用的切磋关卡，但是，玩家当前的实力还暂时不能挑战你，所以你不可发布'切磋'类型的任务。\n"
            )
        else:
            challenge_hint = "- 你拥有可用的切磋关卡，可以发布'切磋'类型的任务。\n"

    return (
        f"{format_npc_role_tagline(npc_name=npc_name, sex=sex, faction=faction, titles=titles)}\n"
        f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。\n"
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，"
        "用简体中文回答玩家本次的发言。\n"
        "非特殊要求下，每次对话长度不必太长。不要自己脑补不存在的设定。\n\n"
        f"【任务发布约束（详情见 skill: task-publishing）】\n"
        f"你作为「{npc_name}」（{faction or '未知阵营'}），在决定是否发布任务时：\n"
        "- 只发布符合你身份和能力范围的任务；\n"
        "- 即使玩家进度较高，你也不应发布超出你角色定位的高难度任务；\n"
        "- 在你的身份允许的情况下，可以优先发布最符合玩家当前进度的任务，其次是低于玩家进度的任务；\n"
        "- 发布任务的动机应自然融入对话，不要生硬地突然提出任务；\n"
        "- 只在对话氛围合适时才考虑发布任务，不是每次对话都需要任务；\n"
        "- 如果你和玩家关系不好/很不熟或你的身份不适合给玩家发布任务，则不要发布任务，并拒绝玩家的发布任务请求；\n"
        f"- {shop_constraint}\n"
        f"{challenge_hint}"
        "（完整的任务类型说明、奖励预算规则、协商流程、对话格式约束等，请按需调用 read_skill(skill_name=\"task-publishing\") / read_skill(\"task-bargaining\") 获取。）\n"
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
            "玩家可能会接受、拒绝、讨价还价或要求变更任务类型/大幅调整。"
            "具体分支处理见 skill: task-bargaining。"
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
    shop_reward_types: Optional[list[str]] = None,
    has_challenge: bool = False,
    player_can_challenge: Optional[bool] = None,
    same_faction_npcs: str = "",
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    mentioned_npcs_str: str = "",
    pending_draft_summary: str = "",
) -> str:
    """组装 Layer 1（固定）+ Layer 2（NPC）+ Layer 3（会话）为 system prompt；固定层在前以利于前缀缓存。"""
    layer1 = build_layer1()
    layer2 = build_layer2(
        npc_name=npc_name,
        sex=sex,
        faction=faction,
        titles=titles,
        emotions=emotions,
        has_shop=has_shop,
        shop_reward_types=shop_reward_types,
        has_challenge=has_challenge,
        player_can_challenge=player_can_challenge,
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

    tagline = format_npc_role_tagline(
        npc_name=npc_name, sex=sex, faction=faction, titles=titles
    )
    core = f"{layer1}\n\n{layer2}\n\n{layer3}"
    return f"{core}\n\n再次强调：{tagline}"


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


# 兼容旧 import：保留 TOOL_USAGE_GUIDE 名称（值现在是动态的简化版）
TOOL_USAGE_GUIDE = build_tool_usage_guide()
