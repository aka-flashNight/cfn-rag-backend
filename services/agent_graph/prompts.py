"""
分层 Prompt 模板（对应文档 6.5.3，2026.04 路线三修订版）。

## 分层策略（最大化前缀缓存）

```
[Layer 1] 世界观骨架（所有 agent 完全一致）
[Layer 2] NPC 扮演层（所有 agent 完全一致；内含任务发布硬约束与 skills 索引指引）
[Layer 3] 会话状态层（所有 agent 完全一致）
[tagline] 再次强调"你是 <NPC>"（所有 agent 完全一致）
--------------------- ↑ prefix cache 命中区 ↑ ---------------------
[agent tail] 专属指令（每个 agent 各自不同）
    - supervisor: 路由规则 + interim_reply 规则 + mood 规则 + JSON schema
    - query:      知识检索 skills 索引 + 工具使用纲领 + 决策轮后缀
    - task:       任务流水线 skills 索引 + 工具使用纲领 + 决策轮后缀
    - dialogue:   对话格式规则 + 生成轮后缀（无工具）
```

关键要点：
- `DIALOGUE_FORMAT_RULES` **只在 dialogue agent 的 tail** 出现——其他 agent 不需要。
- `update_npc_mood` **不暴露**给任何 worker。情绪 / 好感度变化在 supervisor 路由时以 JSON 字段
  （`mood.emotion` / `mood.favorability_change`）一次性决定，等价于调用 `update_npc_mood`。
- 复杂流程（task-publishing / task-bargaining / knowledge-search / skill-discovery）的 **YAML frontmatter
  简介** 作为"索引"**静态**拼进对应 agent 的 tail；详细 body / references 由 `read_skill` / `read_skill_file`
  按需拉取，避免把完整 body 挤进每次 prompt。
- `build_prompt_base(...)` 返回的 Layer 1+2+3+tagline 字符串会写入 state 的 `_prompt_base`，
  每个 agent 只在此基础上拼自己的 tail，便于 LLM API 的 prefix cache 命中。
"""

from __future__ import annotations

from typing import Literal, Optional

from services.game_rag_service import WORLD_BACKGROUND
from services.skills import get_skill_registry


AgentName = Literal["supervisor", "query", "task", "dialogue", "default"]


# ---------------------------------------------------------------------------
# 跨 agent 共用（prefix cache 命中区）
# ---------------------------------------------------------------------------

def build_layer1() -> str:
    """Layer 1：世界观骨架。对所有 agent 完全一致，放最前面。"""
    return "【世界观背景概要】\n" f"{WORLD_BACKGROUND}"


def format_npc_role_tagline(
    *,
    npc_name: str,
    sex: str = "",
    faction: str = "",
    titles: list[str] | None = None,
) -> str:
    sex_desc = f"（性别：{sex}）" if sex else ""
    faction_desc = f"（阵营：{faction}）" if faction else ""
    titles_desc = f"（身份或称呼：{'、'.join(titles)}）" if titles else ""
    return (
        f"你现在扮演游戏角色「{npc_name}」{sex_desc}{faction_desc}{titles_desc}。"
    )


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
    """Layer 2：NPC 扮演。对所有 agent 一致——因为 supervisor 也要以 NPC 口吻做 interim_reply
    和 mood 判断，worker 也要用 NPC 身份调用工具。"""
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
        "【任务发布硬约束（详细流程见 skill: task-publishing / task-bargaining，通过 read_skill 按需加载）】\n"
        f"你作为「{npc_name}」（{faction or '未知阵营'}）："
        "只发布符合你身份和能力范围的任务；即使玩家进度较高，也不应发布超出你角色定位的高难度任务；"
        "发布动机要自然融入对话；"
        "如果你和玩家关系不好/很不熟或你的身份不适合给玩家发布任务，则不要发布任务，并拒绝玩家的发布任务请求；"
        f"{shop_constraint}"
        f"{challenge_hint}"
    )


def build_layer3(
    *,
    same_faction_npcs: str = "",
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    mentioned_npcs_str: str = "",
    pending_draft_summary: str = "",
    history_str: str = "",
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

    # 历史聊天记录对同一轮内的所有 agent 都是共用且基本不变的上下文，
    # 因此放入共享 prompt base 的靠下位置；这样 supervisor / task / query / dialogue
    # 都能看到同一份历史，同时最大化同轮内多次调用的前缀缓存命中。
    if history_str:
        parts.append(history_str.strip())

    return "\n".join(parts)


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


def build_prompt_base(
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
    history_str: str = "",
) -> str:
    """拼装 Layer 1+2+3+tagline——所有 agent 的 **共同前缀**，可被 LLM API 完整缓存。"""
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
        history_str=history_str,
    )
    tagline = format_npc_role_tagline(
        npc_name=npc_name, sex=sex, faction=faction, titles=titles
    )
    return f"{layer1}\n\n{layer2}\n\n{layer3}\n\n再次强调：{tagline}"


# ---------------------------------------------------------------------------
# Per-agent tails（prefix cache 命中区之外的部分）
# ---------------------------------------------------------------------------

# 每个 agent 应看到的 skill 白名单——用来过滤 skills 索引。
# 注意：这里只是**索引**（YAML frontmatter 里的 name+description），正文由 read_skill 按需拉取。
_AGENT_SKILL_WHITELIST: dict[str, frozenset[str]] = {
    "supervisor": frozenset(),  # supervisor 不需要直接看 skill 索引
    "query": frozenset({"knowledge-search", "skill-discovery"}),
    "task": frozenset({"task-publishing", "task-bargaining", "knowledge-search", "skill-discovery"}),
    "dialogue": frozenset(),   # dialogue 不调工具，不需要 skill 索引
    "default": frozenset(),    # 单 agent 兜底：全量
}


def _skills_index_block(agent: AgentName) -> str:
    registry = get_skill_registry()
    rows = registry.index()
    if not rows:
        return ""

    allowed = _AGENT_SKILL_WHITELIST.get(agent)
    if agent == "default" or allowed is None or not allowed:
        # 默认（兼容旧 single-agent）或 supervisor/dialogue：
        # supervisor/dialogue 返回空索引；default 返回全量
        if agent == "default":
            filtered = rows
        else:
            return ""
    else:
        filtered = [r for r in rows if r["name"] in allowed]
        if not filtered:
            return ""

    lines = ["【可用 Skills 索引（需要完整流程时用 read_skill(skill_name) 拉取）】"]
    for r in filtered:
        lines.append(f"- {r['name']}: {r['description']}")
    return "\n".join(lines)


DIALOGUE_FORMAT_RULES = """\
【对话输出规则（仅 dialogue agent 使用）】
1. 只输出角色对话文本，不要有 JSON 或结构化数据；本轮不会提供工具。
2. 你要说的情绪与好感度已由 supervisor 在路由阶段通过 mood 字段记录；请**与下方「本轮已决定的情绪」保持一致**，不要自己改写，也不要再要求调用工具上报。
3. 动作与台词格式：非必要时不出现动作描写；若需要肢体动作/神态/环境描写，必须且只能使用全角粗括号【】包裹；台词直接输出，不加引号；严禁用半角括号 () 或星号 * 描述动作。
   你输出的动作若涉及人称，一律单独起一行，采用第三人称视角：用角色名指代你自己，用「你」指代玩家。
4. 全角粗括号【】仅可用于动作/神态/环境描写，其它内容不得放入。
5. 若上轮 worker 的工具调用失败（例如任务发布失败），要以执行失败为前提继续对话。\
"""


def _supervisor_tail() -> str:
    """Supervisor 的 tail：路由规则 + interim_reply + mood 规则 + JSON schema。
    具体 schema 文案由 ``services/agents/supervisor.py`` 在运行时再拼一遍可变内容（允许情绪、pending_draft 状态等），
    这里只放与 NPC 实例无关的"通用"部分，能被 prefix cache 命中到 tail 的前半段。
    """
    return (
        "【Supervisor 专属指令（其它 agent 不适用）】\n"
        "- 本调用是**路由决策**：根据玩家消息选择下一跳 worker（query / task / dialogue / end）。\n"
        "- 路由到 dialogue 或 end 时，额外在同一 JSON 中给出 mood.emotion / mood.favorability_change，\n"
        "  这等价于一次 update_npc_mood 工具调用——**不要**另外产生 tool_calls。\n"
        "- 可选给出一段不超过 80 字、NPC 口吻的 interim_reply，用于让前端先显示"
        "「NPC 正在说一句短话」遮盖任务准备时长；仅在第一次 route=task 时考虑，否则留空。\n"
        "- 你**不**生成最终对白——最终对白由 dialogue worker 负责。你只能通过 interim_reply 发一句短话。\n"
        "- 具体 JSON schema 与字段约束见下方 user 消息。\n"
    )


def _worker_decision_tail(agent: AgentName) -> str:
    """Query / Task worker 的 tail：工具使用纲领 + skills 索引 + 决策轮后缀。"""
    skills_block = _skills_index_block(agent)
    focus = {
        "query": "你当前是 QueryAgent：仅负责知识检索（search_knowledge / search_stages / search_items）。不要发布任务、不要生成最终对白。",
        "task": "你当前是 TaskAgent：负责任务流水线（prepare_task_context / draft_agent_task / update_task_draft / confirm_agent_task / cancel_agent_task），可辅助查关卡/物品。不要自己写最终对白——对白交给 dialogue agent。",
    }[agent]
    return (
        f"【Agent 角色】{focus}\n\n"
        "【工具使用纲领】\n"
        "- 原子工具的参数 schema 由后端 tools 字段提供。**只使用 schema 里声明的字段**——严禁自行添加 `reason` / `title` / `favorability_delta` 等 schema 之外的字段。\n"
        "- 遇到不熟悉场景时，先 list_skills 看索引，再 read_skill(name) 拉具体流程，必要时 read_skill_file 读 references。\n"
        "- 需要 `ui_hint` 的工具：填一句 ≤12 字的极短『正在……』提示；不确定则留空。\n"
        "- 情绪与好感度由 supervisor 处理，本 agent 不应调用 update_npc_mood；本 agent 的 tools 中也不会出现它。\n\n"
        f"{skills_block}\n\n"
        "本轮请仅判断是否需要调用工具并生成 tool_calls；可调用工具，不要输出对话文本。若不需要任何工具，返回空内容即可。"
    )


def _dialogue_tail() -> str:
    return (
        f"{DIALOGUE_FORMAT_RULES}\n\n"
        "请根据以上信息，以 NPC 身份生成对话回复。必须生成对话内容，不得输出空值。"
    )


def build_agent_tail(agent: AgentName) -> str:
    """返回 agent 专属 tail（不含动态可变部分，如 allowed_emotions、pending_draft 状态等——那些由各 agent 节点自行追加）。"""
    if agent == "supervisor":
        return _supervisor_tail()
    if agent in ("query", "task"):
        return _worker_decision_tail(agent)
    if agent == "dialogue":
        return _dialogue_tail()
    # default（单 agent / 向后兼容）：拼所有 skills 索引 + 对话规则
    skills = _skills_index_block("default")
    return (
        f"{skills}\n\n{DIALOGUE_FORMAT_RULES}\n\n"
        "本轮可调用工具上报本次情绪（仅限兜底场景：agent_enabled=false 降级模式）；"
        "正常 Agent 模式下请把情绪放在决策轮处理。"
    )


# ---------------------------------------------------------------------------
# 节点内后缀（仅传给 user prompt 的最末，用于明确本轮行为）
# ---------------------------------------------------------------------------

DECISION_ROUND_SUFFIX = (
    "本轮请仅判断是否需要调用工具并生成 tool_calls；可调用工具，不要输出对话文本。"
    "若不需要任何工具，返回空内容即可。"
    "本 agent 的 tools 中**不含** update_npc_mood，情绪由 supervisor 处理，不要尝试调用它。"
)

GENERATION_ROUND_SUFFIX = (
    "请根据以上信息，以 NPC 身份生成对话回复。必须生成对话内容，不得输出空值。"
    "本轮不提供任何工具，情绪 / 好感度已由 supervisor 在路由阶段决定，请与之保持一致。"
)


# ---------------------------------------------------------------------------
# 对外统一入口（兼容旧调用）
# ---------------------------------------------------------------------------

def build_system_prompt(
    *,
    agent: AgentName = "default",
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
    history_str: str = "",
) -> str:
    """组装完整 system prompt = 共享 prefix + agent tail。"""
    base = build_prompt_base(
        npc_name=npc_name,
        sex=sex,
        faction=faction,
        titles=titles,
        emotions=emotions,
        has_shop=has_shop,
        shop_reward_types=shop_reward_types,
        has_challenge=has_challenge,
        player_can_challenge=player_can_challenge,
        same_faction_npcs=same_faction_npcs,
        player_identity=player_identity,
        progress_stage_desc=progress_stage_desc,
        favorability=favorability,
        relationship_level=relationship_level,
        mentioned_npcs_str=mentioned_npcs_str,
        pending_draft_summary=pending_draft_summary,
        history_str=history_str,
    )
    tail = build_agent_tail(agent)
    return f"{base}\n\n{tail}"


def build_in_turn_history_appendix(*, npc_name: str, interim_reply: str = "") -> str:
    """构造“本轮新增对话”附加块。

    用途：
    - 历史记录本体已经进入共享 `_prompt_base`；
    - supervisor 在本轮先说出的 interim_reply 属于“同一轮新增的一条 NPC 发言”，
      后续 task/query/dialogue agent 需要把它视为已经发生的上下文；
    - 这条附加块放在共享 base 之后、agent tail 之前，满足“绝大部分不变、底部只加一条”的结构。
    """
    s = (interim_reply or "").strip()
    if not s:
        return ""
    speaker = (npc_name or "该角色").strip() or "该角色"
    return (
        "【本轮新增对话（发生在当前用户消息之后，后续处理时必须视为既成事实）】\n"
        f"{speaker}：{s}"
    )


def build_user_prompt(
    *,
    retrieved_context: str = "",
    user_query: str = "",
    emotion_hint: str = "",
    image_description: str = "",
) -> str:
    prefix_parts: list[str] = []
    if image_description:
        prefix_parts.append(image_description + "。")
    if emotion_hint:
        prefix_parts.append(emotion_hint)
    prefix = "".join(prefix_parts)

    layer4 = build_layer4(
        retrieved_context=retrieved_context,
        history_str="",
        user_query=user_query,
    )
    if prefix:
        return f"{prefix}\n\n{layer4}"
    return layer4


# 兼容旧 import（已废弃）：保留名字但内容空壳化，避免外部误用。
TOOL_USAGE_GUIDE = "（已拆分到 build_agent_tail；请勿再直接引用此常量。）"
