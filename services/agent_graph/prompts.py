"""
分层 Prompt 模板（2026.04 路线三 · 与 OpenAI/Anthropic 常见实践对齐的修订版）。

## 总原则

1. **system** 里只放**跨轮长期稳定**、且**与「当前这一局对话内容」无强绑定**的约束：
   - 全局固定世界观/规则骨架（尽量少）
   - 单个 NPC 的静态设定（同 NPC 在任意会话里不变的部分）
2. **user** 里放**每轮或同一轮多 Agent 会共享、或会随工具调用递增**的上下文：
   - RAG + 关键词/实体检索块（每轮变；同一轮内多 Agent 共享）
   - 聊天历史 + 早期摘要（每轮变；**放在 user 侧**，避免被误认为「系统级永久设定」）
   - 会话态（好感、进度、待确认草案摘要等）；**同阵营 NPC 列表**对固定 NPC 为确定性结果，放在 **system（Layer2）**
   - 工具执行结果（同轮内递增追加）
   - 各 Agent 的职能说明、skills 索引、工具使用方式（放在 user 的靠前或靠后由你们产品决定，但**不应冒充 system 永久设定**）
3. **tools** 在 API 层仍是 `tools=...` 独立参数，不塞进 `system` 字符串；如某模型/兼容层
   只支持单条 user，可把 tools schema 的**文字摘要**复制进 user 作为兼容（本仓库仍以 API tools 为主）。

> 与「最大前缀缓存」不冲突：缓存命中主要看**字节级前缀是否一致**。
> 将「多轮才变一次」的块放在前，「每轮都变」的块放后，依然能整段命中静态前缀。

## 本文件对外主入口

- `build_static_system(...)`：写入 `state["_prompt_base"]`（L1 + L2（含同阵营角色表）+ tagline）
- `build_user_shared_core(...)`：写入 `state["_user_shared"]`（RAG → 对话中提到的其他角色设定 → 会话态 + 历史 + 草案摘要）
  立绘文描与上一轮情绪提示仍由 `call_llm*(..., image_description=, emotion_hint=)` **追加在 user 文本末尾**，避免重复、也与「仅传图像」解耦
- `compose_agent_user_prompt(...)`：在 worker 入口用「shared + agent tail + 当前玩家话」拼出本轮完整 user
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
    same_faction_npcs: str = "",
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

    same_faction_block = (same_faction_npcs or "").strip()
    if same_faction_block:
        same_faction_block = same_faction_block + "\n\n"

    return (
        f"{format_npc_role_tagline(npc_name=npc_name, sex=sex, faction=faction, titles=titles)}\n"
        f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。\n"
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，"
        "用简体中文回答玩家本次的发言。\n"
        "非特殊要求下，每次对话长度不必太长。不要自己脑补不存在的设定。\n\n"
        f"{same_faction_block}"
        "【任务发布硬约束（详细流程见 skill: task-publishing / task-bargaining，通过 read_skill 按需加载）】\n"
        f"你作为「{npc_name}」（{faction or '未知阵营'}）："
        "只发布符合你身份和能力范围的任务；即使玩家进度较高，也不应发布超出你角色定位的高难度任务；"
        "发布动机要自然融入对话；"
        "如果你和玩家关系不好/很不熟或你的身份不适合给玩家发布任务，则不要发布任务，并拒绝玩家的发布任务请求；"
        f"{shop_constraint}"
        f"{challenge_hint}"
    )


def build_user_session_state_block(
    *,
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    pending_draft_summary: str = "",
) -> str:
    """会话内可变状态 + 待确认草案摘要（不含同阵营表——同阵营在 static Layer2）。"""
    parts: list[str] = []

    if player_identity:
        parts.append(f"玩家的身份是：{player_identity}")

    if progress_stage_desc:
        parts.append(progress_stage_desc)

    parts.append(
        f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。"
    )

    if pending_draft_summary:
        parts.append(
            "【待确认的任务草案】\n"
            f"{pending_draft_summary}\n"
            "玩家可能会接受、拒绝、讨价还价或要求变更任务类型/大幅调整。"
            "具体分支处理见 skill: task-bargaining。"
        )

    return "\n".join(parts)


def _rag_block(*, retrieved_context: str) -> str:
    if not (retrieved_context or "").strip():
        return ""
    return (
        "下面是可能与你相关的检索设定和你的过往台词片段"
        "（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
        f"{retrieved_context}"
    )


def _history_block(*, history_str: str) -> str:
    return (history_str or "").strip()


def build_user_shared_core(
    *,
    retrieved_context: str = "",
    history_str: str = "",
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    mentioned_npcs_str: str = "",
    pending_draft_summary: str = "",
) -> str:
    """同一轮内所有 Agent 共享的 user 侧上下文（**不含**玩家当轮发言句）。

    典型顺序：RAG →（对话触发的）其他角色设定摘录 → 会话态 → 历史。
    """
    session_block = build_user_session_state_block(
        player_identity=player_identity,
        progress_stage_desc=progress_stage_desc,
        favorability=favorability,
        relationship_level=relationship_level,
        pending_draft_summary=pending_draft_summary,
    )
    parts: list[str] = []
    rag = _rag_block(retrieved_context=retrieved_context)
    if rag:
        parts.append(rag)
    men = (mentioned_npcs_str or "").strip()
    if men:
        parts.append(men)
    if session_block:
        parts.append(session_block)
    hist = _history_block(history_str=history_str)
    if hist:
        parts.append(hist)
    return "\n\n".join(parts)


build_user_shared_prefix = build_user_shared_core


def build_static_system(
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
) -> str:
    """Layer1 + Layer2（含同阵营角色表）+ tagline。写入 ``state['_prompt_base']``。"""
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
        same_faction_npcs=same_faction_npcs,
    )
    tagline = format_npc_role_tagline(
        npc_name=npc_name, sex=sex, faction=faction, titles=titles
    )
    return f"{layer1}\n\n{layer2}\n\n再次强调：{tagline}"


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
    # 以下历史参数为兼容旧调用保留，**不再进入 system**（请改用 build_user_shared_core）
    same_faction_npcs: str = "",
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    mentioned_npcs_str: str = "",
    pending_draft_summary: str = "",
    history_str: str = "",
) -> str:
    """兼容名：≈ ``build_static_system``；其中 **同阵营表** 会进入 system，其余旧参数仍忽略。"""
    _ = (
        player_identity,
        progress_stage_desc,
        favorability,
        relationship_level,
        mentioned_npcs_str,
        pending_draft_summary,
        history_str,
    )
    return build_static_system(
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
    )


# ---------------------------------------------------------------------------
# Per-agent tails（放在 **user 侧**；多 agent 下 static system 可最大化缓存）
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
        "task": "你当前是 TaskAgent：负责任务流水线（prepare_task_context / draft_agent_task / update_task_draft / confirm_agent_task / cancel_agent_task），可辅助查关卡/物品。你只负责调用工具，不要写对白。",
    }[agent]
    return (
        f"【Agent 角色】{focus}\n\n"
        "【工具使用纲领】\n"
        "- 原子工具的参数 schema 由 tools 字段提供。**只使用 schema 里声明的字段**。\n"
        "- 遇到不熟悉场景时，先 list_skills 看索引，再 read_skill(name) 拉具体流程，必要时 read_skill_file 读 references。\n"
        "- 需要 `ui_hint` 的工具：填一句 ≤12 字的极短『正在……』提示；不确定则留空。\n"
        f"{skills_block}\n"
    )


def _dialogue_tail() -> str:
    # 生成约束句只保留在 ``GENERATION_ROUND_SUFFIX``（nodes 里拼接），避免与生成轮后缀重复。
    return f"{DIALOGUE_FORMAT_RULES}\n"


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
    """兼容旧名：现仅返回 **static system**（= ``build_static_system``）。agent 与各 tail 在 user 侧拼接。"""
    _ = (
        agent,
        player_identity,
        progress_stage_desc,
        favorability,
        relationship_level,
        mentioned_npcs_str,
        pending_draft_summary,
        history_str,
    )
    return build_static_system(
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
    )


def build_in_turn_history_appendix(
    *,
    npc_name: str,
    interim_reply: str = "",
    streamed_content_suffix: str = "",
) -> str:
    """构造「本轮已推送给玩家的可见正文」附加块。

    - **interim_reply**：supervisor 短句，形如「{speaker}：…」之上文。
    - **streamed_content_suffix**：与前端已推送 content 一致的片段（如 ``{任务发布成功}``、
      ``{任务草案拟定完成}`` 等，来自 ``_system_prefix_text`` 的快照），拼在 interim **之后**，
      便于模型与玩家所见同步。

    应放在 **「玩家：…」当轮原话之后**；二者皆空时返回空串。
    """
    s = (interim_reply or "").strip()
    extra = (streamed_content_suffix or "").strip()
    if not s and not extra:
        return ""
    speaker = (npc_name or "该角色").strip() or "该角色"
    parts: list[str] = []
    if s:
        parts.append(
            "【本轮新增对话（已发送给玩家，不要重复发送）】\n"
            f"{speaker}：{s}"
        )
    if extra:
        if s:
            parts.append(
                # "以下片段已作为正文推送给玩家（与前端 content 流一致，含花括号系统提示）：\n"
                f"{extra}"
            )
        else:
            parts.append(
                # "【已推送给玩家的正文片段（与前端 content 流一致）】\n"
                f"{extra}"
            )
    body = "\n\n".join(parts)
    return (
        body
        + "\n\n请在此之后继续生成「"
        + speaker
        + "」的回复。"
    )


def build_user_prompt(
    *,
    retrieved_context: str = "",
    history_str: str = "",
    user_query: str = "",
    same_faction_npcs: str = "",
    player_identity: str = "",
    progress_stage_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    mentioned_npcs_str: str = "",
    pending_draft_summary: str = "",
) -> str:
    """兼容单 agent 路径：拼出 ``_user_shared``（**不含**玩家当轮原话，避免重复）。"""
    _ = (user_query, same_faction_npcs)
    return build_user_shared_core(
        retrieved_context=retrieved_context,
        history_str=history_str,
        player_identity=player_identity,
        progress_stage_desc=progress_stage_desc,
        favorability=favorability,
        relationship_level=relationship_level,
        mentioned_npcs_str=mentioned_npcs_str,
        pending_draft_summary=pending_draft_summary,
    )


def format_player_utterance(*, user_query: str) -> str:
    q = (user_query or "").strip()
    if not q:
        return ""
    return f"玩家：{q}"


def compose_agent_user(
    *blocks: str,
) -> str:
    """将若干 prompt 段用空行拼合，自动跳过空段。"""
    return "\n\n".join((b or "").strip() for b in blocks if (b or "").strip())


# 兼容旧 import（已废弃）：保留名字但内容空壳化，避免外部误用。
TOOL_USAGE_GUIDE = "（已拆分到 build_agent_tail；请勿再直接引用此常量。）"
