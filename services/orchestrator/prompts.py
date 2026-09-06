"""聊天主 Agent 的分层 prompt（对应 docs/v3-developer/01 §6）。

吸收旧 services/agent_graph/prompts.py 的有效分层资产（前缀缓存对齐设计保留）：
- system（静态，跨轮稳定）：L1 世界观 + L2 扮演约束（商店/切磋/同阵营表）+ appearance；
- user shared core（每轮变、同一轮多调用共享）：RAG 块 → 提及 NPC 块 → 会话状态 → 历史；
- user tail：meta 行协议（02 §4.3）+ 输出格式规则 + 玩家当轮发言。

删除：supervisor/worker 决策层 prompt、_TRUNCATE hacks。
汇合调用 #2 不带 meta 协议（meta 每回合只发一次），改为「情绪回显 + 直接说台词」。
"""

from __future__ import annotations

import json
from typing import Optional

from services.game_progress import StageConfig
from services.llm import meta_prompt_block
from services.npc.manager import NPCState

# ---------------------------------------------------------------------------
# L1：世界观骨架（保留自 game_rag_service.WORLD_BACKGROUND 资产）
# ---------------------------------------------------------------------------

WORLD_BACKGROUND = (
    "近未来末世废土世界。曾经历科技爆发的“统合时代”，"
    "因基因编辑实用化和生化技术发展引发阶级大洗牌，社会矛盾激化。"
    "统合政府推行“天网计划”（全球监控网络）与"
    "“诺亚方舟计划”（超级战士研发）以维持统治，"
    "却因系统被病毒入侵、激光武器失控而在2012年触发“审判日”灾难："
    "卫星轰击、地层武器、天网机器人暴走、僵尸病毒、天灾，"
    "导致文明崩溃、环境荒漠化。\n"
    "灾后世界以“尘都”为剧情主要聚焦的区域（地理原型为成都），"
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

# 阵营别名（用于玩家输入模糊匹配，保留资产）
FACTION_ALIASES: dict[str, list[str]] = {
    "联合大学": ["大学"],
    "摇滚公园": ["摇滚"],
    "A兵团元老": ["A兵团"],
    "A兵团": ["A兵团元老"],
}

SKIP_FACTION_SAME_CAMP = "闲杂人等"
PC_CHAR_PLACEHOLDER = "$PC_CHAR"


# ---------------------------------------------------------------------------
# 输出格式规则（保留资产：dialogue format rules）
# ---------------------------------------------------------------------------

DIALOGUE_FORMAT_RULES = """\
【对话输出规则】
1. 第一行是 meta JSON（见上方格式），其余全部是角色台词本身。
2. 非必要时不出现动作描写；若需要肢体动作/神态/环境描写，必须且只能使用全角粗括号【】包裹；台词直接输出，不加引号；严禁用半角括号 () 或星号 * 描述动作。
3. 你输出的动作若涉及人称，一律单独起一行，采用第三人称视角：用角色名指代你自己，用「你」指代玩家。
4. 全角粗括号【】仅可用于动作/神态/环境描写，其它内容不得放入。
5. 用简体中文、陈述语气；不要输出 JSON（第一行除外）、不要提及工具、系统或立绘。
"""


def format_npc_role_tagline(
    *, npc_name: str, sex: str = "", faction: str = "", titles: Optional[list[str]] = None,
) -> str:
    sex_desc = f"（性别：{sex}）" if sex else ""
    faction_desc = f"（阵营：{faction}）" if faction else ""
    titles_desc = f"（身份或称呼：{'、'.join(titles)}）" if titles else ""
    return f"你现在扮演游戏角色「{npc_name}」{sex_desc}{faction_desc}{titles_desc}。"


def build_appearance_block(state: NPCState) -> str:
    """appearance 形象描述块（07）：有则恒入 prompt，与是否发图无关。"""
    if not state.appearance:
        return ""
    return f"【你的形象】\n{state.appearance}"


def build_shop_constraint(has_shop: bool, shop_reward_types: Optional[list[str]] = None) -> str:
    shop_reward_types = shop_reward_types or []
    if has_shop:
        return (
            "你可以将自己商店的物品作为任务奖励（物品等级需匹配玩家进度）。"
            f"当前商店可覆盖的奖励类型包括：[{('、'.join(shop_reward_types) if shop_reward_types else '未知')}]。"
            "如果玩家索要的物品类型在你店里没有/不适配，你应回绝或推荐其向对应的商人索取，而非同意。"
        )
    return (
        "你不经营商店，奖励以金币、经验值、K点、技能点、药剂、弹夹、材料等通用物资为主。"
        "如果玩家向你索要装备，请推荐其向售卖物品的角色提出请求，而非同意。"
    )


def build_challenge_hint(has_challenge: bool, player_can_challenge: Optional[bool] = None) -> str:
    if not has_challenge:
        return "- 你没有切磋关卡，不可发布「切磋」类型的任务。"
    if player_can_challenge is False:
        return "- 你拥有可用的切磋关卡，但玩家当前的实力还暂时不能挑战你，不可发布「切磋」类型的任务。"
    return "- 你拥有可用的切磋关卡，可以发布「切磋」类型的任务。"


def build_static_system(
    *,
    npc_name: str,
    state: NPCState,
    same_faction_npcs: str = "",
    has_shop: bool = False,
    shop_reward_types: Optional[list[str]] = None,
) -> str:
    """L1 世界观 + L2 扮演约束 + 同阵营表 + appearance（静态，前缀缓存命中区）。"""
    emotions_str = "、".join(state.emotions or ["普通"])
    tagline = format_npc_role_tagline(
        npc_name=npc_name, sex=state.sex or "", faction=state.faction or "", titles=state.titles,
    )
    parts = [
        f"【世界观背景概要】\n{WORLD_BACKGROUND}",
        tagline,
        f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。",
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，用简体中文回答玩家本次的发言。",
        "非特殊要求下，每次对话长度不必太长。不要自己脑补不存在的设定。",
    ]
    same = (same_faction_npcs or "").strip()
    if same:
        parts.append(f"【同阵营角色】\n{same}")
    parts.append("【任务发布硬约束（详细流程由后台任务系统处理，你只负责给出方向）】")
    parts.append(
        f"你作为「{npc_name}」（{state.faction or '未知阵营'}）："
        "只发布符合你身份和能力范围的任务；即使玩家进度较高，也不应发布超出你角色定位的高难度任务；"
        "如果你和玩家关系不好/很不熟或你的身份不适合给玩家发布任务，则不要发布任务并拒绝玩家的请求。"
    )
    parts.append(build_shop_constraint(has_shop, shop_reward_types))
    parts.append(build_challenge_hint(bool(state.challenge)))
    appearance = build_appearance_block(state)
    if appearance:
        parts.append(appearance)
    parts.append(f"再次强调：{tagline}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# user shared core
# ---------------------------------------------------------------------------

def _rag_block(retrieved_context: str) -> str:
    if not (retrieved_context or "").strip():
        return ""
    return (
        "下面是可能与你相关的检索设定和你的过往台词片段"
        "（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
        f"{retrieved_context}"
    )


def build_session_state_block(
    *,
    player_identity: str = "",
    progress_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    pending_draft_summary: str = "",
) -> str:
    parts: list[str] = []
    if player_identity:
        parts.append(f"玩家的身份是：{player_identity}")
    if progress_desc:
        parts.append(progress_desc)
    parts.append(f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。")
    if pending_draft_summary:
        parts.append(
            "【待确认的任务草案（已拟定、尚未发布）】\n"
            f"{pending_draft_summary}\n"
            "玩家可能会接受、拒绝、讨价还价或岔开话题。"
        )
    return "\n".join(parts)


def build_user_shared_core(
    *,
    retrieved_context: str = "",
    mentioned_npcs_str: str = "",
    history_str: str = "",
    player_identity: str = "",
    progress_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    pending_draft_summary: str = "",
) -> str:
    """同一轮多调用共享的 user 侧上下文（不含玩家当轮发言与尾部指令）。"""
    parts: list[str] = []
    rag = _rag_block(retrieved_context)
    if rag:
        parts.append(rag)
    men = (mentioned_npcs_str or "").strip()
    if men:
        parts.append(men)
    session_block = build_session_state_block(
        player_identity=player_identity,
        progress_desc=progress_desc,
        favorability=favorability,
        relationship_level=relationship_level,
        pending_draft_summary=pending_draft_summary,
    )
    if session_block:
        parts.append(session_block)
    hist = (history_str or "").strip()
    if hist:
        parts.append(hist)
    return "\n\n".join(parts)


def build_chat_tail(
    *,
    npc_emotions: list[str],
    user_query: str,
    pending_draft: bool = False,
) -> str:
    """user tail：meta 协议（意图路由，双分支）+ prepare 工具指南 + 输出规则 + 玩家当轮发言。"""
    parts = [
        meta_prompt_block(npc_emotions, has_pending_draft=pending_draft),
    ]
    if not pending_draft:
        parts.append(PREPARE_TOOL_GUIDE)
    parts.append(DIALOGUE_FORMAT_RULES)
    parts.append(f"玩家：{user_query}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# prepare_task_context 工具指南（仅无草案时注入；有草案时走 confirm/cancel/update 路由）
# ---------------------------------------------------------------------------

PREPARE_TOOL_GUIDE = """\
【prepare_task_context 工具使用说明（仅在你决定委派 task_draft 时调用）】
你有一个工具 prepare_task_context：筛选任务候选集（关卡/物品/奖励与预算规则）。规则：
1. 先输出 1~2 句过渡话回应玩家，**然后**调用本工具；禁止只调用工具不说话。调用后本轮正文到此为止。
2. task_type 合法值与选择原则（**必须严格按此选择，选错会被打回**）：
   - 问候：让玩家来见你一面/打招呼（无实质劳动）。传话：让玩家给你带话/传信。
   - 通关 / 清理 / 挑战：让玩家去打怪或清场（通关=打副本关卡；清理=清理区域怪物；挑战=高难度战斗）。
   - 切磋：与玩家切磋（仅当你有切磋关卡时可选）。
   - 资源收集：让玩家把物品**带回来交给你**（你缺物资时用）。
   - 装备缴纳：让玩家上缴装备给你。物品持有：让玩家持有/保管某物品。
   - 特殊物品获取：让玩家去获取某件特定物品（不算上缴，拿到即可）。
   - 通关并收集 / 通关并持有：打完关还要收集/持有物品。
   - **核心原则：玩家想要资源/装备时，禁止选「资源收集/装备缴纳」让他交资源（他想要的是得到）；
     你缺什么才让他「收集/缴纳」什么。task_type 是玩家要做的事，reward_types 是玩家得到的东西，不要混淆。**
3. reward_types：{"regular": [金币/经验值], "optional": [其他奖励类型]}。经验值仅挑战类可大量给。
4. requirement_keywords：把玩家提到的物品/关卡/区域关键词填进去（让相关候选排前）；没有就不填。
5. 候选池与奖励预算由工具返回，交给后台任务系统处理——你的台词里不要写任何具体任务内容或数字。
"""


# ---------------------------------------------------------------------------
# 汇合调用 #2（不带 meta 协议；情绪回显）
# ---------------------------------------------------------------------------

def build_merge_user_prompt(
    *,
    npc_name: str,
    player_query: str,
    spoken_text: str,
    emotion: str,
    result_block: str,
    instruction: str,
) -> str:
    """汇合调用 #2 的 user 消息：已说正文 + 结果 + 指令（01 §6.4 情绪回显）。"""
    parts = [f"玩家刚才说：{player_query}"]
    if (spoken_text or "").strip():
        parts.append(
            "【你本轮已经说过的话（已推送给玩家，衔接它继续说，不要重复）】\n"
            f"{npc_name}：{spoken_text.strip()}"
        )
    parts.append(
        f"【本轮你的情绪】{emotion}（保持一致，不要改变语气基调）。"
        "本轮不要再输出第一行 meta JSON，直接输出台词本身。"
    )
    parts.append(result_block)
    parts.append(f"【现在】{instruction}")
    return "\n\n".join(parts)


TASK_MERGE_INSTRUCTION = (
    "后台任务系统已拟好委托草案（结果见上）。请以你的口吻向玩家详细说明这份委托："
    "任务内容、要求、奖励都讲清楚，结尾必须问玩家「你接还是不接？」。"
    "若结果中有「偏离说明」，先用一句话自然解释为何与最初想法不同。"
    "不要说任何系统词汇（草案/后台/系统），就当是你自己斟酌后的安排。"
)

TASK_UPDATE_MERGE_INSTRUCTION = (
    "委托草案已按玩家的新条件调整完毕（结果见上）。请重新说明调整后的方案"
    "（内容/要求/奖励），结尾再次询问玩家是否接受。不要提系统或后台。"
)

SEARCH_MERGE_INSTRUCTION = (
    "请以你的口吻把查到的内容讲给玩家（结果见上，出处括号可省略）。"
    "若结果为空或未找到，就自然带过（如「这我也不太清楚」），不要编造设定。"
)

TASK_FAIL_INSTRUCTION = (
    "后台任务系统这次没能拟出合适的委托（原因见上）。请以你的口吻自然收场"
    "（如「今天手头的事都派完了，改天吧」），不要提任何系统/错误词汇，"
    "也不要向玩家承诺确定时间。"
)


# ---------------------------------------------------------------------------
# confirm 发布文本生成（与聊天 Agent 同源前缀的单次调用；无硬校验，宽松归一化）
# ---------------------------------------------------------------------------

CONFIRM_ARGS_SYSTEM = (
    "你正在以当前扮演角色的身份，为自己刚与玩家谈妥的委托撰写发布文书。"
    "严格按角色口吻撰写。输出一个 JSON 对象（不要代码块、不要解释）。字段："
    'title（任务标题，简洁）、description（任务说明，与任务内容一致）、'
    "get_dialogue（接取对话数组，1~3 条：NPC 向玩家发布任务，可穿插玩家 $PC 的回应，"
    '每条 {"name","title","emotion","text"}，text 为纯对话，动作神态用全角【】），'
    "finish_dialogue（完成对话数组，1~3 条：玩家向 NPC 交付，NPC 验收）。"
    "对话要自然衔接上文，不要与近期发言重复或冲突。"
)


def build_confirm_args_user_prompt(
    *,
    draft_summary: str,
    npc_name: str,
    spoken_text: str = "",
    emotion: str = "",
) -> str:
    """confirm 参数生成（复用聊天 Agent base_messages 前缀后追加的 user 消息）。"""
    parts = []
    if (spoken_text or "").strip():
        parts.append(
            f"【本轮你已对玩家说的话（情绪：{emotion or '普通'}，保持一致）】\n"
            f"{npc_name}：{spoken_text.strip()}"
        )
    parts.append(f"【待发布任务详情】\n{draft_summary}")
    parts.append(
        "现在以你的身份生成发布文本 JSON（title/description/get_dialogue/finish_dialogue）。"
        "对话中的玩家条目 name 用 $PC。"
    )
    return "\n\n".join(parts)


def parse_confirm_args_json(text: str) -> Optional[dict]:
    """宽容解析 confirm 参数 JSON（首个 { 到末个 }）。"""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def fallback_confirm_args(draft: dict, npc_name: str) -> dict:
    """confirm 参数生成的 LLM 失败兜底：模板文本，保证流程不中断。"""
    title = str(draft.get("title") or "委托")
    reqs = "、".join(
        f"{r.get('stage_name', '?')}({r.get('difficulty', '简单')})"
        for r in (draft.get("finish_requirements") or []) if isinstance(r, dict)
    )
    submit = "、".join(
        f"{it.get('item_name', '?')}x{it.get('count', 1)}"
        for it in (draft.get("finish_submit_items") or []) if isinstance(it, dict)
    )
    rewards = "、".join(
        f"{it.get('item_name', '?')}x{it.get('count', 1)}"
        for it in (draft.get("rewards") or []) if isinstance(it, dict)
    )
    desc_parts = []
    if reqs:
        desc_parts.append(f"通关：{reqs}")
    if submit:
        desc_parts.append(f"提交：{submit}")
    if rewards:
        desc_parts.append(f"报酬：{rewards}")
    description = "；".join(desc_parts) or title
    npc_title = npc_name
    get_text = f"这份委托就交给你了。{('要求：' + description) if description else ''}"
    finish_text = "干得漂亮，说好的报酬一分不少。"
    return {
        "title": title,
        "description": description,
        "get_dialogue": [{"name": npc_name, "title": npc_title, "emotion": "", "text": get_text}],
        "finish_dialogue": [{"name": npc_name, "title": npc_title, "emotion": "", "text": finish_text}],
    }


# ---------------------------------------------------------------------------
# 进度描述
# ---------------------------------------------------------------------------

def progress_stage_desc(stage: Optional[int], cfg: Optional[StageConfig]) -> str:
    if not stage or cfg is None:
        return "玩家当前主线进度未知或尚未开始。"
    level = f"等级区间 {cfg.min_level or '?'}-{cfg.max_level or '?'}" if cfg.min_level or cfg.max_level else "等级未知"
    return f"当前主线进度：阶段 {stage}（{cfg.name}，{level}）。"
