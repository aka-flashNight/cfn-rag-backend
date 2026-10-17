"""聊天主 Agent 的分层 prompt（对应 docs/v3-developer/01 §6）。

吸收旧 services/agent_graph/prompts.py 的有效分层资产（前缀缓存对齐设计保留）：
- system（静态，跨轮稳定）：L1 世界观 + 输出规则 → L2 扮演约束（路由指南/商店/切磋/同阵营表）+ appearance；
- user shared core（每轮变、同一轮多调用共享）：RAG 块 → 提及 NPC 块 → 会话状态 → 历史；
- user tail：立绘说明 + 情绪连贯提示 + 玩家当轮发言。

v3 再次重构（路由工具化）：任务选型规则（原 PREPARE_TOOL_GUIDE）整体移交任务 Agent，
聊天主 Agent 只保留「该不该路由、路由给谁」的判断——具体判据写在路由工具的
description 里，本文件的 build_routing_guide 只讲跨工具的公共规则。
汇合调用 #2 不带 meta 协议（meta 每回合只发一次），改为「情绪回显 + 直接说台词」。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from services.game_progress import StageConfig
from services.llm import meta_prompt_block
from services.llm.meta import DEFAULT_EMOTION
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

# 本块属 system 层1，对全部 NPC、全部回合逐字恒定，所以不提路由：关闭 Agent 能力时
# 没有任何路由工具，提了就是悬空指代。「路由轮不写台词」的例外由层2 的
# build_routing_guide（只在 agent_enabled 时注入）承担。
DIALOGUE_FORMAT_RULES = """\
【对话输出规则】
1. 第一行是 meta JSON（格式见下方【输出格式】一节）；其后是角色台词本身。
2. 非必要时不出现动作描写；若需要肢体动作/神态/环境描写，必须且只能使用全角粗括号【】包裹；台词直接输出，不加引号；严禁用半角括号 () 或星号 * 描述动作。
3. 你输出的动作若涉及人称，一律单独起一行，采用第三人称视角：用角色名指代你自己，用「你」指代玩家。
4. 全角粗括号【】仅可用于动作/神态/环境描写，其它内容不得放入。
5. 用简体中文；不要输出 JSON（第一行除外）、不要提及工具、系统或立绘。
6. 说话的那一轮，第一行 JSON 之后**必须输出台词正文，不得输出空回复**；没有合适的长话也要以角色身份自然回应。
"""


# 说话方式（层1，跨 NPC 与跨回合逐字恒定，前缀缓存友好）。
# 治两类看起来相反、实则同源的毛病：过度省略（口语里本该说全的成分被砍掉、
# 人名地名自造简称）与过度啰嗦（日常闲聊也铺满段落）。同源在于模型按「写文章」
# 而不是「当面说话」出话，所以规则一律给可判定的口径——句式完整、称呼照设定、
# 口语词、字数看场合、不重复、不硬凹——不给「自然一点」这类空泛形容词。
SPEECH_STYLE_RULES = """\
【说话方式】
1. 你是在当面说话，不是在写文章。按口语的完整结构说：日常口语中该有的主语、宾语、量词和连接词都要在，不要为了短而砍成断句或电报体。
2. 称呼照设定来：人名、地名、组织、物品在设定与你的过往台词里怎么叫就怎么叫，不要自行截短，不要另造简称或外号。其他日常用词也不要自造简称，而是直接使用口语常用称呼。
3. 用词按口语：说「所以、不过、还有、有点、挺好」，不说「因此、然而、此外、较为、综上所述」；不堆成语，不堆砌用词，不用书面语的总结句收尾。
4. 字数看场合：日常打招呼、随口闲聊就一两句话说完，不展开、不分段、不罗列条目；只有确实需要讲清楚时才多说——介绍人或地方、回忆一段经历、说明委托的内容与报酬、解释原因、答复玩家的追问。人设话多也只在合理范围内话多，人设话少就更要不啰嗦。
5. 同一件事不要换个说法再说一遍：本轮已经说过的、上一轮已经叮嘱过的，都不必重复。
6. 不要为了显得有个性而硬凹：不加耍帅或装深沉的句子，不用夸张比喻堆气氛。在符合人设和当前角色语言习惯的前提下，说话像个正常人，不中二、不机械，不要有AI感。预期参考过往台词。
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


def build_shop_hint(has_shop: bool, shop_reward_types: Optional[list[str]] = None) -> str:
    """商店可行性提示（聊天主 Agent 侧）。

    v3 再次重构后这里只讲「你手里有什么货、别答应给不了的东西」——reward_types
    怎么填、哪些类型合法属于任务 Agent 的选型规则，见
    subagents.prompts.PREPARE_SELECTION_GUIDE。
    """
    shop_reward_types = shop_reward_types or []
    if has_shop:
        types = "、".join(shop_reward_types) if shop_reward_types else "未知"
        return (
            f"你经营商店，可以把自己店里的物品当作委托报酬，货品大致覆盖这些类型：[{types}]"
            "（物品等级要匹配玩家进度）。玩家索要的类型不在其中时，说明你这里没有，"
            "或建议他去找卖这类货的人；不要答应给不了的东西。"
        )
    return (
        "你不经营商店，报酬只能给金币、经验值、K点、技能点、药剂、弹夹、材料这类通用物资，"
        "给不出武器/防具/插件。仅当玩家明确索要这类东西时，才向他说明你没有，"
        "并建议他去找卖这类货的角色；不要无条件把玩家推给别的商人。"
    )


def build_challenge_hint(
    has_challenge: bool,
    player_can_challenge: Optional[bool] = None,
) -> str:
    """切磋提示三态（与任务 Agent 的切磋目标判定同源，见 context 装配）：
    无关卡 / 有关卡但玩家实力不足 / 双方满足。满足时弱化措辞，避免过度引导模型。

    聊天主 Agent 只需要知道「能不能跟玩家提切磋」；具体的切磋目标候选由任务 Agent
    在 prepare 阶段取。
    """
    if not has_challenge:
        return "- 你没有切磋关卡，别跟玩家提「跟我打一场」这类委托。"
    if player_can_challenge is False:
        return (
            "- 你拥有切磋关卡，但玩家当前的实力还暂时不能挑战你，"
            "别跟他提切磋委托。"
        )
    return "- 在对话情景合适时可以考虑提出跟玩家切磋一场。"


def build_routing_guide(*, has_pending_draft: bool) -> str:
    """路由公共规则（层2，按有无待确认草案二分支）。

    判据本体在各路由工具的 description 里——工具表就是模型的路由提示词，
    重复一份只会白占 token 并可能自相矛盾。本块只讲跨工具的公共规则：
    路由轮不说话、草案不等于已发布、后台结果交回来之后怎么接着说。
    """
    lines = [
        "【路由（把本轮交给后台系统办）】",
        "你有几个路由工具，各自说明里写清了什么时候用。公共规则：",
        "1. 调用路由工具的那一轮：第一行 meta JSON 之后**不要写任何台词**，只发工具调用。"
        "后台办完会把结果交回给你，那时你再开口——不要自己抢着说「我想想给你安排点什么」，"
        "也不要在这轮编造任何任务内容或数字。",
        "2. 一轮最多路由一件事。拿不准该不该路由时，就正常说话，不要路由。",
        "3. **草案只是拟定，不是已发布**。玩家点头之前禁止说「任务已发布」。",
    ]
    if has_pending_draft:
        lines.extend([
            "4. 现在有一份待确认的委托草案（内容见下方会话状态）。玩家对它表态时这样分派：",
            "   - 明确接受 → route_task_confirm（他已经点头了，不要再追问第二次）；",
            "   - 明确拒绝/放弃 → route_task_cancel；",
            "   - 讨价还价、要求小幅调整，而你愿意让步 → route_task_update；",
            "   - 要求彻底换一个任务方向 → route_task_draft（新草案会替换旧草案）；",
            "   - 你不愿意让步 → 不调用任何工具，直接以你的口吻拒绝，并问他到底接还是不接。",
            "5. 后台把调整后的方案交回来时，重新说明方案（内容/要求/报酬），并再次征询他是否接取。",
        ])
    else:
        lines.extend([
            "4. 玩家索要委托/工作/差事，或按你的身份和当下情景你决定主动派活 → route_task_draft；",
            "   需要查证设定、角色往事、关卡或物品信息才答得上来 → route_search。",
            "5. 后台把拟好的委托交回来时，向玩家讲清内容/要求/报酬，并征询他是否接取。",
        ])
    return "\n".join(lines)


def build_static_system(
    *,
    npc_name: str,
    state: NPCState,
    same_faction_npcs: str = "",
    scene_block: str = "",
    scene_supplement: str = "",
    has_shop: bool = False,
    shop_reward_types: Optional[list[str]] = None,
    player_can_challenge: Optional[bool] = None,
    has_pending_draft: bool = False,
    agent_enabled: bool = True,
    task_publish_disabled: bool = False,
) -> str:
    """system 分层（前缀缓存对齐，从稳定到易变）：
    层1 跨 NPC 一致（世界观、输出规则、说话方式）→ 层2 单 NPC 一致
    （扮演约束、meta 协议、场景块、appearance、发布硬约束、商店报酬）
    → 稀疏变动（切磋可行性：随玩家等级变）→ 每轮重装配（路由指南：随草案状态变；
    同阵营表与场景补充：随玩家这句话提到的角色变）。

    层2 内部也按这个次序排：把每轮会重装配的三块压到最后，前面那段才能在
    玩家提到别的角色、或草案出现/消失时继续命中前缀缓存。

    v3 再次重构后层1 不再含任务选型指南（那 1229 字整体移交任务 Agent），
    因此层1 对全部 NPC 与全部回合恒定，前缀缓存命中率只受层2 影响。

    agent_enabled=False（玩家关闭 Agent 能力）时，任务发布相关的全部约束块
    （路由指南、发布硬约束、商店报酬、切磋可行性）都不注入，meta 协议块也去掉
    「或者发一个路由工具调用」那半句——它们只在能派发委托时才有意义，留着只会
    白占 token 并诱导模型承诺做不到的事；改为注入 NO_AGENT_CONSTRAINT，让 NPC
    以人设口吻婉拒而不是空口答应；玩家坚持要活时，还要引导他去终端打开接取
    委托的权限（v2 同名提示的口径）。

    task_publish_disabled=True（无驻地 NPC 名单，临时措施）时同样不注入任务
    相关块，但换成 NO_TASK_PUBLISH_CONSTRAINT：以人设口吻直接婉拒，不引导
    玩家去改终端权限（玩家侧没有可开的闸）。
    """
    can_publish = agent_enabled and not task_publish_disabled
    emotions_str = "、".join(state.emotions or ["普通"])
    tagline = format_npc_role_tagline(
        npc_name=npc_name, sex=state.sex or "", faction=state.faction or "", titles=state.titles,
    )
    # ---- 层1：跨 NPC 一致 ----
    parts = [
        worldview_block(),
        DIALOGUE_FORMAT_RULES,
        SPEECH_STYLE_RULES,
    ]
    # ---- 层2：单 NPC 一致 ----
    parts.extend([
        tagline,
        f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。请选择其中最合适的一种作为你当前的情绪立绘。",
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，用简体中文回答玩家本次的发言。",
        "未处于任务委派、描述介绍、事件回忆等特殊语境的日常闲聊中，每次对话长度不必太长，但也不要因此而强行缩写语句或词语，而是保持正常聊天口语的自然流畅的前提下正常说话，不要啰嗦。",
        ANTI_HALLUCINATION_RULE,
        meta_prompt_block(state.emotions or ["普通"], agent_enabled=can_publish),
    ])
    scene = (scene_block or "").strip()
    if scene:
        parts.append(scene)
    appearance = build_appearance_block(state)
    if appearance:
        parts.append(appearance)
    if can_publish:
        parts.append("【任务发布硬约束（具体怎么拟由后台任务系统负责，你只决定要不要派、往哪个方向派）】")
        parts.append(
            f"你作为「{npc_name}」（{state.faction or '未知阵营'}）："
            "只发布符合你身份和能力范围的任务；即使玩家进度较高，也不应发布超出你角色定位的高难度任务；"
            "发布动机要自然融入对话；"
            "如果你和玩家关系不好/很不熟或你的身份不适合给玩家发布任务，"
            "则不要发布任务，并拒绝玩家的发布任务请求；"
        )
        parts.append(build_shop_hint(has_shop, shop_reward_types))
        # ---- 稀疏变动：随玩家等级变（升级跨过切磋门槛时才改写） ----
        parts.append(build_challenge_hint(
            bool(state.challenge), player_can_challenge,
        ))
        # ---- 每轮重装配：随草案状态变 ----
        parts.append(build_routing_guide(has_pending_draft=has_pending_draft))
    elif task_publish_disabled:
        parts.append(NO_TASK_PUBLISH_CONSTRAINT)
    else:
        parts.append(NO_AGENT_CONSTRAINT)
    # ---- 每轮重装配：随玩家这句话提到的角色变（提到谁就把谁从这两块里剔掉） ----
    same = (same_faction_npcs or "").strip()
    if same:
        parts.append(f"【同阵营角色】\n{same}")
    supplement = (scene_supplement or "").strip()
    if supplement:
        parts.append(supplement)
    parts.append(f"再次强调：{tagline}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# user shared core
# ---------------------------------------------------------------------------

# 检索为空时的兜底沿用 v2 口径：明确授权模型按自己的设定发挥，而不是整块留白——
# 留白会让模型退回通用套话，是伪人感的来源之一。
RAG_EMPTY_FALLBACK = "（当前没有检索到任何上下文，你可以根据自己的设定自由发挥，但要保持合理。）"


def rag_block(retrieved_context: str) -> str:
    """检索设定块（聊天主 Agent 与对话修改子 Agent 共用同一份文本）。"""
    body = (retrieved_context or "").strip() or RAG_EMPTY_FALLBACK
    return (
        "下面是可能与你相关的检索设定和你的过往台词片段"
        "（仅用于保持设定与说话风格，请不要逐字复读原文）：\n"
        f"{body}"
    )


def worldview_block() -> str:
    """世界观概要块（层1 第一条；对话修改子 Agent 用同一份，故抽成函数）。"""
    return f"【世界观背景概要】\n{WORLD_BACKGROUND}"


#: 脑补禁令（聊天主 Agent 层2 与对话修改子 Agent 共用同一份文本）
ANTI_HALLUCINATION_RULE = (
    "绝对不要自己脑补不存在的设定，一切没有提供给你的信息，你都不能有任何提及或暗示。"
    "无法把握的模糊地带可以略过或转移话题，不要自己乱加设定，以免出戏。例如，如果前文没有说"
    "某个人在什么地点，或者某个地点有什么东西，你不要擅自脑补，对话中绝对不能自己造信息。"
    "只允许依据准确已知信息来表述。"
)


def _save_player_name_line(save_info: Any) -> str:
    """绑定存档投影里的玩家名字行（B2-1）＋称呼口径；未绑定/投影缺名字＝空串。

    名字由游戏投影原样给出，网名/字符串这类不像名字的取值不能当称呼用，
    所以称呼规则跟名字同行给出。
    """
    if not isinstance(save_info, dict):
        return ""
    character = save_info.get("character")
    if not isinstance(character, dict):
        return ""
    name = str(character.get("name") or "").strip()
    if not name:
        return ""
    return (
        f"玩家的名字是「{name}」。"
        "如果玩家的名字看起来像正常的名字或佣兵代号，你可以在合适的情况下用这个来称呼玩家。"
        "如果名字像非正式的网名或字符，那么你不要用名字作为称呼，"
        "如果一定要称呼，可以用「佣兵」之类的身份来指代。"
    )


def _save_scene_line(save_info: Any) -> str:
    """绑定存档投影里的玩家场景行；未绑定/场景段缺失＝空串。

    战斗中给关卡名（stageName 非战斗时是残留值，须以 inCombat 门控）；
    非战斗给地图名——优先归一后的 label，缺标签回落帧标签（stageFlag）原文。
    """
    if not isinstance(save_info, dict):
        return ""
    scene = save_info.get("scene")
    if not isinstance(scene, dict):
        return ""
    stage_name = str(scene.get("stageName") or "").strip()
    if scene.get("inCombat") is True and stage_name:
        return f"玩家当前正在关卡「{stage_name}」中战斗。"
    label = str(scene.get("label") or "").strip() or str(scene.get("stageFlag") or "").strip()
    return f"玩家当前所在场景：{label}。" if label else ""


def build_session_state_block(
    *,
    player_identity: str = "",
    progress_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    pending_draft_summary: str = "",
    save_info: Any = None,
) -> str:
    parts: list[str] = []
    if player_identity:
        parts.append(f"玩家的身份是：{player_identity}")
    player_name_line = _save_player_name_line(save_info)
    if player_name_line:
        parts.append(player_name_line)
    scene_line = _save_scene_line(save_info)
    if scene_line:
        parts.append(scene_line)
    if progress_desc:
        parts.append(progress_desc)
    if pending_draft_summary:
        parts.append(
            "【待确认的任务草案（已拟定、尚未发布）】\n"
            f"{pending_draft_summary}\n"
            "玩家可能会接受、拒绝、讨价还价或岔开话题。"
        )
    # 好感度是这块里唯一几乎每轮都在变的数字，压在最后：它一变只废掉自己这一行，
    # 不会连带把上面的身份/场景/进度与草案块的前缀缓存一起打断。
    parts.append(f"你目前对玩家的好感度是 {favorability}（{relationship_level}）。")
    return "\n".join(parts)


#: 已完成委托块尾部的长期记忆更新指引（仅在长期记忆已有关键记录时拼接）
LTM_TASK_DONE_HINT = "长期记忆里若仍记着上述委托处于「已发布」，把它更新为「已完成」。"


def build_finished_task_block(raw_block: str, *, has_long_term_facts: bool) -> str:
    """已完成委托块（阶段三）：raw_block 由上下文装配直读磁盘拼好，这里只补一句
    长期记忆更新指引——仅在长期记忆已有关键记录时补，否则这句无可执行对象。"""
    if not raw_block:
        return ""
    if not has_long_term_facts:
        return raw_block
    return f"{raw_block}\n{LTM_TASK_DONE_HINT}"


def build_user_shared_core(
    *,
    retrieved_context: str = "",
    mentioned_npcs_str: str = "",
    summary_str: str = "",
    history_str: str = "",
    player_identity: str = "",
    progress_desc: str = "",
    favorability: int = 0,
    relationship_level: str = "陌生",
    relationship_hint: str = "",
    pending_draft_summary: str = "",
    save_info: Any = None,
    long_term_block: str = "",
    finished_task_block: str = "",
) -> str:
    """同一轮多调用共享的 user 侧上下文（不含玩家当轮发言）。

    排序按前缀缓存对齐（从稳定到易变）：已完成委托 → 长期记忆（跨窗口，改动稀疏）
    → 早期摘要（同为跨轮稀疏，紧跟长期记忆）→ 近期对话（跨轮追加；窗口填满后
    整块前滑，在此之前前缀是稳的）→ 会话态（玩家身份/所在场景/进度/待确认草案，
    块内好感度压尾）→ 关系跨档提示（只在委派工具真的下发那几轮出现）
    → RAG/提及 NPC（跨轮全变、轮内不变）。

    好感度原先排在最前面，于是它每轮一变就把后面所有块的前缀缓存全部打断；
    挪到这些块之后，稳定段才能真正复用。
    """
    parts: list[str] = []
    done = (finished_task_block or "").strip()
    if done:
        parts.append(done)
    long_term = (long_term_block or "").strip()
    if long_term:
        parts.append(long_term)
    summary = (summary_str or "").strip()
    if summary:
        parts.append(summary)
    hist = (history_str or "").strip()
    if hist:
        parts.append(hist)
    session_block = build_session_state_block(
        player_identity=player_identity,
        progress_desc=progress_desc,
        favorability=favorability,
        relationship_level=relationship_level,
        pending_draft_summary=pending_draft_summary,
        save_info=save_info,
    )
    if session_block:
        parts.append(session_block)
    hint = (relationship_hint or "").strip()
    if hint:
        parts.append(hint)
    rag = rag_block(retrieved_context)
    if rag:
        parts.append(rag)
    men = (mentioned_npcs_str or "").strip()
    if men:
        parts.append(men)
    return "\n\n".join(parts)


def build_portrait_note(npc_name: str) -> str:
    """立绘文字说明（v2 口径）：告诉模型这张图是谁，并免掉它硬拿图开话题。

    v3 一度只发图片 part 不发说明，而兜底的 appearance 块在现网数据里恒为空
    （npc_state_db.json 的 85 个 NPC 无一填 appearance），模型等于收到一张
    没头没尾的人像——弱模型会去描述图或被图带跑。
    """
    who = (npc_name or "").strip() or "当前 NPC"
    return f"现在传入了你所扮演的角色「{who}」的肖像（不需要强行通过肖像内容开启话题）。"


def build_emotion_hint(current_emotion: Optional[str]) -> str:
    """上一轮情绪连贯提示（v2 口径）：仅在有非「普通」的历史情绪时注入。"""
    emo = (current_emotion or "").strip()
    if not emo or emo == DEFAULT_EMOTION:
        return ""
    return f"你之前的情绪是「{emo}」。"


def build_player_message(
    user_query: str,
    *,
    npc_name: str = "",
    has_image: bool = False,
    current_emotion: Optional[str] = None,
) -> str:
    """末条 user 消息：玩家当轮发言（最易变，置于一切内容之后；图片 part 附其尾）。

    发言之后按 v2 口径追加立绘说明与情绪连贯提示：这两句是模型生成前最后读到
    的文字，位置最贴近生成点，情绪连贯尤其依赖它（否则每轮情绪都从零起跳）。
    """
    tail = ""
    if has_image:
        tail += build_portrait_note(npc_name)
    tail += build_emotion_hint(current_emotion)
    return f"玩家：{user_query}" + (f"\n\n{tail}" if tail else "")


NO_AGENT_CONSTRAINT = """\
【当前无法派发委托】
你现在不能给玩家派发任何委托、代办或差事，也查不了外部情报——只能凭自己已知的设定交谈。
因此：不要答应「我去安排」「回头给你派个活」这类承诺，也不要编造具体任务内容或报酬数字。
玩家索要任务/委托时，先以符合你身份和当前关系的口吻婉拒（如手头没活、时机不合适、不放心交给他）。
若玩家坚持要接活，则提醒他：得先在自己那台终端上把接取委托的权限打开，你这边才派得出去。
这类提醒同样要用角色口吻说（例：「你那终端的委托权限还关着呢，打开了我才好给你派活」），
不要用界面或系统术语（面板、开关、勾选、参数名之类），也不要把它说成系统故障。
"""


NO_TASK_PUBLISH_CONSTRAINT = """\
【当前无法派发委托】
你不能给玩家派发任何委托、代办或差事，也查不了外部情报——只能凭自己已知的设定交谈。
因此：不要答应「我去安排」「回头给你派个活」这类承诺，也不要编造具体任务内容或报酬数字。
玩家索要任务/委托时，以符合你身份和当前关系的口吻婉拒（如手头没活、时机不合适、不放心交给他）。
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
    "任务内容、要求、奖励都讲清楚，结尾必须征询玩家是否接取。"
    "问法不要固定成同一句：结合当下的语境（事情的轻重缓急、任务的性质、你刚才那句过渡话）与你的性格、"
    "和玩家的关系/好感，从下面两类说法里挑一种，并尽量换着说。"
    "下面这些**仅供参考，不要照抄**——照你的身份和当下情景，用自己的话问："
    "疑问式：「没问题吧？」「能完成吗？」「条件满意吗？」「可以接受吧？」「有意向吗？」"
    "「愿意接吗？」「有信心吗？」「能帮忙吗？」「合你心意吗？」「这活儿你接吗？」；"
    "陈述式：「如果愿意，我就发派任务给你。」「接受的话我就把任务派给你。」"
    "「只要你同意，我就发布委托。」「你点个头，这事就定了。」「想接就说一声，我把委托发给你。」"
    "硬性要求：一次只用一种说法，别把几种堆在一起；必须给玩家留出能明确回答接或拒的口子，"
    "不能含糊带过、不能替玩家拍板，也不要用「没意见我就安排下去」这类默认沉默即同意的说法。"
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

TASK_CONFIRM_MERGE_INSTRUCTION = (
    "这份委托已经正式派给玩家了（结果见上）。请以你的口吻把这事定下来："
    "简短交代他要做什么、办完回来找你，语气按你和他的关系与当下情景来。"
    "不要再问一次他接不接（他已经接了），也不要提系统/后台/草案这类词。"
)

TASK_CANCEL_MERGE_INSTRUCTION = (
    "这份委托已经撤掉了（结果见上）。请以你的口吻自然收场，"
    "不要提系统/后台/草案这类词，也不要承诺什么时候再给他派活。"
)

TASK_CANCEL_NOOP_INSTRUCTION = (
    "玩家想推掉这件事，但其实你手上并没有正在跟他谈的委托。"
    "请以你的口吻自然回应（可以顺着他的话头带过，或问他指的是哪件事），"
    "不要提系统/后台/草案这类词，也不要假装真的撤销了什么。"
)

CONFIRM_FAIL_INSTRUCTION = (
    "刚才那份委托出了点问题，暂时给不了玩家。用 1~3 句话以你的口吻解释并自然收场"
    "（如「这份委托刚才出了点问题……改天再谈」），不要提任何系统词汇。"
)


# ---------------------------------------------------------------------------
# 空正文兜底轮（台词一个字都没出时补的一次开口调用）
# ---------------------------------------------------------------------------

def build_empty_reply_user_prompt(
    *, player_query: str, emotion: str, recorded: str = "", delegated: str = "",
) -> str:
    """空正文兜底轮的 user 消息：带上本轮已落库的长期记忆与已派出的对话修改，只要 NPC 开口。

    这一轮的调用**不带任何工具**（长期记忆工具与委派工具都不再注册），所以这里只交代
    结果、不提记录与委派这两件事本身，避免给模型留一个无处可点的指代。
    """
    parts = [f"玩家刚才说：{player_query}"]
    if recorded:
        parts.append(f"【已记下】{recorded}")
    if delegated:
        parts.append(f"【已派出】{delegated}")
    parts.append(
        f"【本轮你的情绪】{emotion}（保持一致，不要改变语气基调）。"
        "不要输出第一行 meta JSON，直接输出台词本身。"
    )
    parts.append("【现在】把话接上，以你的身份正常回复玩家刚才那句话。")
    return "\n\n".join(parts)


#: 兜底轮也没能生成台词时的静态收场话（玩家侧不能留空白）
EMPTY_REPLY_FALLBACK_TEXT = "嗯，我听着呢，你说。"


# 汇合说明要点按任务类型条件拼接：草案摘要给的是「有什么」，这里补的是「讲的时候
# 该把哪几件事讲清」——不同类型的委托容易漏讲的点不一样（传话漏说找谁、
# 通关并收集只讲一半等）。未知类型返回空串，不给模型添噪声。
TASK_TYPE_EXPLAIN_HINTS: dict[str, str] = {
    "问候": "这类委托是让他替你去见个人、打个照面，说明时讲清去找谁、见了做什么就行。",
    "传话": "这类委托是让他替你带话，说明时讲清找谁、带什么话，别把话的内容说成别的。",
    "通关": "这类委托是让他去打通某个关卡，说明时讲清是哪个地方、要打到什么程度。",
    "清理": "这类委托是让他去清一片地方的怪，说明时讲清是哪片地方、要清到什么程度。",
    "挑战": "这是硬仗，说明时把难度和风险讲在前头，别让他以为是轻松活。",
    "切磋": "这是你跟他之间的一场比试，说明时是你自己在跟他约，不是派他去别处打。",
    "资源收集": "这类委托是让他把东西弄来交给你，说明时讲清要什么、要多少、交给谁。",
    "装备缴纳": "这类委托是让他上缴装备，说明时讲清要哪类装备、什么档次、交给谁。",
    "物品持有": "这类委托是让他替你保管东西，说明时讲清拿着什么、别弄丢、怎样算办完。",
    "特殊物品获取": "这类委托是让他去弄一件特定的稀罕物，说明时讲清是什么、上哪弄、拿到后交给你。",
    "通关并收集": "这类委托既要他打通关卡又要带回东西，说明时两件事都得讲清，别只说一半。",
    "通关并持有": "这类委托既要他打通关卡又要拿着某物，说明时两件事都得讲清，别只说一半。",
}


def task_merge_hint(task_type: str) -> str:
    """按任务类型取说明要点（未知类型返回空串）。"""
    return TASK_TYPE_EXPLAIN_HINTS.get((task_type or "").strip(), "")


# ---------------------------------------------------------------------------
# confirm 发布文本生成（与聊天 Agent 同源前缀的单次调用；无硬校验，宽松归一化）
# ---------------------------------------------------------------------------

CONFIRM_ARGS_SYSTEM = (
    "你正在以当前扮演角色的身份，为自己刚与玩家谈妥的委托撰写发布文书。"
    "严格按角色口吻撰写。输出一个 JSON 对象（不要代码块、不要解释）。字段："
    'title（任务标题，简洁）、description（任务说明，与任务内容一致）、'
    "get_dialogue（接取对话数组：发布 NPC 向玩家发布任务，可穿插玩家回应，建议 2~4 条），"
    "finish_dialogue（完成对话数组：玩家向 NPC 交付，NPC 验收，建议 2~4 条）。"
    '每条格式 {"name","emotion","text"}，撰写规则：'
    "1. text 只能是纯对话内容——**不能包含任何动作/神态/旁白描述，不能使用【】中括号或任何括号补注**；"
    "神态一律写在 emotion 字段（从你的可用情绪中选，可为空字符串）。"
    "emotion 逐条各填：每句话都有自己的情绪，随这段对话的推进换，不要整段共用同一个。"
    "写入任务系统的对话会被游戏直接展示，中括号内容会被后端剥离。"
    "2. 玩家条目 name 用 $PC，不要写死玩家名字；NPC 条目 name 用你的名字。"
    "3. 对话要自然衔接上文，不要与近期发言重复，不要复用其它任务的对话文本。"
    "4. dialogue_rewrite 填这份委托是否非常关键或有长期较大影响、玩家完成后你的说话方式应当随之改变"
    "（true/false）；为 true 时在 dialogue_rewrite_instruction 里写你希望自己的对话怎么变。"
)

# 完成 NPC 与接取 NPC 不同时才使用：多出「回接取 NPC 领奖」这一段的字段与规则。
# 两者相同时逐字走 CONFIRM_ARGS_SYSTEM，绝不把回访字段带进模型视野。
CONFIRM_ARGS_PAIRING_SYSTEM = (
    "你正在以当前扮演角色的身份，为自己刚与玩家谈妥的委托撰写发布文书。"
    "这份委托的完成 NPC 与接取 NPC 不同。"
    "严格按角色口吻撰写。输出一个 JSON 对象（不要代码块、不要解释）。字段："
    'title（任务标题，简洁）、description（任务说明，与任务内容一致）、'
    "get_dialogue（任务一接取对话数组：发布 NPC 向玩家发布任务，可穿插玩家回应，建议 2~4 条）、"
    "finish_dialogue（任务一完成对话数组：玩家向完成 NPC 交付，完成 NPC 验收，建议 2~4 条）、"
    "return_for_reward（布尔）、return_description（回访任务说明）、"
    "return_get_dialogue（任务二接取对话数组）、"
    "return_finish_dialogue（任务二完成对话数组）。"
    '每条格式 {"name","emotion","text"}，撰写规则：'
    "1. text 只能是纯对话内容——**不能包含任何动作/神态/旁白描述，不能使用【】中括号或任何括号补注**；"
    "神态一律写在 emotion 字段（从你的可用情绪中选，可为空字符串）。"
    "emotion 逐条各填：每句话都有自己的情绪，随这段对话的推进换，不要整段共用同一个。"
    "写入任务系统的对话会被游戏直接展示，中括号内容会被后端剥离。"
    "2. 任何对话中，NPC 说话人必须是该场对话在场的 NPC：任务一接取对话＝接取 NPC，"
    "任务一完成对话＝完成 NPC，任务二接取对话＝完成 NPC，任务二完成对话＝接取 NPC；"
    "玩家条目 name 用 $PC，不要写死玩家名字。"
    "3. 完成 NPC 与接取 NPC 不同时，你可以选择是否让玩家在任务一完成后回到接取 NPC 领取真实报酬，"
    "用 return_for_reward 表示。return_for_reward 为 false 时，玩家是直接从完成 NPC 获取报酬，"
    "只需要 get_dialogue + finish_dialogue 这两组任务对话；return_for_reward 为 true 时，"
    "玩家是最后回到接取 NPC 获取报酬，此时需要 get_dialogue + finish_dialogue + "
    "return_get_dialogue + return_finish_dialogue 四组任务对话，依次触发。"
    "4. return_for_reward 为 true 时，return_description 填任务二的说明，"
    "讲清玩家回接取 NPC 处复命领报酬；为 false 时留空。"
    "5. return_for_reward 为 true 时，必须同时给出 return_get_dialogue"
    "（任务二在完成 NPC 处接取）和 return_finish_dialogue（任务二回到接取 NPC 处完成），"
    "两组都不能为空。"
    "6. return_for_reward 为 false 时，return_get_dialogue 与 return_finish_dialogue "
    "都必须留空数组。"
    "7. 对话要自然衔接上文，不要与近期发言重复，不要复用其它任务的对话文本。"
    "8. dialogue_rewrite 填这份委托是否非常关键或有长期较大影响、玩家完成后你的说话方式应当随之改变"
    "（true/false）；为 true 时在 dialogue_rewrite_instruction 里写你希望自己的对话怎么变。"
)


def confirm_args_system(*, allow_return: bool) -> str:
    """按「接取 NPC 与完成 NPC 是否不同」选取 confirm 系统提示词。

    相同时走单任务版（无任何回访字段），避免模型被不存在的选项误导。
    """
    return CONFIRM_ARGS_PAIRING_SYSTEM if allow_return else CONFIRM_ARGS_SYSTEM


def build_finish_npc_reference(
    *,
    npc_name: str,
    lines: list[str] | None = None,
    emotions: list[str] | None = None,
    favorability: int | None = None,
    relationship_level: str = "",
) -> str:
    """完成 NPC 的写作参考（完成 NPC ≠ 当前说话人时拼在 confirm user 末尾）。

    三段：过往台词（参考说话风格）、可用情绪标签、与玩家的关系/好感度。
    查不到的段直接省略；三段都为空返回空串（不拼空块）。
    """
    name = (npc_name or "").strip()
    if not name:
        return ""
    parts: list[str] = []
    picked = [str(t).strip() for t in (lines or []) if str(t or "").strip()]
    if picked:
        parts.append(
            f"{name}的过往台词如下（请参考其说话风格编写其对话）：\n"
            + "\n".join(picked)
        )
    pool = [str(e).strip() for e in (emotions or []) if str(e or "").strip()]
    if pool:
        parts.append(f"{name}的可用情绪标签：[{'、'.join(pool)}]")
    if favorability is not None:
        level = (relationship_level or "").strip() or "陌生"
        parts.append(f"{name}与玩家的关系/好感度：好感度 {int(favorability)}（{level}）")
    return "\n\n".join(parts)


def build_confirm_args_user_prompt(
    *,
    draft_summary: str,
    npc_name: str,
    spoken_text: str = "",
    emotion: str = "",
    allow_return: bool = False,
    finish_npc_reference: str = "",
) -> str:
    """confirm 参数生成（复用聊天 Agent base_messages 前缀后追加的 user 消息）。

    ``allow_return``＝接取 NPC 与完成 NPC 不同：只有此时字段清单才列出回访四件套。
    ``finish_npc_reference`` 是完成 NPC 的台词/情绪/关系参考，按作者要求拼在最下面。
    """
    parts = []
    if (spoken_text or "").strip():
        parts.append(
            f"【本轮你已对玩家说的话（情绪：{emotion or '普通'}，保持一致）】\n"
            f"{npc_name}：{spoken_text.strip()}"
        )
    parts.append(f"【待发布任务详情】\n{draft_summary}")
    fields = (
        "title/description/get_dialogue/finish_dialogue/"
        "return_for_reward/return_description/return_get_dialogue/return_finish_dialogue/"
        "dialogue_rewrite/dialogue_rewrite_instruction"
        if allow_return else
        "title/description/get_dialogue/finish_dialogue/"
        "dialogue_rewrite/dialogue_rewrite_instruction"
    )
    parts.append(
        f"现在以你的身份生成发布文本 JSON（{fields}）。"
        "对话中的玩家条目 name 用 $PC。"
    )
    reference = (finish_npc_reference or "").strip()
    if reference:
        parts.append(reference)
    return "\n\n".join(parts)


def build_confirm_retry_prompt(issues: list[dict]) -> str:
    """confirm 未通过校验时的修正消息：逐条给出问题与修法，要求重新输出完整 JSON。"""
    lines = ["上一次输出未通过校验："]
    for issue in issues:
        field = str(issue.get("field") or "").strip()
        message = str(issue.get("message") or "").strip()
        hint = str(issue.get("fix_hint") or "").strip()
        head = f"- {field}：{message}" if field else f"- {message}"
        lines.append(head)
        if hint:
            lines.append(f"  修改建议：{hint}")
    lines.append("请修正后重新输出完整 JSON（字段与上一次要求相同）。")
    return "\n".join(lines)


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
    """confirm 参数生成的 LLM 失败兜底：模板文本，保证流程不中断。

    两段对话的说话人分别取接取 NPC 与完成 NPC——跨 NPC 委托时都用当前 NPC
    会把完成对话说成接取人，说话人校验会直接判错。
    """
    from services.agent_tools.task_tools import resolve_task_npcs

    get_npc, finish_npc = resolve_task_npcs(draft, npc_name)
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
    get_text = f"这份委托就交给你了。{('要求：' + description) if description else ''}"
    finish_text = "干得漂亮，说好的报酬一分不少。"
    return {
        "title": title,
        "description": description,
        "get_dialogue": [{"name": get_npc, "emotion": "", "text": get_text}],
        "finish_dialogue": [{"name": finish_npc, "emotion": "", "text": finish_text}],
        "return_for_reward": False,
        "return_description": "",
        "return_get_dialogue": [],
        "return_finish_dialogue": [],
    }


# ---------------------------------------------------------------------------
# 进度描述
# ---------------------------------------------------------------------------

def progress_stage_desc(stage: Optional[int], cfg: Optional[StageConfig]) -> str:
    if not stage or cfg is None:
        return "玩家当前主线进度未知或尚未开始。"
    level = f"等级区间 {cfg.min_level or '?'}-{cfg.max_level or '?'}" if cfg.min_level or cfg.max_level else "等级未知"
    return f"当前主线进度：阶段 {stage}（{cfg.name}，{level}）。"
