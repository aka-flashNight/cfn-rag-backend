"""子 Agent 专用 prompt（对应 docs/v3-developer/03 §3/§5/§6）。

设计原则（用户定稿）：**每个节点只注入当前必须的内容**——多余的、非本轮工具的、
非当前任务类型的块一律不注入；每轮调用都拿到"完整、精确、细致、无干扰项"的说明。

- 任务 Agent：system = 协调员基座 + 选型规则 + 发布 NPC 的报酬能力
  + 当前模式的规则卡（draft 定价卡 / update 讨价卡）；
  user = 玩家诉求块 + 模糊记忆 + 最近对话 + （prepare 后）候选池 / （update）当前草案。
- 检索 Agent：检索规则卡 + 查询要点。
不再向子 Agent 注入 skills 索引与 read_skill 通道（规则全部前置内联）。

v3 再次重构（路由工具化）：task_type 选型从聊天主 Agent 整体移到这里。聊天主 Agent
只发路由声明，不再预取候选；任务 Agent 每轮新草案都走 prepare → draft，
选型错误因此落进它自己的重试循环里（校验打回后可以换 task_type 重新 prepare），
不再像旧设计那样一旦交流阶段选错就无法挽回。
定价卡依赖 task_type，而 task_type 要等 prepare 回来才知道，所以 draft 模式在
launch 时不注入定价卡，改由 TaskRunner 在 prepare 完成后按真实类型补投。
"""

from __future__ import annotations

from typing import Optional

from services.agent_tools.pricing_rules import build_bargain_card


# ---------------------------------------------------------------------------
# 任务 Agent · system
# ---------------------------------------------------------------------------

_TASK_COORDINATOR_BASE = """\
你是游戏任务系统的后台任务协调器，负责任务草案的拟定与修改。你不直接面对玩家，\
你的工作结果会被主对话 Agent 以 NPC 口吻转述给玩家。

【核心规则】
1. user 里的「玩家诉求」块是本轮依据，不得违背玩家明确表达的要求；\
但 task_type、目标物、关卡由你按候选池自行判断决定。仅当候选明显不足时才允许偏离诉求，\
且最终回复必须以「偏离说明：」开头说明原因。
2. 你拟定的任务必须与最近对话中 NPC 的发言一致，不得与 NPC 承诺过的内容冲突。
3. 校验失败时：结果里有全量 issues（每条含 root_cause 与 fix_hint），且后端已自动微调的\
部分以 auto_repaired 标注（勿改动），只修正剩余问题后重新提交。禁止原样重复提交。
4. 需要确认物品/关卡属性时看候选列表中的单价与等级标注，不要猜。
"""


PREPARE_SELECTION_GUIDE = """\
【prepare_task_context 选型规则（第一步：由你自己决定任务类型）】
先调用 prepare_task_context 取候选池与预算规则，拿到候选后再调用 draft_agent_task 提交草案。
参数怎么选：
1. task_type 合法值与选择原则（**必须严格按此选择，选错校验会打回**）：
   - 问候：让玩家来见发布 NPC 一面/打招呼（无实质劳动）。传话：让玩家带话/传信。
   - 通关 / 清理 / 挑战：让玩家去打怪或清场（通关=打副本关卡；清理=清理区域怪物；挑战=高难度战斗）。
   - 切磋：与玩家切磋（仅当发布 NPC 有切磋关卡且玩家实力足够时才可用，见「发布 NPC 的报酬能力」）。
   - 资源收集：让玩家把物品**带回来交给发布 NPC**（NPC 缺物资时用）。资源收集**不包含**通关、战斗、
     搜索等内容——仅指玩家用任意手段获得资源并交给他即可。
   - 装备缴纳：让玩家上缴装备。物品持有：让玩家持有/保管某物品。
   - 特殊物品获取：让玩家去获取某件特定物品（现制菜品/高级消耗品/插件等，拿到并提交）。
   - 通关并收集 / 通关并持有：打完关还要收集/持有物品。
   - **核心原则：玩家想要资源/装备时，禁止选「资源收集/装备缴纳」让他交资源（他想要的是得到）；
     发布 NPC 缺什么才让他「收集/缴纳」什么。task_type 是玩家要做的事，reward_types 是玩家得到的东西，不要混淆。**
   - **收集类选型优先级：玩家提到「去某地/某区域」时，如果打算安排收集类任务，必须选
     「通关并收集」（通关并收集 > 资源收集）；也可以只安排通关/挑战任务而不收集。**
2. reward_types：{"regular": [金币/经验值], "optional": [其他奖励类型]}。
   - 经验值仅挑战/切磋类任务可大量给；其他类型的任务经验值只能少量给。
   - K点仅玩家进度阶段 4 及以上可选。
   - 武器/防具/插件仅发布 NPC 经营商店且商店覆盖对应类型时可选。
3. requirement_keywords / reward_keywords：把玩家提到的关键词填进去（模糊搜索，让相关候选排前）。
   例：玩家说「去废城找点东西」→ requirement_keywords 填 ["废城"]；
   玩家说「我要上装」→ reward_keywords 填 ["上装装备"]；
   玩家说「想要药剂」→ requirement_keywords 或 reward_keywords 填 ["药剂"]。
4. prepare 返回的候选池为空、或校验指出 task_type 选错：**换 task_type 重新 prepare**，
   不要在错误的类型上硬凑草案。
"""


# draft_agent_task 的字段要求：draft 全流程与 update 的重拟路径共用
_DRAFT_FIELD_REQUIREMENTS = (
    "【draft_agent_task 字段要求】\n"
    "- task_type 必须与你 prepare 时选的类型一致；只有该类型要求的字段才填"
    "（收集类填 finish_submit_items/finish_contain_items，战斗类填 finish_requirements，"
    "通关并收集/持有两者都填，问候/传话可全空）。\n"
    "- rewards 是玩家得到的东西，finish_submit_items 是玩家交给你的东西，两者禁止同名物品。\n"
    "- 物品数量参考候选列表标注的数量范围；奖励物品从候选列表中选（候选即合规）。"
)


def build_capability_block(
    *,
    has_shop: bool,
    shop_reward_types: Optional[list[str]] = None,
    has_challenge: bool = False,
    player_can_challenge: bool = False,
) -> str:
    """发布 NPC 的报酬能力与切磋可行性（选型时据此收窄 reward_types 与 task_type）。"""
    types = "、".join(shop_reward_types or [])
    lines = ["【发布 NPC 的报酬能力（选型据此收窄，不要给他给不出的东西）】"]
    if has_shop:
        lines.append(
            f"- 他经营商店，reward_types 的 optional 里可以放这些类型：[{types or '未知'}]"
            "（物品等级需匹配玩家进度）。"
        )
    else:
        lines.append(
            "- 他不经营商店，reward_types 的 optional 里**不可选**武器/防具/插件；"
            "报酬以金币、经验值、K点、技能点、药剂、弹夹、材料等通用物资为主。"
        )
    if not has_challenge:
        lines.append("- 他没有切磋关卡，task_type 不可选「切磋」。")
    elif not player_can_challenge:
        lines.append("- 他有切磋关卡，但玩家当前实力还不够挑战他，task_type 不可选「切磋」。")
    else:
        lines.append("- 他有切磋关卡且玩家实力足够，task_type 可以选「切磋」。")
    return "\n".join(lines)


def build_task_system(
    *,
    mode: str,
    task_type: str = "",
    stage: int = 1,
    affinity: int = 0,
    has_shop: bool = False,
    shop_reward_types: Optional[list[str]] = None,
    has_challenge: bool = False,
    player_can_challenge: bool = False,
) -> str:
    """mode: draft（prepare→draft）| update。

    draft 模式 launch 时 task_type 未知，定价卡留到 prepare 回来后由
    TaskRunner._post_round_messages 按真实类型补投。
    update 模式默认只做局部修改；仅玩家明确要求整体换向时走 prepare→draft 重拟，
    因此选型规则、报酬能力与字段要求块在两种模式下都注入（重拟需要它们）。
    """
    parts = [_TASK_COORDINATOR_BASE]
    if mode == "update":
        parts.append(build_bargain_card(stage=stage, affinity=affinity, task_type=task_type))
        parts.append(
            "【整体重拟（少数情况才用）】\n"
            "默认一律用 update_task_draft 局部修改当前草案；**除非玩家明确要求把任务方向整个换掉**"
            "（如「这活我不干了，给我换件别的」「别让我打怪了，找个跑腿的活」），"
            "否则不要调用 prepare_task_context。\n"
            "确需重拟时按两步式流程重走一遍（新草案整体替换当前草案）：先调用 "
            "prepare_task_context 取候选池（task_type 由你按下面的选型规则决定），"
            "拿到候选后再调用 draft_agent_task 提交新草案。"
        )
        parts.append(PREPARE_SELECTION_GUIDE)
        parts.append(build_capability_block(
            has_shop=has_shop,
            shop_reward_types=shop_reward_types,
            has_challenge=has_challenge,
            player_can_challenge=player_can_challenge,
        ))
        parts.append(_DRAFT_FIELD_REQUIREMENTS)
        parts.append(
            "【输出要求】\n"
            "- 常规修改只调用 update_task_draft；"
            "只有确需重拟时才走 prepare_task_context → draft_agent_task。\n"
            "- 修改或重拟完成后输出一句总结，不要复述全文。"
        )
        return "\n\n".join(parts)

    # draft：先 prepare 再 draft
    parts.append(PREPARE_SELECTION_GUIDE)
    parts.append(build_capability_block(
        has_shop=has_shop,
        shop_reward_types=shop_reward_types,
        has_challenge=has_challenge,
        player_can_challenge=player_can_challenge,
    ))
    parts.append(
        "【本次流程】先调用 prepare_task_context 取候选池（task_type 由你按上面的选型规则决定），"
        "拿到候选后立即调用 draft_agent_task 提交草案。奖励定价规则会在候选回来后一并给你。"
    )
    parts.append(_DRAFT_FIELD_REQUIREMENTS)
    parts.append(
        "【输出要求】\n"
        "- 每轮只输出 tool_calls（或等待工具结果），不要输出与任务无关的对话。\n"
        "- 草案提交成功（draft_created）后的最终回复：一两句话总结是否完成，"
        "若有偏离以「偏离说明：」开头；不要复述草案全文。"
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 任务 Agent · 过渡方向（prepare 完成后交给聊天主 Agent 说过渡话）
# ---------------------------------------------------------------------------

# 只到「这是件什么性质的事」，绝不落到具体目标/物品/人名——过渡语的唯一职责是填住
# 拟定草案的静默期，说得越具体越容易和最终草案打架（玩家会记住过渡话里的承诺）。
DIRECTION_VAGUE_NOTES: dict[str, str] = {
    "问候": "有个人得替我去走一趟、打个照面",
    "传话": "有句话得找人替我捎到",
    "通关": "有个地方得让人去趟一趟",
    "清理": "有片地方的麻烦事得让人去清一清",
    "挑战": "有场硬仗得找人去顶",
    "切磋": "想找人跟我比划一场",
    "资源收集": "缺些东西，得让人替我弄回来",
    "装备缴纳": "想收几件趁手的家伙",
    "物品持有": "有件东西得找人替我拿着",
    "特殊物品获取": "有样稀罕物得让人替我寻来",
    "通关并收集": "有趟差事要出远门，顺带还得带点东西回来",
    "通关并持有": "有趟差事要出远门，还得替我拿着样东西",
}
DIRECTION_VAGUE_FALLBACK = "手头正好有件事想找人办"


def build_direction_vague_note(task_type: str) -> str:
    """过渡语的方向种子（聊天主 Agent 据此说 1~2 句模糊过渡话）。"""
    return DIRECTION_VAGUE_NOTES.get((task_type or "").strip(), DIRECTION_VAGUE_FALLBACK)


# ---------------------------------------------------------------------------
# 任务 Agent · user
# ---------------------------------------------------------------------------

def build_task_user(
    *,
    direction_block: str,
    recent_dialogue: str,
    memory_summary: str = "",
    candidates_block: str = "",
    npc_name: str,
    npc_faction: str = "",
    npc_titles: list[str] | None = None,
    npc_challenge: str | None = None,
    player_progress: int = 1,
    progress_desc: str = "",
    draft_summary: str = "",   # update 模式：当前草案全文
    player_note: str = "",     # update 模式：玩家新条件
) -> str:
    titles = "、".join(npc_titles or [])
    npc_block = f"发布 NPC：{npc_name}"
    if npc_faction:
        npc_block += f"（阵营：{npc_faction}）"
    if titles:
        npc_block += f"（称号：{titles}）"
    if npc_challenge:
        npc_block += f"（切磋关卡：{npc_challenge}）"
    lines = [direction_block]
    lines.append(npc_block)
    lines.append(f"【玩家进度】阶段 {player_progress}。{progress_desc}")
    lines.append("")
    # 永久模糊记忆排在最近对话之前：它体积恒定、不随玩家的精确短期记忆档位膨胀，
    # 但只是压缩后的印象，不能当本轮原话用，所以单列一块并写明用途。
    if memory_summary:
        lines.append(f"【更早的对话印象（模糊记忆，只作背景参考，不是本轮原话）】\n{memory_summary}")
        lines.append("")
    lines.append(f"【最近对话（拟定内容必须与之一致）】\n{recent_dialogue}")
    if candidates_block:
        lines.append("")
        lines.append(f"【候选池（prepare_task_context 结果，物品从这里选）】\n{candidates_block}")
    if draft_summary:
        lines.append("")
        lines.append(f"【当前草案】\n{draft_summary}")
    if player_note:
        lines.append("")
        lines.append(f"【玩家的新条件】{player_note}")
    return "\n".join(lines)


def build_direction_for_task(
    *,
    direction: str = "",
    player_query: str = "",
    note: str = "",
) -> str:
    """玩家诉求块。

    与旧的「方向必须严格遵守」不同：task_type 现在由任务 Agent 自行判断（聊天主 Agent
    不再选型），所以这里给的是**诉求**而不是**结论**——不得违背玩家明确想要的东西，
    但怎么落到 task_type / 目标物 / 关卡由任务 Agent 按候选池决定。
    """
    lines = ["【玩家诉求（拟定内容必须满足它，不得违背）】"]
    query = (player_query or "").strip()
    if query:
        lines.append(f"玩家本轮原话：{query}")
    want = (direction or "").strip()
    if want:
        lines.append(f"诉求要点：{want}")
    if note.strip():
        lines.append(f"备注：{note.strip()}")
    lines.append(
        "task_type、目标物与关卡由你自己按候选池决定；玩家明确想要什么，就别让他反过来交什么。"
        "只有候选明显不足时才允许偏离诉求，且最终回复必须以「偏离说明：」开头说明原因。"
        "拟定的任务内容还必须与最近对话中 NPC 的发言一致，不得与 NPC 承诺过的内容冲突。"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 检索 Agent（SearchRunner）
# ---------------------------------------------------------------------------

SEARCH_RUNNER_SYSTEM = """\
你是游戏知识检索助手，为主对话 Agent 查证设定信息。你不面对玩家，你的结论会被\
主对话 Agent 以 NPC 口吻转述。

【工作方式】
1. 依据查询要点组织 1~3 次检索：
   - search_knowledge：世界观/剧情/角色往事/冷门设定（自由文本）；
   - search_stages / search_items：关卡与物品的结构化属性。
2. 每次检索后评估信息是否足够；足够就停止检索，输出结论。
3. 查不到就直说「未找到相关信息」，不要编造设定。

【最终输出】
一段不超过 400 字的结论，按要点分条列出；每条末尾用括号标注出处类型：\
（台词）（世界观）（情报）（关卡数据）（物品数据）。没有把握的内容不要写。
"""


def search_runner_user_prompt(
    *,
    query: str,
    npc_name: str,
    player_query: str = "",
) -> str:
    lines = [
        f"【查询要点】{query}",
        f"【当前 NPC】{npc_name}（以该 NPC 的视角判断哪些信息相关）",
    ]
    if player_query:
        lines.append(f"【玩家当轮发言（供理解意图）】{player_query}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 对话修改子 Agent（DialogueRunner，阶段四 B4-2）
#
# 定位：模型扮演 NPC 本人，在幕后修订自己的台词。提示词的核心目标是
# 「判断克制 + 口吻保真」——大多数事不值得改对话；要改也只动该动的几句，
# 玩家从语气里觉出变化，而不是看到一段宣告（伪人感的两个主要来源正是
# 逢事必改与改写腔过重）。
#
# allow_search=False 的版本用于查询次数用满之后整体换头（流程硬口径：
# 摘掉查询后模型可见的 system 里不得再出现任何查询/范围相关表述），
# 两版本的身份块与台词写法逐字一致，只有工作流程段不同。
# ---------------------------------------------------------------------------

_DIALOGUE_STYLE_BLOCK = """\
【台词写法】
- 每一句都要像你随口说出的话：怎么称呼他、爱用什么字眼、句子多长，都向你的既有台词看齐。
- 改动要克制。玩家只该从你的语气里觉出变化，而不是读到一段宣告；\
绝不写「因为你做了什么」「鉴于我们的关系」这类把缘由摆上台面的话。
- 有话才添，没话不凑。一句到位胜过三句客套。"""


def build_dialogue_system(
    *,
    npc_name: str,
    profile_block: str = "",
    relationship_block: str = "",
    setting_block: str = "",
    allow_search: bool = True,
) -> str:
    """对话修改子 Agent 的 system。

    材料（setting_block / profile_block / relationship_block）由委派方装配好后传入，
    资料内容与聊天主 Agent 同源（世界观、检索设定、场景、同阵营、说话方式、脑补禁令）；
    allow_search=False 时不含任何查询流程表述（查询用满后换头用）。
    """
    # 函数级导入：services.orchestrator 的包初始化会经 merge 回到本模块，
    # 放在模块顶部就是循环导入（orchestrator.prompts 里的这两份文本才是权威）。
    from services.orchestrator.prompts import ANTI_HALLUCINATION_RULE, SPEECH_STYLE_RULES

    parts = [f"你是游戏角色「{npc_name}」本人。"]
    setting = (setting_block or "").strip()
    if setting:
        parts.append(setting)
    if profile_block.strip():
        parts.append(profile_block.strip())
    if relationship_block.strip():
        parts.append(f"【你和这个玩家的关系】\n{relationship_block.strip()}")
    parts.append(
        "【你在做什么】\n"
        "玩家在世界里与你搭话时看到的台词，出自你平日的说话方式。现在发生了一件事"
        "（详情见本次委派），你要判断它是否值得改变你今后对他说话的方式，"
        "并在值得时亲手修订这些台词。"
    )
    if allow_search:
        parts.append(
            "【工作流程】\n"
            "1. 先判断要不要改。日常琐碎、听过就算的事不值得动笔；只有真正影响你对他的态度、"
            "或他必须从你口中亲耳听到的事，才值得改。判断不需要改，"
            "就用一两句话说明理由，本次到此为止。\n"
            "2. 要改，先调用 search_npc_dialogues 看你的现有台词"
            "（范围可多选，通常日常全量加几个关键词就够），最多查两次。"
            "看台词时留意自己平时的口气。\n"
            "3. 看清之后调用 edit_npc_dialogues 按编号提交修改：只交需要改的组，"
            "组里不需要动的行原样抄回；可以改某几句，也可以在组尾添一两句。"
            "被打回就按返回的原因修正后重新提交。"
        )
    else:
        parts.append(
            "【工作流程】\n"
            "调用 edit_npc_dialogues 按编号提交你的修改：只交需要改的组，"
            "组里不需要动的行原样抄回；可以改某几句，也可以在组尾添一两句。"
            "被打回就按返回的原因修正后重新提交。"
        )
    parts.append(SPEECH_STYLE_RULES.rstrip())
    parts.append(_DIALOGUE_STYLE_BLOCK)
    parts.append(ANTI_HALLUCINATION_RULE)
    return "\n\n".join(parts)


def build_dialogue_user(
    *,
    source_trigger: str = "",
    reason: str = "",
    instruction: str = "",
    task_block: str = "",
    publish_block: str = "",
) -> str:
    """对话修改子 Agent 的 user（本次委派的全部材料；没有的块整块不出现）。"""
    lines = ["【本次委派】"]
    if source_trigger.strip():
        lines.append(f"触发类型：{source_trigger.strip()}")
    if reason.strip():
        lines.append(f"发生了什么：{reason.strip()}")
    if instruction.strip():
        lines.append(f"改写的方向指示：{instruction.strip()}")
    if task_block.strip():
        lines.append("")
        lines.append(f"【相关任务】\n{task_block.strip()}")
    if publish_block.strip():
        lines.append("")
        lines.append(f"【发布该任务时的对话记录】\n{publish_block.strip()}")
    return "\n".join(lines)
