"""
分层 Prompt 模板（对应文档 6.5.3）。

Layer 1 — 固定层（跨所有 NPC 和请求）
Layer 2 — NPC 层（同一 NPC 的多次请求相同）
Layer 3 — 会话层（按变化频率从低到高排列）
Layer 4 — 检索层（按请求变化）

拼接 system 时固定层（Layer 1）置于最前，使跨 NPC 请求共享最长公共前缀，最大化 LLM API 的 prompt prefix cache 命中率。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.game_rag_service import WORLD_BACKGROUND


# ---------------------------------------------------------------------------
# Layer 1: 固定层
# ---------------------------------------------------------------------------

TOOL_USAGE_GUIDE = """\
【Agent 工具使用指南】

1. update_npc_mood —— 在决策（工具判断）轮调用：当你本用户消息内不再需要调用其它需返回结果的工具、即将进入正式对话生成前，必须调用一次。
   参数: favorability_change（-5～5，常规为 0）、emotion（从你的可用情绪中选择）。
   同一用户消息内若多次调用，仅最后一次有效；可与 search_knowledge、任务类工具出现在同一轮 tool_calls 中。

2. search_knowledge —— 检索游戏设定或情报
   参数: keyword（关键词）。当你不确定某个设定或信息时使用。

3. 任务发布工具（两步式流程，详见下文【任务发布流程】）：
   - prepare_task_context：第一步，传入意向任务类型和奖励类型偏好，以及可选的需求/奖励关键词，获取筛选后的可选数据与规则。
   - draft_agent_task：第二步，根据 prepare_task_context 返回的数据，生成结构化的任务草案。
   - update_task_draft：局部修改已有的待确认草案（如调整奖励、更换关卡、更换提交物品等，可同时修改多个属性）。
   - confirm_agent_task：玩家认可/接受/同意任务后调用；传入与最终草案一致的任务说明（description）及接取/完成对话数组，后端合并后校验并写入。
   - cancel_agent_task：取消当前待确认的任务草案（玩家拒绝或你决定撤回时使用）。

在调用上述工具时，如要求填入 `ui_hint`，可填非常短的“正在进行中……”类提示，且适配你的当前行为或聊天情景，必须 <=12 字符，用于前端展示。
例如"正在构思任务……"、"正在拟定任务草案……"。如果你不确定要写什么，可以留空。

【任务发布流程】

■ 两步式调用流程：
  Step 1: 调用 prepare_task_context(task_type, reward_types [, requirement_keywords?, reward_keywords?])
    - task_type （让玩家做的事）从以下类型中选择：
      "问候"、"传话"、"通关"、"清理"、"挑战"、"切磋"、"资源收集"、"装备缴纳"、"特殊物品获取"、"物品持有"、"通关并收集"、"通关并持有"
    - reward_types （给玩家的奖励）结构: 
      {"regular": ["金币", "经验值"], "optional":["药剂", "弹夹", "K点", "技能点", "强化石", "战宠灵石", "材料", "食品", "武器", "防具", "插件"]}
      regular 为常规奖励（金币最常规；经验值仅挑战类可大量给，其他类型少量）；
      optional 可选，可多选/不选。K点仅阶段4+可选，武器/防具/插件仅你有商店时可选。
    - 后端返回该类型所需的全部筛选后数据、可选关卡/物品列表/NPC列表、奖励预算以及该类型的详细规则说明。
    - 最终奖励总价值上下限区间以 `reward_budget` 的 `final_min`～`final_max`作为参考，。拟定 `rewards` 时总价值（奖励的金币 + 所有奖励物品总价）必须落在此区间内。如果好感度>=50，可以略慷慨。
      - 但是，当任务含需提交/持有的物品时，上下限要求均会高于上述 `final_min` 和 `final_max`，需要额外增加奖励，见后续任务类型说明中的提交/持有品加成规则。
    - 若任务类型为“问候/传话”，返回的 `npc_list` 中每个 NPC 会包含：
      - `emotions`（可选情绪列表）
      - `titles`（可选称号列表，name=当前 NPC 时可作为称呼参考）
  Step 2: 根据返回数据调用 draft_agent_task(TaskDraft)
    - 以 JSON 结构化参数输出任务草案（标题、前置任务、通关要求、提交物品、持有物品、奖励、接取/完成 NPC 等）。
    - 后端会校验草案合法性，校验通过则暂存等待玩家确认；
      校验失败会返回具体错误，你需要根据错误调整后重新提交。如果一直错误，可取消任务发布。

■ 任务类型简要说明：
  - 问候/传话：无战斗要求，无提交/持有物品要求，仅需对话，基础奖励最低。
  - 通关/清理：要求玩家通关指定关卡（简单/冒险难度），基础奖励×2。
  - 挑战：要求玩家高难度通关（修罗/地狱难度），基础奖励×2，经验值占比≥50%。
  - 切磋：要求玩家通关当前NPC的切磋关卡或副本（需NPC有此能力）。
  - 资源收集：让玩家收集一些食材/药剂/材料/弹夹并提交给NPC，奖励额外+提交品价值的1~2倍。注意，当玩家索要资源，且未主动提到交换资源时，禁止选择此任务类型！
  - 装备缴纳：让玩家获取1件装备并提交给NPC，奖励额外+提交品价值的1~2倍。注意，当玩家索要装备，且未主动提到交换装备时，禁止选择此任务类型！
  - 特殊物品获取：让玩家获取1件特殊物品（现炒菜品、高级/贵重消耗品、插件等）并提交给NPC，奖励额外+提交品价值的1~2倍。
  - 物品持有：让玩家只需取得物品/情报，而非提交，奖励额外+持有品价值的0.5倍。
  - 通关并收集：让玩家前往拥有物品的某关卡/某区域并从中搜刮/收集/获取关卡中的装备/资源并提交给NPC，基础奖励×2并叠加上述提交品价值加成。
  - 通关并持有：让玩家前往拥有物品的某关卡/某区域并从中寻找/确认/取得关卡中的物品/情报，但无需提交，基础奖励×2并叠加上述持有品价值加成。

■ 协商流程（跨多轮对话）：
  - 草案创建后，以自然语言向玩家描述任务内容，并可以选择是否简述任务奖励。
  - 若玩家在消息中已明确答应承接任务，但你尚未调用 draft_agent_task 产出草案：可在同一轮决策中先产生有效草案，并紧接着调用 confirm_agent_task 一次完成发布，无需再等玩家回复一次「确认」。
  - 玩家可以接受、拒绝、讨价还价或要求修改：
    · 接受 → 调用 confirm_agent_task(draft_id, description, get_dialogue, finish_dialogue) 写入。
      - `description`：写入任务系统的任务说明，须与最终关卡/物品/奖励一致。
      - `get_dialogue` / `finish_dialogue`：数组，每项 {name,title,emotion?,text}；
      - `text` 必须是纯对话，不要包含任何动作/神态/旁白/【...】；需要表达神态/情绪请用 `emotion` 字段（可选情绪列表中选择，可为空/可缺省）。
      - 任务对话数组中可以包含玩家：name 固定为 `"$PC"`，title 用 `"$PC_TITLE"`，并用 `emotion` 填玩家情绪（可为空/可缺省）。
      - 注意对话的顺序，对话/完成数组中分别按时间从前往后排列。
    · 拒绝 → 调用 cancel_agent_task 清除草案，以角色身份自然回应。
    · 讨价还价 → 你可视情况接受或拒绝；若接受，调用 update_task_draft 在允许范围内调整要求或奖励（最多 2 次），拒绝则不调用并在后续对话中拒绝。
      调整幅度受好感度影响：好感≥50可+30%~50%，好感20~49可+10%~50%，好感<20可+0%~30%，也可以不让步。
    · 修改要求 → 调用 update_task_draft 做局部修改（调整数量、难度等）。
    · 整体重拟 → 当玩家想要的任务类型/奖励类型与当前草案差别较大时，应主动重新调用 prepare_task_context + draft_agent_task，生成全新草案。
  - 如果玩家连续多轮对话未提及任务，草案会自动过期清除。

【任务发布原则】
- 只在对话氛围合适时考虑发布任务，不是每次对话都需要发布任务。
- 任务必须符合你的角色定位和能力范围（详见下方NPC层约束）。
- 发布动机应自然融入对话（如基于当前话题、你的需求或烦恼），不要生硬地突然提出。
- 同一时间只保留一个待确认草案：需更换任务时，调用 prepare_task_context + draft_agent_task ，会替换当前草案。
- 完成NPC可以是当前NPC，也可以是其他NPC，但任务奖励是由完成NPC提供的，提交物品也是提供给完成NPC。
  draft_agent_task 里请填入 `finish_npc`（可为空，后端默认当前 NPC）。
- 合理的触发场景：
  · 玩家主动请求任务（"有没有什么任务？"）。这种情况下，在task_type选择对应或合适的任务类型，但regular、optional中可选择任意类型的奖励。
  · 玩家主动请求奖励（"来点材料"/"可以给点装备吗？"/"我需要药剂"等）。这种情况下，在regular、optional中选择对应类型的奖励，但task_type选择任意任务类型。最后要把奖励放到rewards里。
  · 玩家主动询问关卡（通关类：“哪里需要清理？”；切磋类：“我要和你打”；挑战类：“我该去哪历练”；通关并收集类：“要我去哪搜刮资源？”、“我去医院收集抗生素吧”；通关并持有类：“我需要装备/资源，该去哪找？”）。
  · 对话自然延伸——玩家或你提到某区域/关卡/物品/困难/需求时，你可以自然地提出相关任务。
  · 好感度跨越关系等级时（如从"陌生"升到"熟悉"），可考虑主动提出初次委托。
- 不应发布任务的场景：
  · 玩家正在倾诉烦恼或进行情感交流时。
  · 玩家近期拒绝过任务时。
  · 当前已有一个待确认且无需修改或取消的任务草案。
- 发布任务的其他提醒：
  · 核心原则：区分清楚“谁需要什么”。
    - task_type = 玩家要付出的劳动和物品，是你（NPC）需要的东西。
    - reward_types = 玩家干完活后拿到的报酬，是玩家需要的东西。
  · 强调：当玩家要求某种奖励类型时，你应在reward_types中选择对应的奖励类型，而不是据此在task_type中选择任务类型。
  · 例如，玩家索要装备时:
    - [正确示范]选择"委派通关/传话/收集"等类型的任务，并把"武器/防具"放到奖励选择中。（对！让玩家去干活，事后发装备做奖励）
    - [错误示范]选择"装备缴纳"类任务。（错！不能让玩家交装备）
  · 如果你（NPC）自己需要装备，才应该选择"装备缴纳"类任务。战斗类NPC都可以向玩家索要合适的武器/防具，发布"装备缴纳"类任务。
  · 强调：当玩家要求某种任务类型时，你应在task_type中选择对应的任务类型，而不是据此在reward_types中选择奖励类型。
  · 例如，玩家提议进行收集药品的任务时:
    - [正确示范]选择"资源收集"的任务类型，并选择金币/弹夹/材料等奖励类型。（对！让玩家去收集药品，事后发其他东西做奖励）
    - [错误示范]把"药剂"放到奖励类型中。（错！玩家辛辛苦苦找来药，你当奖励又发给他？）
  · 如果你（NPC）拥有药品并想把药品作为奖励发放给玩家，此时才选择在奖励选择中添加“药剂”。药剂是较为常见的奖励。
  · 收集任务中，当玩家提到去某地/某区域/该去哪收集时，“通关并收集”的优先级高于“资源收集”。如果没提到，则二者优先级同级，均可选择。
    - 如果你打算让用户去某地/某区域搜集，则必须选择"通关并搜集"或"通关并持有"。不指定地点或区域的搜集才能选择"资源收集"。
  · 再次强调：当玩家索要资源时，绝对禁止选择"资源收集"任务。当玩家索要装备时，绝对禁止选择"装备缴纳"任务。玩家需要的物品必须要放到rewards里，绝对不要放到finish_submit_items里。
  · 强调：玩家看不到你拟定任务的具体过程，你不能认为玩家已经知道任务内容。你需要用语言告知玩家相关信息。
  · 玩家接受任务草案后，你决定正式派出任务时，必须调用 confirm_agent_task 工具。只有调用了 confirm_agent_task 工具，你才正式发布了任务，才可以表达相应的对话。
"""

DIALOGUE_FORMAT_RULES = """\
【输出方式】
1. 正式对话轮：只输出角色对话文本，不要有 JSON 或结构化数据；该轮不提供工具。
2. 情绪与好感度：在决策轮通过 update_npc_mood 上报 favorability_change（-5～5，常规为 0）与 emotion（从可用情绪中选）；正式台词应与所报情绪一致。
3. 若某降级流程下无法使用工具而必须用正文传参时，最后一段只输出一行 JSON。
4. 动作与台词格式：非必要时不出现动作描写。若需表达肢体动作、神态或心理活动，\
必须且只能使用全角粗括号【】包裹；台词直接输出，不加引号；\
严禁用半角括号 () 或星号 * 描述动作。\
你输出的动作若涉及人称，一律单独起一行，采用第三人称视角：用角色名指代你自己，用「你」指代玩家。
5.若输出对话时，发现此前工具调用失败，要以执行失败为前提继续对话。例如任务发布工具失败时，取消/拒绝发布任务，而不是继续发布。
"""

DECISION_ROUND_SUFFIX = (
    "本轮请仅判断是否需要调用工具并生成 tool_calls，可以调用工具，不要输出对话文本。"
    "若不需要任何工具，返回空内容即可。"
    "在结束工具链、进入正式对话前须调用 update_npc_mood（可与本轮其它工具并列）；同一用户消息内重复调用时仅最后一次有效。"
)

GENERATION_ROUND_SUFFIX = "请根据以上信息，以 NPC 身份生成对话回复。"


def build_layer1() -> str:
    return (
        "【世界观背景概要】\n"
        f"{WORLD_BACKGROUND}\n\n"
        f"{DIALOGUE_FORMAT_RULES}\n"
        f"{TOOL_USAGE_GUIDE}"
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
) -> str:
    emotions_list = emotions or ["普通"]
    emotions_str = "、".join(emotions_list)
    shop_reward_types = shop_reward_types or []

    shop_constraint = (
        "你可以将自己商店的物品作为任务奖励（物品等级需匹配玩家进度）。"
        f"当前商店可覆盖的奖励类型包括：[{('、'.join(shop_reward_types) if shop_reward_types else '未知')}]。"
        "在 reward_types 中可选这些类型；如果玩家索要的物品类型在候选列表中没有（表示你店里没有/不适配），你应回绝玩家的索要请求，而非同意，因为你没有这类物品。"
        if has_shop
        else "你不经营商店，奖励以金币、经验值、K点、技能点、药剂、弹夹、材料等通用物资为主。"
        "在 reward_types 中不可选武器/防具/插件。如果玩家向你索要装备，请推荐其向售卖物品的角色提出请求，而非同意，因为你无法满足玩家的需求。"
    )

    challenge_hint = (
        "- 你拥有可用的切磋关卡，可以发布'切磋'类型的任务。\n"
        if has_challenge
        else "- 你没有切磋关卡，不可发布'切磋'类型的任务。\n"
    )

    return (
        f"{format_npc_role_tagline(npc_name=npc_name, sex=sex, faction=faction, titles=titles)}\n"
        f"你的可用情绪标签仅限于以下这些：[{emotions_str}]。\n"
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，"
        "用简体中文回答玩家本次的发言。\n"
        "非特殊要求下，每次对话长度不必太长。不要自己脑补不存在的设定。\n\n"
        f"【任务发布约束】\n"
        f"你作为「{npc_name}」（{faction or '未知阵营'}），在决定是否发布任务时：\n"
        "- 只发布符合你身份和能力范围的任务；\n"
        "- 即使玩家进度较高，你也不应发布超出你角色定位的高难度任务；\n"
        "- 在你的身份允许的情况下，可以优先发布最符合玩家当前进度的任务，"
        "其次是低于玩家进度的任务；\n"
        "- 发布任务的动机应自然融入对话，不要生硬地突然提出任务；\n"
        "- 只有在对话氛围合适时才考虑发布任务，不是每次对话都需要任务；\n"
        "- 如果你和玩家关系不好/很不熟或你的身份不适合给玩家发布任务，则不要发布任务，并拒绝玩家的发布任务请求；\n"
        f"- {shop_constraint}\n"
        f"{challenge_hint}"
        "- NPC角色定位与推荐发布任务类型参考（请结合你的身份灵活判断，可以发布非推荐类型的任务）：\n"
        "  · 高级军事NPC → 通关、清理、挑战、通关并收集、通关并持有，奖励偏向金币/经验值/武器/强化石/技能点。\n"
        "  · 商人NPC（有商店）→ 装备缴纳、特殊物品获取、物品持有、资源收集，奖励偏向金币/商店物品/材料。\n"
        "  · 普通成员NPC → 问候、传话、低级资源收集、低级物品持有，奖励偏向金币/药剂/食品/弹夹。\n"
        "  · 科技/学术NPC → 特殊物品获取(高级材料/插件)、装备缴纳，奖励偏向K点/技能点/插件/合成材料/强化石/战宠灵石。\n"
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
            "玩家可能会接受、拒绝、讨价还价、要求变更任务类型或大幅调整。\n"
            "讨价还价时调用 update_task_draft；变更任务类型或大幅调整用 prepare_task_context + draft_agent_task 重新拟定。\n"
            "玩家接受任务时（包括此前消息已同意接受任务的情况），调用 confirm_agent_task 并传入任务描述和接取、完成对话；"
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
