"""插话提示词与工具策略接口。

日常真 LLM 路径使用 `build_daily_system_prompt`——复用终端的稳定块
（世界观/说话方式/扮演/NPC 状态/场景/同阵营），但**不**引入终端的 meta 行、
路由轮、说话爆发、过渡语、任务草案块。

任务真 LLM 路径使用 `build_task_system_prompt`：复用同一套稳定前缀，换成任务
纪律、任务 JSONL 输出格式与任务工具说明（只有检索与长期记忆）。

工具说明只写行为要求，不解释工具机制（不出现“后台/Host/参数不会出现在台词里”
这类系统视角描述）；委派只在注册闸放行时出现，且要求“非常必要才调用”。
"""

from __future__ import annotations

from typing import Any, Optional

from services.dialogue_interject.context import build_user_context
from services.dialogue_interject.schemas import InterjectRequest
from services.npc.manager import NPCState
from services.orchestrator.prompts import (
    ANTI_HALLUCINATION_RULE,
    SPEECH_STYLE_RULES,
    build_appearance_block,
    build_challenge_hint,
    build_publish_constraint,
    build_session_state_block,
    build_shop_hint,
    format_npc_role_tagline,
    worldview_block,
)

#: 日常输出的兜底 system（仅在调用方没有传稳定块时使用；B1 真路径用 build_daily_system_prompt）
DAILY_SYSTEM_PROMPT = """【身份】
你是当前对话中的 NPC，只输出角色台词。

【输出格式（最高优先级）】
一行一个 JSON 对象，不要 Markdown、不要代码围栏、不要解释文字。
{"add":"台词","emo":"情绪"}
{"closing":"结束语","emo":"情绪"}

【纪律】
- 单条不超过 135 字；可自行拆成多条。
- 第一句想好就立即输出，不要等整段想完。
- 只允许输出对话内容，禁止任何动作描写或括号内容。
- 原对话剩余内容仅供参考，不要机械续写。
- closing 可留空；为空时最后一条 add 即结束语。
"""

TASK_SYSTEM_PROMPT = """【身份】
你是当前任务对话中的 NPC。玩家插话后，你必须回到任务原有脉络。

【输出格式（最高优先级）】
一行一个 JSON 对象，不要 Markdown、不要代码围栏、不要解释文字。
{"type":"mute"}
{"type":"add","no":0,"text":"新增台词","emo":"情绪"}
{"type":"edit","no":1,"text":"改写台词","emo":"情绪"}
{"type":"keep","no":2}
{"type":"remove","no":3}
{"type":"continue","no":4}
{"type":"end"}

【字段与操作】
- mute：可选，需要禁言时必须作为第一行输出；不输出即默认 false。
- add：在当前位置插入当前 NPC 的新台词，no 固定为 0，可多条，不占原编号。
- edit：用 text 替换原第 no 条；同一 no 可连续输出多条 edit，表示拆成多条。
- keep：原第 no 条保持原文。
- remove：移除原第 no 条。
- continue：从原第 no 条起到组末全部保持原文。
- end：不再保留任何未处理原台词；只有原剩余已全部处理完或本来没有原剩余时使用。
- add、edit 必须填 text 和 emo；其他操作不填 text、emo。
- edit、keep、remove、continue 的 no 必须与【原始剩余台词（按编号）】的方括号编号一致。

【顺序与纪律】
- 从原第 1 条开始按编号递增处理；continue 前每条都必须明确 edit、keep 或 remove，不得跳号、倒序或重复。连续 edit 拆分同一条是唯一例外。
- continue 和 end 必须且只能输出一个，并放在最后一行；没有原剩余时只能用 end。
- 后续能直接接回原文时必须用 continue，不做无意义微调；要保留中间一条并继续改后面时用 keep。
- 只改当前 NPC 的台词；其他 NPC 用 keep 或 continue 保留，不得替他们新增台词。
- 单条不超过 135 字；禁止动作描写、旁白、心理和括号内容。
- 任务目标、奖励、完成条件以提供的任务上下文为准，不得遗漏或编造。
- 只有玩家说出严重离谱、完全无法接话且无法正常回到主线的内容时才输出 mute。禁言时第一条 add 按 NPC 人设选择无视、打断或化解，再继续后续对话。
"""


def build_daily_output_rules(prompt_max: int = 135) -> str:
    """日常 JSONL 输出纪律（协议 5.3 / D-11 / D-13）。"""
    return "\n".join([
        "【输出格式（最高优先级）】",
        "一行一个 JSON 对象，不要 Markdown、不要代码围栏、不要解释文字。",
        '{"add":"台词","emo":"情绪"}',
        '{"closing":"结束语","emo":"情绪"}',
        "字段：",
        "- add：本轮新说的一句台词；可多条，按输出顺序播放。",
        "- closing：可选结束语；没有就留空，最后一条 add 即结束语。",
        "纪律：",
        f"1. 单条不超过 {prompt_max} 字；要说得更多就拆成多条 add。",
        "2. 你是当前 NPC，只输出角色台词，不写旁白、动作、心理或任何括号内容。",
        "3. 想好第一句就立即输出第一行，不要等整段想完。",
        "4. 原对话剩余台词只作参考，不要机械续写；玩家已经切到 AI 聊天。",
        "5. 不要输出说话人名字前缀，说话人由系统决定。",
        "6. 需要调用工具时按工具说明执行；不要在台词里提工具，也不要解释自己是怎么做到的。",
    ])


def build_daily_tool_guide(
    *,
    has_pending_draft: bool = False,
    task_publish_disabled: bool = False,
    delegate_enabled: bool = False,
    player_can_challenge: bool = False,
    item_tools: bool = False,
    has_item_claim: bool = False,
) -> str:
    """日常可用工具说明（协议 7.4 / 实施指引 §5、§10）。

    只写本轮真实注册的工具；不解释工具机制，不用“后台/Host”等系统词。
    任务发布三块能力（发布硬约束/商店报酬/切磋）在 system 稳定区，这里只写工具判据。
    有 pending claim 时物品工具只剩取消，任务工具一律不出现（先处理索取）。
    """
    if has_item_claim:
        return "\n".join([
            "【工具（本轮可用）】",
            "- route_search：玩家问设定、人物、关卡或物品，或你需要查证时调用。"
            "调用工具的那一轮不要输出 add/closing；结果回来后用 JSONL 回答。",
            "- update_long_term_memory：需要记住重要信息时调用；可以在说话的同时调用。",
            "- cancel_item_request：玩家明确拒绝当前索取时调用；调用后正常回话。",
            "【物品往来纪律】",
            "1. 当前有一笔待处理的索取：玩家拒绝就取消；玩家同意就等实际交付，"
            "没收到之前不要承认收到。",
            "2. 口头答应不等于交付；只有东西真正到你手上才算。",
            "3. 玩家声称已经送了，但物品往来记忆没有记录时，不要当成已经收到，"
            "可以请玩家实际交付。",
        ])
    lines = [
        "【工具（本轮可用）】",
        "- route_search：玩家问设定、人物、关卡或物品，或你需要查证时调用。"
        "调用工具的那一轮不要输出 add/closing；结果回来后用 JSONL 回答。",
        "- update_long_term_memory：需要记住重要信息时调用；可以在说话的同时调用。",
    ]
    if item_tools:
        lines.extend([
            "- give_item_to_player：关系不错、想答谢或讨好玩家、手头宽裕等情景下，"
            "可以送玩家东西；不想送就拒绝，好感度低时更倾向于拒绝；"
            "拒绝就不调用工具，直接以不赠送为前提回复。玩家向你要东西（“给我个药”“送我件装备”）时，"
            "你决定给就调用；只有办理结果确认已交付，才能说给了。"
            "调用这一轮只发工具调用，不要输出台词。",
            "- request_item_from_player：通常要东西应走委托；只有玩家主动要给你且不要报酬、"
            "玩家声称已经送了、你想要但不想付报酬、或只是价值不高的小东西时，才调用。"
            "玩家说要送你东西（“这个给你”“拿着”）时，用它核实接收；"
            "只有办理结果确认收到，才能承认收到。"
            "调用这一轮只发工具调用，不要输出台词。",
            "【物品往来纪律】",
            "1. 玩家向你要东西：给不给由你决定；拒绝也要承认玩家开过口，只说明你这边没给成。",
            "2. 玩家说要送你东西：必须调用 request_item_from_player 完成真实交接；"
            "没收到就说没收到。",
            "3. 口头答应不等于交付；只有东西真正到你手上才算。",
            "4. 玩家声称已经送了，但物品往来记忆没有记录时，不要当成已经收到，"
            "可以请玩家实际交付。",
            "5. 赠送要看关系、财力和当下情景；不合适的索取或赠送要推辞。",
        ])
    if not task_publish_disabled:
        draft_rule = (
            "- route_task_draft：玩家明确索要委托/工作/差事，或按你的身份和当下情景你决定主动派活时调用。"
        )
        if player_can_challenge:
            draft_rule += "玩家提出想跟你切磋、对战、比划或直接动手，也走这个入口。"
        draft_rule += "调用这一轮只发工具调用，不要输出任何 add/closing；草案拟好后，再用 JSONL 说明委托。"
        lines.append(draft_rule)
        if has_pending_draft:
            lines.append(
                "- route_task_update / route_task_confirm / route_task_cancel："
                "针对当前待确认草案的讨价还价/接受/拒绝。调用轮不要输出台词。"
            )
        lines.extend([
            "【任务草案纪律】",
            "1. 触发草案时不要输出“我去准备一下”之类的过渡语。",
            "2. 草案拟好后，只输出最终确认对话（add）和结束语（closing）；"
            "结束语表达“不接就算了/这活就搁着”，不要求固定措辞。",
            "3. 草案只是拟定，不是已发布；玩家点头前不要说“任务已发布”。",
            "4. 确认成功后可以说已交给玩家；任务内容/要求/报酬以草案为准，不要编造数字。",
            "5. 一轮最多只派一件事；拿不准该不该派时，就正常说话，不要调用工具。",
        ])
    else:
        lines.append("- 你当前不能发布委托；玩家索要时以角色口吻回应或婉拒。")
    if delegate_enabled:
        lines.append(
            "- delegate_dialogue_rewrite：只有非常必要时才调用（你的长期立场或以后说话方式"
            "会因此改变时才用）；调用那一轮不要输出 add/closing；不需要就绝不调用。"
        )
    return "\n".join(lines)


def build_daily_system_prompt(
    *,
    npc_name: str,
    state: NPCState,
    scene_block: str = "",
    same_faction_block: str = "",
    save_info: Any = None,
    progress_desc: str = "",
    prompt_max: int = 135,
    equipment_block: str = "",
    has_pending_draft: bool = False,
    task_publish_disabled: bool = False,
    delegate_enabled: bool = False,
    has_shop: bool = False,
    shop_reward_types: Optional[list[str]] = None,
    player_can_challenge: Optional[bool] = None,
    item_tools: bool = False,
    has_item_claim: bool = False,
    include_tool_guide: bool = True,
) -> str:
    """日常插话 system：稳定块在前，随草案/好感度变化的状态块在后（前缀缓存友好）。

    任务发布能力与聊天主 Agent 共用同一份文本（`build_publish_constraint` /
    `build_shop_hint` / `build_challenge_hint`）——插话同样能派活，所以「给不给得出
    什么报酬」「能不能约切磋」这两条边界必须一致；任务发布关闭时不注入。
    """
    emotions = "、".join(state.emotions or ["普通"])
    tagline = format_npc_role_tagline(
        npc_name=npc_name, sex=state.sex or "", faction=state.faction or "", titles=state.titles,
    )
    parts: list[str] = [
        worldview_block(),
        build_daily_output_rules(prompt_max),
        SPEECH_STYLE_RULES,
        tagline,
        f"你的可用情绪标签仅限于以下这些：[{emotions}]。每条输出的 emo 必须从中选择最合适的一种。",
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，用简体中文说话。",
        ANTI_HALLUCINATION_RULE,
    ]
    # 层2：单 NPC 恒定项（发布硬约束、商店报酬）→ 稀疏项（切磋可行性随玩家等级变）
    if not task_publish_disabled:
        parts.append(build_publish_constraint(
            npc_name, state.faction or "", mention_system=False,
        ))
        parts.append(build_shop_hint(has_shop, shop_reward_types))
        parts.append(build_challenge_hint(bool(state.challenge), player_can_challenge))
    if include_tool_guide:
        parts.append(build_daily_tool_guide(
            has_pending_draft=has_pending_draft,
            task_publish_disabled=task_publish_disabled,
            delegate_enabled=delegate_enabled,
            player_can_challenge=bool(player_can_challenge),
            item_tools=item_tools,
            has_item_claim=has_item_claim,
        ))
    parts.append(build_session_state_block(
        progress_desc=progress_desc,
        favorability=state.favorability,
        relationship_level=state.relationship_level,
        save_info=save_info,
    ))
    scene = (scene_block or "").strip()
    if scene:
        parts.append(scene)
    appearance = build_appearance_block(state)
    if appearance:
        parts.append(appearance)
    equipment = (equipment_block or "").strip()
    if equipment:
        parts.append(equipment)
    same = (same_faction_block or "").strip()
    if same:
        parts.append(f"【同阵营角色】\n{same}")
    return "\n\n".join(parts)


def build_task_output_rules(prompt_max: int = 135) -> str:
    """任务 JSONL 输出格式：统一 type/no/text，continue/end 互斥收尾。"""
    return "\n".join([
        "【输出格式（最高优先级）】",
        "一行一个 JSON 对象，不要 Markdown、不要代码围栏、不要解释文字。",
        "只允许以下七种操作：",
        '{"type":"mute"}',
        '{"type":"add","no":0,"text":"新增台词","emo":"情绪"}',
        '{"type":"edit","no":1,"text":"改写台词","emo":"情绪"}',
        '{"type":"keep","no":2}',
        '{"type":"remove","no":3}',
        '{"type":"continue","no":4}',
        '{"type":"end"}',
        "字段要求：",
        "- type：必填，只能是 mute、add、edit、keep、remove、continue、end。",
        "- no：add 固定填 0；edit、keep、remove、continue 必须填原剩余台词的编号，"
        "与【原始剩余台词（按编号）】里的方括号编号完全一致；end、mute 不填 no。",
        "- text：add、edit 必填；keep、remove、continue、end、mute 不填 text。",
        "- emo：add、edit 必填；其他操作不填 emo。",
        "操作含义：",
        "- add：在当前位置插入一条当前 NPC 的新台词，可放在任意两个操作之间，可多条；不占原编号。",
        "- edit：用 text 替换原第 no 条；同一个 no 可连续输出多条 edit，表示把这一条拆成多条。",
        "- keep：原第 no 条保持原文。",
        "- remove：移除原第 no 条。",
        "- continue：从原第 no 条开始，直到组末全部保持原文；这是收尾操作。",
        "- end：本组到此结束，不再保留任何未处理的原台词；只有原剩余已全部逐条处理完、"
        "或本轮本来就没有原剩余时才使用。",
        "- mute：可选，表示本组后续不再让玩家说话；需要时必须作为整个输出的第一行，"
        "不输出此操作即默认 mute=false。",
        "输出顺序：",
        "1. 从原第 1 条开始，按编号递增处理；continue 之前的每条原台词都必须明确用 "
        "edit、keep 或 remove 处理，不得跳号、倒序或重复。连续 edit 拆分同一条是唯一例外。",
        "2. continue 和 end 必须且只能输出其中一个，并放在最后一行；输出后立即停止，"
        "后面不得再有任何操作。没有原剩余时只能用 end。",
        "3. 示例（假设原剩余共 4 条：保留第 1 条、改写第 2 条、移除第 3 条、"
        "第 4 条及其后保持原文）：",
        '{"type":"add","no":0,"text":"你先听我说。","emo":"普通"}',
        '{"type":"keep","no":1}',
        '{"type":"edit","no":2,"text":"改写的第 2 条","emo":"普通"}',
        '{"type":"remove","no":3}',
        '{"type":"continue","no":4}',
        "台词纪律：",
        f"1. 单条不超过 {prompt_max} 字；要说得更多就拆成多条 add，或用连续 edit 拆分同一条。",
        "2. 你是当前 NPC，只输出角色台词，不写旁白、动作、心理或任何括号内容。",
        "3. 想好第一句就立即输出第一行，不要等整段想完。",
        "4. 不要输出说话人名字前缀，说话人由系统决定。",
        "5. 需要调用工具时按工具说明执行；不要在台词里提工具，也不要解释自己是怎么做到的。",
    ])


def build_task_discipline(npc_name: str) -> str:
    """任务对话内容纪律：只写行为要求，不解释宿主或后端机制。"""
    return "\n".join([
        "【任务对话纪律】",
        "1. 无论玩家说什么，最后都必须回到当前任务对话的原有脉络，不能聊开就不管后续。",
        f"2. 只改{npc_name}自己的台词；其他 NPC 的台词用 keep，或直接交给 continue 保留，"
        "不得替其他 NPC 新增台词。",
        "3. 想保留中间某条、但还要修改更后面的台词时，中间那条必须用 keep；"
        "只有从某条起到组末都不用改时，才能用 continue 提前收尾。",
        "4. 后续内容能直接接回原文时必须用 continue 提前结束，"
        "不要对无需修改的后续对话做没有实际意义的微调；continue 只按原剩余编号计，"
        "不把本轮 add 计入。",
        "5. 如果没有编辑，且下一条无法直接接上玩家的对话时，建议先 add 一条来回应玩家。",
        "6. 任务目标、奖励、完成条件以【任务上下文】为准；改写时不得漏掉原句里的任务信息"
        "（目标/地点/数量/报酬），也不要编造。",
        "7. 只有玩家说出严重离谱、完全无法接话且无法正常回到任务主线的内容时，"
        "才在最开头输出一次 mute；其他情况一律不输出，保持默认 false。",
        "8. 禁言时，第一条 add 必须在保持 NPC 人设和身份的前提下，选择无视、打断或化解"
        "玩家的离谱内容，再按正常顺序处理原剩余、回到后续对话并用 continue 或 end 收尾。",
    ])


def build_task_tool_guide() -> str:
    """任务可用工具说明（协议 4.1 / D-14）：只有检索与长期记忆。"""
    return "\n".join([
        "【工具（本轮可用）】",
        "- route_search：玩家问设定、人物、关卡或物品，或你需要查证任务细节时调用。"
        "调用工具的那一轮不要输出任何任务 JSONL 操作；结果回来后用 JSONL 回答。",
        "- update_long_term_memory：需要记住重要信息时调用；可以在说话的同时调用。",
    ])


def build_task_item_rules() -> str:
    """任务插话的物品推辞口径（协议 §7.5.1）：只写行为，不提工具名。"""
    return "\n".join([
        "【物品往来】",
        "玩家要送你东西、向你讨东西，或问你能不能给时，按情景推辞："
        "任务紧急就说任务紧急、先处理要事；事情不急就说等忙完再说。",
        "玩家声称已经送过东西，不要当真。",
    ])


def build_task_system_prompt(
    *,
    npc_name: str,
    state: NPCState,
    scene_block: str = "",
    same_faction_block: str = "",
    save_info: Any = None,
    progress_desc: str = "",
    prompt_max: int = 135,
    equipment_block: str = "",
    include_tool_guide: bool = True,
) -> str:
    """任务插话 system：复用日常稳定前缀，换成任务纪律与任务工具说明。"""
    emotions = "、".join(state.emotions or ["普通"])
    tagline = format_npc_role_tagline(
        npc_name=npc_name, sex=state.sex or "", faction=state.faction or "", titles=state.titles,
    )
    parts: list[str] = [
        worldview_block(),
        build_task_output_rules(prompt_max),
        build_task_discipline(npc_name),
        SPEECH_STYLE_RULES,
        tagline,
        f"你的可用情绪标签仅限于以下这些：[{emotions}]。每条 add、edit 的 emo 都必须从中选择最合适的一种。",
        "请始终以符合该角色身份、口吻、记忆、立场、当前好感度和所选情绪的语气，用简体中文说话。",
        ANTI_HALLUCINATION_RULE,
    ]
    if include_tool_guide:
        parts.append(build_task_tool_guide())
    parts.extend([
        build_task_item_rules(),
        build_session_state_block(
            progress_desc=progress_desc,
            favorability=state.favorability,
            relationship_level=state.relationship_level,
            save_info=save_info,
        ),
    ])
    scene = (scene_block or "").strip()
    if scene:
        parts.append(scene)
    appearance = build_appearance_block(state)
    if appearance:
        parts.append(appearance)
    equipment = (equipment_block or "").strip()
    if equipment:
        parts.append(equipment)
    same = (same_faction_block or "").strip()
    if same:
        parts.append(f"【同阵营角色】\n{same}")
    return "\n\n".join(parts)


def build_system_prompt(mode: str) -> str:
    """按模式返回兜底 system（仅在调用方没有传稳定块时使用）。"""
    return TASK_SYSTEM_PROMPT if mode == "task" else DAILY_SYSTEM_PROMPT


def build_messages(
    request: InterjectRequest,
    *,
    daily_system: str | None = None,
    rag_context_text: str = "",
    long_term_block: str = "",
    pending_draft_summary: str = "",
    item_memory_block: str = "",
    gift_detail_block: str = "",
    item_recent_block: str = "",
    pending_claim_block: str = "",
) -> list[dict[str, str]]:
    """组装本轮 messages：稳定 system + 现场 user（session 历史不参与）。"""
    system = daily_system or build_system_prompt(request.mode)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_user_context(
                request,
                rag_context_text=rag_context_text,
                long_term_block=long_term_block,
                pending_draft_summary=pending_draft_summary,
                item_memory_block=item_memory_block,
                gift_detail_block=gift_detail_block,
                item_recent_block=item_recent_block,
                pending_claim_block=pending_claim_block,
            ),
        },
    ]


def build_daily_messages(request: InterjectRequest, blocks: Any) -> list[dict[str, str]]:
    """B2 真路径：system 稳定块 + user 现场（RAG/长期记忆/草案摘要）。"""
    return build_messages(
        request,
        daily_system=blocks.system_prompt,
        rag_context_text=getattr(blocks, "rag_context_text", ""),
        long_term_block=getattr(blocks, "long_term_block", ""),
        pending_draft_summary=getattr(blocks, "pending_draft_summary", ""),
        item_memory_block=getattr(blocks, "item_memory_block", ""),
        gift_detail_block=getattr(blocks, "gift_detail_block", ""),
        item_recent_block=getattr(blocks, "item_recent_block", ""),
        pending_claim_block=getattr(blocks, "pending_claim_block", ""),
    )


def build_task_messages(request: InterjectRequest, blocks: Any) -> list[dict[str, str]]:
    """P4 任务路径：system 任务块 + user 现场（RAG/长期记忆/任务上下文/played/remaining）。

    任务模式不拼草案摘要与组内物品块；跨对话物品账本进入 user 现场。
    """
    return build_messages(
        request,
        daily_system=blocks.system_prompt,
        rag_context_text=getattr(blocks, "rag_context_text", ""),
        long_term_block=getattr(blocks, "long_term_block", ""),
        item_memory_block=getattr(blocks, "item_memory_block", ""),
    )


def tool_matrix_for(
    mode: str,
    *,
    has_draft: bool = False,
    delegate_enabled: bool = False,
    item_tools: bool = False,
    has_item_claim: bool = False,
) -> tuple[str, ...]:
    """工具策略接口（协议 §7.4 / §7.5.2）。

    日常：查询/检索、长期记忆、任务草案及草案存在时的 confirm/cancel/update；
    无 claim 且无草案时注册赠送/索取，有 claim 时只留取消索取；
    委派工具只在注册闸放行（有存档、无并发改写）时加入。
    任务：只有查询/检索与长期记忆；无任务类工具、无物品工具、无委派（协议 4.1）。
    """
    if mode == "task":
        return ("route_search", "update_long_term_memory")
    if mode != "daily":
        return ()
    if has_item_claim:
        return ("route_search", "update_long_term_memory", "cancel_item_request")
    names = ["route_task_draft", "route_search", "update_long_term_memory"]
    if has_draft:
        names.extend(["route_task_confirm", "route_task_cancel", "route_task_update"])
    elif item_tools:
        names.extend(["give_item_to_player", "request_item_from_player"])
    if delegate_enabled:
        names.append("delegate_dialogue_rewrite")
    return tuple(names)
