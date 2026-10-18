"""插话提示词与工具策略接口。

B1：日常真 LLM 最小路径使用 `build_daily_system_prompt`——复用终端的稳定块
（世界观/说话方式/扮演/NPC 状态/场景/同阵营），但**不**引入终端的 meta 行、
路由轮、说话爆发、过渡语、任务草案块。

工具：B1 不向模型暴露任何工具（`tool_matrix_for` 返回空表），P2 再按协议 §7.4
接入查询/记忆/草案；本轮避免半成品任务与委派链。
"""

from __future__ import annotations

from typing import Any

from services.dialogue_interject.context import build_user_context
from services.dialogue_interject.schemas import InterjectRequest
from services.npc.manager import NPCState
from services.orchestrator.prompts import (
    ANTI_HALLUCINATION_RULE,
    SPEECH_STYLE_RULES,
    build_appearance_block,
    build_session_state_block,
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
{"mute":true}
{"add":"台词","emo":"情绪"}
{"n":1,"nochange":true}
{"n":2,"line":"改写台词","emo":"情绪"}
{"n":3,"remove":true}
{"end":4}

【纪律】
- 新增排最前；按原始编号从前往后检查；不需要改就 nochange 或直接 end。
- 能接回原文时必须用 end；end 不输出＝整组替换。
- 只改当前 NPC；其他 NPC 一般 nochange 或交给 end。
- 单条不超过 135 字；禁止动作描写/括号内容。
- 信息不能漏；任务目标/奖励以提供的任务上下文为准。
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
        "6. 本轮没有可用工具；不要输出工具调用或 JSON 之外的任何结构。",
        "7. 本轮只能对话：玩家索要委托/工作时以角色口吻回应或婉拒，不要声称已经安排或发布。",
    ])


def build_daily_system_prompt(
    *,
    npc_name: str,
    state: NPCState,
    scene_block: str = "",
    same_faction_block: str = "",
    save_info: Any = None,
    progress_desc: str = "",
    prompt_max: int = 135,
) -> str:
    """日常插话 system：稳定块在前，随好感度变化的状态块在后（前缀缓存友好）。"""
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
        build_session_state_block(
            progress_desc=progress_desc,
            favorability=state.favorability,
            relationship_level=state.relationship_level,
            save_info=save_info,
        ),
    ]
    scene = (scene_block or "").strip()
    if scene:
        parts.append(scene)
    appearance = build_appearance_block(state)
    if appearance:
        parts.append(appearance)
    same = (same_faction_block or "").strip()
    if same:
        parts.append(f"【同阵营角色】\n{same}")
    return "\n\n".join(parts)


def build_system_prompt(mode: str) -> str:
    """按模式返回兜底 system（任务模式 B1 不接真模型）。"""
    return TASK_SYSTEM_PROMPT if mode == "task" else DAILY_SYSTEM_PROMPT


def build_messages(
    request: InterjectRequest, *, daily_system: str | None = None,
) -> list[dict[str, str]]:
    """组装本轮 messages：稳定 system + 现场 user（session 历史不参与）。"""
    system = daily_system or build_system_prompt(request.mode)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_context(request)},
    ]


def tool_matrix_for(mode: str, *, has_draft: bool = False) -> tuple[str, ...]:
    """工具策略接口（协议 §7.4）。

    B1 不向模型暴露任何工具：查询/记忆/草案/委派都留到 P2 按矩阵接入。
    保留函数签名，调用方不必在本轮感知“工具尚未接线”。
    """
    return ()
