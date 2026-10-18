"""插话提示词骨架与工具策略接口（P0）。

P0 只立接口与最小输出纪律；P1/P2/P4 再按实施指引 §6 填实日常/任务两套提示词、
接入真实工具注册表。这里不引用任何工具实现，避免 P0 与工具链耦合。
"""

from __future__ import annotations

from services.dialogue_interject.context import build_user_context
from services.dialogue_interject.schemas import InterjectRequest

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


def build_system_prompt(mode: str) -> str:
    """按模式返回 system 提示词（P0 骨架）。"""
    return TASK_SYSTEM_PROMPT if mode == "task" else DAILY_SYSTEM_PROMPT


def build_messages(request: InterjectRequest) -> list[dict[str, str]]:
    """组装本轮 messages：稳定 system + 现场 user（session 历史不参与）。"""
    return [
        {"role": "system", "content": build_system_prompt(request.mode)},
        {"role": "user", "content": build_user_context(request)},
    ]


def tool_matrix_for(mode: str, *, has_draft: bool = False) -> tuple[str, ...]:
    """工具策略接口（协议 §7.4）。

    P0 只返回白名单名字，不解析工具注册表；P1 再接真实注册与执行。
    - 日常：终端全量（含任务草案/确认/取消/改案、查询、记忆、对话改写委派）。
    - 任务：查询/检索 + 长期记忆写入；任务类工具与委派一律禁止。
    """
    if mode == "task":
        return ("route_search", "update_long_term_memory")
    tools = ["route_task_draft", "route_search", "update_long_term_memory", "delegate_dialogue_rewrite"]
    if has_draft:
        tools.extend(["confirm_agent_task", "cancel_agent_task", "update_task_draft"])
    return tuple(tools)
