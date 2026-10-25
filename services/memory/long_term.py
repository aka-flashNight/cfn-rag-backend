"""跨对话长期记忆（B3-2/B3-3/B3-4）：作用域解析、注入块、聊天侧调整工具。

与滚动摘要是两套东西：摘要按会话存、只总结对话内容；长期记忆按 (作用域, NPC) 存，
跨会话与跨对话窗口保留。每个作用域一条基础长期记忆（NPC 对玩家的整体印象）＋
至多 LTM_MAX_FACT_ENTRIES 条关键记录（重要事件或信息，带作用域内编号）。
作用域＝当前绑定的存档槽位键；未绑定存档时用 STANDALONE_SCOPE，与各存档互不混淆。

聊天侧永远只有一个工具名 update_long_term_memory，其 schema 与描述按作用域当前状态三选一：
- 已有基础印象、关键记录未满：编号可选（不填＝新记一条，填了＝改写该编号那条）；
- 尚无基础印象：必须同时给出基础印象与一条关键记录；
- 关键记录已满：只接受编号＋改写后的内容。
工具不进全局 ToolRegistry（注册表的 description/parameters_schema 是类属性字面量，
做不到同名按状态换形），由编排层按状态手拼 tool dict 并在回合内直接执行。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from services.integration.state import get_integration_state
from services.memory.store import (
    LTM_KIND_BASE,
    LTM_KIND_FACT,
    LTM_MAX_CHARS,
    LTM_MAX_FACT_ENTRIES,
    STANDALONE_SCOPE,
    LongTermEntry,
    MemoryStore,
)

logger = logging.getLogger(__name__)

#: 聊天主 Agent 侧的长期记忆工具名（三形态共用一个名字）
CHAT_TOOL_NAME = "update_long_term_memory"

#: 滚动摘要轮侧的基础印象工具名（只在摘要那次调用里下发）
SUMMARY_TOOL_NAME = "set_base_long_term_memory"

#: 提示词里给的期望长度（入库上限统一是 LTM_MAX_CHARS，超出截断）
#: 首次写入只给 100 字：基础印象之后还要靠摘要轮逐次补全，一开始写满会挤掉后续空间。
BASE_EXPECT_CHARS = 300
BASE_FIRST_EXPECT_CHARS = 100
FACT_EXPECT_CHARS = 100


# ---------------------------------------------------------------------------
# 作用域与开关
# ---------------------------------------------------------------------------

def scope_key(slot_key: str | None) -> str:
    """长期记忆作用域键：绑定存档＝槽位键；未绑定＝STANDALONE_SCOPE（独立一份）。"""
    key = str(slot_key or "").strip()
    return key or STANDALONE_SCOPE


def resolve_enabled(requested: bool | None) -> bool:
    """本回合是否启用长期记忆。

    游戏内（embedded 口径：--embedded 启动或已被 bind）恒定启用，不受理关闭；
    非游戏打开时按前端每次请求带的开关值，缺省＝关闭。
    """
    if get_integration_state().mode == "embedded":
        return True
    return bool(requested)


def normalize_content(value: Any) -> str:
    """入库前归一：去首尾空白，超过 LTM_MAX_CHARS 截断。"""
    text = str(value or "").strip()
    return text[:LTM_MAX_CHARS]


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

@dataclass
class LongTermSnapshot:
    """一个作用域内某 NPC 的全部长期记忆（一轮开始时读一次，轮内复用）。"""

    slot_key: str
    npc: str
    base: str = ""
    facts: list[LongTermEntry] = field(default_factory=list)

    @property
    def fact_count(self) -> int:
        return len(self.facts)

    @property
    def is_full(self) -> bool:
        return self.fact_count >= LTM_MAX_FACT_ENTRIES

    @property
    def is_empty(self) -> bool:
        return not self.base and not self.facts

    def fact_by_no(self, entry_no: int) -> LongTermEntry | None:
        return next((f for f in self.facts if f.entry_no == entry_no), None)


async def load_snapshot(store: MemoryStore, *, slot_key: str | None, npc: str) -> LongTermSnapshot:
    """读作用域内某 NPC 的长期记忆（基础印象 + 全部关键记录）。"""
    scope = scope_key(slot_key)
    npc_name = (npc or "").strip()
    if not npc_name:
        return LongTermSnapshot(slot_key=scope, npc="")
    rows = await store.ltm_list(scope, npc_name)
    base = ""
    facts: list[LongTermEntry] = []
    for row in rows:
        if row.kind == LTM_KIND_BASE:
            base = row.content
        elif row.kind == LTM_KIND_FACT:
            facts.append(row)
    return LongTermSnapshot(slot_key=scope, npc=npc_name, base=base, facts=facts)


# ---------------------------------------------------------------------------
# 提示词块
# ---------------------------------------------------------------------------

def format_block(snapshot: LongTermSnapshot) -> str:
    """聊天主 Agent 的长期记忆注入块；作用域内一条都没有时整块不注入。"""
    if snapshot.is_empty:
        return ""
    parts = [
        "【长期记忆（跨对话窗口保留）】",
        "你在以往对话里对这个玩家积累的记忆：以基础印象为准把握你们当前的关系，"
        "关键记录用于避免把同一件事重复记录。",
    ]
    if snapshot.base:
        parts.append(f"基础印象：{snapshot.base}")
    if snapshot.facts:
        parts.append("关键记录（编号在调整时引用）：")
        parts.extend(f"{f.entry_no}. {f.content}" for f in snapshot.facts)
    return "\n".join(parts)


def format_summary_block(snapshot: LongTermSnapshot) -> str:
    """滚动摘要轮的长期记忆独立块（含与摘要的隔离说明）；空则不注入。"""
    if snapshot.is_empty:
        return ""
    parts = [
        "【长期记忆（跨对话窗口保留）】",
        "以下是这个 NPC 在以往对话窗口里积累下来的长期记忆，"
        "与本次要撰写的对话摘要是两套独立的内容。",
    ]
    if snapshot.base:
        parts.append(f"基础印象：{snapshot.base}")
    if snapshot.facts:
        parts.append("关键记录：")
        parts.extend(f"{f.entry_no}. {f.content}" for f in snapshot.facts)
    parts.append(
        "长期记忆不参与对话摘要的撰写：摘要只总结本次给出的对话记录内容，"
        "不要把长期记忆里的信息写进摘要。"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 工具（手拼 OpenAI tool dict）
# ---------------------------------------------------------------------------

def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# 三形态共用的说话要求。措辞刻意对着路由轮的「调用路由工具的那一轮……只发工具调用」
# 反着写：模型会把那条规则套到本工具上，发完调用就一个字台词都不给，玩家侧只剩空气泡。
CHAT_TOOL_SPEECH_REQUIREMENT = (
    "调用本工具的那一轮：第一行 meta JSON 之后照常输出台词正文，"
    "绝不能只发这个调用、不写台词。"
)


def build_chat_tool(snapshot: LongTermSnapshot) -> dict[str, Any]:
    """按作用域当前状态给出聊天侧工具（三形态，同名不同形）。"""
    if not snapshot.base:
        return _tool(
            CHAT_TOOL_NAME,
            (
                f"{CHAT_TOOL_SPEECH_REQUIREMENT}"
                "记录你对这个玩家的长期记忆（跨对话窗口保留）。调用时必须同时给出两项："
                "base_content＝玩家与你的整体关系情况、你对玩家的整体印象的总结式状态描述"
                "（必须精简，如果当前对话中有效信息较少，可以只输出一句简短的总结，例如15字以内。"
                f"若关键信息较多，可视情况增加。总字数不得超过{BASE_FIRST_EXPECT_CHARS} 字）；"
                "content＝一条关键记录，只记重要、重大、影响明显的"
                f"事件或信息，日常寒暄与随口闲聊不记，{FACT_EXPECT_CHARS} 字以内，只写结论性事实。"
            ),
            {
                "base_content": {
                    "type": "string",
                    "description": "玩家与你的整体关系情况、你对玩家的整体印象（总结式状态描述）",
                },
                "content": {
                    "type": "string",
                    "description": "要记录的关键事件或信息（重要、重大、影响明显）",
                },
            },
            ["base_content", "content"],
        )

    if snapshot.is_full:
        return _tool(
            CHAT_TOOL_NAME,
            (
                f"{CHAT_TOOL_SPEECH_REQUIREMENT}"
                "维护你对这个玩家的长期记忆（跨对话窗口保留）。长期记忆分两部分：一条基础印象"
                "（玩家与你的整体关系情况和你对玩家的整体印象），以及 "
                f"{LTM_MAX_FACT_ENTRIES} 条关键记录（重要、重大、影响明显的事件或信息），"
                "编号见提示词中的长期记忆块。调用本工具改写其中一条关键记录：entry_no 指定编号，"
                "content 给出改写后的完整内容。要记录的信息与某条既有记录相关时融合进那条；"
                "确实比现有某条更重要时，改写那条不重要的记录。"
                f"每条关键记录 {FACT_EXPECT_CHARS} 字以内，只写结论性事实。"
            ),
            {
                "entry_no": {
                    "type": "integer",
                    "description": "要改写的关键记录编号（取自长期记忆块）",
                },
                "content": {
                    "type": "string",
                    "description": "改写后的完整内容",
                },
            },
            ["entry_no", "content"],
        )

    return _tool(
        CHAT_TOOL_NAME,
        (
            f"{CHAT_TOOL_SPEECH_REQUIREMENT}"
            "维护你对这个玩家的长期记忆（跨对话窗口保留）。长期记忆分两部分：一条基础印象"
            "（玩家与你的整体关系情况和你对玩家的整体印象），以及若干条关键记录"
            "（重要、重大、影响明显的事件或信息），编号见提示词中的长期记忆块。"
            "日常寒暄、随口闲聊、普通小委托都不记"
            "（除非是陌生状态下的首次任务，可记录一次。"
            "已有任何一条任务/委托的长期记忆后，不再记忆不重要小任务，"
            "只可记录关键、重要、有长期影响的任务）。"
            "要记录的信息与某条既有关键记录相关时，"
            "填 entry_no 把新信息融合进那条记录、content 给出融合后的完整内容，不要另记一条；"
            "与任何既有记录都无关时才留空 entry_no 新记一条。"
            f"每条关键记录 {FACT_EXPECT_CHARS} 字以内，只写结论性事实。"
        ),
        {
            "entry_no": {
                "type": "integer",
                "description": "要改写的既有关键记录编号；留空表示新记一条",
            },
            "content": {
                "type": "string",
                "description": "关键记录内容（改写时为融合后的完整内容）",
            },
        },
        ["content"],
    )


def build_summary_tool(snapshot: LongTermSnapshot) -> dict[str, Any]:
    """滚动摘要轮的基础印象工具：已有基础印象时只在明显变化才调用（无变化＝不发调用）。"""
    if not snapshot.base:
        return _tool(
            SUMMARY_TOOL_NAME,
            (
                "记录你对这个玩家的基础印象——跨对话窗口保留的一条总结式状态描述，"
                "只写玩家与你的整体关系情况和你对玩家的整体印象。"
                "根据本次给出的对话内容写，当可写的信息少时就写得短，例如只有简短的一句话；"
                f"有效关键信息较多时可适当增多，最大{BASE_FIRST_EXPECT_CHARS} 字以内。"
            ),
            {
                "content": {
                    "type": "string",
                    "description": "玩家与你的整体关系情况、你对玩家的整体印象（总结式状态描述）",
                },
            },
            ["content"],
        )
    return _tool(
        SUMMARY_TOOL_NAME,
        (
            "更新你对这个玩家的基础印象——跨对话窗口保留的一条总结式状态描述，只写玩家与你的"
            f"整体关系情况和你对玩家的整体印象（{BASE_EXPECT_CHARS} 字以内）。"
            "仅当近期对话让整体关系或整体印象发生明显变化时调用，没有明显变化就不调用。"
            "调用时给出更新后的完整内容：与现有内容不冲突的部分必须完整保留，"
            "已过时的部分按近期对话更新，只需补充时做最小化补充，"
            "关键记录里已经记着的事不复述。"
        ),
        {
            "content": {
                "type": "string",
                "description": "更新后的完整基础印象",
            },
        },
        ["content"],
    )


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def parse_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    """取工具调用参数（arguments 是 JSON 字符串；畸形按空 dict）。"""
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    raw = func.get("arguments") if isinstance(func, dict) else None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("长期记忆工具参数不是合法 JSON，忽略本次调用: %.120s", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_call_name(tool_call: dict[str, Any]) -> str:
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    return str(func.get("name") or "") if isinstance(func, dict) else ""


async def apply_chat_tool(
    store: MemoryStore,
    tool_call: dict[str, Any],
    *,
    snapshot: LongTermSnapshot,
    session_id: str,
) -> str | None:
    """执行聊天侧长期记忆工具调用；返回落库结果描述（失败/无效＝None）。"""
    if tool_call_name(tool_call) != CHAT_TOOL_NAME:
        return None
    args = parse_arguments(tool_call)
    content = normalize_content(args.get("content"))
    base_content = normalize_content(args.get("base_content"))

    if base_content:
        await store.ltm_save_base(snapshot.slot_key, snapshot.npc, base_content, session_id)
        snapshot.base = base_content

    if not content:
        return "已写入基础印象" if base_content else None

    raw_no = args.get("entry_no")
    entry_no: int | None = None
    if raw_no is not None and str(raw_no).strip() != "":
        try:
            entry_no = int(raw_no)
        except (TypeError, ValueError):
            entry_no = None

    if entry_no is not None:
        updated = await store.ltm_update_fact(
            snapshot.slot_key, snapshot.npc, entry_no, content, session_id,
        )
        if updated is None:
            logger.warning(
                "长期记忆调整落空：作用域 %s / NPC %s 没有编号 %d 的关键记录",
                snapshot.slot_key, snapshot.npc, entry_no,
            )
            return None
        target = snapshot.fact_by_no(entry_no)
        if target is not None:
            target.content = content
        return f"已改写关键记录 {entry_no}"

    if snapshot.is_full:
        logger.warning(
            "长期记忆关键记录已满（%d 条），忽略本次无编号调用：作用域 %s / NPC %s",
            snapshot.fact_count, snapshot.slot_key, snapshot.npc,
        )
        return None

    entry = await store.ltm_append_fact(snapshot.slot_key, snapshot.npc, content, session_id)
    # 快照就地跟上：同一轮里模型连发两次调用时，第二次要看到最新的条数与编号
    snapshot.facts.append(entry)
    return f"已记录关键记录 {entry.entry_no}"


async def apply_summary_tool(
    store: MemoryStore,
    tool_calls: list[dict[str, Any]],
    *,
    slot_key: str,
    npc: str,
    session_id: str,
) -> bool:
    """执行摘要轮的基础印象更新；没有调用或内容为空＝False（基础印象保持原样）。"""
    for tool_call in tool_calls or []:
        if tool_call_name(tool_call) != SUMMARY_TOOL_NAME:
            continue
        content = normalize_content(parse_arguments(tool_call).get("content"))
        if not content:
            continue
        await store.ltm_save_base(slot_key, npc, content, session_id)
        logger.info(
            "基础长期记忆已更新：作用域 %s / NPC %s（%d 字）", slot_key, npc, len(content),
        )
        return True
    return False
