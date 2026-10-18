"""插话上下文的纯文本装配（P0 骨架，P1/P2 接真模型时复用）。

原则（实施指引 §4.1）：请求现场是权威——`played` / `remaining` /
`reference_continuation` / `player_input` / `task_context` 全部来自游戏；
session 历史不参与本轮 prompt 拼装。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from services.dialogue_interject.schemas import (
    InterjectRequest,
    PlayedLine,
    ReferenceLine,
    RemainingLine,
    TaskSummary,
)

#: 上下文预算省略标记（协议 4.4：编号仍按原索引，不能打乱）
OMISSION_MARK = "（中间省略）"

_TASK_KIND_LABEL = {"task_pre": "任务前", "task_post": "任务后"}


def render_played(lines: Sequence[PlayedLine], *, npc: str, max_lines: int | None = None) -> list[str]:
    """已播历史逐行渲染：当前 NPC/ai → `NPC：…`；玩家 → `玩家：…`；其他 NPC → `【…】`。"""
    rendered: list[str] = []
    for line in _trim(lines, max_lines):
        if line is None:
            rendered.append(OMISSION_MARK)
            continue
        text = line.text.strip()
        if not text:
            continue
        if line.role in ("npc", "ai"):
            rendered.append(f"{npc}：{text}")
        elif line.role == "player":
            rendered.append(f"玩家：{text}")
        else:
            rendered.append(f"【{line.name or '其他NPC'}：{text}】")
    return rendered


def render_remaining(lines: Sequence[RemainingLine], *, max_lines: int | None = None) -> list[str]:
    """原始剩余台词：`[n] NPC名：…`；任务编号必须保留原 index。"""
    rendered: list[str] = []
    for line in _trim(lines, max_lines):
        if line is None:
            rendered.append(OMISSION_MARK)
            continue
        rendered.append(f"[{line.index}] {line.name}：{line.text.strip()}")
    return rendered


def render_reference(lines: Sequence[ReferenceLine]) -> list[str]:
    """参考续写：明确标“上一轮未播”，不参与编号、不进已播历史。"""
    rendered: list[str] = []
    for line in lines:
        text = line.text.strip()
        if line.kind == "add":
            rendered.append(f"上一轮未播新增：{text}")
        elif line.kind == "closing":
            if text:
                rendered.append(f"上一轮未播结束语：{text}")
        elif line.kind == "edit":
            rendered.append(f"上一轮原第 {line.index} 条改写：{text}")
        elif line.kind == "remove":
            rendered.append(f"上一轮原第 {line.index} 条移除")
    return rendered


def render_task_summary(summary: TaskSummary, *, label: str) -> list[str]:
    """任务摘要一块：标题/NPC/目标/奖励/（已完成）玩家做了什么。"""
    lines = [f"{label}：{summary.title.strip() or '（未命名任务）'}"]
    if summary.task_id is not None:
        lines.append(f"任务 id：{summary.task_id}")
    if summary.npc.strip():
        lines.append(f"发布 NPC：{summary.npc.strip()}")
    if summary.objectives:
        lines.append("目标：" + "；".join(item.strip() for item in summary.objectives if item.strip()))
    if summary.rewards:
        lines.append("奖励：" + "；".join(item.strip() for item in summary.rewards if item.strip()))
    if summary.outcome.strip():
        lines.append(f"玩家完成了什么：{summary.outcome.strip()}")
    return lines


def build_user_context(request: InterjectRequest) -> str:
    """按实施指引 §4.2/§4.3 的建议顺序拼装 user 上下文（P0 骨架）。"""
    blocks: list[str] = []
    dialogue = request.dialogue

    if dialogue.played:
        blocks.append("【本组已播历史】\n" + "\n".join(render_played(dialogue.played, npc=request.npc)))

    if dialogue.remaining:
        header = "【原对话剩余台词（编号仅作定位参考）】" if request.mode == "daily" else "【原始剩余台词（按编号）】"
        blocks.append(header + "\n" + "\n".join(render_remaining(dialogue.remaining)))

    if dialogue.reference_continuation:
        blocks.append(
            "【参考续写（上一轮未播，仅供参考，可不采用）】\n"
            + "\n".join(render_reference(dialogue.reference_continuation))
        )

    if request.mode == "task":
        task_lines: list[str] = []
        context = request.task_context
        if context is not None:
            if context.completed is not None:
                task_lines.extend(render_task_summary(context.completed, label="已完成任务"))
            if context.upcoming is not None:
                task_lines.extend(render_task_summary(context.upcoming, label="将完成任务"))
        kind_label = _TASK_KIND_LABEL.get(dialogue.kind, "任务")
        if task_lines:
            blocks.append(f"【任务上下文（{kind_label}）】\n" + "\n".join(task_lines))
        else:
            blocks.append(f"【任务上下文（{kind_label}）】\n（本次没有可用的已完成/将完成任务摘要）")

    blocks.append("【玩家刚刚说的话】\n" + request.player_input)
    return "\n\n".join(blocks)


def _trim(items: Sequence, max_lines: int | None) -> Iterable:
    """预算截断：保留开头少量 + 最近 M 条，中间插省略标记。"""
    if max_lines is None or max_lines <= 0 or len(items) <= max_lines:
        return list(items)
    head = min(2, max_lines // 3)
    tail = max_lines - head
    return [*items[:head], None, *items[-tail:]]
