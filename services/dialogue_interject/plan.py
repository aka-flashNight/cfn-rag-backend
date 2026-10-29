"""任务 plan 归一化（协议 §4.4 / §8）。

模型按输出顺序给出 add/edit/keep/remove，并以 continue 或 end 收尾：
- add 的 no=0，可插在任意两个操作之间；首个原编号操作之前的 add 已即时推送，
  不重复进入 plan，其余 add 按原顺序进入 steps。
- edit/keep/remove 的 no 必须按原剩余编号递增；跳过的编号补成 remove。
- 同一 no 的连续 edit 合并为一条改写成多段；其他重复或倒序指令忽略并记录。
- continue(no) 表示从原第 no 条起到组末保留原文；end 表示不再保留未处理原文。
- 两者只取先出现的一条，之后的内容全部忽略。缺失或 continue 非法时按错误兜底：
  从最后处理的原编号之后保留原文；尚未处理任何原编号则保留全部原文。

SSE 统一为 {mute, resume, steps}：resume 是原剩余接回点 1..N+1，N+1 表示不接回。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from services.dialogue_interject.parser import ParsedEntry
from services.dialogue_interject.schemas import (
    PlanLinePayload,
    PlanPayload,
    PlanStepPayload,
)


@dataclass(frozen=True)
class PlanRejection:
    """被忽略或兜底修正的指令与原因（写入日志，不进 SSE）。"""

    kind: str
    n: int | None
    reason: str


@dataclass
class NormalizedPlan:
    """归一化后的任务计划（内部字段供日志/测试，不进入 SSE）。"""

    mute: bool = False
    resume: int = 1
    steps: list[PlanStepPayload] = field(default_factory=list)
    fallback_continue: bool = False
    rejected: list[PlanRejection] = field(default_factory=list)

    def to_payload(self) -> PlanPayload:
        return PlanPayload(mute=self.mute, resume=self.resume, steps=list(self.steps))

    def to_wire(self) -> dict:
        return self.to_payload().to_wire()


def _payload_lines(entry: ParsedEntry) -> list[PlanLinePayload]:
    return [
        PlanLinePayload(text=seg.text, emotion=seg.emotion, split=seg.split)
        for seg in entry.segments
    ]


def normalize_task_plan(entries: list[ParsedEntry], remaining_count: int) -> NormalizedPlan:
    """把有序任务指令归一化为 Host 可直接顺序播放的计划。"""
    plan = NormalizedPlan(mute=any(entry.kind == "mute" for entry in entries))
    steps: list[PlanStepPayload] = []
    last_n = 0
    original_started = False
    can_extend_edit = False
    terminated = False
    resume: int | None = None

    def fill_skipped(before_n: int) -> None:
        """把 before_n 之前尚未处理的原编号补成 remove。"""
        nonlocal last_n, can_extend_edit
        for missing in range(last_n + 1, before_n):
            steps.append(PlanStepPayload(n=missing, kind="remove"))
            plan.rejected.append(
                PlanRejection("remove", missing, "原编号未提及，按 remove 兜底")
            )
        if before_n - 1 > last_n:
            can_extend_edit = False
        last_n = max(last_n, before_n - 1)

    for entry in entries:
        kind = entry.kind
        if kind == "mute":
            continue
        if terminated:
            plan.rejected.append(
                PlanRejection(kind, entry.n, "位于 continue/end 之后，已忽略")
            )
            continue
        if kind == "add":
            if original_started:
                steps.append(PlanStepPayload(n=0, kind="add", lines=_payload_lines(entry)))
                can_extend_edit = False
            # 前缀 add 已通过 line 事件即时推送，不能重复进入 plan。
            continue

        original_started = True
        if kind == "end":
            terminated = True
            fill_skipped(remaining_count + 1)
            resume = remaining_count + 1
            continue
        if kind == "continue":
            terminated = True
            n = entry.n
            if n is not None and last_n < n <= remaining_count:
                fill_skipped(n)
                resume = n
            else:
                plan.fallback_continue = True
                resume = min(last_n + 1, remaining_count + 1)
                plan.rejected.append(
                    PlanRejection(
                        "continue", n,
                        f"continue 编号无效，按最后处理编号后保留原文（resume={resume}）",
                    )
                )
            continue

        n = entry.n
        if kind not in ("edit", "keep", "remove") or n is None:
            plan.rejected.append(PlanRejection(kind, n, "未知任务指令，已忽略"))
            continue
        if not (1 <= n <= remaining_count):
            plan.rejected.append(
                PlanRejection(kind, n, f"no 越界（当前剩余 1..{remaining_count}）")
            )
            continue
        if n < last_n:
            plan.rejected.append(
                PlanRejection(kind, n, f"原编号倒退（当前已处理到 {last_n}），已忽略")
            )
            continue
        if n == last_n:
            if kind == "edit" and can_extend_edit and steps:
                steps[-1].lines.extend(_payload_lines(entry))
            else:
                plan.rejected.append(
                    PlanRejection(kind, n, "原编号重复；只有连续 edit 可用于拆分，已忽略")
                )
            continue

        fill_skipped(n)
        if kind == "edit":
            steps.append(PlanStepPayload(n=n, kind="edit", lines=_payload_lines(entry)))
            can_extend_edit = True
        else:
            steps.append(PlanStepPayload(n=n, kind=kind))
            can_extend_edit = False
        last_n = n

    if resume is None:
        plan.fallback_continue = True
        resume = min(last_n + 1, remaining_count + 1)
        plan.rejected.append(
            PlanRejection(
                "continue", None,
                f"缺少 continue/end，按最后处理编号后保留原文（resume={resume}）",
            )
        )

    plan.resume = resume
    plan.steps = steps
    return plan
