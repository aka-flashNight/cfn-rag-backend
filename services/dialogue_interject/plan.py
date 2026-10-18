"""任务 plan 归一化（协议 §4.4 / §8）。

职责：把逐行 JSONL 解析出的任务指令，收敛成 Host 可直接应用的
`{mute, end, edits[]}`。只做结构/编号/说话人/情绪/字数层面的宽容校验，
不做内容层面的“漏信息”判断。

关键语义（照协议，不自行解释）：
- `n` 与请求 `remaining[].index` 同口径（1..N）；越界忽略并记录。
- 没有 `end` ＝整组替换，本轮新增＋改写构成剩余内容，不再接回原文。
- `end` 指向被移除条目 → 跳到下一条未移除原文；没有则结束（end=null）。
- `end` 及之后的编辑/移除一律忽略（协议 4.5：从 end 到组末用原文）。
- `end` 之前未提及的条目视为隐含 nochange（宽容），但不写进 plan.edits。
- 同一 n 多条 line 合并为“替换多条”；line 与 remove/nochange 冲突时按 edit 处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from services.dialogue_interject.parser import ParsedEntry, TextSegment
from services.dialogue_interject.schemas import (
    PlanEditPayload,
    PlanLinePayload,
    PlanPayload,
)


@dataclass(frozen=True)
class PlanRejection:
    """被忽略的指令与原因（写入日志，不进 SSE）。"""

    kind: str
    n: int | None
    reason: str


@dataclass
class NormalizedPlan:
    """归一化后的任务计划（内部字段供日志/测试，不进入 SSE）。"""

    mute: bool = False
    end: int | None = None
    edits: list[PlanEditPayload] = field(default_factory=list)
    implied_nochange: list[int] = field(default_factory=list)
    rejected: list[PlanRejection] = field(default_factory=list)

    def to_payload(self) -> PlanPayload:
        return PlanPayload(mute=self.mute, end=self.end, edits=list(self.edits))

    def to_wire(self) -> dict:
        return self.to_payload().to_wire()


@dataclass
class _Bucket:
    """某个原编号上收到的指令（按到达顺序累积）。"""

    lines: list[TextSegment] = field(default_factory=list)
    remove: bool = False
    nochange: bool = False


def normalize_task_plan(entries: list[ParsedEntry], remaining_count: int) -> NormalizedPlan:
    """把任务指令序列归一化为 Host 计划。

    `entries` 为解析器输出的全部行（含 add）；add 不进入 plan（已通过 line 事件推送）。
    `remaining_count` 为请求 `remaining` 的条数 N。
    """
    plan = NormalizedPlan(mute=any(entry.kind == "mute" for entry in entries))
    buckets: dict[int, _Bucket] = {}
    end_value: int | None = None
    end_seen = False

    for entry in entries:
        if entry.kind in ("mute", "add"):
            continue

        if entry.kind == "end":
            if end_seen:
                plan.rejected.append(PlanRejection("end", entry.end, "重复 end，只取第一条"))
                continue
            end_seen = True
            if entry.end is None or not (1 <= entry.end <= remaining_count):
                plan.rejected.append(
                    PlanRejection("end", entry.end, f"end 越界（当前剩余 1..{remaining_count}）")
                )
                continue
            end_value = entry.end
            continue

        n = entry.n
        if n is None or not (1 <= n <= remaining_count):
            plan.rejected.append(
                PlanRejection(entry.kind, n, f"n 越界（当前剩余 1..{remaining_count}）")
            )
            continue

        # 协议 4.5：end 及之后的编辑/移除一律忽略（从 end 起用原文）。
        # 这里先按“声明顺序”记录，等 end 解析完成后再统一过滤，避免重复判断。
        bucket = buckets.setdefault(n, _Bucket())
        if entry.kind == "line":
            bucket.lines.extend(entry.segments)
        elif entry.kind == "remove":
            bucket.remove = True
        elif entry.kind == "nochange":
            bucket.nochange = True

    # 先按本次计划里的 remove 解析 end：指向被移除条目时跳到下一条未移除原文
    if end_value is not None:
        removed = {n for n, bucket in buckets.items() if bucket.remove and not bucket.lines}
        while end_value in removed:
            end_value += 1
        if end_value > remaining_count:
            plan.rejected.append(
                PlanRejection("end", end_value, "end 指向被移除条目且后面没有可接回原文")
            )
            end_value = None
    plan.end = end_value

    edits: list[PlanEditPayload] = []
    for n in sorted(buckets):
        bucket = buckets[n]
        # end 及之后不应用编辑/移除；未提及项由 Host 按原文播放
        if plan.end is not None and n >= plan.end:
            plan.rejected.append(PlanRejection("edit", n, f"位于 end={plan.end} 及之后，接回原文"))
            continue
        if bucket.lines:
            if bucket.remove:
                plan.rejected.append(PlanRejection("remove", n, "同一 n 同时有 line，按 edit 处理"))
            if bucket.nochange:
                plan.rejected.append(
                    PlanRejection("nochange", n, "同一 n 同时有 line，按 edit 处理")
                )
            edits.append(
                PlanEditPayload(
                    n=n,
                    kind="edit",
                    lines=[
                        PlanLinePayload(text=seg.text, emotion=seg.emotion, split=seg.split)
                        for seg in bucket.lines
                    ],
                )
            )
        elif bucket.remove:
            edits.append(PlanEditPayload(n=n, kind="remove"))
        elif bucket.nochange:
            edits.append(PlanEditPayload(n=n, kind="nochange"))

    plan.edits = edits
    if plan.end is not None:
        edited = {edit.n for edit in edits}
        plan.implied_nochange = [n for n in range(1, plan.end) if n not in edited]
    return plan
