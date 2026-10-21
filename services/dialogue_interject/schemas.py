"""对话内嵌插话 · 请求 / SSE / JSONL / plan 的 Pydantic 模型。

真源：游戏仓 `docs/AI对话内嵌插话-需求与协议-2026-10-18.md`（下称《协议》）§5；
共享 fixture：`docs/AI对话内嵌插话-交接与契约fixture-2026-10-18.md`。
本文件只做结构校验，不解释业务语义；字段名与协议逐字一致，不增不减。

P0 边界：请求/事件/plan 的形状先立起来，提示词与真 LLM 编排在 P1/P2/P4 填实。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: 协议 5.1：dialogue_instance_id 只允许 UUID/hex 一类安全字符，长度受限
DIALOGUE_INSTANCE_ID_PATTERN = r"^[A-Za-z0-9_-]{8,128}$"
#: 协议 5.1：turn_id 非空；同一 turn 重发幂等
TURN_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"

#: 协议 D-11：玩家输入上限 150 字
PLAYER_INPUT_MAX = 150
#: 协议 D-11：单条台词与结束语硬上限 165，提示词上限 135
LINE_HARD_MAX = 165
LINE_PROMPT_MAX = 135


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------


class PlayedLine(BaseModel):
    """已播行（请求 played[]，finalize transcript[] 同结构）。"""

    seq: int = Field(..., ge=0, description="0 基单调序号；同一 session 内不重复")
    role: Literal["npc", "player", "other_npc", "ai"]
    name: str = ""
    text: str
    emotion: str | None = None


class RemainingLine(BaseModel):
    """原始剩余台词（相对编号 1..N，与模型 n 同口径）。"""

    index: int = Field(..., ge=1)
    name: str = ""
    text: str
    emotion: str | None = None


class ReferenceLine(BaseModel):
    """参考续写（上一轮未播的新增/结束语/改写/移除）。"""

    kind: Literal["add", "closing", "edit", "remove"]
    name: str = ""
    text: str = ""
    emotion: str | None = None
    index: int | None = Field(default=None, ge=1, description="edit/remove 的原编号")

    @model_validator(mode="after")
    def _check_shape(self) -> "ReferenceLine":
        if self.kind in ("edit", "remove") and self.index is None:
            raise ValueError(f"reference_continuation.kind={self.kind} 必须带 index")
        if self.kind in ("add", "edit") and not self.text.strip():
            raise ValueError(f"reference_continuation.kind={self.kind} 的 text 不能为空")
        return self


class DialogueBlock(BaseModel):
    """dialogue 段：kind / played / remaining / reference_continuation。"""

    kind: Literal["daily", "task_pre", "task_post"]
    group_id: str = ""
    played: list[PlayedLine] = Field(default_factory=list)
    remaining: list[RemainingLine] = Field(default_factory=list)
    reference_continuation: list[ReferenceLine] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_numbering(self) -> "DialogueBlock":
        # 协议 5.1：remaining[].index 固定为 1..N（不跳号、不重号）
        indexes = [line.index for line in self.remaining]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("dialogue.remaining[].index 必须是 1..N 连续编号")
        # played seq 唯一（同一 session 内不重复）
        seqs = [line.seq for line in self.played]
        if len(set(seqs)) != len(seqs):
            raise ValueError("dialogue.played[].seq 不能重复")
        # reference 的 edit/remove 编号必须在当前 remaining 的 1..N 内；
        # 越界属 Host 记账错误，422 拒绝，不做宽容渲染（《协议》5.1 / 实施指引 3.1）
        remaining_count = len(indexes)
        for item in self.reference_continuation:
            if item.kind in ("edit", "remove") and item.index is not None:
                if item.index > remaining_count:
                    raise ValueError(
                        f"reference_continuation.{item.kind}.index={item.index} "
                        f"超出当前 remaining 的 1..{remaining_count}"
                    )
        return self


class TaskSummary(BaseModel):
    """任务摘要的一段（completed / upcoming）。"""

    task_id: int | None = None
    title: str = ""
    npc: str = ""
    objectives: list[str] = Field(default_factory=list)
    rewards: list[str] = Field(default_factory=list)
    outcome: str = ""


class TaskContext(BaseModel):
    """任务上下文（协议 4.2）：游戏侧上传，后端不反查任务表。"""

    completed: TaskSummary | None = None
    upcoming: TaskSummary | None = None


class Limits(BaseModel):
    """字数口径由 Host 随请求下发；缺省用协议冻结值。"""

    line_hard_max: int = Field(default=LINE_HARD_MAX, ge=1, le=1000)
    line_prompt_max: int = Field(default=LINE_PROMPT_MAX, ge=1, le=1000)


class InterjectRequest(BaseModel):
    """`POST /api/game/dialogue-interject` 请求体（协议 5.1）。"""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["daily", "task"]
    dialogue_instance_id: str = Field(..., pattern=DIALOGUE_INSTANCE_ID_PATTERN)
    turn_id: str = Field(..., pattern=TURN_ID_PATTERN)
    npc: str = Field(..., min_length=1)
    player_input: str
    session_title: str = ""
    dialogue: DialogueBlock
    task_context: TaskContext | None = None
    limits: Limits = Field(default_factory=Limits)

    @field_validator("npc", "player_input")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("不能为空")
        return stripped

    @field_validator("player_input")
    @classmethod
    def _limit_player_input(cls, value: str) -> str:
        if len(value) > PLAYER_INPUT_MAX:
            raise ValueError(f"玩家输入不得超过 {PLAYER_INPUT_MAX} 字（当前 {len(value)}）")
        return value

    @field_validator("session_title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check_mode_kind(self) -> "InterjectRequest":
        # 协议 5.1：mode=daily 只能 daily；mode=task 只能 task_pre/task_post
        if self.mode == "daily" and self.dialogue.kind != "daily":
            raise ValueError("mode=daily 时 dialogue.kind 必须为 daily")
        if self.mode == "task" and self.dialogue.kind not in ("task_pre", "task_post"):
            raise ValueError("mode=task 时 dialogue.kind 必须为 task_pre 或 task_post")
        # 协议 5.1：任务模式必须有 task_context（两段都可为空，但对象必须在）
        if self.mode == "task" and self.task_context is None:
            raise ValueError("mode=task 时必须提供 task_context")
        return self


class FinalizeRequest(BaseModel):
    """`POST /api/game/dialogue-interject/finalize` 请求体（协议 5.1 末段）。"""

    model_config = ConfigDict(extra="ignore")

    dialogue_instance_id: str = Field(..., pattern=DIALOGUE_INSTANCE_ID_PATTERN)
    turn_id: str | None = Field(default=None, pattern=TURN_ID_PATTERN)
    transcript: list[PlayedLine] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_seq_unique(self) -> "FinalizeRequest":
        seqs = [line.seq for line in self.transcript]
        if len(set(seqs)) != len(seqs):
            raise ValueError("transcript[].seq 不能重复")
        return self


class FinalizeResponse(BaseModel):
    """finalize 回执（协议未定义字段；后端只回执对账结果，供 Host 观测）。"""

    ok: bool = True
    dialogue_instance_id: str
    session_id: str
    appended: int = 0
    total: int = 0


# ---------------------------------------------------------------------------
# SSE 事件
# ---------------------------------------------------------------------------


class StatusPayload(BaseModel):
    stage: Literal["connecting", "thinking", "searching", "drafting", "waiting"]
    #: 子 Agent 的 ≤12 字进度提示；缺省时 Host 回落 stage 文案。
    ui_hint: str = ""


class MutePayload(BaseModel):
    mute: bool = True


class SplitMark(BaseModel):
    """拆分标记（协议 5.2）：仅当一条正文被 165 硬上限拆成多段时出现。"""

    index: int = Field(..., ge=0)
    count: int = Field(..., ge=2)


class LinePayload(BaseModel):
    """SSE `line` 事件数据：只用于当前可播放的 add / closing。"""

    kind: Literal["add", "closing"]
    text: str
    emotion: str
    speaker: str
    seq: int = Field(..., ge=0)
    split: SplitMark | None = None
    #: 草案等待期的过渡话：Host 据此在玩家越过此句后继续显示 ui_hint 占位。
    interim: bool = False

    def to_wire(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
            "emotion": self.emotion,
            "speaker": self.speaker,
            "seq": self.seq,
        }
        if self.split is not None:
            data["split"] = self.split.model_dump()
        if self.interim:
            data["interim"] = True
        return data


class PlanLinePayload(BaseModel):
    """plan.edits[].lines[]：改写后的逐段正文。"""

    text: str
    emotion: str
    split: SplitMark | None = None

    def to_wire(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text, "emotion": self.emotion}
        if self.split is not None:
            data["split"] = self.split.model_dump()
        return data


class PlanEditPayload(BaseModel):
    n: int = Field(..., ge=1)
    kind: Literal["nochange", "edit", "remove"]
    lines: list[PlanLinePayload] = Field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        data: dict[str, Any] = {"n": self.n, "kind": self.kind}
        if self.lines:
            data["lines"] = [line.to_wire() for line in self.lines]
        return data


class PlanPayload(BaseModel):
    """SSE `plan` 事件数据（协议 8）：end 缺省＝整组替换，因此线上始终保留 end（可为 null）。"""

    mute: bool = False
    end: int | None = None
    edits: list[PlanEditPayload] = Field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        return {
            "mute": self.mute,
            "end": self.end,
            "edits": [edit.to_wire() for edit in self.edits],
        }


class DonePayload(BaseModel):
    dialogue_instance_id: str
    session_id: str
    usage: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    code: str
    message: str
    valid_lines: int = Field(default=0, ge=0)


class SSEEvent(BaseModel):
    """一条 SSE 帧：event 名 + data 对象（编码见 `encode`）。"""

    event: Literal["status", "mute", "line", "plan", "done", "error"]
    data: dict[str, Any]

    def encode(self) -> bytes:
        import json

        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event}\ndata: {payload}\n\n".encode("utf-8")


def status_event(stage: str, ui_hint: str = "") -> SSEEvent:
    return SSEEvent(
        event="status",
        data=StatusPayload(stage=stage, ui_hint=(ui_hint or "").strip()).model_dump(),
    )


class StreamStatus(str):
    """流内旁路信号：不进入 JSONL 解析，由编排器转成 SSE `status`。

    真 LLM 适配器在进入工具轮前和子 Agent 进度事件到达时插入；假流（纯文本）
    不需要也不受影响。继承 str 且内容为空串：调用方若直接 `"".join(chunks)`
    不会破坏原文本。
    """

    stage: str
    ui_hint: str

    def __new__(cls, stage: str, ui_hint: str = "") -> "StreamStatus":
        obj = super().__new__(cls, "")
        obj.stage = stage
        obj.ui_hint = (ui_hint or "").strip()
        return obj

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"StreamStatus(stage={self.stage!r}, ui_hint={self.ui_hint!r})"


def mute_event() -> SSEEvent:
    return SSEEvent(event="mute", data=MutePayload().model_dump())


def line_event(payload: LinePayload) -> SSEEvent:
    return SSEEvent(event="line", data=payload.to_wire())


def plan_event(payload: PlanPayload) -> SSEEvent:
    return SSEEvent(event="plan", data=payload.to_wire())


def done_event(payload: DonePayload) -> SSEEvent:
    return SSEEvent(event="done", data=payload.model_dump(exclude_none=True))


def error_event(code: str, message: str, *, valid_lines: int = 0) -> SSEEvent:
    return SSEEvent(
        event="error",
        data=ErrorPayload(code=code, message=message, valid_lines=valid_lines).model_dump(),
    )
