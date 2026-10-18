"""插话编排骨架：LLM 流 → 逐行解析 → SSE 事件（P0 不接真模型）。

数据流（协议 §5.2 / §5.7）：
1. 流式 chunk 进解析器，凑齐一行立即校验/拆分；
2. 首条有效 add/closing 立即推 `line`，不等 plan/done；
3. 任务模式在流正常结束后归一化 `plan`（mute/add 不重复进入 plan）；
4. 零有效可做一次修复性重试；仍零 → `error(no_valid_lines)`，不补台词；
5. 部分有效后断流 → 已发 `line` 保留，`error(stream_failed)`，不发 plan。

P0 的真实 LLM 接线由调用方注入 `stream_factory`（测试用假流）；P1 换成
`services/llm/client.py` 的 chat_stream。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Mapping, Sequence

from services.dialogue_interject.parser import (
    JSONLStreamParser,
    ParsedEntry,
    ParseOptions,
)
from services.dialogue_interject.plan import normalize_task_plan
from services.dialogue_interject.schemas import (
    DonePayload,
    InterjectRequest,
    LinePayload,
    done_event,
    error_event,
    line_event,
    mute_event,
    plan_event,
    status_event,
)
from services.llm.meta import DEFAULT_EMOTION

logger = logging.getLogger(__name__)

#: 假流/真流统一接口：给请求，返回文本增量迭代器
StreamFactory = Callable[[InterjectRequest], AsyncIterator[str]]

_TEXT_KINDS = ("add", "closing", "line")


@dataclass
class _DriveState:
    """一次流回放的结果汇总。"""

    entries: list[ParsedEntry] = field(default_factory=list)
    valid_rows: int = 0
    failed: bool = False
    error: Exception | None = None
    error_code: str = ""


class InterjectOrchestrator:
    """一次插话请求的编排器（P0：假流可完整跑通协议链路）。"""

    def __init__(
        self,
        request: InterjectRequest,
        *,
        session_id: str,
        stream_factory: StreamFactory,
        repair_factory: StreamFactory | None = None,
        npc_emotions: Sequence[str] = (DEFAULT_EMOTION,),
        speaker_emotions: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.request = request
        self.session_id = session_id
        self._stream_factory = stream_factory
        self._repair_factory = repair_factory
        self._options = ParseOptions(
            mode=request.mode,
            npc=request.npc,
            npc_emotions=tuple(npc_emotions) or (DEFAULT_EMOTION,),
            speaker_emotions=dict(speaker_emotions or {}),
            remaining={line.index: line for line in request.dialogue.remaining},
            hard_max=request.limits.line_hard_max,
        )
        self._seq = 0
        self._mute_sent = False

    async def run(self) -> AsyncIterator:
        """产出 SSEEvent 序列（调用方负责编码/取消）。"""
        yield status_event("thinking")

        state = _DriveState()
        async for event in self._drive(self._stream_factory, state):
            yield event

        if state.failed:
            yield error_event(
                state.error_code or "stream_failed",
                self._error_message(state.error),
                valid_lines=state.valid_rows,
            )
            return

        # 协议 5.4-8：整轮零条有效输出，可做一次修复性重试（不阻塞已收到的句子）
        if not state.entries and self._repair_factory is not None:
            logger.warning(
                "[interject] 零有效输出，做一次修复性重试 dialogue=%s turn=%s",
                self.request.dialogue_instance_id, self.request.turn_id,
            )
            repair_state = _DriveState()
            async for event in self._drive(self._repair_factory, repair_state):
                yield event
            if repair_state.failed:
                yield error_event(
                    repair_state.error_code or "stream_failed",
                    self._error_message(repair_state.error),
                    valid_lines=repair_state.valid_rows,
                )
                return
            state = repair_state

        if not state.entries:
            yield error_event(
                "no_valid_lines",
                "模型没有产出可解析的 JSONL 行",
                valid_lines=0,
            )
            return

        if self.request.mode == "task":
            plan = normalize_task_plan(state.entries, len(self.request.dialogue.remaining))
            logger.info(
                "[interject] plan dialogue=%s turn=%s end=%s edits=%d implied_nochange=%s rejected=%s",
                self.request.dialogue_instance_id, self.request.turn_id,
                plan.end, len(plan.edits), plan.implied_nochange,
                [(item.kind, item.n, item.reason) for item in plan.rejected],
            )
            yield plan_event(plan.to_payload())

        yield done_event(
            DonePayload(
                dialogue_instance_id=self.request.dialogue_instance_id,
                session_id=self.session_id,
                usage={},
            )
        )

    # ------------------------------------------------------------------

    async def _drive(self, factory: StreamFactory, state: _DriveState) -> AsyncIterator:
        """回放一次流：增量解析并即时推送可播事件。"""
        parser = JSONLStreamParser(self._options)
        try:
            async for chunk in factory(self.request):
                for entry in parser.feed(chunk):
                    state.entries.append(entry)
                    async for event in self._entry_event(entry, state):
                        yield event
            for entry in parser.finish():
                state.entries.append(entry)
                async for event in self._entry_event(entry, state):
                    yield event
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 流失败统一降级，不向 Host 抛栈
            logger.warning("[interject] 流中断: %s", exc, exc_info=True)
            state.failed = True
            state.error = exc
            # 配置/上下文类错误带自己的 code（如 llm_not_configured），其余按流中断
            state.error_code = str(getattr(exc, "code", "") or "")
            return
        if parser.issues:
            logger.info(
                "[interject] 跳过 %d 行无效输出：%s",
                len(parser.issues),
                [(issue.reason, issue.raw[:80]) for issue in parser.issues],
            )

    async def _entry_event(self, entry: ParsedEntry, state: _DriveState) -> AsyncIterator:
        if entry.kind == "mute":
            if not self._mute_sent:
                self._mute_sent = True
                yield mute_event()
            return

        if entry.kind in _TEXT_KINDS:
            state.valid_rows += 1

        if entry.kind in ("add", "closing"):
            for segment in entry.segments:
                payload = LinePayload(
                    kind=entry.kind,
                    text=segment.text,
                    emotion=segment.emotion,
                    speaker=segment.speaker,
                    seq=self._seq,
                    split=segment.split,
                )
                self._seq += 1
                yield line_event(payload)

    @staticmethod
    def _error_message(exc: Exception | None) -> str:
        if exc is None:
            return "流中断"
        text = str(exc).strip()
        return text[:200] or exc.__class__.__name__
