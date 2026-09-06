"""汇合逻辑（对应 docs/v3-developer/03 §2 MERGE_WAIT / §4 时序）。

规则（硬性产品规则，实现在此）：
- 宽限 merge_grace_ms 内拿到 final → 直接汇合回复 #2（不发过渡语）；
- 超时且有中间结果且未说过过渡语 → 过渡语 #2a（只说一次、只说大概、**不报数字**），
  之后继续等 final → 汇合 #2b；
- 无中间结果 → 等待（SSE 保活注释帧）→ final → 汇合；
- 子 Agent 失败/超时 → FAIL_REPLY（NPC 人设话术，不甩错误码）。

LLM 调用预算：聊天主 Agent 每回合 ≤ 3 次（#1 + 过渡 #2a + 汇合 #2b）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional

from services.llm import ChatRequest, LLMClient, split_meta_events
from services.orchestrator.events import (
    SSEEvent,
    accumulate_usage,
    agent_status_event,
    content_event,
    KEEP_ALIVE,
)
from services.orchestrator.prompts import (
    SEARCH_MERGE_INSTRUCTION,
    TASK_FAIL_INSTRUCTION,
    TASK_MERGE_INSTRUCTION,
    TASK_UPDATE_MERGE_INSTRUCTION,
    build_merge_user_prompt,
)

logger = logging.getLogger(__name__)

# 长等待时保活注释帧的间隔
_KEEPALIVE_INTERVAL_S = 10.0


@dataclass
class MergeOutcome:
    """汇合结果（turn.py 的 POST_PROCESS 使用）。"""

    status: Literal["final", "failed", "timeout"] = "final"
    agent: str = "task"
    payload: dict[str, Any] = field(default_factory=dict)
    spoke_interim: bool = False
    reply_text: str = ""          # 汇合回复正文（记忆用）
    # 分段回复：每次 LLM 说话爆发（过渡语 #2a / 汇合 #2b / FAIL_REPLY）各占一段。
    # 段与段之间隔着 agent_status 等待，记忆落库时各存一条 assistant 记录，
    # 前端历史渲染即每条一个气泡（对应「连续说话分段」产品规则）。
    reply_segments: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None


class MergeCoordinator:
    """单个后台子 Agent 的汇合协调器。"""

    def __init__(
        self,
        *,
        handle: Any,  # SubagentHandle
        llm: LLMClient,
        base_messages: list[dict[str, Any]],
        npc_name: str,
        player_query: str,
        spoken_text: str,
        emotion: str,
        grace_s: float,
        timeout_s: float,
    ) -> None:
        self.handle = handle
        self.llm = llm
        self.base_messages = base_messages
        self.npc_name = npc_name
        self.player_query = player_query
        self.spoken_text = spoken_text
        self.emotion = emotion
        self.grace_s = grace_s
        self.timeout_s = timeout_s
        self.agent = handle.agent
        self.saw_intermediate = False
        self.spoke_interim = False
        self._vague_note_cache = ""
        self._final_event: Any = None
        self.outcome = MergeOutcome(agent=self.agent)

    # ------------------------------------------------------------------
    # 事件排水（agent_status 事件即时转发；final/failed 记录）
    # ------------------------------------------------------------------

    def _phase_for(self, ev: Any) -> str:
        if ev.kind == "intermediate":
            return "repairing" if self.agent == "task" else "searching"
        if ev.kind == "failed":
            return "failed"
        if self.agent == "search":
            return "searching"
        return "repairing" if "修正" in (ev.ui_hint or "") else "drafting"

    async def _drain(self, timeout: float) -> AsyncIterator[SSEEvent]:
        """在 timeout 内消费子 Agent 事件；final/failed 记入 self._final_event 并返回。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                ev = await self.handle.wait_event(min(remaining, _KEEPALIVE_INTERVAL_S))
            except asyncio.TimeoutError:
                if loop.time() >= deadline - 1e-6:
                    return
                yield KEEP_ALIVE
                continue
            if ev.kind in ("final", "failed"):
                self._final_event = ev
                phase = "done" if ev.kind == "final" else "failed"
                yield agent_status_event(self.agent, phase, ev.ui_hint)
                return
            if ev.kind == "intermediate":
                self.saw_intermediate = True
                if ev.vague_note:
                    self._vague_note_cache = ev.vague_note
            yield agent_status_event(self.agent, self._phase_for(ev), ev.ui_hint)

    # ------------------------------------------------------------------
    # LLM 调用（流式，跳过模型可能冒出的 meta 行）
    # ------------------------------------------------------------------

    async def _stream_reply(self, user_prompt: str) -> AsyncIterator[SSEEvent]:
        messages = [*self.base_messages, {"role": "user", "content": user_prompt}]
        req = ChatRequest(messages=messages, purpose="chat", send_image=False)
        try:
            stream = self.llm.chat_stream(req)
            meta, events = await split_meta_events(stream, [])
            collected: list[str] = []
            async for ev in events:
                if ev.kind == "content":
                    if ev.text:
                        collected.append(ev.text)
                        yield content_event(ev.text)
                elif ev.kind == "usage" and ev.usage:
                    self.outcome.usage = accumulate_usage(self.outcome.usage, ev.usage)
            self.outcome.reply_text += "".join(collected)
            if collected:
                self.outcome.reply_segments.append("".join(collected))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("汇合回复调用失败: %s", exc)

    async def _interim_reply(self, vague_note: str) -> AsyncIterator[SSEEvent]:
        """过渡语 #2a：1~2 句模糊过渡，禁止数字/物品名/金额（03 §3 硬性规则）。"""
        self.spoke_interim = True
        self.outcome.spoke_interim = True
        result_block = (
            "【后台进度】相关事项还在处理中（这是中间状态）。"
            "硬性要求：本轮**绝对不要**说出任何具体数字、物品名、金额、任务细节。"
        )
        prompt = build_merge_user_prompt(
            npc_name=self.npc_name,
            player_query=self.player_query,
            spoken_text=self.spoken_text,
            emotion=self.emotion,
            result_block=result_block,
            instruction=(
                f"只输出 1~2 句过渡话（大意：{vague_note or '还在准备'}），"
                "说完就停，不要展开。"
            ),
        )
        async for ev in self._stream_reply(prompt):
            yield ev

    def _result_block(self, payload: dict[str, Any]) -> str:
        if self.agent == "search":
            findings = (payload.get("findings") or "").strip()
            return "【后台结果：检索到的资料】\n" + (findings or "（没有查到相关信息）")
        lines = ["【后台结果：任务草案】"]
        summary = (payload.get("draft_summary") or "").strip()
        lines.append(summary or "（草案信息缺失）")
        if payload.get("fallback"):
            lines.append("（注意：这是按最保守规则拟定的方案，说明时保持自然，不要显得敷衍）")
        deviation = (payload.get("deviation_note") or "").strip()
        if deviation:
            lines.append(f"偏离说明：{deviation}")
        return "\n".join(lines)

    def _merge_instruction(self) -> str:
        if self.agent == "search":
            return SEARCH_MERGE_INSTRUCTION
        if self.handle.kind == "task_update":
            return TASK_UPDATE_MERGE_INSTRUCTION
        return TASK_MERGE_INSTRUCTION

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[SSEEvent]:
        # 1) 宽限期：拿到 final 直接汇合（不发过渡语）
        async for ev in self._drain(self.grace_s):
            yield ev

        final_ev = self._final_event
        if final_ev is None:
            # 2) 有中间结果且未说过过渡语 → 过渡语 #2a（只说一次）
            if self.saw_intermediate and not self.spoke_interim:
                async for ev in self._interim_reply(self._last_vague_note):
                    yield ev
            # 3) 继续等最终结果（保活由 _drain 产生）
            async for ev in self._drain(self.timeout_s):
                yield ev
            final_ev = self._final_event

        if final_ev is None:
            self.outcome.status = "timeout"
            self.outcome.payload = {"reason": "等待后台任务超时"}
            yield agent_status_event(self.agent, "failed", "等待超时")
            async for ev in self._fail_reply("等待超时"):
                yield ev
            return

        if final_ev.kind == "failed":
            self.outcome.status = "failed"
            self.outcome.payload = dict(final_ev.payload or {})
            async for ev in self._fail_reply(
                str((final_ev.payload or {}).get("reason") or "处理失败"),
            ):
                yield ev
            return

        self.outcome.status = "final"
        self.outcome.payload = dict(final_ev.payload or {})
        prompt = build_merge_user_prompt(
            npc_name=self.npc_name,
            player_query=self.player_query,
            spoken_text=self.spoken_text,
            emotion=self.emotion,
            result_block=self._result_block(self.outcome.payload),
            instruction=self._merge_instruction(),
        )
        async for ev in self._stream_reply(prompt):
            yield ev

    async def _fail_reply(self, reason: str) -> AsyncIterator[SSEEvent]:
        """FAIL_REPLY：NPC 人设话术收场，不甩错误码。"""
        result_block = (
            f"【内部原因（不要向玩家提及，也不要提及系统/错误）】{reason}"
        )
        prompt = build_merge_user_prompt(
            npc_name=self.npc_name,
            player_query=self.player_query,
            spoken_text=self.spoken_text,
            emotion=self.emotion,
            result_block=result_block,
            instruction=TASK_FAIL_INSTRUCTION,
        )
        before = self.outcome.reply_text
        async for ev in self._stream_reply(prompt):
            yield ev
        if self.outcome.reply_text == before:
            # LLM 调用也失败：静态人设话术兜底
            text = "今天手头的事都派完了，改天吧。"
            self.outcome.reply_text += text
            self.outcome.reply_segments.append(text)
            yield content_event(text)

    @property
    def _last_vague_note(self) -> str:
        return self._vague_note_cache or "委托内容还在调整"
