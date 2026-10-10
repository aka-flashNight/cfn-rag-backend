"""汇合逻辑（对应 docs/v3-developer/03 §2 MERGE_WAIT / §4 时序）。

规则（硬性产品规则，实现在此）：
- 宽限 merge_grace_ms 内拿到 final → 直接汇合回复 #2（不发过渡语）；
- 宽限期过后拿到中间结果且未说过过渡语 → 过渡语 #2a（只说一次、只说方向、
  **不报数字**），之后继续等 final → 汇合 #2b（过渡语说完并入 spoken_text，
  汇合提示词带「你本轮已经说过的话」块，第二次生成要顺着它说）；
- 中间结果晚于宽限期到达时同样要说（bridge 排水遇到 intermediate 立即返回），
  否则玩家会一直对着空白等草案拟完；
- 始终无中间结果 → 等待（SSE 保活注释帧）→ final → 汇合；
- 子 Agent 失败/超时 → FAIL_REPLY（NPC 人设话术，不甩错误码）。

过渡语的口径（v3 再次重构定稿）：NPC **心里有大致方向**，只是细节还没敲定。
既不能表现得茫然（像别人塞给它的活），也不能假装细节已经定了；只讲方向，
不涉及物品名/数量/金额/人名/地名。方向措辞来自子 Agent 的 vague_note
（TaskRunner 按 prepare 选出的真实 task_type 生成，见
subagents.prompts.build_direction_vague_note）。

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
    task_merge_hint,
)
from services.subagents.prompts import DIRECTION_VAGUE_FALLBACK

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
        """agent_status 的 phase 一律由 ui_hint 推：只有「校验修正中」算 repairing。

        v3 再次重构后 intermediate 有两种来源——prepare 完成（方向已定，仍在拟定）
        和校验失败（确实在返工），前者不该报 repairing。
        """
        if ev.kind == "failed":
            return "failed"
        if self.agent == "search":
            return "searching"
        return "repairing" if "修正" in (ev.ui_hint or "") else "drafting"

    async def _drain(
        self, timeout: float, *, bridge_on_intermediate: bool = False,
    ) -> AsyncIterator[SSEEvent]:
        """在 timeout 内消费子 Agent 事件；final/failed 记入 self._final_event 并返回。

        bridge_on_intermediate=True（宽限期过后的排水）时，收到 intermediate 立即
        返回，把控制权交回 stream() 去说过渡语——否则晚于宽限期到达的中间结果
        永远等不到说话时机。宽限期内的排水不带该标志：那段时间要先看 final
        会不会紧跟着到，抢着说过渡话反而多余。
        """
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
                # 首条方向种子优先：任务流里 prepare 完成时推的是「这件事大致是
                # 什么性质」，之后校验失败推的是「还在调整」。过渡语的职责是报方向，
                # 被后到的调整话覆盖会让玩家觉得 NPC 自己也说不清要派什么活。
                if ev.vague_note and not self._vague_note_cache:
                    self._vague_note_cache = ev.vague_note
            yield agent_status_event(self.agent, self._phase_for(ev), ev.ui_hint)
            if bridge_on_intermediate and ev.kind == "intermediate":
                return

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
        """过渡语 #2a：1~2 句报方向的模糊过渡，禁止数字/物品名/金额（03 §3）。"""
        self.spoke_interim = True
        self.outcome.spoke_interim = True
        direction = (vague_note or "").strip() or DIRECTION_VAGUE_FALLBACK
        result_block = (
            "【当前状况】这事是你自己在盘算的：方向你心里有数，只是具体怎么安排还没定下来，"
            "也还没跟玩家说全。别表现得茫然（这不是谁塞给你的活），"
            "也别假装细节已经敲定了。"
            "硬性要求：本轮**绝对不要**说出任何具体数字、物品名、金额、人名、地名或任务细节。"
        )
        prompt = build_merge_user_prompt(
            npc_name=self.npc_name,
            player_query=self.player_query,
            spoken_text=self.spoken_text,
            emotion=self.emotion,
            result_block=result_block,
            instruction=(
                f"只输出 1~2 句过渡话，把方向点到为止：{direction}。"
                "口吻是「先跟你打个招呼，容我把这事安排出来」，说完就停，"
                "不要展开、不要追问玩家、不要许诺任何具体内容。"
            ),
        )
        before = self.outcome.reply_text
        async for ev in self._stream_reply(prompt):
            yield ev
        if self.outcome.reply_text == before:
            # 过渡话调用也失败：补一句中性话填住静默期，别让玩家对着空白等草案
            text = "稍等，容我把这事想周全。"
            self.outcome.reply_text += text
            self.outcome.reply_segments.append(text)
            yield content_event(text)
        # 过渡语已推送给玩家：并入 spoken_text，汇合 #2b / 失败话术的
        # 「你本轮已经说过的话」块要带上它，第二次生成才能顺着过渡句说
        interim_text = self.outcome.reply_text[len(before):].strip()
        if interim_text:
            prev = self.spoken_text.strip()
            self.spoken_text = f"{prev}\n{interim_text}" if prev else interim_text

    def _result_block(self, payload: dict[str, Any]) -> str:
        if self.agent == "search":
            findings = (payload.get("findings") or "").strip()
            return "【后台结果：检索到的资料】\n" + (findings or "（没有查到相关信息）")
        lines = ["【后台结果：任务草案】"]
        summary = (payload.get("draft_summary") or "").strip()
        lines.append(summary or "（草案信息缺失）")
        deviation = (payload.get("deviation_note") or "").strip()
        if deviation:
            lines.append(f"偏离说明：{deviation}")
        # 按任务类型条件拼接说明要点：摘要给的是「有什么」，这里补「讲的时候别漏什么」
        hint = task_merge_hint(str(payload.get("task_type") or ""))
        if hint:
            lines.append(f"说明要点：{hint}")
        if payload.get("fallback"):
            lines.append("（注意：这是按最保守规则拟定的方案，说明时保持自然，不要显得敷衍）")
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
        loop = asyncio.get_running_loop()
        # 总预算 = 宽限期 + 子 Agent 超时；过渡语说完后剩余部分全用来等 final
        deadline = loop.time() + self.grace_s + self.timeout_s

        # 1) 宽限期：拿到 final 直接汇合（不发过渡语）
        async for ev in self._drain(self.grace_s):
            yield ev

        # 2) 宽限期已过仍无终态：等到「可说话的时机」——过渡语只说一次，
        #    bridge 排水遇到 intermediate 立即返回，晚到的中间结果也不会漏说
        while self._final_event is None:
            if self.saw_intermediate and not self.spoke_interim:
                async for ev in self._interim_reply(self._vague_note_cache):
                    yield ev
                async for ev in self._drain(self._remaining(deadline)):
                    yield ev
                break
            remaining = self._remaining(deadline)
            if remaining <= 0:
                break
            # 过渡语说过之后不再 bridge：一路等到终态或预算耗尽
            async for ev in self._drain(
                remaining, bridge_on_intermediate=not self.spoke_interim,
            ):
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

    @staticmethod
    def _remaining(deadline: float) -> float:
        """距总预算截止还剩多少秒（不为负）。"""
        return max(0.0, deadline - asyncio.get_running_loop().time())
