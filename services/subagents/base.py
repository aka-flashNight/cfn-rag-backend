"""后台子 Agent 基座（对应 docs/v3-developer/03 §3）。

fire-and-steer 模式：
- 每个子 Agent 是一个 asyncio.Task，通过 ``asyncio.Queue`` 向 orchestrator 推送
  progress / intermediate / final / failed 四类事件；
- steer 能力本期仅 ``cancel()``（同会话新回合取消旧任务）；update 结构预留；
- LLM 工具循环为共享实现：非流式调用 + 同批 tool_calls 并行执行（ToolRegistry.dispatch_batch）；
- 子 Agent 不发图、不接触玩家可见正文，LLM 调用 purpose="subagent"。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from services.llm import ChatRequest, LLMClient
from services.tools.base import ToolContext, ToolRegistry
from services.llm.errors import LLMError, is_tools_unsupported_error

logger = logging.getLogger(__name__)

SubagentKind = Literal["task_draft", "task_update", "search"]
AgentLabel = Literal["task", "search"]


@dataclass
class SubagentEvent:
    """子 Agent → orchestrator 的事件（03 §3）。"""

    kind: Literal["progress", "intermediate", "final", "failed"]
    ui_hint: str = ""            # ≤12 字，给前端进度条：「拟定委托中」「校验修正中」「查资料中」
    vague_note: str = ""         # intermediate 专用：只说类型不说数字的一句话
    payload: dict[str, Any] = field(default_factory=dict)  # final: draft_summary/findings；failed: reason


class SubagentHandle:
    """asyncio.Task 包装：orchestrator 只接触 Handle，不直接操作协程。"""

    def __init__(
        self,
        kind: SubagentKind,
        agent: AgentLabel,
        coro: Any,
        *,
        events: Optional[asyncio.Queue[SubagentEvent]] = None,
        on_finish: Optional[Callable[["SubagentHandle"], None]] = None,
    ) -> None:
        self.kind = kind
        self.agent = agent
        self.events: asyncio.Queue[SubagentEvent] = events or asyncio.Queue()
        loop = asyncio.get_running_loop()
        self.task: asyncio.Task = loop.create_task(coro)
        self._on_finish = on_finish
        self._replay: list[SubagentEvent] = []  # wait_final 跳过的 progress/intermediate
        self.task.add_done_callback(self._done)

    def _done(self, _task: asyncio.Task) -> None:
        if self._on_finish is not None:
            try:
                self._on_finish(self)
            except Exception:  # pragma: no cover
                logger.exception("子 Agent 收尾回调失败")

    @property
    def done(self) -> bool:
        return self.task.done()

    def cancel(self) -> None:
        """取消后台任务（同会话新回合 / 回合收尾时调用；副作用仅内存与草案，可安全丢弃）。"""
        if not self.task.done():
            self.task.cancel()

    async def wait_event(self, timeout: float) -> SubagentEvent:
        """取下一个事件；超时抛 asyncio.TimeoutError。"""
        return await asyncio.wait_for(self.events.get(), timeout=timeout)

    async def next_event(self, timeout: float) -> SubagentEvent:
        """取下一个事件（含 wait_final 期间跳过的 progress/intermediate）。"""
        if self._replay:
            return self._replay.pop(0)
        return await self.wait_event(timeout)

    async def wait_final(self, timeout: float) -> SubagentEvent:
        """持续消费事件直到 final/failed；期间 progress/intermediate 暂存（可用 next_event 取回）。

        超时抛 asyncio.TimeoutError。
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            ev = await self.wait_event(remaining)
            if ev.kind in ("final", "failed"):
                return ev
            self._replay.append(ev)


class SubagentBase:
    """LLM 工具循环共享实现（03 §5/§6）。"""

    kind: SubagentKind = "search"
    agent: AgentLabel = "search"
    default_ui_hint = "查资料中"

    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        tool_names: list[str],
        system_prompt: str,
        user_prompt: str,
        ctx: ToolContext,
        max_rounds: int,
        queue: asyncio.Queue[SubagentEvent],
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.tool_names = tool_names
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.ctx = ctx
        self.max_rounds = max_rounds
        self.queue = queue
        self.rounds_used = 0

    # ------------------------------------------------------------------

    def _tools_schema(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name in self.tool_names:
            tool = self.registry.get(name)
            if tool is None:
                continue
            out.append(tool.to_openai_tool())
        return out

    def _push(self, ev: SubagentEvent) -> None:
        try:
            self.queue.put_nowait(ev)
        except Exception:  # pragma: no cover
            pass

    def _terminal_payload(self, results: list[tuple[str, "ToolResult", ToolContext]]) -> Optional[dict[str, Any]]:
        """子类覆盖：从本轮工具结果中识别终态（如 draft_created），返回 final payload。"""
        return None

    async def _on_exhausted(self) -> Optional[dict[str, Any]]:
        """子类覆盖：轮限耗尽且无终态时的兜底（TaskRunner 的 fallback 草案）。"""
        return None

    def _on_round_results(self, results: list[tuple[str, "ToolResult", ToolContext]]) -> None:
        """子类覆盖：每轮工具结果后的进度/中间事件。"""

    def _final_from_text(self, text: str) -> dict[str, Any]:
        """子类覆盖：模型停止调用工具时，把文本转成 final payload。"""
        return {"findings": text}

    async def run(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        last_text = ""
        terminal: Optional[dict[str, Any]] = None

        self._push(SubagentEvent(kind="progress", ui_hint=self.default_ui_hint))

        try:
            for _round in range(self.max_rounds):
                self.rounds_used = _round + 1
                # 每轮动态取工具表（TaskRunner 的 prepare→draft 收窄依赖此）
                tools = self._tools_schema()
                result = await self.llm.chat(ChatRequest(
                    messages=messages,
                    tools=tools or None,
                    purpose="subagent",
                ))
                if result.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": result.content or "",
                        "tool_calls": result.tool_calls,
                    })
                elif result.content:
                    last_text = result.content
                    break  # 模型停止调用工具 → 视为结束

                batch = await self.registry.dispatch_batch(result.tool_calls, self.ctx)
                for _call_id, tr, new_ctx in batch:
                    self.ctx = new_ctx
                    messages.append({
                        "role": "tool",
                        "tool_call_id": _call_id,
                        "content": tr.result_json,
                    })
                self._on_round_results(batch)

                terminal = self._terminal_payload(batch)
                if terminal is not None:
                    break
            else:
                logger.warning("%s 达到轮限 %d", self.__class__.__name__, self.max_rounds)

            if terminal is not None:
                self._push(SubagentEvent(kind="final", payload=terminal))
                return

            fallback_payload = await self._on_exhausted()
            if fallback_payload is not None:
                self._push(SubagentEvent(kind="final", payload=fallback_payload))
                return

            self._last_model_text = last_text
            payload = self._final_from_text(last_text)
            self._push(SubagentEvent(kind="final", payload=payload))
        except asyncio.CancelledError:
            # 同会话新回合取消：副作用仅内存与草案，安全丢弃（01 §4）
            raise
        except Exception as exc:
            logger.exception("子 Agent %s 执行失败", self.kind)
            reason = (
                "当前模型不支持工具调用，该功能暂不可用"
                if isinstance(exc, LLMError) and is_tools_unsupported_error(exc)
                else f"{type(exc).__name__}: {exc}"
            )
            self._push(SubagentEvent(kind="failed", payload={"reason": reason}))


# ---------------------------------------------------------------------------
# 会话级子 Agent 登记（每会话最多 1 TaskRunner + 1 SearchRunner，01 §4）
# ---------------------------------------------------------------------------

class SessionSubagents:
    """单会话的活跃子 Agent 句柄。"""

    def __init__(self) -> None:
        self.task: Optional[SubagentHandle] = None
        self.search: Optional[SubagentHandle] = None

    def cancel_unfinished(self) -> list[SubagentKind]:
        """取消所有未完成的子 Agent，返回被取消的 kind 列表。"""
        cancelled: list[SubagentKind] = []
        for attr in ("task", "search"):
            handle: Optional[SubagentHandle] = getattr(self, attr)
            if handle is not None and not handle.done:
                handle.cancel()
                cancelled.append(handle.kind)
            setattr(self, attr, None)
        return cancelled

    def register(self, handle: SubagentHandle) -> None:
        if handle.agent == "task":
            self.task = handle
        else:
            self.search = handle


_SESSION_SUBAGENTS: dict[str, SessionSubagents] = {}
_SUBAGENTS_LOCK = asyncio.Lock()


async def get_session_subagents(session_id: str) -> SessionSubagents:
    async with _SUBAGENTS_LOCK:
        bucket = _SESSION_SUBAGENTS.get(session_id)
        if bucket is None:
            bucket = SessionSubagents()
            _SESSION_SUBAGENTS[session_id] = bucket
        return bucket


def reset_session_subagents() -> None:
    """测试用：清空登记表。"""
    _SESSION_SUBAGENTS.clear()


def count_pending_subagent_tasks() -> int:
    """测试断言用：全进程未完成的子 Agent 任务数。"""
    n = 0
    for bucket in _SESSION_SUBAGENTS.values():
        for handle in (bucket.task, bucket.search):
            if handle is not None and not handle.done:
                n += 1
    return n
