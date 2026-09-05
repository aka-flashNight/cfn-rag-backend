"""SSE 事件构造（对应 docs/v3-developer/01 §5 契约、09-前端适配说明 §1）。

本模块是 SSE 事件契约的**实现权威**：
- meta          {emotion, favorability_change, favorability, relationship_level}  每回合一次、必先于 content
- content       {delta}                                                           纯台词增量
- tool_status   {tool, status, ui_hint}                                           status ∈ running/success/failed
- agent_status  {agent, phase, ui_hint}                                           agent ∈ task/search；phase ∈ drafting/repairing/searching/done/failed
- system_notice {text}                                                            替代旧 {花括号} 约定
- done          {session_id, usage}                                               回合结束
- error         {code, message, retryable}                                        不可恢复错误
- keep_alive    无 data（SSE 注释帧）                                              等待子 Agent 时的保活
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SSEEvent:
    """单条 SSE 事件（event 名 + data dict）。"""

    event: str
    data: dict[str, Any]

    def encode(self) -> bytes:
        """编码为 SSE 帧；keep_alive 编码为注释帧。"""
        if self.event == "keep_alive":
            return b": keep-alive\n\n"
        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event}\ndata: {payload}\n\n".encode("utf-8")


def meta_event(
    *,
    emotion: str,
    favorability_change: int,
    favorability: int,
    relationship_level: str,
) -> SSEEvent:
    return SSEEvent(event="meta", data={
        "emotion": emotion,
        "favorability_change": favorability_change,
        "favorability": favorability,
        "relationship_level": relationship_level,
    })


def content_event(delta: str) -> SSEEvent:
    return SSEEvent(event="content", data={"delta": delta})


def tool_status_event(tool: str, status: str, ui_hint: str = "") -> SSEEvent:
    return SSEEvent(event="tool_status", data={
        "tool": tool,
        "status": status,
        "ui_hint": ui_hint,
    })


def agent_status_event(agent: str, phase: str, ui_hint: str = "") -> SSEEvent:
    return SSEEvent(event="agent_status", data={
        "agent": agent,
        "phase": phase,
        "ui_hint": ui_hint,
    })


def system_notice_event(text: str) -> SSEEvent:
    return SSEEvent(event="system_notice", data={"text": text})


def done_event(session_id: str, usage: dict[str, Any] | None = None) -> SSEEvent:
    data: dict[str, Any] = {"session_id": session_id}
    if usage:
        data["usage"] = usage
    return SSEEvent(event="done", data=data)


def error_event(code: str, message: str, *, retryable: bool = False) -> SSEEvent:
    return SSEEvent(event="error", data={
        "code": code,
        "message": message,
        "retryable": retryable,
    })


KEEP_ALIVE = SSEEvent(event="keep_alive", data={})


def accumulate_usage(total: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any] | None:
    """跨调用累计 usage（prompt/completion/cached 求和，total 派生）。"""
    if not incoming:
        return total
    base = dict(total or {})
    for key in ("prompt_tokens", "completion_tokens"):
        base[key] = int(base.get(key) or 0) + int(incoming.get(key) or 0)
    if incoming.get("cached_tokens") is not None:
        base["cached_tokens"] = int(base.get("cached_tokens") or 0) + int(incoming["cached_tokens"])
    if base.get("prompt_tokens") or base.get("completion_tokens"):
        base["total_tokens"] = int(base.get("prompt_tokens") or 0) + int(base.get("completion_tokens") or 0)
    return base
