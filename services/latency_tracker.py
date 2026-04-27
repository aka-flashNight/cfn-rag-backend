"""
首字延迟分步检测工具。

提供异步上下文管理器 LatencyTracker，用于追踪每一步动作的耗时。
仅在 CFN_AGENT_DEBUG_LATENCY 环境变量为 true/1/yes/on 时生效，
关闭时几乎零开销（仅一次 bool 检查）。
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Optional


def _latency_debug_enabled() -> bool:
    """检查是否启用了首字延迟调试。"""
    try:
        from core.config import get_settings

        if get_settings().cfn_agent_debug_latency:
            return True
    except Exception:
        pass
    import os

    v = (os.environ.get("CFN_AGENT_DEBUG_LATENCY") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _emit_latency(step_name: str, elapsed: float) -> None:
    """输出单步耗时到 stderr。"""
    print(
        f"[CFN_AGENT_DEBUG_LATENCY] {step_name} — {elapsed:.3f}s",
        file=sys.stderr,
        flush=True,
    )


class LatencyTracker:
    """异步上下文管理器，用于追踪步骤耗时。

    用法::

        async with LatencyTracker("RAG 多路检索"):
            result = await do_rag()

        # 或手动控制：
        tracker = LatencyTracker.start("某步骤")
        await do_something()
        tracker.end()
    """

    def __init__(self, step_name: str):
        self.step_name = step_name
        self._start: Optional[float] = None
        self._enabled = _latency_debug_enabled()

    async def __aenter__(self):
        if self._enabled:
            self._start = time.perf_counter()
        return self

    async def __aexit__(self, *args):
        if self._enabled and self._start is not None:
            _emit_latency(self.step_name, time.perf_counter() - self._start)

    @staticmethod
    def start(step_name: str) -> "LatencyTracker":
        """手动开始计时（返回 tracker，用 .end() 结束）。"""
        tracker = LatencyTracker(step_name)
        if tracker._enabled:
            tracker._start = time.perf_counter()
        return tracker

    def end(self) -> None:
        """手动结束计时并输出耗时。"""
        if self._enabled and self._start is not None:
            _emit_latency(self.step_name, time.perf_counter() - self._start)


async def latency_sleep_zero():
    """在追踪步骤之间插入微睡眠，使 stderr 输出顺序与异步事件循环一致。"""
    if _latency_debug_enabled():
        await asyncio.sleep(0)
