"""插话动作等待与 Host 回执。

Host 收到 SSE `action` 后执行 AS2，再 POST `/action-result`；本模块按
action_id 把回执交回正在等待的流。等待超时后迟到的回执走 late callback，
由物品工具补记账本但不补发本轮台词。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

from services.dialogue_interject.schemas import ActionResultRequest

logger = logging.getLogger(__name__)

#: 等待 Host 回执的上限（协议 §7.5.10 / 实施指引 §21.7）
ACTION_WAIT_TIMEOUT_S = 15.0
#: 幂等结果保留时间；超过后只靠持久表/进程重启前的窗口兜底
RESULT_TTL_S = 24 * 60 * 60


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    success: bool
    error: str = ""
    actual_quantity: int | None = None
    overflow_money: int | None = None
    received_at: float = 0.0

    @classmethod
    def from_request(cls, payload: ActionResultRequest) -> "ActionResult":
        return cls(
            action_id=payload.action_id,
            success=bool(payload.success),
            error=(payload.error or "").strip(),
            actual_quantity=payload.actual_quantity,
            overflow_money=payload.overflow_money,
            received_at=time.time(),
        )


class ActionRegistry:
    """进程内 action 等待表；一次插话进程内只保留一个回执。"""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[ActionResult]] = {}
        self._results: dict[str, ActionResult] = {}
        self._late_callbacks: dict[str, Callable[[ActionResult], None]] = {}

    def reset(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._results.clear()
        self._late_callbacks.clear()

    def register(self, action_id: str) -> asyncio.Future[ActionResult]:
        key = (action_id or "").strip()
        if not key:
            raise ValueError("action_id 不能为空")
        previous = self._pending.get(key)
        if previous is not None and not previous.done():
            return previous
        fut: asyncio.Future[ActionResult] = asyncio.get_running_loop().create_future()
        self._pending[key] = fut
        return fut

    def resolve(self, result: ActionResult) -> bool:
        """交回一次回执；返回 True＝命中正在等待的流。重复 action_id 只生效一次。"""
        key = (result.action_id or "").strip()
        if not key:
            return False
        if key in self._results:
            logger.info("[interject] action-result 重复回执已忽略 action_id=%s", key)
            return False
        self._results[key] = result
        self._prune_results()
        fut = self._pending.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(result)
            return True
        callback = self._late_callbacks.pop(key, None)
        if callback is not None:
            try:
                callback(result)
            except Exception:  # noqa: BLE001 - 迟到回执不能反向打断流
                logger.exception("[interject] 迟到 action-result 回调失败 action_id=%s", key)
        else:
            logger.info("[interject] action-result 无等待者，已留作幂等结果 action_id=%s", key)
        return False

    async def wait(
        self,
        action_id: str,
        *,
        timeout: float = ACTION_WAIT_TIMEOUT_S,
        on_late: Callable[[ActionResult], None] | None = None,
    ) -> ActionResult:
        key = (action_id or "").strip()
        fut = self._pending.get(key)
        if fut is None:
            # 回执可能已先到：直接取幂等结果
            cached = self._results.get(key)
            if cached is not None:
                return cached
            fut = self.register(key)
        # 超时/取消都可能错过回执：提前挂迟到回调，正常收到时在 finally 清掉
        if on_late is not None:
            self._late_callbacks[key] = on_late
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(key, None)
            if fut.done() and not fut.cancelled():
                self._late_callbacks.pop(key, None)

    def _prune_results(self) -> None:
        cutoff = time.time() - RESULT_TTL_S
        stale = [key for key, result in self._results.items() if result.received_at < cutoff]
        for key in stale:
            self._results.pop(key, None)


#: 生产/测试共用的进程级注册表；测试 teardown 调用 ACTIONS.reset()
ACTIONS = ActionRegistry()


def resolve_action_result(payload: ActionResultRequest) -> bool:
    return ACTIONS.resolve(ActionResult.from_request(payload))


def is_loopback_host(host: str | None) -> bool:
    text = (host or "").strip().lower()
    # TestClient 的默认 host；生产 Host 只走 127.0.0.1/::1。
    return text in {"", "127.0.0.1", "::1", "localhost", "testclient"}
