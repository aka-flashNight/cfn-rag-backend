"""索取 claim 状态机、本组赠送明细与最近成功/失败信息。

只保存进程内会话状态；跨对话物品数量在 `item_ledger.py` 的 SQLite 账本。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

#: 协议 §7.5.6：pending claim 10 分钟超时
CLAIM_TIMEOUT_S = 10 * 60
#: 最近成功/失败保留轮数
RECENT_TURNS = 3


class ClaimExistsError(RuntimeError):
    """同一会话已有待处理索取。"""


@dataclass
class ItemClaim:
    claim_id: str
    npc: str
    item_name: str
    quantity: int
    turn_no: int
    created_at: float = field(default_factory=time.time)

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) - self.created_at > CLAIM_TIMEOUT_S


@dataclass(frozen=True)
class GiftRecord:
    turn_no: int
    item_name: str
    quantity: int


class ItemClaimStore:
    """会话级状态；一次只允许一个 pending claim。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: dict[str, ItemClaim] = {}
        self._gifts: dict[str, list[GiftRecord]] = {}
        self._success: dict[str, Deque[str]] = {}
        self._failure: dict[str, Deque[str]] = {}
        #: 组内轮次登记：turn_id → 第 N 次请求（同一 turn 重发不重复计数）
        self._turns: dict[str, list[str]] = {}

    # -- 轮次 -----------------------------------------------------------

    def turn_number(self, session_id: str, turn_id: str) -> int:
        """本组内第 N 次插话请求（从 1 开始）；同一 turn_id 幂等。"""
        key = (session_id or "").strip()
        turn = (turn_id or "").strip()
        if not key or not turn:
            return 1
        with self._lock:
            turns = self._turns.setdefault(key, [])
            try:
                return turns.index(turn) + 1
            except ValueError:
                turns.append(turn)
                return len(turns)

    # -- claim ---------------------------------------------------------

    def pending(self, session_id: str) -> ItemClaim | None:
        key = (session_id or "").strip()
        if not key:
            return None
        with self._lock:
            claim = self._pending.get(key)
            if claim is None:
                return None
            if claim.expired():
                self._pending.pop(key, None)
                return None
            return claim

    def create(
        self,
        session_id: str,
        *,
        npc: str,
        item_name: str,
        quantity: int,
        turn_no: int,
    ) -> ItemClaim:
        key = (session_id or "").strip()
        if not key:
            raise ValueError("session_id 不能为空")
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None and not existing.expired():
                raise ClaimExistsError("同一会话已有待处理索取")
            claim = ItemClaim(
                claim_id=str(uuid.uuid4()),
                npc=(npc or "").strip(),
                item_name=(item_name or "").strip(),
                quantity=max(1, int(quantity or 1)),
                turn_no=max(1, int(turn_no or 1)),
            )
            self._pending[key] = claim
            return claim

    def close(self, session_id: str, claim_id: str | None = None) -> ItemClaim | None:
        key = (session_id or "").strip()
        with self._lock:
            claim = self._pending.get(key)
            if claim is None:
                return None
            if claim_id and claim.claim_id != claim_id:
                return None
            self._pending.pop(key, None)
            return claim

    # -- 本组赠送明细 ---------------------------------------------------

    def record_gift(
        self,
        session_id: str,
        *,
        turn_no: int,
        item_name: str,
        quantity: int,
    ) -> None:
        key = (session_id or "").strip()
        if not key or quantity <= 0:
            return
        with self._lock:
            rows = self._gifts.setdefault(key, [])
            for index, row in enumerate(rows):
                if row.turn_no == turn_no and row.item_name == item_name:
                    rows[index] = GiftRecord(
                        turn_no=row.turn_no,
                        item_name=row.item_name,
                        quantity=row.quantity + quantity,
                    )
                    return
            rows.append(GiftRecord(turn_no=turn_no, item_name=item_name, quantity=quantity))

    def gift_detail_block(self, session_id: str) -> str:
        key = (session_id or "").strip()
        with self._lock:
            rows = list(self._gifts.get(key, []))
        if not rows:
            return ""
        rows.sort(key=lambda row: (row.turn_no, row.item_name))
        lines = [
            f"第{row.turn_no}轮：{row.quantity}个{row.item_name}"
            for row in rows
        ]
        return "【本组已赠送】\n" + "\n".join(lines)

    # -- 最近成功/失败 ---------------------------------------------------

    def record_success(self, session_id: str, text: str) -> None:
        self._push(self._success, session_id, text)

    def record_failure(self, session_id: str, text: str) -> None:
        self._push(self._failure, session_id, text)

    def recent_block(self, session_id: str) -> str:
        key = (session_id or "").strip()
        with self._lock:
            success = list(self._success.get(key, ()))
            failure = list(self._failure.get(key, ()))
        lines: list[str] = []
        if success:
            lines.extend(success)
        if failure:
            lines.extend(f"上次物品往来失败：{text}" for text in failure)
        return "【最近物品往来】\n" + "\n".join(lines) if lines else ""

    # -- 清理 -----------------------------------------------------------

    def reset(self, session_id: str) -> None:
        key = (session_id or "").strip()
        if not key:
            return
        with self._lock:
            self._pending.pop(key, None)
            self._gifts.pop(key, None)
            self._success.pop(key, None)
            self._failure.pop(key, None)
            self._turns.pop(key, None)

    def reset_all(self) -> None:
        with self._lock:
            self._pending.clear()
            self._gifts.clear()
            self._success.clear()
            self._failure.clear()
            self._turns.clear()

    def _push(self, bucket: dict[str, Deque[str]], session_id: str, text: str) -> None:
        key = (session_id or "").strip()
        value = (text or "").strip()
        if not key or not value:
            return
        with self._lock:
            queue = bucket.get(key)
            if queue is None:
                queue = deque(maxlen=RECENT_TURNS)
                bucket[key] = queue
            queue.append(value)


#: 生产/测试共用的进程级状态；测试 teardown 调用 reset_all()
ITEM_CLAIMS = ItemClaimStore()
