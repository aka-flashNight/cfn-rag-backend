"""集成运行时状态（单例）：启动模式与存档绑定投影。

- 启动模式：launcher.py 解析 `--embedded` 后置环境变量 `CFN_RAG_LAUNCH_MODE`，
  本模块导入时读取（进程级固定）；
- 运行时模式口径（需求文档 §3）：由启动参数 + 运行时 bind 状态共同决定——
  standalone 启动的 exe 被游戏 bind 后即按 embedded 口径工作；
- bind 载荷均为游戏投影的只读信息，exe 永不主动拉取；同槽位重复 bind＝投影刷新。
"""

from __future__ import annotations

import os
import re
import threading
from typing import Callable

#: 槽位键校验：镜像游戏侧 SaveSlotKey.cs:18（^[A-Za-z0-9_-]{1,128}$），拒绝路径分隔符
SLOT_KEY_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
_SLOT_KEY_RE = re.compile(SLOT_KEY_PATTERN)

#: 本进程前端页面的回环地址（7077 同时伺服 API 与前端，见开发计划 B1-1）
FRONTEND_BASE_URL = "http://127.0.0.1:7077"


def is_valid_slot_key(slot_key: str) -> bool:
    """槽位键是否为合法的存档标识（拒绝空串、路径分隔符、超长）。"""
    return bool(slot_key) and _SLOT_KEY_RE.fullmatch(slot_key) is not None


def frontend_url_for(mode: str) -> str:
    """按当前模式给前端页面 URL：embedded 带 `?embedded=1`（前端据此切换紧凑布局）。"""
    if mode == "embedded":
        return f"{FRONTEND_BASE_URL}/?embedded=1"
    return f"{FRONTEND_BASE_URL}/"


class IntegrationState:
    """进程内集成状态。bind/unbind 可来自任何线程，读写加锁。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.launch_mode = (
            "embedded"
            if os.environ.get("CFN_RAG_LAUNCH_MODE") == "embedded"
            else "standalone"
        )
        self._slot_key: str | None = None
        self._character: dict | None = None
        self._progress: dict | None = None

    @property
    def mode(self) -> str:
        """对外口径模式：启动即 embedded，或已被游戏 bind。"""
        with self._lock:
            return self._mode_locked()

    @property
    def slot_key(self) -> str | None:
        with self._lock:
            return self._slot_key

    def _mode_locked(self) -> str:
        if self.launch_mode == "embedded" or self._slot_key is not None:
            return "embedded"
        return "standalone"

    def bind(
        self,
        slot_key: str,
        character: dict | None = None,
        progress: dict | None = None,
    ) -> None:
        """绑定/刷新当前存档投影（幂等：同槽位覆盖式刷新，换槽位等价 re-bind）。"""
        with self._lock:
            self._slot_key = slot_key
            self._character = character
            self._progress = progress

    def unbind(self) -> None:
        """解除绑定（回标题/无存档状态）；数据行为退化为 standalone。"""
        with self._lock:
            self._slot_key = None
            self._character = None
            self._progress = None

    def snapshot(self) -> dict:
        """一次性快照（health/status 与后续阶段消费）。"""
        with self._lock:
            return {
                "launch_mode": self.launch_mode,
                "mode": self._mode_locked(),
                "bound_slot": self._slot_key,
                "character": self._character,
                "progress": self._progress,
            }


_STATE = IntegrationState()

_shutdown_lock = threading.Lock()
_process_shutdown_callback: Callable[[], None] | None = None


def set_process_shutdown_callback(callback: Callable[[], None]) -> None:
    """注册进程优雅退出回调（由 launcher.py 在启动 uvicorn 前调用）。"""
    global _process_shutdown_callback
    with _shutdown_lock:
        _process_shutdown_callback = callback


def request_process_shutdown() -> bool:
    """请求本进程优雅退出；未注册回调（非 launcher 托管）时返回 False。"""
    with _shutdown_lock:
        callback = _process_shutdown_callback
    if callback is None:
        return False
    callback()
    return True


def get_integration_state() -> IntegrationState:
    """获取进程内集成状态单例。"""
    return _STATE
