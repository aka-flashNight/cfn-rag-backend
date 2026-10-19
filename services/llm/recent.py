"""最近一次生效的 LLM 连接配置（进程内，本地单用户）。

配置的唯一持久化位置是终端前端的 localStorage；后端只保留内存副本：
- 终端页每次 ask 都带页面配置，这里记住最近一次非空配置；
- 游戏内配置小窗（同源 /config）保存时也 POST 到 /api/game/llm-config，
  后端据此更新内存，供插话复用。
exe 重启后内存为空；小窗会从 localStorage 读回配置并重新 POST，不需要重新填写。
"""

from __future__ import annotations

import threading
import time

from services.llm.client import LLMConfig

_lock = threading.Lock()
_recent: LLMConfig | None = None
_updated_at: int = 0


def remember_llm_config(config: LLMConfig | None, updated_at: int | None = None) -> None:
    """记住一次带 api_key 的配置；空配置不覆盖已有值。"""
    global _recent, _updated_at
    if config is None or not config.api_key.strip():
        return
    with _lock:
        _recent = config
        _updated_at = int(updated_at if updated_at and updated_at > 0 else time.time() * 1000)


def replace_llm_config(config: LLMConfig | None, updated_at: int | None = None) -> None:
    """显式保存（含清空）时整体替换内存配置，不再回落到旧值。"""
    global _recent, _updated_at
    with _lock:
        _recent = config
        _updated_at = int(updated_at if updated_at and updated_at > 0 else time.time() * 1000)


def get_recent_llm_config() -> LLMConfig | None:
    with _lock:
        return _recent


def get_recent_updated_at() -> int:
    with _lock:
        return _updated_at
