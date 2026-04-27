"""
SSE 会话心跳续期（路线四 · Server Profile）。

Server profile 下，每个 SSE 长连接持有一个 ``sess:{session_id}:alive`` Redis key
（TTL 90s）。每 15s 由服务端 SSE comment frame 触发一次续期。

客户端断开 90s 后 key 自动过期，Worker 或监控组件可据此判断活跃会话数。

设计要点：
- Local profile 下所有函数均为 no-op，零开销
- 心跳 tick 在 SSE 生成器内通过 ``asyncio.sleep(15)`` 驱动
- 不创建独立的心跳协程，避免泄漏
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)

# 心跳参数
ALIVE_KEY_PREFIX = "sess:"
ALIVE_KEY_SUFFIX = ":alive"
ALIVE_TTL = 90  # 秒，客户端断开后 key 存活时间
HEARTBEAT_INTERVAL = 15  # 秒，续期间隔


def _alive_key(session_id: str) -> str:
    return f"{ALIVE_KEY_PREFIX}{session_id}{ALIVE_KEY_SUFFIX}"


async def start_heartbeat(session_id: str, cache: Any | None = None) -> None:
    """SSE 开始时注册会话存活 key。

    仅在 server profile（cache 为 RedisCache）时生效。
    """
    if cache is None:
        return
    settings = get_settings()
    if not settings.use_redis:
        return
    try:
        await cache.set(_alive_key(session_id), "1", ttl=ALIVE_TTL)
        logger.debug("Heartbeat started: %s", session_id)
    except Exception:
        logger.warning("Heartbeat start failed for %s", session_id, exc_info=True)


async def tick_heartbeat(session_id: str, cache: Any | None = None) -> None:
    """SSE 传输期间每 HEARTBEAT_INTERVAL 调用一次，续期 TTL。

    不抛异常：心跳失败不应中断 SSE 流。
    """
    if cache is None:
        return
    settings = get_settings()
    if not settings.use_redis:
        return
    try:
        await cache.expire(_alive_key(session_id), ALIVE_TTL)
    except Exception:
        logger.debug("Heartbeat tick failed for %s", session_id, exc_info=True)


async def stop_heartbeat(session_id: str, cache: Any | None = None) -> None:
    """SSE 结束时删除会话存活 key。"""
    if cache is None:
        return
    settings = get_settings()
    if not settings.use_redis:
        return
    try:
        await cache.delete(_alive_key(session_id))
        logger.debug("Heartbeat stopped: %s", session_id)
    except Exception:
        logger.debug("Heartbeat stop failed for %s", session_id, exc_info=True)


async def is_session_alive(session_id: str, cache: Any | None = None) -> bool:
    """检查指定会话是否有活跃的 SSE 连接。"""
    if cache is None:
        return False
    settings = get_settings()
    if not settings.use_redis:
        return False
    try:
        return await cache.exists(_alive_key(session_id))
    except Exception:
        return False


async def heartbeat_generator(
    session_id: str,
    cache: Any | None = None,
) -> None:
    """在 SSE 生成器中使用：每 HEARTBEAT_INTERVAL 发一次 comment 帧 + 续期。

    用法::

        async for _ in heartbeat_generator(session_id, cache):
            yield b":\\n\\n"  # SSE comment frame
    """
    if cache is None:
        return
    settings = get_settings()
    if not settings.use_redis:
        return
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        await tick_heartbeat(session_id, cache)
        yield
