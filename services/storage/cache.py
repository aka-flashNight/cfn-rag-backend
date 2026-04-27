"""
缓存后端抽象：MemoryCache（local profile）与 RedisCache（server profile）。

所有后端遵循统一的 CacheBackend Protocol，调用方无需关心底层实现。
新增缓存操作均支持 LatencyTracker（与 CFN_AGENT_DEBUG_LATENCY 联动）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

from core.config import Settings
from services.latency_tracker import LatencyTracker


@runtime_checkable
class CacheBackend(Protocol):
    """缓存后端接口（结构化鸭子类型）。"""

    async def get(self, key: str) -> str | None:
        """读取缓存值，不存在返回 None。"""
        ...

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """写入缓存，可选 TTL（秒）。"""
        ...

    async def delete(self, key: str) -> None:
        """删除缓存键。"""
        ...

    async def exists(self, key: str) -> bool:
        """检查键是否存在。"""
        ...

    async def expire(self, key: str, ttl: int) -> None:
        """续期已有键的 TTL。"""
        ...

    async def incr(self, key: str) -> int:
        """原子递增计数器，返回新值。"""
        ...

    async def setnx(self, key: str, value: str, ttl: int | None = None) -> bool:
        """SETNX：仅键不存在时写入，返回是否写入成功。"""
        ...

    async def publish(self, channel: str, message: str) -> None:
        """Pub/Sub 发布消息（预留，MemoryCache 为空操作）。"""
        ...

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        """Pub/Sub 订阅消息（预留）。"""
        ...

    async def close(self) -> None:
        """释放连接资源。"""
        ...


# ---------------------------------------------------------------------------
# MemoryCache：基于 in-memory dict，local profile 默认
# ---------------------------------------------------------------------------


class MemoryCache:
    """基于 dict + asyncio.Lock 的轻量缓存，无外部依赖。"""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        async with self._lock:
            expires_at = (time.monotonic() + ttl) if ttl is not None else None
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            _, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._store[key]
                return False
            return True

    async def expire(self, key: str, ttl: int) -> None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store[key] = (entry[0], time.monotonic() + ttl)

    async def incr(self, key: str) -> int:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._store[key] = ("1", None)
                return 1
            val_str, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                self._store[key] = ("1", expires_at)
                return 1
            try:
                new_val = int(val_str) + 1
            except (ValueError, TypeError):
                new_val = 1
            self._store[key] = (str(new_val), expires_at)
            return new_val

    async def setnx(self, key: str, value: str, ttl: int | None = None) -> bool:
        async with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                _, expires_at = entry
                if expires_at is None or time.monotonic() <= expires_at:
                    return False
            expires_at = (time.monotonic() + ttl) if ttl is not None else None
            self._store[key] = (value, expires_at)
            return True

    async def publish(self, channel: str, message: str) -> None:
        pass  # in-memory 不做 pub/sub

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        # 永不产出（in-memory 无跨进程 pub/sub）
        if False:
            yield ""

    async def close(self) -> None:
        async with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# RedisCache：基于 redis.asyncio，server profile 默认
# ---------------------------------------------------------------------------


class RedisCache:
    """基于 redis.asyncio.Redis 的缓存后端。

    特性：
    - 连接池复用
    - 所有 key 自动添加 ``cfn:`` 前缀避免命名冲突
    - 支持 Pub/Sub（供未来跨 pod SSE 广播预留）
    """

    KEY_PREFIX = "cfn:"

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: Any = None
        self._pubsub: Any = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    def _k(self, key: str) -> str:
        return f"{self.KEY_PREFIX}{key}"

    async def get(self, key: str) -> str | None:
        with LatencyTracker("cache.get"):
            r = await self._ensure_client()
            return await r.get(self._k(key))

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        with LatencyTracker("cache.set"):
            r = await self._ensure_client()
            await r.set(self._k(key), value, ex=ttl)

    async def delete(self, key: str) -> None:
        with LatencyTracker("cache.delete"):
            r = await self._ensure_client()
            await r.delete(self._k(key))

    async def exists(self, key: str) -> bool:
        r = await self._ensure_client()
        return bool(await r.exists(self._k(key)))

    async def expire(self, key: str, ttl: int) -> None:
        r = await self._ensure_client()
        await r.expire(self._k(key), ttl)

    async def incr(self, key: str) -> int:
        with LatencyTracker("cache.incr"):
            r = await self._ensure_client()
            return await r.incr(self._k(key))

    async def setnx(self, key: str, value: str, ttl: int | None = None) -> bool:
        r = await self._ensure_client()
        if ttl is not None:
            return bool(await r.set(self._k(key), value, nx=True, ex=ttl))
        return bool(await r.set(self._k(key), value, nx=True))

    async def publish(self, channel: str, message: str) -> None:
        r = await self._ensure_client()
        await r.publish(self._k(channel), message)

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        r = await self._ensure_client()
        pubsub = r.pubsub()
        await pubsub.subscribe(self._k(channel))
        try:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    yield msg["data"]
        finally:
            await pubsub.unsubscribe(self._k(channel))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_cache(settings: Settings) -> CacheBackend:
    """根据 Settings 创建对应的缓存后端实例。"""
    backend = settings.effective("cache_backend")
    if backend == "redis":
        return RedisCache(url=settings.redis_url)
    return MemoryCache()
