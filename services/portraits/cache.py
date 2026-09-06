"""立绘处理结果进程内缓存（对应 docs/v3-developer/07 §3，修 F2）。

key = (npc_key, 命中情绪, 源文件 mtime)：
- npc_key / 情绪：manifest 查表实际命中的规范 key 与表情；
- mtime：源 PNG 修改时间——manifest 随烘焙再生成（hash 目录名变化/图片更新）后
  自动 miss 重算，不缓存跨版本路径（协议 §6.7）。

线程安全 LRU，≤64 项（WebP data URL 与 PNG bytes 均为几十~几百 KB，内存可控）。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional

from services.portraits.manifest_lookup import DEFAULT_EXPRESSION

DEFAULT_MAX_SIZE = 64


class PortraitImageCache:
    """简单线程安全 LRU（OrderedDict move_to_end）。"""

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self._max_size = max(1, int(max_size))
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def portrait_cache_key(npc_key: str, expression_used: Optional[str], mtime: float) -> tuple:
    """缓存 key：(npc_key, 命中情绪, 源文件 mtime)。情绪空值归一为「普通」。"""
    return (npc_key, expression_used or DEFAULT_EXPRESSION, mtime)


_CACHE = PortraitImageCache()


def get_cache() -> PortraitImageCache:
    """全局缓存单例（测试可用 PortraitImageCache 自建实例绕开）。"""
    return _CACHE
