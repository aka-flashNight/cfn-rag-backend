"""
存储抽象层（路线四）。

对外暴露统一的工厂函数，根据 Settings 返回对应的后端实现：

- ``create_cache(settings) -> CacheBackend``
- ``create_message_store(settings) -> MessageStore``
- ``create_session_store(settings) -> SessionStore``
- ``create_draft_store(settings) -> DraftStore``
- ``create_vector_backend(settings) -> VectorBackend``

用法::

    from core.config import get_settings
    from services.storage import create_cache

    settings = get_settings()
    cache = create_cache(settings)
    await cache.set("key", "value", ttl=300)
"""

from services.storage.cache import (
    CacheBackend,
    MemoryCache,
    RedisCache,
    create_cache,
)
from services.storage.db import (
    ChatMessage,
    DraftStore,
    MessageStore,
    SessionInfo,
    SessionStore,
    SqliteBackend,
    TaskDraft,
    create_draft_store,
    create_message_store,
    create_session_store,
)
from services.storage.vector import (
    LlamaIndexLocalBackend,
    VectorBackend,
    create_vector_backend,
)

__all__ = [
    # cache
    "CacheBackend",
    "MemoryCache",
    "RedisCache",
    "create_cache",
    # db
    "ChatMessage",
    "DraftStore",
    "MessageStore",
    "SessionInfo",
    "SessionStore",
    "SqliteBackend",
    "TaskDraft",
    "create_draft_store",
    "create_message_store",
    "create_session_store",
    # vector
    "LlamaIndexLocalBackend",
    "VectorBackend",
    "create_vector_backend",
]
