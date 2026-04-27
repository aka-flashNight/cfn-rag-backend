"""
向量存储后端抽象。

- LlamaIndexLocalBackend：封装现有本地 VectorStoreIndex（local profile 默认）
- QdrantBackend：基于 Qdrant 向量数据库（server profile 默认，见 qdrant_vector.py）
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """向量存储后端接口（结构化鸭子类型）。"""

    async def retrieve(
        self,
        query: str,
        filters: Any = None,
        top_k: int = 20,
    ) -> list[Any]:
        """检索最相关的 top_k 个节点，返回 NodeWithScore 列表。"""
        ...

    async def is_ready(self) -> bool:
        """检查向量索引是否就绪可用。"""
        ...

    async def rebuild_index(self) -> None:
        """强制重建索引（从原始数据重新构建）。"""
        ...

    async def close(self) -> None:
        """释放连接资源。"""
        ...


# ---------------------------------------------------------------------------
# LlamaIndexLocalBackend：封装现有 ai_engine 本地索引
# ---------------------------------------------------------------------------


class LlamaIndexLocalBackend:
    """基于 LlamaIndex VectorStoreIndex 的本地向量后端。

    直接复用 ``ai_engine.game_data_loader.get_cached_index()`` 的索引缓存。
    所有方法委托给同步 LlamaIndex API，通过 ``asyncio.to_thread`` 避免阻塞。
    """

    def __init__(self) -> None:
        self._index: Any = None

    def _get_index(self) -> Any:
        if self._index is None:
            from ai_engine.game_data_loader import get_cached_index

            self._index = get_cached_index()
        return self._index

    def invalidate_cache(self) -> None:
        self._index = None

    async def retrieve(
        self,
        query: str,
        filters: Any = None,
        top_k: int = 20,
    ) -> list[Any]:
        import asyncio

        index = self._get_index()
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        return await asyncio.to_thread(retriever.retrieve, query)

    async def is_ready(self) -> bool:
        import asyncio

        from ai_engine.game_data_loader import is_vector_index_valid, get_vector_index_dir

        return await asyncio.to_thread(is_vector_index_valid, get_vector_index_dir())

    async def rebuild_index(self) -> None:
        import asyncio

        from ai_engine.game_data_loader import rebuild_vector_index

        await asyncio.to_thread(rebuild_vector_index)
        self.invalidate_cache()

    async def close(self) -> None:
        self._index = None


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_vector_backend(settings: "core.config.Settings") -> VectorBackend:
    """根据 Settings 创建对应的向量后端实例。"""
    from core.config import Settings

    backend = settings.effective("vector_backend")
    if backend == "qdrant":
        from services.storage.qdrant_vector import QdrantBackend

        return QdrantBackend(url=settings.qdrant_url, collection=settings.qdrant_collection)
    return LlamaIndexLocalBackend()
