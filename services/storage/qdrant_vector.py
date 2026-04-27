"""
Qdrant 向量后端（路线四 · Server Profile）。

实现 ``VectorBackend`` Protocol，使用 ``qdrant-client`` 连接 Qdrant 向量数据库。

Collection 设计：
- 向量维度：384（bge-small-zh-v1.5）
- 距离度量：Cosine
- Payload 包含 LlamaIndex 节点元数据（type, character, task_source 等）
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# bge-small-zh-v1.5 输出维度
EMBEDDING_DIM = 384


class QdrantBackend:
    """基于 Qdrant 的向量检索后端。

    检索逻辑等价于 ``VectorStoreIndex.as_retriever().retrieve()``：
    传入 query embedding → Qdrant 搜索 → 返回带相似度分数的节点列表。
    """

    def __init__(self, url: str, collection: str = "cfn_game") -> None:
        self._url = url
        self._collection_name = collection
        self._client: Any = None
        self._embed_model: Any = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(url=self._url)
        return self._client

    def _get_embed_model(self) -> Any:
        if self._embed_model is None:
            from ai_engine.game_data_loader import ensure_embed_model
            from llama_index.core import Settings

            ensure_embed_model(offline=True)
            self._embed_model = Settings.embed_model
        return self._embed_model

    async def _ensure_collection(self) -> None:
        client = await self._ensure_client()
        from qdrant_client.models import Distance, VectorParams

        exists = await client.collection_exists(self._collection_name)
        if not exists:
            await client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("创建 Qdrant collection: %s", self._collection_name)

    async def retrieve(
        self,
        query: str,
        filters: Any = None,
        top_k: int = 20,
    ) -> list[Any]:
        import asyncio

        collection = self._collection_name
        await self._ensure_collection()
        embed_model = self._get_embed_model()

        query_embedding = await asyncio.to_thread(
            embed_model.get_text_embedding, query
        )

        client = await self._ensure_client()

        from qdrant_client.models import Filter

        qdrant_filter = _build_qdrant_filter(filters) if filters is not None else None

        results = await client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        nodes = []
        for hit in results:
            payload = hit.payload or {}
            node = _hit_to_node(hit.id, hit.score, payload)
            nodes.append(node)
        return nodes

    async def is_ready(self) -> bool:
        try:
            client = await self._ensure_client()
            return await client.collection_exists(self._collection_name)
        except Exception:
            return False

    async def rebuild_index(self) -> None:
        # Qdrant 重建由 Worker 任务执行（见 worker/tasks.py）
        logger.info("Qdrant rebuild_index 应由 Worker rebuild_vector_index 任务触发")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Qdrant filter 转换
# ---------------------------------------------------------------------------


def _build_qdrant_filter(filters: Any) -> Any:
    """将 LlamaIndex MetadataFilters 转换为 Qdrant Filter。

    MetadataFilters 结构：
      - filters: list[MetadataFilter]
      - condition: "and" | "or"

    MetadataFilter 结构：
      - key: str
      - value: str | list[str]
      - operator: "==" | "in" | ...
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must_conditions = []

    # 兼容多种 MetadataFilters 结构
    filter_list = getattr(filters, "filters", None)
    if filter_list is None:
        return None

    for mf in filter_list:
        key = getattr(mf, "key", None)
        value = getattr(mf, "value", None)
        if key is None or value is None:
            continue

        # 简化为 exact match（Qdrant 的 MatchValue）
        if isinstance(value, list):
            # 复合条件：用 should (OR)
            should_conditions = [
                FieldCondition(key=key, match=MatchValue(value=v))
                for v in value
            ]
            must_conditions.append(Filter(should=should_conditions))
        else:
            must_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    if not must_conditions:
        return None

    condition = getattr(filters, "condition", "and")
    if condition == "or":
        return Filter(should=must_conditions)
    return Filter(must=must_conditions)


def _hit_to_node(hit_id: Any, score: float, payload: dict) -> Any:
    """将 Qdrant 搜索结果转换为兼容 LlamaIndex NodeWithScore 的对象。"""
    from llama_index.core.schema import NodeWithScore, TextNode

    text = payload.pop("_node_content", payload.pop("text", ""))
    metadata = {k: v for k, v in payload.items()}

    node = TextNode(
        id_=str(hit_id),
        text=str(text),
        metadata=metadata,
    )
    return NodeWithScore(node=node, score=score)
