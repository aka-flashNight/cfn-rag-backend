"""
BM25 稀疏检索与 Dense/BM25 的 RRF 融合。

与向量索引共用同一批 LlamaIndex 节点的 `node_id`，便于检索层评估与 golden set 对齐。
持久化：`resources/tools/bm25_index/`（与 vector_index 同级目录）。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from rank_bm25 import BM25Okapi

from ai_engine.game_data_loader import cjk_tokenizer, get_db_path, iter_docstore_nodes


def get_bm25_index_dir() -> Path:
    return get_db_path().parent / "bm25_index"


def _node_text(node: BaseNode) -> str:
    t = getattr(node, "text", None) or ""
    return str(t).strip()


def filter_nodes_by_metadata(
    nodes: Sequence[BaseNode],
    filters: MetadataFilters | None,
) -> list[BaseNode]:
    if not filters or not getattr(filters, "filters", None):
        return list(nodes)

    def match(meta: dict[str, Any]) -> bool:
        for f in filters.filters:
            key = getattr(f, "key", None) or getattr(f, "filter_key", None)
            val = getattr(f, "value", None) or getattr(f, "filter_value", None)
            if key is None:
                continue
            if meta.get(key) != val:
                return False
        return True

    result: list[BaseNode] = []
    for n in nodes:
        meta = dict(n.metadata or {})
        if match(meta):
            result.append(n)
    return result


class BM25IndexCache:
    """在节点子集上构建 BM25；支持 pickle 缓存（按节点 id 集合指纹）。"""

    def __init__(self, nodes: Sequence[BaseNode]) -> None:
        self.nodes: list[BaseNode] = list(nodes)
        self._node_ids = [n.node_id for n in self.nodes]
        corpus_tokens = [cjk_tokenizer(_node_text(n)) for n in self.nodes]
        # rank_bm25 不接受空 token 列表行
        corpus_tokens = [t if t else [""] for t in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 20) -> list[NodeWithScore]:
        q_tokens = cjk_tokenizer(query or "")
        if not q_tokens:
            q_tokens = [""]
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            range(len(self.nodes)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        out: list[NodeWithScore] = []
        for i in ranked:
            s = float(scores[i])
            out.append(NodeWithScore(node=self.nodes[i], score=s))
        return out


_bm25_cache: dict[str, BM25IndexCache] = {}


def get_bm25_cache_for_nodes(nodes: Sequence[BaseNode]) -> BM25IndexCache:
    key = _fingerprint_nodes(nodes)
    if key not in _bm25_cache:
        _bm25_cache[key] = BM25IndexCache(nodes)
    return _bm25_cache[key]


def _fingerprint_nodes(nodes: Sequence[BaseNode]) -> str:
    ids = sorted({n.node_id for n in nodes})
    return f"{len(ids)}:" + ":".join(ids[:50])  # 截断防 key 过长


def clear_bm25_memory_cache() -> None:
    _bm25_cache.clear()


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[NodeWithScore]],
    k: int = 60,
    top_k: int = 20,
) -> list[NodeWithScore]:
    """
    RRF: score(d) = sum_i 1/(k + rank_i(d))。
    """
    scores: dict[str, float] = {}
    best_node: dict[str, BaseNode] = {}
    for rlist in ranked_lists:
        for rank, nws in enumerate(rlist):
            node = getattr(nws, "node", nws)
            nid = node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
            best_node[nid] = node

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        NodeWithScore(node=best_node[nid], score=sc)
        for nid, sc in merged
        if nid in best_node
    ]


def dense_retrieve(
    index: VectorStoreIndex,
    query: str,
    filters: MetadataFilters | None,
    top_k: int = 20,
) -> list[NodeWithScore]:
    r = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )
    return list(r.retrieve(query))


def bm25_retrieve(
    index: VectorStoreIndex,
    query: str,
    filters: MetadataFilters | None,
    top_k: int = 20,
) -> list[NodeWithScore]:
    all_nodes = iter_docstore_nodes(index)
    subset = filter_nodes_by_metadata(all_nodes, filters)
    if not subset:
        return []
    cache = get_bm25_cache_for_nodes(subset)
    return cache.retrieve(query, top_k=top_k)


def hybrid_rrf_retrieve(
    index: VectorStoreIndex,
    query: str,
    filters: MetadataFilters | None,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    d_nodes = dense_retrieve(index, query, filters, top_k=max(top_k * 3, 30))
    b_nodes = bm25_retrieve(index, query, filters, top_k=max(top_k * 3, 30))
    return reciprocal_rank_fusion([d_nodes, b_nodes], k=rrf_k, top_k=top_k)


def metadata_filters_for_golden_row(row: dict[str, Any]) -> MetadataFilters | None:
    """由 golden jsonl 行构造 MetadataFilters（单类型单字段）。"""
    t = (row.get("filter_type") or row.get("type") or "").strip()
    if not t:
        return None
    flist: list[MetadataFilter] = [MetadataFilter(key="type", value=t)]
    ch = row.get("filter_character")
    if ch is not None and str(ch).strip():
        flist.append(
            MetadataFilter(key="character", value=str(ch).strip().lower())
        )
    return MetadataFilters(filters=flist)
