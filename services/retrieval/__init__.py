"""services/retrieval：检索与向量模型（embedder / store / hybrid / pools / config / loader）。

对应 docs/v3-developer/04-检索与向量模型.md。对外主要入口：
- OnnxEmbedder / get_default_embedder
- VectorStore / Node / IndexStaleError
- RetrievalEngine / RetrievalInput / RetrievalBundle / rrf_fuse / cjk_tokenizer
- load_corpus / compute_corpus_fingerprint
- format_retrieval_context
"""

from services.retrieval.embedder import OnnxEmbedder, get_default_embedder, set_default_embedder
from services.retrieval.hybrid import (
    BM25Index,
    RetrievalBundle,
    RetrievalEngine,
    RetrievalInput,
    ScoredNode,
    build_queries,
    cjk_tokenizer,
    format_retrieval_context,
    format_sections,
    get_retrieval_engine,
    rrf_fuse,
    set_retrieval_engine,
)
from services.retrieval.store import (
    IndexStaleError,
    Node,
    VectorStore,
    get_index_dir,
)

__all__ = [
    "OnnxEmbedder",
    "get_default_embedder",
    "set_default_embedder",
    "BM25Index",
    "RetrievalBundle",
    "RetrievalEngine",
    "RetrievalInput",
    "ScoredNode",
    "build_queries",
    "cjk_tokenizer",
    "format_retrieval_context",
    "format_sections",
    "get_retrieval_engine",
    "rrf_fuse",
    "set_retrieval_engine",
    "IndexStaleError",
    "Node",
    "VectorStore",
    "get_index_dir",
]
