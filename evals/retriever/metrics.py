"""检索指标纯函数（与 RAG 指标解耦）。"""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    if not expected:
        return 1.0
    top = set(retrieved[:k])
    return len(expected & top) / len(expected)


def precision_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for x in top if x in expected)
    return hits / k


def mrr_at(expected: set[str], retrieved: list[str], max_rank: int) -> float:
    for i, rid in enumerate(retrieved[:max_rank], start=1):
        if rid in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """二元相关性：命中为 1，未命中为 0。"""
    gains = [1.0 if doc_id in expected else 0.0 for doc_id in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    rel_count = min(len(expected), k)
    ideal = [1.0] * rel_count + [0.0] * max(0, k - rel_count)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def aggregate_mean(rows: Iterable[float]) -> float:
    xs = [x for x in rows]
    if not xs:
        return 0.0
    return sum(xs) / len(xs)
