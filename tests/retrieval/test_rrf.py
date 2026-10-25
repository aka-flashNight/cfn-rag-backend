"""RRF 融合测试（04 §7.4）：构造已知排序的伪分数，验证融合序。"""

from __future__ import annotations

import numpy as np

from services.retrieval.hybrid import BM25Index, _ranks, rrf_fuse
from services.retrieval.store import Node


def _score(v: dict[int, float], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for i, s in v.items():
        out[i] = s
    return out


def test_ranks_order():
    scores = _score({0: 0.9, 1: 0.7, 2: 0.8}, 3)
    ranks = _ranks(scores)
    assert ranks[0] == 0  # 最高分
    assert ranks[2] == 1
    assert ranks[1] == 2


def test_rrf_known_ordering():
    """dense 排序 A>B>C，bm25 排序 B>C>A → 融合后 B 第一（两路排名加权最优）。

    k=60，rank r → 1/(61+r)：
      A: 1/61 + 1/63 = 0.0322664
      B: 1/62 + 1/61 = 0.0325224  ← 第一
      C: 1/63 + 1/62 = 0.0320020
    """
    n = 3
    dense = _score({0: 0.9, 1: 0.8, 2: 0.7}, n)   # A(rank0) > B(rank1) > C(rank2)
    bm25 = _score({1: 8.0, 2: 5.0, 0: 1.0}, n)    # B(rank0) > C(rank1) > A(rank2)
    fused = rrf_fuse(dense, bm25, k=60)
    order = np.argsort(-fused, kind="stable")
    assert list(order) == [1, 0, 2]
    assert fused[1] == pytest_approx(1 / 62 + 1 / 61)
    assert fused[0] == pytest_approx(1 / 61 + 1 / 63)
    assert fused[2] == pytest_approx(1 / 63 + 1 / 62)


def pytest_approx(expected: float):
    import pytest

    return pytest.approx(expected, rel=1e-6)


def test_rrf_k_parameter():
    n = 2
    dense = _score({0: 1.0, 1: 0.5}, n)
    bm25 = _score({1: 1.0, 0: 0.5}, n)
    fused_k1 = rrf_fuse(dense, bm25, k=1)
    assert fused_k1[0] == pytest_approx(1 / 2 + 1 / 3)
    assert fused_k1[1] == pytest_approx(1 / 3 + 1 / 2)


def test_bm25_index_scores():
    nodes = [
        Node(id="d1", text="收集猫爪交给铁匠", type="task", character="smith"),
        Node(id="d2", text="副本任务的推荐等级", type="task", character="andy"),
        Node(id="d3", text="世界大崩坏的历史", type="world_lore"),
    ]
    index = BM25Index(nodes)
    scores = index.scores("猫爪 铁匠")
    assert float(scores[0]) > float(scores[1])  # 含关键词的文档得分更高
    assert index.scores("").shape == (3,)  # 空 query 不崩溃
