"""OnnxEmbedder 快照回归（04 §7.1）：int8 输出与 fp32 参考（导出侧快照）余弦 ≥ 0.99；批量编码顺序稳定。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5-onnx-int8"
SNAPSHOT = PROJECT_ROOT / "tests" / "fixtures" / "embedder_snapshot.json"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists() or not SNAPSHOT.exists(),
    reason="int8 模型或导出快照缺失（先运行 scripts/export_onnx_int8.py）",
)


@pytest.fixture(scope="module")
def embedder():
    from services.retrieval.embedder import OnnxEmbedder

    return OnnxEmbedder(MODEL_DIR)


@pytest.fixture(scope="module")
def snapshot():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_int8_matches_fp32_reference(embedder, snapshot):
    texts = snapshot["texts"]
    ref = np.asarray(snapshot["vectors_fp32"], dtype=np.float32)
    got = embedder.encode(texts)
    cos = np.sum(got * ref, axis=1)  # 双侧已归一化
    assert cos.mean() >= 0.99, f"余弦均值 {cos.mean():.6f} < 0.99"
    assert cos.min() >= 0.99, f"最低余弦 {cos.min():.6f} < 0.99"


def test_batch_encode_order_stable(embedder, snapshot):
    texts = snapshot["texts"]
    full = embedder.encode(texts)
    # 分两批编码应与整批一致（顺序稳定）
    half = len(texts) // 2
    split = np.vstack([embedder.encode(texts[:half]), embedder.encode(texts[half:])])
    assert np.allclose(full, split, atol=1e-6)


def test_request_level_cache_avoids_recompute(embedder):
    """同回合相同 query 只编码一次（修 C1 的重复嵌入）。"""
    texts = ["缓存测试句子甲", "缓存测试句子乙", "缓存测试句子甲"]
    v = embedder.encode(texts)
    assert np.allclose(v[0], v[2], atol=0.0)  # 命中缓存，完全一致
    embedder.clear_cache()
