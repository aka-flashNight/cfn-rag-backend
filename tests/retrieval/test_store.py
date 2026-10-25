"""VectorStore 测试（04 §7.2）：指纹不匹配触发重建语义；npy 损坏报错并按 IndexStaleError 处理。"""

from __future__ import annotations

import numpy as np
import pytest

from services.retrieval.embedder import OnnxEmbedder
from services.retrieval.store import IndexStaleError, Node, VectorStore

from tests.conftest import FakeEmbedder


def _make_nodes(n=8) -> list[Node]:
    return [
        Node(id=f"n{i}", text=f"测试语料第{i}条：铁匠铺可以强化装备。", type="dialogue", character="smith")
        for i in range(n)
    ]


def test_build_and_save_load_roundtrip(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore.build(_make_nodes(), embedder, expected_fingerprint="fp-abc")
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path, expected_fingerprint="fp-abc")
    assert loaded.dim == FakeEmbedder.dim
    assert [n.id for n in loaded.nodes] == [n.id for n in store.nodes]
    # fp16 落盘有舍入，但矩阵应近似一致
    assert np.allclose(loaded.matrix, store.matrix, atol=1e-2)
    # 磁盘上应为 fp16（体积减半）
    disk = np.load(tmp_path / "vectors.npy", mmap_mode="r")
    assert disk.dtype == np.float16


def test_fingerprint_mismatch_raises_stale(tmp_path):
    store = VectorStore.build(_make_nodes(), FakeEmbedder(), expected_fingerprint="fp-old")
    store.save(tmp_path)
    with pytest.raises(IndexStaleError):
        VectorStore.load(tmp_path, expected_fingerprint="fp-new")


def test_missing_files_raise_stale(tmp_path):
    with pytest.raises(IndexStaleError):
        VectorStore.load(tmp_path, "fp-x")


def test_corrupted_npy_raises_stale(tmp_path):
    store = VectorStore.build(_make_nodes(), FakeEmbedder(), expected_fingerprint="fp-c")
    store.save(tmp_path)
    (tmp_path / "vectors.npy").write_bytes(b"garbage-not-a-npy-file")
    with pytest.raises(IndexStaleError):
        VectorStore.load(tmp_path, "fp-c")


def test_meta_row_count_mismatch_raises_stale(tmp_path):
    store = VectorStore.build(_make_nodes(), FakeEmbedder(), expected_fingerprint="fp-d")
    store.save(tmp_path)
    import json

    meta_path = tmp_path / "meta.json"
    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps(rows[:-1], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(IndexStaleError):
        VectorStore.load(tmp_path, "fp-d")


def test_build_empty_corpus_raises():
    with pytest.raises(ValueError):
        VectorStore.build([], FakeEmbedder(), "fp-e")


def test_real_onnx_embedder_dim_is_512():
    """维度从模型输出推导（消灭 C7 的 384 笔误）。模型缺失时跳过。"""
    import os
    from pathlib import Path

    model_dir = Path(__file__).resolve().parents[2] / "models" / "bge-small-zh-v1.5-onnx-int8"
    if not (model_dir / "model.onnx").exists():
        pytest.skip("int8 模型未导出")
    embedder = OnnxEmbedder(model_dir)
    assert embedder.dim == 512
    vecs = embedder.encode(["一句话", "另一句话"])
    assert vecs.shape == (2, 512)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
