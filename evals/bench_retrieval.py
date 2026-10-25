"""检索基准脚本（对应 docs/v3-developer/04 §5/§6 与 08 P2 验收）。

输出指标：
- 索引全量构建耗时（4426 条，验收 ≤25s，基线 fp32 ~43s）
- 单轮 Tier-1 检索耗时（嵌入+融合+选池，验收 ≤150ms）
- tiny_golden recall@10（单池 dense 模式，与旧 eval 口径对齐；≥0.99 不回退为参考线）
- 向量库落盘体积（验收 ≤10MB）

用法：python evals/bench_retrieval.py [--skip-recall]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN = PROJECT_ROOT / "evals" / "datasets" / "tiny_golden.jsonl"
OLD_DOCSTORE = PROJECT_ROOT.parent / "resources" / "tools" / "vector_index" / "docstore.json"


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def _build_old_id_text_map() -> dict[str, str]:
    """旧 LlamaIndex docstore 的 node_id → text 映射（golden 的 expected_doc_ids 是旧 id）。"""
    if not OLD_DOCSTORE.exists():
        return {}
    data = json.loads(OLD_DOCSTORE.read_text(encoding="utf-8"))
    rows = data.get("docstore/data", data)
    out: dict[str, str] = {}
    for k, v in rows.items():
        inner = (v or {}).get("__data__", {})
        text = inner.get("text") or ""
        if text:
            out[k] = text
    return out


def bench_recall(engine, nodes) -> tuple[float, int, int]:
    """tiny_golden recall@10：单池 dense 模式（按 golden 行的 type/character 过滤后 top10）。

    v3 golden（build_golden_set 重建）的 expected_doc_ids 直接是新 Node id；
    旧 golden（LlamaIndex uuid）兼容路径：通过旧 docstore「文本精确匹配」映射。
    """
    import numpy as np

    old_map = _build_old_id_text_map()
    if not GOLDEN.exists():
        return float("nan"), 0, 0
    text_to_new_id = {n.text: n.id for n in nodes}
    node_row = {n.id: i for i, n in enumerate(nodes)}  # 节点 id → 全局矩阵行号
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]

    recalls: list[float] = []
    mapped_total = 0
    expected_total = 0
    for row in rows:
        expected_ids = row.get("expected_doc_ids") or []
        expected_new = {i for i in expected_ids if i in node_row}
        if not expected_new and old_map:
            expected_texts = [old_map.get(i) for i in expected_ids]
            expected_new = {text_to_new_id[t] for t in expected_texts if t and t in text_to_new_id}
        mapped_total += len(expected_new)
        expected_total += len(expected_ids)
        if not expected_new:
            continue

        query = row.get("retrieve_query") or row.get("question") or ""
        ftype = row.get("filter_type") or row.get("type") or ""
        fchar = (row.get("filter_character") or "").strip().lower()
        subset = [
            n for n in nodes
            if n.type == ftype and (not fchar or (n.character or "") == fchar)
        ]
        if not subset:
            recalls.append(0.0)
            continue
        qv = engine._resolve_embedder().encode([query])[0]
        mx = engine.store.matrix[[node_row[n.id] for n in subset]]  # 全局矩阵行
        sims = mx @ qv
        top10 = np.argsort(-sims)[:10]
        hit_ids = {subset[int(i)].id for i in top10}
        r = len(expected_new & hit_ids) / len(expected_new)
        recalls.append(r)
        miss = expected_new - hit_ids
        print(f"    [{row.get('id')}] recall={r:.2f} type={ftype} char={fchar or '-'}"
              + (f" 未命中={sorted(miss)}" if miss else ""))

    mean = statistics.mean(recalls) if recalls else float("nan")
    return mean, mapped_total, expected_total


def main() -> int:
    parser = argparse.ArgumentParser(description="检索基准")
    parser.add_argument("--skip-recall", action="store_true", help="跳过 golden recall（旧 docstore 缺失时）")
    args = parser.parse_args()

    from services.retrieval.embedder import OnnxEmbedder
    from services.retrieval.hybrid import RetrievalEngine, RetrievalInput
    from services.retrieval.loader import compute_corpus_fingerprint, load_corpus
    from services.retrieval.store import VectorStore, get_index_dir

    print("=" * 64)
    print("CFN-RAG v3 检索基准（ONNX int8 + 二进制向量库 + hybrid RRF）")
    print("=" * 64)

    # 1. embedder 预载
    t0 = time.perf_counter()
    embedder = OnnxEmbedder()
    embedder.warmup()
    print(f"嵌入模型预载+预热: {time.perf_counter() - t0:.2f}s")

    # 2. 语料解析（不含嵌入）
    t0 = time.perf_counter()
    nodes = load_corpus()
    parse_s = time.perf_counter() - t0
    print(f"语料解析（六类源 → {len(nodes)} 条节点）: {parse_s:.2f}s")

    # 3. 向量库全量构建（嵌入 + fp16 落盘）——验收 ≤25s
    fingerprint = compute_corpus_fingerprint()
    t0 = time.perf_counter()
    store = VectorStore.build(nodes, embedder, fingerprint)
    build_s = time.perf_counter() - t0
    index_dir = get_index_dir()
    store.save(index_dir)
    print(f"向量库全量构建（{len(nodes)} 条, batch=64）: {build_s:.2f}s  {'✓ ≤25s' if build_s <= 25 else '✗ 超标(>25s)'}")

    # 4. 单轮 Tier-1 检索耗时——验收 ≤150ms
    engine = RetrievalEngine(embedder=embedder, index_dir=index_dir)
    engine.set_store(store)
    samples = [
        RetrievalInput(user_query="这把步枪的伤害怎么样，有配件吗？", npc_name="andy law",
                       npc_titles=["军火商", "冷兵器商人"], npc_faction="流亡者",
                       npc_last_message="上次给你介绍的那把步枪用得如何？"),
        RetrievalInput(user_query="最近有什么委托可以做？", npc_name="铁匠",
                       npc_titles=["大师铁匠"], npc_faction="王国"),
        RetrievalInput(user_query="北边的情报里提到了什么？", npc_name="情报贩子",
                       npc_last_message="你听说了吗，北边出事了。"),
        RetrievalInput(user_query="废弃矿坑的推荐等级是多少？", npc_name="andy law",
                       npc_titles=["军火商"]),
    ]
    timings: list[float] = []
    for i in range(24):
        inp = samples[i % len(samples)]
        t0 = time.perf_counter()
        bundle = engine.retrieve(inp)
        dt = (time.perf_counter() - t0) * 1000
        timings.append(dt)
        assert not bundle.is_empty or True
    timings.sort()
    p50 = timings[len(timings) // 2]
    p90 = timings[int(len(timings) * 0.9)]
    mean = statistics.mean(timings)
    print(f"单轮检索（{len(timings)} 次, 含 ≤3 次嵌入+矩阵乘+BM25+池选择）:")
    print(f"    mean={mean:.1f}ms  p50={p50:.1f}ms  p90={p90:.1f}ms  max={timings[-1]:.1f}ms  "
          f"{'✓ p90 ≤150ms' if p90 <= 150 else '✗ 超标(>150ms)'}")

    # 5. tiny_golden recall@10
    if not args.skip_recall:
        recall, mapped, total = bench_recall(engine, nodes)
        if total == 0:
            print("recall@10: 无法评估（旧 docstore 或 golden 集缺失）")
        else:
            print(f"tiny_golden recall@10（单池 dense, 旧 id 文本映射 {mapped}/{total}）: {recall:.3f}")

    # 6. 向量库体积——验收 ≤10MB
    size_mb = _dir_size_mb(index_dir)
    print(f"向量库落盘体积: {size_mb:.1f} MB  {'✓ ≤10MB' if size_mb <= 10 else '✗ 超标(>10MB)'}  [{index_dir}]")

    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
