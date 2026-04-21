"""
检索层评估：dense / bm25 / hybrid_rrf 对比，不调 LLM。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# 保证可 `python -m evals.retriever.eval_retriever` 从仓库根运行
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_engine.bm25_retrieval import metadata_filters_for_golden_row
from evals.retriever.metrics import (
    aggregate_mean,
    mrr_at,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from services.game_rag_service import GameRAGService


def _node_id(nws: object) -> str:
    node = getattr(nws, "node", nws)
    return str(getattr(node, "node_id", "") or "")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _git_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def evaluate_one_mode(
    service: GameRAGService,
    mode: str,
    rows: list[dict],
    top_k: int = 20,
) -> dict:
    k_list = (5, 10, 20)
    prec_k = (5, 10)
    per_type: dict[str, list[dict]] = defaultdict(list)

    agg_r = {k: [] for k in k_list}
    agg_p = {k: [] for k in prec_k}
    agg_mrr: list[float] = []
    agg_ndcg: list[float] = []

    for row in rows:
        q = (row.get("retrieve_query") or row.get("question") or "").strip()
        filters = metadata_filters_for_golden_row(row)
        expected = set(row.get("expected_doc_ids") or [])
        typ = (row.get("type") or row.get("filter_type") or "unknown").strip()

        nws_list = service.retrieve_nodes_for_eval(
            mode, q, filters, top_k=top_k
        )
        retrieved = [_node_id(n) for n in nws_list]

        metrics_row = {
            "recall": {f"@{k}": recall_at_k(expected, retrieved, k) for k in k_list},
            "precision": {f"@{k}": precision_at_k(expected, retrieved, k) for k in prec_k},
            "mrr@10": mrr_at(expected, retrieved, 10),
            "ndcg@10": ndcg_at_k(expected, retrieved, 10),
        }
        for k in k_list:
            agg_r[k].append(metrics_row["recall"][f"@{k}"])
        for k in prec_k:
            agg_p[k].append(metrics_row["precision"][f"@{k}"])
        agg_mrr.append(metrics_row["mrr@10"])
        agg_ndcg.append(metrics_row["ndcg@10"])
        per_type[typ].append(metrics_row)

    summary = {
        "mode": mode,
        "n": len(rows),
        "recall": {f"@{k}": aggregate_mean(agg_r[k]) for k in k_list},
        "precision": {f"@{k}": aggregate_mean(agg_p[k]) for k in prec_k},
        "mrr@10": aggregate_mean(agg_mrr),
        "ndcg@10": aggregate_mean(agg_ndcg),
    }
    return {"summary": summary, "per_type": {k: v for k, v in per_type.items()}}


def _write_report(
    path: Path,
    modes: list[str],
    results: list[dict],
) -> None:
    lines = [
        f"# Retriever 评估报告",
        f"",
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        f"",
        "## 总体对比",
        f"",
        "| mode | n | R@5 | R@10 | R@20 | P@5 | P@10 | MRR@10 | nDCG@10 |",
        "|------|---|-----|------|------|-----|------|--------|---------|",
    ]
    for r, mode in zip(results, modes):
        s = r["summary"]
        lines.append(
            f"| {mode} | {s['n']} | "
            f"{s['recall']['@5']:.4f} | {s['recall']['@10']:.4f} | {s['recall']['@20']:.4f} | "
            f"{s['precision']['@5']:.4f} | {s['precision']['@10']:.4f} | "
            f"{s['mrr@10']:.4f} | {s['ndcg@10']:.4f} |"
        )
    lines.append("")
    lines.append("## 按 type 拆分（仅列出 recall@10）")
    lines.append("")
    all_types = set()
    for r in results:
        all_types.update(r.get("per_type", {}).keys())
    for t in sorted(all_types):
        lines.append(f"### {t}")
        lines.append("")
        lines.append("| mode | recall@10 |")
        lines.append("|------|-----------|")
        for r, mode in zip(results, modes):
            pt = r.get("per_type", {}).get(t) or []
            if not pt:
                rec = 0.0
            else:
                rec = aggregate_mean([x["recall"]["@10"] for x in pt])
            lines.append(f"| {mode} | {rec:.4f} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retriever-only evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/golden_v1.jsonl",
        help="jsonl golden 路径",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="0=全量；>0 只取前 N 条",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="dense,bm25,hybrid_rrf",
        help="逗号分隔",
    )
    args = parser.parse_args()

    ds_path = _ROOT / args.dataset
    if not ds_path.is_file():
        print(f"[eval] 数据集不存在: {ds_path}", file=sys.stderr)
        sys.exit(1)

    rows = _load_jsonl(ds_path)
    if args.sample and args.sample > 0:
        rows = rows[: args.sample]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    service = GameRAGService()

    results: list[dict] = []
    for mode in modes:
        t0 = time.perf_counter()
        results.append(evaluate_one_mode(service, mode, rows))
        print(f"[eval] mode={mode} elapsed={time.perf_counter() - t0:.2f}s")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gh = _git_short()
    out_md = _ROOT / "evals" / "reports" / f"retriever_{ts}_{gh}.md"
    out_json = _ROOT / "evals" / "reports" / f"retriever_{ts}_{gh}.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    _write_report(out_md, modes, results)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {"modes": modes, "dataset": str(ds_path), "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[eval] wrote {out_md}")
    print(f"[eval] wrote {out_json}")


if __name__ == "__main__":
    main()
