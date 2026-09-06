"""
评估入口：检索层（v3 仅保留此轨道）。

示例::

    python -m evals.runners.run_all --suite retriever --dataset evals/datasets/tiny_golden.jsonl --sample 0
    python -m evals.runners.build_golden_set --tiny   # 先重建 golden 集（expected_doc_ids 对齐新索引）

说明：v2 的 Ragas 端到端轨道（evals/rag/）在 v3 中已移除——其实现深绑旧
GameRAGService / llm_client / NPCManager / MemoryManager 等已删除模块，且 ragas
依赖不属于运行时 venv。如需端到端质量评估，需基于 services.orchestrator +
services.llm 重建（见 docs/v3-developer/验收报告.md 遗留项）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("retriever",),
        default="retriever",
        help="v3 仅保留检索层评估轨道（Ragas 轨道已随旧架构移除）",
    )
    parser.add_argument(
        "--dataset",
        default="evals/datasets/golden_v1.jsonl",
        help="retriever 用的 jsonl；调试请显式指定 tiny_golden.jsonl",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="0=全量；>0 只评测前 N 条",
    )
    parser.add_argument(
        "--modes",
        default="dense,bm25,hybrid_rrf",
        help="逗号分隔",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "evals.retriever.eval_retriever",
        "--dataset",
        args.dataset,
        "--modes",
        args.modes,
    ]
    if args.sample:
        cmd.extend(["--sample", str(args.sample)])
    r = subprocess.run(cmd, cwd=str(_ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
