"""
评估入口：检索层 / Ragas / 全部。

示例::

    python -m evals.runners.run_all --suite retriever --sample 0
    python -m evals.runners.run_all --suite rag --dataset evals/datasets/tiny_golden.jsonl --sample 5
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
        choices=("retriever", "rag", "all"),
        default="retriever",
    )
    parser.add_argument(
        "--dataset",
        default="evals/datasets/golden_v1.jsonl",
        help="retriever/rag 共用的 jsonl（rag 默认可改为 tiny_golden.jsonl）",
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
        help="仅 retriever：逗号分隔",
    )
    args = parser.parse_args()

    if args.suite in ("retriever", "all"):
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

    if args.suite in ("rag", "all"):
        cmd = [
            sys.executable,
            "-m",
            "evals.rag.eval_rag",
            "--dataset",
            args.dataset,
        ]
        if args.sample:
            cmd.extend(["--sample", str(args.sample)])
        r = subprocess.run(cmd, cwd=str(_ROOT))
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
