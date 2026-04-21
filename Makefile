# Windows 可用: mingw32-make 或 nmake；亦可直接复制命令到 PowerShell。

.PHONY: eval-retriever eval-rag eval-all golden-tiny golden-full

golden-tiny:
	python -m evals.runners.build_golden_set --tiny

golden-full:
	python -m evals.runners.build_golden_set --full

eval-retriever:
	python -m evals.runners.run_all --suite retriever --dataset evals/datasets/tiny_golden.jsonl --sample 0

eval-rag:
	python -m evals.runners.run_all --suite rag --dataset evals/datasets/tiny_golden.jsonl --sample 0

eval-all:
	python -m evals.runners.run_all --suite all --dataset evals/datasets/tiny_golden.jsonl --sample 0
