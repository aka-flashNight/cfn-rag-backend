# 评估体系（路线 1）

本目录包含**两条独立轨道**，请勿混用指标含义：

## 1. 检索层评估（Retriever Eval）

- **评什么**：给定 query，retriever 是否召回到「标注相关」的 chunk（按 `node_id`）。
- **指标**：`recall@k`、`precision@k`、`MRR@n`、`nDCG@k`（k ∈ {5,10,20} 等）。
- **模式**：`dense`（向量） / `bm25`（稀疏） / `hybrid_rrf`（RRF 融合，k=60）。
- **特点**：**不调 LLM**，无 API 费用，适合频繁回归。

## 2. RAG 端到端评估（Ragas）

- **评什么**：检索 + 生成整体质量（上下文是否支撑回答、回答是否切题等）。
- **指标**：`faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`；设定类题目可加 `answer_correctness`（需参考句）。
- **特点**：需要 **Judge LLM**（使用项目 `.env` 中的 OpenAI 兼容端点），有费用。

## 目录说明

| 路径 | 说明 |
|------|------|
| `datasets/` | Golden 集（如 `golden_v1.jsonl`、`tiny_golden.jsonl`） |
| `retriever/` | 检索层评估脚本与指标纯函数 |
| `rag/` | Ragas 端到端评估 |
| `runners/` | 入口：`build_golden_set.py`、`run_all.py` |
| `reports/` | 运行产物（`retriever_<ts>_<git>.md`、`rag_<ts>_<git>.md`），可 `.gitignore` 大文件 |

## 快速运行

```bash
pip install -r requirements.txt
# 确保已构建向量索引（首次启动应用或运行知识库构建）

python -m evals.runners.run_all --suite retriever --sample 0
python -m evals.runners.run_all --suite rag --sample 10
```

`--sample N`：`N=0` 表示全量；`N>0` 只评测前 N 条（调试用）。

更多说明见仓库根目录 [README.md](../README.md)「评估体系」章节。
