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
- **特点**：需要 **Judge LLM**（`.env` 中 OpenAI 兼容端点）与 **本地 BGE 嵌入**（与主项目 `models/bge-small-zh-v1.5` 一致）；有 API 费用（按 Judge 端点计费）。
- **与线上一致性**：检索与 prompt 来自 `GameRAGService`；**生成**在 `eval_rag.py` 中使用 **`call_llm_eval_text_only`**（纯文本），**不**走 `call_llm` 的立绘与 tools，避免评估与正式多模态链路互相干扰。

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

# 默认 --dataset 为 evals/datasets/golden_v1.jsonl（全量约 80 条）；调试请显式指定 tiny：
python -m evals.runners.run_all --suite retriever --dataset evals/datasets/tiny_golden.jsonl --sample 0
python -m evals.runners.run_all --suite rag --dataset evals/datasets/tiny_golden.jsonl --sample 5
```

`--sample N`：`N=0` 表示全量；`N>0` 只评测前 N 条（调试用）。

**Ragas 的 API Key**：建议在 `.env` 显式配置 `LLM_API_KEY`（本机 Ollama 可填 `ollama`）。仅 `eval_rag` 在本机 Ollama 风格 `LLM_API_BASE` 且 Key 为空时，会临时使用占位符 `ollama`；正式对话服务仍须非空 Key。详见根目录 [README.md](../README.md)「评估体系」。

**条目数**：`--tiny` 固定 **5** 条；`--full` 固定 **130** 条（台词 50、世界观 30、任务 30、情报 20），写入 `golden_v1.jsonl`。**不要**让模型选题；用固定 jsonl 才能保证可复现。

**耗时与成本**：RAG 一次会 1×生成 + Ragas **4 指标**（每指标多轮 Judge 调用）。单条样本端到端常在 **数分钟** 量级；全量 130 条粗估 **数小时**，先 `--sample 5` 再外推。详见根目录 [README.md](../README.md)「评估体系」中的「为何调很多次 / 条目怎么选」。

更多说明见仓库根目录 [README.md](../README.md)「评估体系」章节。
