# 评估体系（v3：检索层）

v3 仅保留**检索层评估**轨道（不调 LLM，无 API 费用，适合频繁回归）。
v2 的 Ragas 端到端轨道（`evals/rag/`）已随旧架构（GameRAGService / llm_client /
NPCManager 等）一并移除；如需端到端质量评估，需基于 `services/orchestrator` +
`services/llm` 重建（见 docs/v3-developer/验收报告.md 遗留项）。

## 检索层评估（Retriever Eval）

- **评什么**：给定 query，retriever 是否召回到「标注相关」的 chunk（按新索引 `Node.id`）。
- **指标**：`recall@k`、`precision@k`、`MRR@n`、`nDCG@k`（k ∈ {5,10,20} 等）。
- **模式**：`dense`（向量） / `bm25`（稀疏） / `hybrid_rrf`（RRF 融合，k=60）。
- **评估口径**：`RetrievalEngine.retrieve_for_eval`（v3 新增的评估专用入口）——
  按 golden 行的 `filter_type` / `filter_character` 做单池过滤后取 top-k，
  **不做业务阈值过滤**（阈值会截断召回，评估口径必须放开）。

## 目录说明

| 路径 | 说明 |
|------|------|
| `datasets/` | Golden 集（`golden_v1.jsonl` 全量 130 条、`tiny_golden.jsonl` 微型 5 条） |
| `retriever/` | 检索层评估脚本与指标纯函数 |
| `runners/` | 入口：`build_golden_set.py`（重建 golden 集）、`run_all.py` |
| `reports/` | 运行产物（`retriever_<ts>_<git>.md` / `.json`） |

## 快速运行

```bash
pip install -r requirements.txt

# 1. 重建 golden 集（expected_doc_ids 与当前索引 Node.id 对齐；语料更新后需重建）
python -m evals.runners.build_golden_set --tiny   # 5 条
python -m evals.runners.build_golden_set --full   # 130 条（台词 50 / 世界观 30 / 任务 30 / 情报 20）

# 2. 跑检索层评估（默认 --dataset 为 golden_v1.jsonl；调试请显式指定 tiny）
python -m evals.runners.run_all --suite retriever --dataset evals/datasets/tiny_golden.jsonl --sample 0
python -m evals.runners.run_all --suite retriever --dataset evals/datasets/golden_v1.jsonl --sample 0
```

`--sample N`：`N=0` 表示全量；`N>0` 只评测前 N 条（调试用）。

**条目数**：`--tiny` 固定 **5** 条；`--full` 固定 **130** 条。**不要**让模型选题；
用固定 jsonl 才能保证可复现（采样种子固定 42，但语料变化后行内容会变，需整集重建）。

**构建 golden 前置**：需要游戏 `resources/` 数据可访问（与主服务一致，自动探测）；
排除「彩蛋」「成员」阵营 NPC 的样本（与旧口径一致）。

更多说明见仓库根目录 [README.md](../README.md)「评估」章节。
