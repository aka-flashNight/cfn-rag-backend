# Skills / 工具重构后的评估回归说明

在将任务与检索工具迁移到 `services/skills/`、并调整流式 `mood_update` 与 JSON 兜底开关后，**检索与 RAG 评估链路应保持不变**（Retriever 仍走 `game_rag_service` / `ai_engine`，Golden 集与指标定义未改）。

## LangGraph Checkpointer 与 HITL（本阶段不接入）

- **讨价还价 / 确认与取消**：当前已是合法 **Human-in-the-loop**——每轮玩家消息对应一次 `ask`，多轮协商 = 多次 HTTP；后端校验失败返回 `validation_failed` 属于 **LLM 自修正/约束生成**，与「interrupt 等人点按钮」不是同一套机制。
- **SqliteSaver / PostgresSaver + `interrupt_before(...)`**：属于总计划中的 **v2**（与路线 4 服务端 profile、草案持久化统一升级），**不在 2.2.x 本次范围接入**。现在接入会牵动 API 契约（`pending_confirmation`、resume 端点）与部署形态，应在专门迭代中做。
- **结论**：本阶段保持 **无 Checkpointer 的 `graph.compile()`**；待路线 4 再统一加持久化 checkpointer 与可选 interrupt 续跑。

## 2.2.4 / 2.2.5 / 2.2.6 落地摘要（与总计划对齐）

- **2.2.4**：决策轮仍全量 skills；`mood` 与其它工具分流由 `decision_node` 与 `_should_continue` 体现；生成轮仅 mood tools；**去掉**流式正文里基于前缀的截断（`_TRUNCATE_PREFIXES` / `_earliest_truncate_at`），改为 **原生流式 `content` + `mood_update`**；**缩短**生成轮 user 提示（`_generation_round_tool_suffix`），决策轮已带可解析的 `update_npc_mood` 时提示勿再调工具；`parse_mood_node` 中文本 mood 解析与 **末尾 JSON 剥离** 仅在 `CFN_ENABLE_MOOD_TEXT_FALLBACK` 开启时启用。
- **2.2.5**：`dispatch_tool_call` 保留在 `tool_executor.py`；各 `execute_*` 迁至 **`services/agent_tools/skill_tool_executors.py`**；草案摘要与实体行格式化迁至 **`services/agent_tools/draft_formatting.py`**；skills 的 handler 直接从 `skill_tool_executors` 引用；`tool_executor` 仍再导出符号以兼容旧路径。
- **2.2.6**：见下方「已执行校验」。

## 建议流程

1. 记录当前基线（可选）：在重构前或已知良好提交上运行一次，保留 `evals/reports/` 下生成的 `retriever_*.md` / `rag_*.md`。
2. 在当前分支上同样运行：
   ```bash
   python -m evals.runners.run_all --suite retriever --dataset evals/datasets/tiny_golden.jsonl --sample 0 --modes dense,bm25,hybrid_rrf
   python -m evals.runners.run_all --suite rag --dataset evals/datasets/tiny_golden.jsonl --sample 0
   ```
   全量 golden 可将 `--dataset` 换为 `evals/datasets/golden_v1.jsonl`（需已生成该文件）。Ragas 需配置可用的 Judge LLM（如 `.env` 中 `LLM_API_KEY` 等，以项目 `evals/rag/eval_rag.py` 为准）。
3. 对比两次报告的 `recall@k`、`MRR`、`nDCG`（Retriever）以及 Ragas 各均值；**合理波动**来自数据集抽样、Judge LLM 随机性；若 Retriever 指标断崖下降，优先检查 `hybrid_rrf` 配置、索引与 `game_data_loader` 是否一致。

## 与本次重构的关系

- **Retriever / Ragas**：不依赖 Anthropic 风格 skill 注册表的业务路径；回归用于确认未误改检索与 `_AskContext` 注入逻辑。
- **流式 mood**：由 SSE `mood_update` 与 `done` 承载，不在上述评估脚本中覆盖；需对 `/ask?stream=true` 做手动或集成测试。

## 已执行校验（2026-04-21，本仓库）

- **导入 / 图编译**：`get_full_graph()`、`get_decision_loop()`、`get_skill_registry()` 通过。
- **Retriever（路线 1）**：在 `tiny_golden.jsonl` 上 `--modes dense,bm25,hybrid_rrf` 全量跑通，生成报告 `evals/reports/retriever_20260421_111347_54726d3.{md,json}`。
- **Ragas**：本机未配置 `LLM_API_KEY` 时跳过；配置好 Judge 后执行上节 `rag` 命令即可补全。
