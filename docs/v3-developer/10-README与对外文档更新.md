# 10 · README 与对外文档更新

> 实施阶段 P9。README 需要重写而非修补：架构、技术栈、平台推荐、立绘章节均已过时。
> 问题清单 G 节列出的 7 处「README 与实现不符」借重写一并消灭。

---

## 1. README 重写要点

| 章节 | 动作 |
|---|---|
| 功能特点 | 删 LangGraph/Supervisor/「防护机制四件套」/「BM25+Hybrid RRF（此前名不副实）」等表述；改为：前台聊天 + 后台子 Agent 并行编排、hybrid 检索、ONNX int8 轻量向量、任务全流程 HITL |
| 架构概览 | 换成 01 的新架构图（TurnOrchestrator + 后台子 Agent）；技术栈表更新（删 langgraph/langchain/llama-index/torch/redis/qdrant/pg/arq/prometheus，加 onnxruntime/tokenizers） |
| 双 Profile 云原生 | **整章删除**（Server Profile 下线）；deploy/Docker/K8s 相关内容删除 |
| 嵌入模型 | bge-small-zh-v1.5，**512 维**（修正旧的 384 笔误），ONNX int8 运行时 |
| 立绘包 | **整章重写**：立绘由游戏项目的对话立绘 manifest 自动提供（说明 resources/游戏项目目录关系与 `CFN_GAME_PROJECT_DIR`）；删除 illustration.zip、SWF/ffdec、tools 目录全部说明；新增 npc_state_db.json `appearance` 可选字段说明 |
| 常见问题 | 删「立绘如何获取」「第一次构建 10-30 秒」（改为 ≤25 秒且仅指纹变化时重建）；「模型下载」仅 dev 需要 |
| MCP Server | 若保留仓库则标注「独立开发工具，不属于运行链路」；否则整章删除 |
| 评估/压测 | 评估保留（dev）；压测章节删除（loadtest 下线） |
| 发布版本说明 | 体积数字更新（exe ≤160MB 目标、Full zip 相应减小） |

## 2. 免费体验平台章节重写（含联网核实任务）

> **执行注意**：本节涉及「当前哪些平台可免费体验」的时效信息。实施 P9 时，开发 AI **必须先联网核实**下列各平台的现行免费政策与模型可用性（用户指示：此项需先做联网调研，不要凭本手册的旧快照直接写），再落笔。

推荐顺序（重写后的结构）：

1. **火山引擎（火山方舟）——首选体验平台**
   - 理由：每个模型赠送 50 万 token 免费额度；候选模型齐全（doubao-seed-2-0-lite 等）；**支持工具调用，可体验完整 Agent 功能（任务发布/讨价还价）**；国内访问稳定；OpenAI 兼容端点。
   - 注意：额度有限非无限免费，用完后为低价计费。
2. **各家官方 API（便宜模型的正规渠道）**
   - DeepSeek 开放平台（deepseek-v4-flash / vision-exp，可关思考）
   - 智谱开放平台（glm-5.3-flash，极致低价，思考恒开需低档位）
   - 阿里云百炼（qwen3.8-flash，enable_thinking 可关）
   - OpenAI（gpt-5.6-luna，reasoning none 档）
   - 写明：以上按 token 计费但单价极低，适合长期使用。
3. **Google Gemini——免费额度充足但需代理**（保留旧说明，更新模型名与额度，需联网核实）。
4. **ModelScope 魔搭社区——放最后并明确标注限制**：
   > 魔搭推理 API **不支持工具调用**，因此**只能体验聊天功能，无法体验任务发布/讨价还价等 Agent 功能**（后端会检测到工具不可用并以 NPC 话术说明）。仅建议用于零成本快速试聊。
   - 保留原获取步骤说明（绑定阿里云实名 → 访问令牌 → 选模型），模型清单需联网刷新（kimi-k2.5 已下线等）。
5. **Kimi**：注明目前无免费赠送（需联网核实最新政策），仅作为付费备选列出。

默认配置示例改为：
```env
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL_NAME=deepseek-v4-flash-vision-exp
```

## 3. AGENTS.md / data_files_overview.md

- AGENTS.md：补充「激活 venv 后 pytest 跑 tests/」「运行时依赖只装 requirements.txt」两条约定。
- data_files_overview.md：任务规则部分不变；追加 `agent_tasks.json`/`agent_text.json` 写入流程在 v3 中的调用位置（orchestrator 同步动作）。

## 4. 发布说明模板（v3.0.0）

- 亮点：体积 357MB → ≤160MB；聊天一次调用出情绪+正文；任务/搜索后台并行不再干等；任务拟定不再被校验卡死；立绘自动读取游戏最新资产。
- 升级注意：与 v2 前端不兼容（需同步升级前端）；首次启动自动重建向量索引（旧索引不迁移）；Server Profile 部署能力移除。
