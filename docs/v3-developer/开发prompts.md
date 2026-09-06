# v3 重构 · 分窗口开发 Prompts

> 用法：4 个窗口**按顺序**执行（②依赖①的新模块，③依赖①②，④最后）。
> 每个窗口开在新会话，模型建议 glm-5.3-flash 或同级。每个 prompt 自包含，直接整段复制。
> 通用约定（已写进各 prompt）：工作目录 `E:\program\cfn-rag-backend`；执行 python 前先 `.\venv\Scripts\Activate.ps1`；代码注释用简体中文；发现手册与现实冲突时以代码现实为准，并记录到 `docs/v3-developer/实施手记.md`；每完成一个模块 git commit 一次。

---

## 窗口 ① 基础设施（分支+清理 / LLM 接入层 / 向量检索 / 存储启动）

````
你是资深 Python 后端工程师，负责 CFN-RAG Backend（NPC 角色扮演对话+任务发布 Agent 服务，FastAPI，工作目录 E:\program\cfn-rag-backend）v3 重构的【基础设施阶段】。

【必读文档，先读完再动手】
- docs/v3-developer/README.md（总览与红线）
- docs/v3-developer/01-总体架构与核心决策.md（§7 删除清单、§8 配置）
- docs/v3-developer/02-LLM接入层.md（本阶段核心）
- docs/v3-developer/04-检索与向量模型.md（本阶段核心）
- docs/v3-developer/06-存储启动与打包瘦身.md 的 §1~§2
- docs/v3-developer/08-实施阶段与验收.md 的 P0/P1/P2/P6
- docs/v3-searching/01-现有实现调研.md、03-问题清单.md（背景）
- docs/v3-searching/02-联网调研-向量模型.md（ONNX int8 依据）

【任务清单】
1. git checkout -b v3-refactor，后续全部工作在此分支，分模块提交。
2. 按 01 §7 删除清单 git rm：deploy/、worker/、alembic/、alembic.ini、docker-compose.yml、Dockerfile、loadtest/、mcp_server/、tools/、services/agents/、services/agent_graph/、services/storage/、services/session_heartbeat.py。（删完后 api/game_api.py、services/game_rag_service.py 会 import 失败，属预期，下一窗口重写它们，本窗口不动这两个文件。）
3. scripts/export_onnx_int8.py：把 models/bge-small-zh-v1.5 导出 ONNX 并动态 int8 量化到 models/bge-small-zh-v1.5-onnx-int8/（含 tokenizer），按 04 §1.2 校验余弦相似度 ≥0.99。需要 pip install optimum[onnxruntime] torch(cpu版) transformers（装到 venv 即可，后面会从运行时依赖中剔除）。
4. services/llm/（client.py / profiles.py / meta.py / errors.py）：按 02 全文实现，包括模型 Profile 注册表、思考参数降级链、图像降级、reasoning_content 隔离、usage 回收、meta 行解析器。删除 services/llm_client.py、services/npc_mood_agent.py。
5. services/retrieval/（embedder.py / store.py / hybrid.py / pools.py / config.py / loader.py）：按 04 实现 OnnxEmbedder、二进制 VectorStore（vectors.npy fp16 + meta.json + bm25.pkl + fingerprint.json）、hybrid RRF 检索、池配置化。文档加载/切分逻辑从 ai_engine/game_data_loader.py 与 services/game_rag_service.py 的检索部分吸收（这两处是你理解现有语料规则的唯一来源，读透再写）。完成后删除 ai_engine/。
6. services/npc/manager.py + services/memory/（store.py / summarize.py）：按 06 §1 实现（单例+锁+防抖落盘、SQLite 单连接 WAL、摘要有界队列、appearance 字段 forward-compatible）。删除 memory_manager.py、npc_manager.py、task_draft_store.py（功能并入 memory/）。
7. core/startup.py 重写（并行初始化、logging 化）；core/config.py 按 01 §8 精简为 local-only。
8. requirements.txt 按 06 §3.1 精简、新建 requirements-dev.txt 按 06 §3.2。
9. tests/ 下按 02 §6、04 §7、06 §5 写 pytest 并跑通（mock httpx / fake 数据，不依赖真实 LLM key）。

【验收（全过才算完）】
- pytest 全绿；检索 bench 脚本（evals/bench_retrieval.py 自建）输出：构建 ≤25s、单次检索 ≤150ms；
- int8 模型校验余弦 ≥0.99；向量库 ≤10MB；
- requirements.txt 中无 torch/transformers/llama-index/langgraph/langchain/redis/qdrant/sqlalchemy/arq/prometheus/structlog/locust/modelscope；
- git log 有清晰的分模块提交。

注意：不要写 api/、services/orchestrator/、services/subagents/、agent_tools 的改造（后续窗口的事）。完成后输出：改动文件清单、bench 数据、遗留问题。
````

---

## 窗口 ② 编排与子 Agent（orchestrator / subagents / 校验管线 / API 重写）

````
你是资深 Python 后端工程师，负责 CFN-RAG Backend（工作目录 E:\program\cfn-rag-backend）v3 重构的【编排与任务管线阶段】。基础设施阶段（services/llm、services/retrieval、services/npc、services/memory、core/startup）已由上一窗口完成，先读代码了解现状。

【必读文档，先读完再动手】
- docs/v3-developer/README.md、01-总体架构与核心决策.md
- docs/v3-developer/03-对话编排与多Agent并行.md（本阶段核心）
- docs/v3-developer/05-任务管线与校验反馈.md（本阶段核心）
- docs/v3-developer/02-LLM接入层.md 的 §4~§5（meta 协议与 act 语义）
- docs/v3-searching/01-现有实现调研.md 的 §4~§7（旧的 prompt 分层、任务管线、Skills 机制是你要吸收的资产）
- docs/v3-developer/实施手记.md（上一窗口的偏差记录）

【任务清单】
1. services/tools/ 整理：删除 mood 类工具；query/task/system 工具保留给子 Agent；ToolRegistry.dispatch 改 async 且支持同批 tool_calls 并行（asyncio.gather）。
2. services/agent_tools/ 按 05 改造：validator 聚合模式 + 8 类规则增强反馈（ValidationIssue/ValidationReport）；新增 repair.py 自动修复；新增 fallback_draft 后端兜底；draft_summary 失败时也返回草案快照；bargain_count 独立成列。原子写入逻辑（05 §1 保留清单）原样保留。
3. services/subagents/（base.py / task_runner.py / search_runner.py / prompts.py）：按 03 §3/§5/§6 实现，Skills 三级加载接入（services/skills/ 保留沿用）。
4. services/orchestrator/（turn.py / context.py / events.py / merge.py）：按 03 §1/§2/§4/§7 实现完整状态机与 SSE 新契约（01 §5 事件表）。prompt 体系：吸收 services/agent_graph/prompts.py 的有效分层资产（世界观 L1、扮演约束、输出格式规则），重写为「聊天 prompt（含 meta 协议说明，02 §4.3）+ 子 Agent 独立小 prompt」。
5. api/game_api.py 重写：/api/ask（SSE）走 TurnOrchestrator 单一路径；/api/ask/confirm 等旧路由按新契约简化或移除；删除 LangGraph/legacy 双路径全部残留；删除 services/game_rag_service.py；apply_proxy_config 改客户端级（06 §1.4）。assets_api.py 暂不动（窗口③处理）。
6. tests/ 按 03 §9、05 §8 写 e2e（Fake LLM 脚本化流）并跑通。

【验收（全过才算完）】
- pytest 全绿（含：七条主路径、中间结果只说一次且不报数字、confirm 失败丢正文重生成、子 Agent 取消、聚合校验一次报多错、自动修复、兜底草案）；
- 启动 python main.py 能起服务，用 curl/脚本打 /api/ask 验证 SSE 事件序列符合 09-前端适配说明.md §2 时序；
- git 分模块提交清晰。

完成后输出：改动文件清单、SSE 事件实测序列样例、遗留问题，并把偏差记到 docs/v3-developer/实施手记.md。
````

---

## 窗口 ③ 立绘与打包（portraits / assets_api / 依赖与 exe）

````
你是资深 Python 后端工程师，负责 CFN-RAG Backend（工作目录 E:\program\cfn-rag-backend）v3 重构的【立绘与打包阶段】。前两个窗口已完成基础设施与编排层，先读代码了解现状（重点 services/orchestrator/context.py 的装配点、services/llm/profiles.py 的 vision 标记）。

【必读文档，先读完再动手】
- docs/v3-developer/07-立绘与多模态.md（本阶段核心）
- docs/v3-developer/06-存储启动与打包瘦身.md 的 §3~§4
- docs/v3-developer/01-总体架构与核心决策.md、README.md
- 游戏项目的立绘查表协议（权威，必须逐条遵守）：
  E:\Steam\steamapps\common\CRAZYFLASHER7StandAloneStarter\project\CrazyFlashNight\docs\对话立绘查表映射-外部Python读取指南-2026-09-05.md

【任务清单】
1. services/portraits/（manifest_lookup.py / provider.py / cache.py）：按 07 实现——manifest 查表（角色名归一化三注册、情绪回退链、uri 解析、heroKeys 特例）、bounds 裁剪 + 长边 ≤512 + WebP q80 + base64、按 (npc, 情绪, mtime) 进程内缓存。manifest 缺失时整体降级无图模式不报错。
2. 接入编排层：上下文装配时按 07 §4 规则决定是否带图（仅 purpose=chat 且 vision=true）；appearance 字段注入聊天 prompt（有则恒入）；立绘情绪跟随当回合 meta 的 emo。
3. api/assets_api.py 改造：立绘接口改走 manifest 查表返回裁剪后 PNG；删除旧的立绘生成/解压相关接口。
4. 删除 scripts/extract_portraits_from_swf.py、services/portrait_utils.py、core/startup.py 中立绘解压任务（如还有残留）。
5. scripts/build_exe.py 按 06 §4 重写（PyInstaller onefile、hidden imports onnxruntime/tokenizers、exclude torch/transformers/llama_index/langchain 等、打包 dist/ + int8 模型 + backup_resources）。
6. 全新 venv 只装 requirements.txt，跑 python main.py 冒烟；然后实际构建 exe，双击冒烟（浏览器打开、聊天一回合、发任务流程走通到草案说明）。
7. tests/ 按 07 §8 补 pytest 并跑通。

【验收（全过才算完）】
- pytest 全绿；exe ≤ 160MB 且双击全功能冒烟通过；
- 视觉模型对话能正确带图（用任一可识图模型的真实 key 验证一次，或至少验证图片正确进入请求体）；
- 非 vision Profile 模型全程无图不失败；
- git 分模块提交清晰。

完成后输出：改动文件清单、exe 实测体积、冒烟结果、遗留问题，并把偏差记到 docs/v3-developer/实施手记.md。
````

---

## 窗口 ④ 文档与回归（README 重写 / evals / 最终验收）

````
你是资深后端工程师兼技术文档作者，负责 CFN-RAG Backend（工作目录 E:\program\cfn-rag-backend）v3 重构的【文档与回归阶段】。代码重构已由前三个窗口完成。

【必读文档】
- docs/v3-developer/10-README与对外文档更新.md（本阶段核心）
- docs/v3-developer/09-前端适配说明.md（检查它与代码 events.py 是否一致）
- docs/v3-developer/08-实施阶段与验收.md（P9/P10）
- docs/v3-developer/实施手记.md（全部偏差记录）
- 现状 README.md、AGENTS.md、data_files_overview.md

【任务清单】
1. 【必须先联网调研再写】核实以下时效信息并注明核实日期：火山方舟各模型免费额度现状与 doubao-seed-2-0-lite 可用性；ModelScope 魔搭免费额度及「API 不支持工具调用」是否仍然成立；DeepSeek/智谱/阿里百炼/OpenAI 官方平台低价模型与价格；Kimi 是否还有免费额度；另搜一下当前还有什么可靠（来源稳定）的免费体验 API 渠道。
2. 按 10 §1~§2 重写 README.md（架构/技术栈/立绘/平台推荐/常见问题全改；删除 Server Profile、MCP、压测、立绘包、ffdec 章节；嵌入模型写 512 维 ONNX int8）。
3. 更新 AGENTS.md、data_files_overview.md；起草 release_notes_v3.0.0.md（按 10 §4）。
4. evals/ 适配新检索接口（如窗口①②未做），跑 tiny + full golden，与 docs/v3-developer/基线记录.md 对比，recall@10 不回退是红线。
5. 对照 08 的人工验收清单，把所有自动化可验的项跑掉，输出验收报告（哪些通过、哪些需人工）。
6. 检查 09-前端适配说明.md 与 services/orchestrator/events.py 实际事件字段逐条一致，不一致以代码为准修订文档。

【验收】
- README 按步骤可复现安装配置；evals recall 不回退；验收报告落 docs/v3-developer/验收报告.md；git 提交清晰。

完成后输出：联网核实结论摘要（附来源链接）、验收报告、遗留问题。
````

---

## 执行顺序与注意事项

| 窗口              | 依赖   | 产出                                                          |
| ----------------- | ------ | ------------------------------------------------------------- |
| ① 基础设施       | 无     | v3-refactor 分支、llm/retrieval/npc/memory、requirements 拆分 |
| ② 编排与任务管线 | ①     | orchestrator/subagents/校验管线/API、可跑通的 SSE 服务        |
| ③ 立绘与打包     | ①②   | portraits、exe ≤160MB 冒烟                                   |
| ④ 文档与回归     | ①②③ | README、evals 对比、验收报告                                  |

1. 窗口之间通过**代码 + 实施手记.md**交接，不要让后面的窗口重读全部调研文档（prompt 里只给了它该读的）。
2. 每个窗口结束后你看一眼它的输出摘要和 git log，确认验收项再开下一个。
3. 若某窗口验收不过，把它的「遗留问题」原样贴回同一窗口继续修，不要直接开新窗口。
