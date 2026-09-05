# CFN-RAG Backend v3 重构开发手册

> 2026-09-05。本手册是 v3 重构的**唯一执行依据**，供后续开发 AI 按部就班实施。
> 前置阅读：`docs/v3-searching/`（现状调研 + 联网调研 + 问题清单），本手册不再重复其中证据，只引用结论编号（如 A2、C1、F1）。
> 部署路线：**仅本地打包 exe 分发**。Server Profile（Postgres/Qdrant/Redis/arq/K8s）整体下线，不再维护。

---

## 1. 重构目标

| 目标 | 现状（证据） | 目标值 |
|---|---|---|
| 包体 | exe 357MB，主因 torch/transformers 栈（F1） | **单 exe ≤ 160MB** |
| 冷启动 | 模型+索引加载慢，首请求阻塞 | 索引已建时 **启动到可服务 ≤ 5s** |
| 首字延迟（TTFT） | 串行 supervisor→worker→dialogue，2~3 次串行 LLM（B2） | 常规聊天 **≤ 2.5s**（flash 模型、无思考/低思考） |
| 单轮检索开销 | 12~16 次串行嵌入+检索，query 重复嵌入（C1） | **≤ 3 次嵌入 + 1 次矩阵乘，≤ 150ms** |
| 任务流程 | 校验一次只报一个错、无自动修复、可卡死（D1/D2/D3） | 错误一次报全 + 小幅自动修复 + 后端兜底，**草案拟定不卡死** |
| 架构 | 双路径分叉、防护全失效、大量死代码（B1/B4/B6） | 单一路径、前台聊天 + 后台子 Agent 并行、无死代码 |

## 2. 业务功能红线（必须保持正常）

1. **NPC 角色扮演聊天**：AI 绝对扮演 NPC，获取该 NPC 设定信息，遵守人设、符合身份与形象聊天；好感度/情绪/关系等级系统正常；情绪必须先于对话正文到达前端（B5 开发者注：前端先切立绘再显示对话）。
2. **任务发布全流程**：任务拟定 → 讨价还价（可选）→ 任务发布/取消；两步式工具调用（prepare→draft）、草案协商（update）、确认原子写入 `agent_tasks.json` / `agent_text.json`（confirm）、取消（cancel）；HITL：草案必须经玩家明确确认才发布，**拟定 ≠ 发布**。
3. RAG 检索：覆盖角色台词、任务台词、世界观、情报、loading 文案、物品/关卡实体六类语料。
4. 会话管理：多会话、历史、滚动摘要、删除/重命名。
5. 多模态：按游戏最新立绘 manifest 查表取图（见 `07-立绘与多模态.md`），模型不支持识图时回退纯文本（含形象描述），不失败。

## 3. 核心架构决策（速览，详见 01）

| # | 决策 | 替代掉的旧设计 |
|---|---|---|
| D1 | **废弃 LangGraph 与 legacy 路径**，自研轻量 `TurnOrchestrator`（纯 asyncio，单一路径，流式优先） | LangGraph 双路径（B1）、四重失效防护（B4）、`agent_graph/` 死代码（B6） |
| D2 | **聊天主 Agent 一次调用多产出**：流式首行输出紧凑 meta JSON（情绪/好感/委派指令），其余为对话正文 | supervisor 路由 LLM + 正则解析 JSON（A6）、情绪与正文分离调用 |
| D3 | **后台子 Agent（fire-and-steer）**：TaskRunner / SearchRunner 作为 asyncio 后台任务与前台聊天并行；结果分「中间结果/最终结果」回填 | worker 串行硬等（B2） |
| D4 | 聊天主 Agent **不使用 function calling**做决策（决策全走 meta 行）；**function calling 只保留给后台子 Agent**（工具循环） | 流式+tools 互斥回退矩阵（A1/A2）、正文字符串截断（A4） |
| D5 | 统一 `LLMClient`：连接复用、超时重试、模型 Profile 注册表（思考参数/视觉能力）、参数报错自动降级重试、usage 统计 | 每次新建客户端（A5）、400/422 一刀切判不支持工具（A1） |
| D6 | 嵌入链路换 **ONNX Runtime + int8**（模型不换，仍 bge-small-zh-v1.5）；向量存储改二进制；检索改「单次嵌入 + 全库矩阵乘 + 池内过滤 + BM25 RRF」 | torch+transformers（F1）、LlamaIndex JSON 向量库（50MB→9.1MB）、12~16 次串行检索（C1） |
| D7 | 校验管线：**全量收集错误 + 结构化修复指引 + 小幅自动修复 + 最终后端兜底** | 串行短路（D1）、无自动修正（D2）、打回信息不足（D3） |
| D8 | 依赖大裁剪：删 langgraph/langchain/llama-index/redis/qdrant/sqlalchemy/arq/prometheus 等；requirements 分 runtime/dev 两组 | 全家桶同装（F1） |
| D9 | 立绘：**直接读游戏项目 `launcher/web/assets/dialogue-portraits/manifest.json` 查表**；废弃立绘包解压与 SWF/ffdec 导出 | illustration.zip、tools/ffdec.jar、extract_portraits_from_swf.py |
| D10 | 预留 `services/gamebridge/` 接口 stub（游戏进程检测/存档绑定/任务后对话增量），**只定义接口不实现** | — |

## 4. 手册结构（按实施顺序）

| 文档 | 内容 | 对应实施阶段 |
|---|---|---|
| [01-总体架构与核心决策.md](01-总体架构与核心决策.md) | 架构全景、TurnOrchestrator 设计、SSE 事件契约、目录结构 | 全部阶段的依据 |
| [02-LLM接入层.md](02-LLM接入层.md) | LLMClient、模型 Profile、思考控制与降级、meta 协议、reasoning_content | P1 |
| [03-对话编排与多Agent并行.md](03-对话编排与多Agent并行.md) | 回合状态机、任务/搜索并行时序、讨价还价、HITL、中间结果规则 | P3/P4 |
| [04-检索与向量模型.md](04-检索与向量模型.md) | ONNX int8 导出、二进制向量库、hybrid 检索、配置化 | P2 |
| [05-任务管线与校验反馈.md](05-任务管线与校验反馈.md) | 错误聚合、自动修复、兜底、TaskRunner 工具集 | P5 |
| [06-存储启动与打包瘦身.md](06-存储启动与打包瘦身.md) | SQLite/NPC 状态/代理/启动并行/依赖裁剪/PyInstaller | P6/P8 |
| [07-立绘与多模态.md](07-立绘与多模态.md) | manifest 查表、appearance 字段、视觉回退、图片缓存 | P7 |
| [08-实施阶段与验收.md](08-实施阶段与验收.md) | P0~P10 分阶段任务清单与验收标准、测试要求 | 全程 |
| [09-前端适配说明.md](09-前端适配说明.md) | SSE 事件契约变更、时序图、给前端项目的改造清单 | P9 交付物 |
| [10-README与对外文档更新.md](10-README与对外文档更新.md) | README 重写要点、免费体验平台调整（含联网核实任务） | P9 |

## 5. 开发纪律

1. **先看 01 再动手**；每个阶段完成后按 `08` 的验收清单自测，不达标不进入下一阶段。
2. 严格保持「业务红线」（§2）不回归；任务文件只能原子写 `agent_tasks.json` / `agent_text.json`，其余游戏数据只读。
3. 不允许重新引入：torch/transformers、LangGraph/LangChain、LlamaIndex、任何 server-profile 依赖。
4. 所有面向玩家的可见文案走 SSE 事件契约，**禁止**再向 content 里塞 `{花括号}` 系统消息（09 有替代事件）。
5. 代码注释与文档使用简体中文。
