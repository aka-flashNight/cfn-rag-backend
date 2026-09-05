# 联网调研：2026 年多 Agent 并行编排与后台子 Agent 模式现状（v3 重构前置调研 · 二之二）

> 只报告业界现状与事实，不含本项目改造建议。检索时间：2026-09-05；近 3 个月信息为主，少量 2026-04 左右的一手资料标注日期。

---

## 1. 结论速览

1. **「派发子 Agent 后主 Agent 继续对话」已成为主流框架的一等公民能力**：LangChain 2026-04 官方发布 async subagents（后台子 Agent）模式，明确将传统 inline 子 Agent 的「同步阻塞」定性为缺陷；OpenAI、Microsoft Agent Framework、DeepAgents 等均有对应形态。
2. LangGraph 层面的并行原语是 **Send() API（map-reduce 扇出）** 与多节点并行执行；supervisor 框架（langgraph-supervisor）官方讨论确认可用 Send() 并行跑多个 worker，限制在共享状态。
3. 前端事件流方面，**AG-UI 协议**（~17 种类型化事件：token 增量、工具调用、状态增量等）正在成为 Agent↔前端 流式通信的标准化选项；多路并行流的常见做法是单条 SSE 连接 + 事件打标（agent_id/run_id）+ 前端按标路由。
4. 后台任务的结果获取以**轮询**为主（官方博客明确未提供主动推送/回调）；生产化关注点集中在代理缓冲、断线重连（Last-Event-ID）、背压。

---

## 2. 后台子 Agent 模式（用户点名要查的模式）

### 2.1 LangChain「Running Subagents in the Background」（2026-04-16 官方博客）

- **针对的问题**：inline 子 Agent 通过工具调用触发，agent loop 中工具调用是同步的——「子 Agent 没返回，supervisor 无法思考任何其他事情」；阻塞期间用户输入、其他子 Agent 结果、中途纠偏三个信息渠道全部不可用。
- **新模式**：`fire-and-steer`（派发后仍可干预），而非 `fire-and-forget`。
- **机制**：
  - 派发：主 Agent 调 `start_async_task` 启动远程子 Agent，立即返回 task ID，继续自己的 reasoning loop；
  - 子 Agent 在独立进程、独立状态中运行（可部署到 LangSmith deployments 或自托管 Docker/K8s，经 Agent Protocol 标准接口管理）；
  - 主 Agent 期间可自由与用户对话、再派发新任务；
  - 结果回填：**轮询** `check_async_task`；干预：`update_async_task`（补充指令）、`cancel_async_task`（取消）、`list_async_tasks`（枚举）；
  - 构建于 **Agent Protocol**（框架无关的远程 Agent 管理 API 规范：创建线程、启动 run、轮询状态、发送更新、记忆管理）。
- **明确未覆盖**：官方博客未讨论多任务流并行走前端的 UI 聚合方案。
- 来源：[Running Subagents in the Background — LangChain Blog](https://www.langchain.com/blog/running-subagents-in-the-background)（2026-04-16）、[Subagents — LangChain Docs](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)。

### 2.2 DeepAgents 动态子 Agent

- 主 Agent 通过 task 工具调用委派，**单轮可发出多个 task 调用并行执行**；子 Agent 配置项含 `name` / `description` / `graphId` / `url`。
- 来源：[Subagents (Deep Agents) — LangChain Docs](https://docs.langchain.com/oss/python/deepagents/subagents)、[官方视频讲解](https://www.youtube.com/watch?v=5AkdMangfNk)。

### 2.3 其他框架对应形态（简记）

- **Microsoft Agent Framework + AG-UI**：官方 demo 展示 React 前端消费 SSE 流实时渲染多 Agent 工作流状态（活动 Agent 指示、实时状态）。
- **Parallel.ai Task API**：SSE 直接推送任务运行进度/推理/状态，「无需轮询」（平台侧推送）。
- 来源：[Microsoft Agent Framework devblog](https://devblogs.microsoft.com/agent-framework/ag-ui-multi-agent-workflow-demo/)、[Parallel.ai blog](https://parallel.ai/blog/sse-for-tasks)。

---

## 3. LangGraph 并行原语（项目当前所用框架）

| 原语 | 说明 | 来源 |
|---|---|---|
| **Send() API** | 条件边内派发 N 个并行实例（map-reduce 扇出），是 LangGraph 并行子任务的核心机制 | [Scaling LangGraph Agents](https://aipractitioner.substack.com/p/scaling-langgraph-agents-parallelization)、[langgraph-supervisor #189](https://github.com/langchain-ai/langgraph-supervisor-py/discussions/189) |
| **多节点并行** | 同一 super-step 内多个节点同时执行（shared state 读写由 reducer 决定） | 同上 |
| **supervisor 并行 worker** | langgraph-supervisor 官方讨论确认可用 Send() 并行跑多个 worker；**限制在共享状态**（worker 写同一 state 需 reducer 合并） | [GitHub Discussion #189](https://github.com/langchain-ai/langgraph-supervisor-py/discussions/189) |
| **asyncio.gather** | 节点/工具内部并发执行 IO 任务（非图结构层面的并行） | 社区通用实践 |

- 社区对比文章将多 Agent 形态分为 Skills pattern（共享状态）与 Router pattern（并行 Agent、无共享历史）两类。
- 来源：[LangGraph in Production — Medium](https://medium.com/@subratpati/langgraph-in-production-choosing-and-building-multi-agent-systems-c2b955f16429)、[Reddit r/LangChain 讨论（社区）](https://www.reddit.com/r/LangChain/comments/1cthrqz/agents_working_in_parallel_with_langgraph/)。

---

## 4. 多路流式输出给前端（并行 Agent 的 UX 侧）

1. **单连接多路复用**：一条 SSE 连接，每个事件带 `agent_id`/`run_id` 标签，前端 reducer 把 token 路由进各 Agent 的缓冲区——是并行 Agent 流的主流形态。
2. **AG-UI 协议**：开放事件协议，~17 种类型化事件（`TEXT_MESSAGE_CONTENT` token 增量、工具调用开始/结束、状态增量 delta 等），可走 SSE/WebSocket/HTTP；已被 Microsoft Agent Framework、CopilotKit 等采用。
3. **生产化要点**（dev.to / TianPan.co 总结）：
   - 事件 schema 先行：所有 Agent 输出必须匹配统一契约；
   - 常见故障：代理静默缓冲/吃掉流、标签页关闭杀死 Agent、背压、断线重连（Last-Event-ID）；
   - HTTP/2 下多条 SSE 可多路复用同一 TCP 连接，规避 HTTP/1.1 浏览器 6 连接限制（Centrifugo 2026-03 博客）。
4. **Fan-out/Fan-in 检索特例**：有文章专述「triage 后并行跑全部 RAG 专家再汇聚」与顺序路由的差异（并发事件流 vs 串行）。
- 来源：[AG-UI Events 规范](https://docs.ag-ui.com/concepts/events)、[CopilotKit 17 事件解析](https://webflow.copilotkit.ai/blog/master-the-17-ag-ui-event-types-for-building-agents-the-right-way)、[dev.to 生产故障分析](https://dev.to/priyank_agrawal/your-multi-agent-sse-stream-works-in-dev-heres-what-kills-it-in-production-458i)、[TianPan.co](https://tianpan.co/blog/2026/04/10/streaming-real-time-agent-uis-sse-backpressure-reconnection)、[Centrifugo](https://centrifugal.dev/blog/2026/03/01/scaling-ai-token-streams-with-centrifugo)、[Fan-out/Fan-in — Medium](https://medium.com/@junewine/concurrent-fan-out-fan-in-running-all-rag-specialists-in-parallel-7514fa5fec3c)。

---

## 5. 与本项目现状的对应事实（仅列差距）

1. 本项目 supervisor 每条消息只路由到 **1 个** worker，worker 完成后静态边进 dialogue；无任何并行派发（无 Send()、无 gather、无后台任务）。
2. LangGraph 版本 1.1.9 已具备 Send()/并行节点/asyncio 生态，框架能力未被使用。
3. 本项目 SSE 事件体系为自定义五种事件（content/mood_update/tool_status/agent_status/done），无 run_id/agent 维度打标；「interim 伪流式 + 阶段性 tool_status」是串行流水线的 UX 补偿设计。
4. 后台子 Agent 模式（长任务不阻塞主对话）在本项目中的对应场景是「任务发布/确认」类流程——当前实现为同请求内串行完成（HITL 靠 HTTP 轮次天然暂停）。
5. 工具执行层面：一次 LLM 返回多个 tool calls 时，业界常规为并行执行；本项目按流水线排序后串行同步执行。
