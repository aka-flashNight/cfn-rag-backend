# 03 · 对话编排与多 Agent 并行（services/orchestrator/ + services/subagents/）

> 实施阶段 P3/P4。本章定义回合状态机、任务全流程时序、讨价还价、搜索并行、HITL 与「中间结果规则」。
> 依据：`docs/v3-searching/02-联网调研-Agent编排.md`（fire-and-steer 后台子 Agent 模式）+ 用户的聊天/任务并行设计方案。

---

## 1. 组件

```
services/orchestrator/
├── turn.py       # TurnOrchestrator：一个玩家消息 = 一个 Turn
├── context.py    # 上下文装配（NPC/记忆/检索/立绘/appearance，并行）
├── events.py     # SSE 事件构造（契约见 01 §5）
└── merge.py      # 汇合逻辑（宽限/中间结果/最终结果/超时）

services/subagents/
├── base.py       # SubagentHandle：asyncio.Task 包装 + 进度 Queue + cancel
├── task_runner.py    # TaskRunner：任务拟定/修改工具循环
├── search_runner.py  # SearchRunner：检索工具循环
└── prompts.py    # 子 Agent 专用小 prompt（与聊天 prompt 完全分离）
```

## 2. TurnOrchestrator 状态机

```
IDLE
  → BUILD_CONTEXT        并行装配（目标 ≤300ms，检索 ≤150ms 见 04）
  → BURST_STREAMING      调用 #1：缓冲解析 meta → 发 meta 事件 → 流式转发正文
       │ meta.act == null              → POST_PROCESS → DONE
       │ act == task_confirm/cancel    → SYNC_ACTION（成功后继续放正文）→ POST_PROCESS → DONE
       │ act == task_draft/update/search → 启动子 Agent，继续放正文
  → MERGE_WAIT           正文流完，子 Agent 未完成
       │ 宽限 merge_grace_ms 内完成     → MERGE_REPLY（调用 #2，携带结果）
       │ 有中间结果且未说过过渡语       → INTERIM_REPLY（调用 #2a，1~2 句模糊过渡，只说一次）
       │                                → 继续等最终结果 → MERGE_REPLY（调用 #2b）
       │ 无中间结果                     → 等（SSE 保活注释帧）→ 最终结果 → MERGE_REPLY
       │ 子 Agent 失败/超时             → FAIL_REPLY（人设话术说明，不甩错误码）
  → POST_PROCESS         写记忆/好感/草案触碰计数/usage/done
  → DONE
```

**LLM 调用预算（替代旧 token 死熔断，B4）**：每回合聊天主 Agent ≤ 3 次调用（#1 + 过渡 #2a + 汇合 #2b）；TaskRunner ≤ 4 轮、SearchRunner ≤ 3 轮工具循环（config 可调）。任何超限立即走失败话术路径，绝不无限循环。

## 3. 后台子 Agent 基座（base.py）

```python
class SubagentHandle:
    kind: Literal["task_draft", "task_update", "search"]
    task: asyncio.Task
    events: asyncio.Queue[SubagentEvent]   # progress / intermediate / final / failed
    async def wait_final(self, timeout: float) -> SubagentEvent: ...
    def cancel(self) -> None: ...

@dataclass
class SubagentEvent:
    kind: Literal["progress", "intermediate", "final", "failed"]
    ui_hint: str = ""            # ≤12 字，给前端进度条：「拟定委托中」「校验修正中」「查资料中」
    vague_note: str = ""         # intermediate 专用：只说类型不说数字的一句话
    payload: dict = field(default_factory=dict)  # final: draft_summary / findings；failed: reason
```

- **intermediate 的语义**（用户硬性规则）：TaskRunner 在「草案校验未通过、正在修复」时发 intermediate，`vague_note` 只允许说任务类型/方向（如"委托内容在调整了"），**禁止包含任何数量、物品名、金额**。orchestrator 每回合最多消费一次 intermediate（说完过渡语后只等 final）。
- 子 Agent 不发图、不接触玩家可见正文；其 LLM 调用 `purpose="subagent"`。

## 4. 任务全流程时序（核心业务流）

### 4.1 首次发布（task_draft）

```
玩家："最近有什么活给我干吗？"
├─ 调用 #1（流式）：
│    meta: {"emo":"微笑","fav":0,"act":{"kind":"task_draft","direction":"收集类，目标是猫爪，3个左右，报酬金币"}}
│    正文（过渡，1~2 句）："哦？想找活干？我看看手头有什么适合你的……"
│    ↑ meta 解析成功瞬间：mood 事件发前端；TaskRunner 立即后台启动（方向严格注入）
├─ TaskRunner（后台，与上面正文同时跑）：
│    R1: prepare_task_context(收集类, 金币系) → 候选
│    R2: draft_agent_task(...) → 校验
│         ├─ 通过 → final(draft_summary)          → agent_status「委托已拟好」
│         ├─ 小幅偏差 → 自动修复（见 05 §4）→ final（附 repaired_notes）
│         └─ 未通过 → intermediate(vague) + 聚合错误反馈 → R3 修正重试 → …
│            （4 轮仍不过 → 后端兜底草案，见 05 §5，final(fallback_summary)）
└─ MERGE（正文流完后）：
     ├─ 宽限内拿到 final → 调用 #2：NPC 详细说明草案（任务内容/要求/奖励），
     │    结尾必须问「你接还是不接？」→ system_notice「任务草案已拟定，等待确认」
     ├─ 超时但有 intermediate → 调用 #2a（1~2 句模糊过渡："在给你琢磨报酬了，稍等。"）
     │    → 等 final → 调用 #2b 详细说明
     └─ 失败（兜底也不可用，极端） → NPC 人设话术："今天手头的事都派完了，改天吧。"
```

### 4.2 玩家回应（HITL，下一回合）

聊天主 Agent 依据 meta.act 直接处理，**不经子 Agent**：

| 玩家意图 | act              | 动作                                                                                                                                         |
| -------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 接受     | `task_confirm` | 同步 confirm（校验+原子写+分配 ID），成功 → 正文确认 + tool_status(success) + system_notice「委托已发布」                                   |
| 拒绝     | `task_cancel`  | 同步 cancel，正文符合人设地送客                                                                                                              |
| 讨价还价 | `task_update`  | 启动 TaskRunner（update 模式）→ 完成 → 调用#2 重新说明方案并再次询问。**讨价还价不需要 intermediate 过渡**（调整简单，直接等 final） |
| 岔开话题 | null             | 草案保留；草案触碰计数器 +1，达到`draft_keep_turns`（默认 3）自动 cancel + system_notice「过期的委托草案已取消」                           |

confirm 失败（草案过期/校验回归，极少）：丢弃已生成正文 → 补救调用让 NPC 解释（"这份委托刚才出了点问题……"）+ system_notice 真实原因。

### 4.3 搜索并行（search）

```
玩家："你知道安迪以前的事吗？"
├─ 调用 #1：meta act=search(query)，正文过渡："安迪啊……让我想想。"
├─ SearchRunner 后台：search_knowledge 等工具循环 ≤3 轮 → final(findings ≤400字)
└─ MERGE：调用 #2 携带 findings → NPC 以人设口吻讲述。
```

## 5. TaskRunner 内部（task_runner.py）

```python
class TaskRunner(SubagentBase):
    tools = [prepare_task_context, draft_agent_task, update_task_draft,
             search_items, search_stages, list_skills, read_skill, read_skill_file]
```

- **节点精确注入**（每轮只注入当前必须内容）：system = 协调员基座 + 当前模式规则卡
  （draft：定价卡+字段说明，与校验器同源生成；update：讨价还价卡）；user = 方向继承块
  （prepare 参数还原）+ 最近 5 轮对话（含 NPC 过渡话，要求拟定与发言一致）+ 候选池全文 / 当前草案。
  skills 索引与 read_skill 通道已移除（规则全部前置内联）。
- **情形分流**：prepare 成功（交流 Agent 工具调用已执行）→ tools=[draft] 直接拟；
  prepare 失败/缺失 → tools=[prepare] 先重取候选（继承对话思路，不回炉交流 Agent），
  拿到候选后**轮内动态收窄**为 [draft]。
- 循环：LLM（非流式，tools）→ 执行 tool_calls（**同批多个并行 asyncio.gather**，修 B3 的串行；
  工具本身是 CPU 轻操作 + 内存检索，无需 to_thread）→ 结果回填 → 直至 draft 成功/失败或达轮限。
- 校验反馈形态由 `05` 定义：聚合错误 + 修复指引 + 候选清单；数值类问题已被后端微调
  （auto_repaired 标注），打回只含选择类问题。
- update 模式：只调 `update_task_draft(draft_id, modify_fields)`；讨价还价计数（上限 2 次）保留。
- 每轮向 events 队列发 progress；校验失败发 intermediate；结束发 final/failed。
- **逐轮 INFO 日志**：轮次、工具表、结果 status、校验失败逐条 rule/field/message（可观测性）。

## 6. SearchRunner 内部（search_runner.py）

```python
class SearchRunner(SubagentBase):
    tools = [search_knowledge, search_items, search_stages,
             list_skills, read_skill, read_skill_file]
```

- 输入：query + NPC 视角简述。输出 final：`findings`（≤400 字结论，附关键出处类型）。
- 这是「检索 Agent」的落地（用户问题 4）：Tier-1 预检索保底（每回合必有），SearchRunner 负责模型主动发起的深挖。

## 7. 上下文装配（context.py）

并行 `asyncio.gather`：

| 项              | 来源                       | 说明                                                               |
| --------------- | -------------------------- | ------------------------------------------------------------------ |
| NPC 状态        | `services/npc` 内存单例  | 阵营/称号/情绪列表/好感/关系/商店/切磋（不再每请求读 JSON，见 06） |
| 会话记忆        | SQLite                     | 近 N 条 + 滚动摘要 + pending_draft 摘要块 + 玩家身份/进度          |
| Tier-1 检索     | `services/retrieval`     | 见 04：≤3 个 query 变体，单次嵌入，矩阵乘 + 池内过滤 + RRF        |
| 提及 NPC 块     | 子串匹配（沿用）           | 保留现状逻辑，移入 retrieval                                       |
| 立绘            | `services/portraits`     | 仅 vision 模型；manifest 查表 + 缓存（07）                         |
| appearance 文本 | npc_state_db.json 可选字段 | 有则恒入 prompt（07）                                              |
| save_info       | gamebridge stub            | 恒 None（预留，01 §9）                                            |

## 8. 新旧行为对照（必须做到的改善）

| 场景      | 旧                                                  | 新                                      |
| --------- | --------------------------------------------------- | --------------------------------------- |
| 纯聊天    | supervisor→(可能 worker)→dialogue，2~3 次串行 LLM | **1 次流式调用**，情绪同步前置    |
| 发任务    | 串行等待 TaskAgent 全程，玩家干等                   | 正文先出，TaskRunner 后台并行，汇合说明 |
| 查资料    | 路由→QueryAgent→dialogue 串行                     | 正文先出，SearchRunner 后台并行         |
| 确认/取消 | 重跑整条 agent 链路                                 | meta 直发，后端同步执行 <300ms          |
| 情绪      | supervisor 决定+回显约束                            | meta 行自带，解析即发，天然先于正文     |

## 9. 测试要求（P3/P4 验收）

1. Fake LLM（脚本化流）驱动 orchestrator：纯聊天/发任务/确认/取消/讨价还价/岔开话题保留与过期/搜索 七条主路径 e2e。
2. 中间结果规则：构造「校验失败 1 次后成功」的 TaskRunner，断言过渡语只说一次、不含数字。
3. 宽限路径：TaskRunner 在宽限内完成 → 无过渡语直接汇合。
4. confirm 失败补救：mock confirm 抛错 → 断言正文被丢弃、补救调用发生、system_notice 含原因。
5. 子 Agent 取消：回合中玩家新消息到达 → 旧子 Agent cancelled，无泄漏（task 数归零）。
6. 并发：同会话两请求不交叉（会话级锁）；NPC 状态更新不丢（06 的锁）。
