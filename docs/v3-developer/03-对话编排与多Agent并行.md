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
└── prompts.py    # 子 Agent 专用小 prompt（与聊天 prompt 完全分离）+ 过渡方向种子

services/tools/route/          # 聊天主 Agent 的路由声明工具（category="route"，零副作用）
├── _routing_base.py           # RoutingTool 基类（下划线前缀 → discover() 不当作工具模块）
├── route_task_draft.py        # 判据写在各自的 description 里：工具表即路由提示词
├── route_task_update.py
├── route_task_confirm.py
├── route_task_cancel.py
└── route_search.py
```

## 2. TurnOrchestrator 状态机

```
IDLE
  → BUILD_CONTEXT        并行装配（目标 ≤300ms，检索 ≤150ms 见 04）
  → BURST_STREAMING      调用 #1：缓冲解析 meta → 发 meta 事件 → 流式转发正文 → 读 finish 里的路由声明
       │ 无路由调用                        → POST_PROCESS → DONE
       │ route_task_confirm / route_task_cancel
       │                                  → SYNC_ACTION（后端办完 → 收场调用 #2）→ POST_PROCESS → DONE
       │ route_task_draft / route_task_update / route_search
       │                                  → 启动子 Agent（路由轮不说话，正文为空）
  → MERGE_WAIT           子 Agent 未完成
       │ 宽限 merge_grace_ms 内完成     → MERGE_REPLY（调用 #2，携带结果）
       │ 有中间结果且未说过过渡语       → INTERIM_REPLY（调用 #2a，1~2 句只点方向，只说一次）
       │                                → 继续等最终结果 → MERGE_REPLY（调用 #2b）
       │ 无中间结果                     → 等（SSE 保活注释帧）→ 最终结果 → MERGE_REPLY
       │ 子 Agent 失败/超时             → FAIL_REPLY（人设话术说明，不甩错误码）
  → POST_PROCESS         写记忆/好感/草案触碰计数/usage/done
  → DONE
```

**LLM 调用预算（替代旧 token 死熔断，B4）**：每回合聊天主 Agent ≤ 3 次调用（#1 + 过渡 #2a + 汇合 #2b；SYNC_ACTION 路径是 #1 + 收场 #2）；TaskRunner ≤ 4 轮、SearchRunner ≤ 3 轮工具循环（config 可调）。任何超限立即走失败话术路径，绝不无限循环。

**路由轮为什么静默**：`client.py` 只在 `finish` 事件里聚合 `tool_calls`，正文开始流出时
编排层还不知道本轮会不会路由。缓冲整个调用 #1 再决定，会让 100% 的回合（含纯聊天）
失去 token 流式体验，代价远大于收益。取舍（用户裁定）：正文无条件直发，「路由轮不说话」
由提示词与工具 `description` 约束；模型违规说话时只记 warning（`路由轮（X）仍输出了台词`），
**不吞掉玩家已经看到的内容**。因此 confirm/cancel 必须补一次收场调用，任务/搜索路径的
第一句台词来自过渡或汇合调用——静默期由 `agent_status` / `tool_status` 的等待占位承担。

**请求级开关（玩家设置项，经 `NPCChatRequest` 传入）**：

- `agent_enabled=false`：本回合只可能走「无路由调用」那条分支。三处闸门——
  ① 不下发任何路由工具（模型连调用入口都没有，`ChatRequest.tools is None`）；
  ② system prompt 去掉路由指南与全部任务发布约束（发布硬约束/商店奖励类型/切磋可行性），
  改注入 `NO_AGENT_CONSTRAINT`，让 NPC 以人设口吻婉拒而不是空口答应；玩家坚持要活时
  还要引导他去终端打开接取委托的权限（沿用 v2 `game_rag_service` 同口径的提示，
  但要求用角色口吻说，不报界面术语）；
  ③ meta 协议块去掉「或者发一个路由工具调用」那半句（`meta_prompt_block(agent_enabled=False)`），
  否则就成了悬空指代，反而诱导模型凭空编一个调用。
  模型仍越权硬发 `route_*` 时记 warning 并按纯聊天处理（`_parse_route` 直接短路成 None）。
  残留的待确认草案不进 prompt（模型既无法确认也无法取消，摆着只会诱导它承诺做不到的事），
  但仍照常装配以累计 `draft_keep_turns` 计数被自然清理。
- `summarize_interval`：精确短期记忆档位，见 §7.1。

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

- **intermediate 的语义**（用户硬性规则）：TaskRunner 在两个时点发 intermediate——
  ① `prepare` 拿到候选后（`vague_note` = 该 task_type 的**方向种子**，见 `subagents/prompts.py`
  的 `DIRECTION_VAGUE_NOTES`，只到「这是件什么性质的事」，绝不落到具体目标/物品/人名）；
  ② 草案校验未通过、正在修复时（`vague_note` = "委托内容还在调整，稍等一下。"）。
  两者的 `vague_note` 都**禁止包含任何数量、物品名、金额**。orchestrator 每回合最多消费
  一次 intermediate（说完过渡语后只等 final），且**首条方向种子优先**——被后到的「还在调整」
  覆盖会让玩家觉得 NPC 自己也说不清要派什么活。
- `ui_hint` 取值：「准备候选中」/「修改方案中」（启动）、「拟定委托中」（prepare 完成）、
  「校验修正中」（打回修复）、「生成保底方案」（兜底）、「查资料中」（SearchRunner）。
  `agent_status.phase` 一律由 `ui_hint` 推：只有含「修正」的算 `repairing`，其余任务侧
  事件都是 `drafting`，检索侧是 `searching`，终态是 `done`/`failed`。
- 子 Agent 不发图、不接触玩家可见正文；其 LLM 调用 `purpose="subagent"`。

## 4. 任务全流程时序（核心业务流）

### 4.1 首次发布（route_task_draft）

```
玩家："最近有什么活给我干吗？"
├─ 调用 #1（流式，路由轮 → 无台词）：
│    meta: {"emo":"微笑","fav":0}
│    tool_calls: route_task_draft(note="玩家想找点活干，报酬想要金币")
│    ↑ meta 解析成功瞬间：meta 事件发前端；流结束后读到路由声明 → TaskRunner 立即后台启动
├─ TaskRunner（后台，选型也在这里做）：
│    R1: prepare_task_context(task_type, reward_types) → 候选
│         └─ 立刻 intermediate(vague_note=方向种子) + progress「拟定委托中」
│    R2: draft_agent_task(...) → 校验（定价卡在 R1 结果回来后按真实 task_type 补投）
│         ├─ 通过 → final(draft_summary + task_type)  → agent_status「委托已拟好」
│         ├─ 小幅偏差 → 自动修复（见 05 §4）→ final（附 repaired_notes）
│         └─ 未通过 → intermediate「校验修正中」+ 聚合错误反馈 → R3 修正重试 → …
│            （4 轮仍不过 → 后端兜底草案，见 05 §5，final(fallback_summary)）
└─ MERGE：
     ├─ 宽限内拿到 final → 调用 #2：NPC 详细说明草案（任务内容/要求/奖励），
     │    结尾必须问「你接还是不接？」→ system_notice「任务草案已拟定，等待确认」
     ├─ 宽限外但已见 intermediate → 调用 #2a（1~2 句只点方向："缺些东西，得让人替我
     │    弄回来，稍等我一下。"）→ 等 final → 调用 #2b 详细说明
     └─ 失败（兜底也不可用，极端） → NPC 人设话术："今天手头的事都派完了，改天吧。"
```

过渡语的口吻要求（`merge.py` 的 `_interim_reply`）：这事是 NPC **自己在盘算的**——
方向心里有数、细节还没敲定。既不能表现得茫然（不是谁塞给他的活），也不能假装细节
已经定了；硬性禁止说出任何具体数字、物品名、金额、人名、地名。方向种子缺失时回落
`DIRECTION_VAGUE_FALLBACK`（"手头正好有件事想找人办"），模型空回复时再用静态兜底话术，
保证玩家侧不会对着空白等草案。

### 4.2 玩家回应（HITL，下一回合）

编排层依据路由声明直接处理，**不经子 Agent**：

| 玩家意图 | 路由 | 动作 |
| -------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 接受     | `route_task_confirm` | 同步 confirm（校验+原子写+分配 ID）→ tool_status(success) → 收场调用 #2 让 NPC 交代 → system_notice「委托已发布」 |
| 拒绝     | `route_task_cancel`  | 同步 cancel → tool_status(success) → 收场调用符合人设地送客 → system_notice「委托已取消」；无草案可撤不算失败，NPC 顺着话头自然回应、不发系统行 |
| 讨价还价 | `route_task_update`  | 启动 TaskRunner（update 模式）→ 完成 → 调用#2 重新说明方案并再次询问。**讨价还价不需要 intermediate 过渡**（调整简单，直接等 final）。路由错发 update 而实际无草案时降级为 draft 模式重拟 |
| 岔开话题 | 无路由调用 | 草案保留；草案触碰计数器 +1，达到`draft_keep_turns`（默认 3）自动 cancel + system_notice「过期的委托草案已取消」 |

「不愿意让步」不是一种路由：模型不调任何工具，直接以角色口吻拒绝并问玩家到底接还是不接
（判据写在 `route_task_update` / `route_task_cancel` 的 description 里）。

confirm 失败（草案过期/校验回归，极少）：tool_status(failed) → system_notice 给真实原因 →
补救调用让 NPC 解释（"这份委托刚才出了点问题……"）。路由轮本来就没有台词，所以不存在
「丢弃已生成正文」这一步；草案保留不清。收场/补救调用若模型空回复，用静态兜底话术补上。

### 4.3 搜索并行（route_search）

```
玩家："你知道安迪以前的事吗？"
├─ 调用 #1（路由轮，无台词）：meta {"emo":"普通","fav":0} + route_search(query="安迪的过去")
├─ SearchRunner 后台：search_knowledge 等工具循环 ≤3 轮 → final(findings ≤400字)
└─ MERGE：调用 #2 携带 findings → NPC 以人设口吻讲述。
```

### 4.4 等待预算与 bridge 排水（merge.py）

总预算 = `merge_grace_ms` + `subagent_timeout_s`，一个 `deadline` 管到底：

1. **宽限期**内排水。拿到 `final` → 直接汇合，不说过渡语（快路径，玩家只看到一次开口）。
2. 宽限期过了仍无终态 → 进入「等到可说话的时机」循环：
   - 已见 intermediate 且还没说过过渡语 → 立刻说（调用 #2a），然后把**剩余全部预算**用来等 final；
   - 否则继续排水，且 `bridge_on_intermediate=True`：**intermediate 一到就返回**去说过渡语。
     这一条是关键——`prepare` 常常晚于宽限期才回来，若只按固定片轮询，方向种子就会
     被跳过，玩家对着空白一直等到草案拟完（等于回到重构前的体验）。
   - 过渡语说过之后不再 bridge：一路等到终态或预算耗尽（「中间结果只说一次话」是硬规则）。
3. 预算耗尽仍无终态 → `agent_status(failed,"等待超时")` + 人设话术，不甩错误码。

## 5. TaskRunner 内部（task_runner.py）

工具表**每轮动态取**（`SubagentBase.run()` 每轮重新调 `_tools_schema()`），只有两种模式：

```python
MODE_DRAFT:  轮1 [prepare_task_context] → 轮2+ [draft_agent_task]
MODE_UPDATE: 每轮 [update_task_draft]
```

- **节点精确注入**（每轮只注入当前必须内容）：system = 协调员基座 + 当前模式规则卡
  （draft：任务选型规则 `PREPARE_SELECTION_GUIDE` + 字段说明；update：讨价还价卡）；
  user = 方向块（路由 `note`，缺省回落玩家原话）+ 「更早的对话印象」（滚动摘要，
  见 §7.1）+ 最近 8 条对话（近 4 轮，含本回合已流出的过渡话，要求拟定与发言一致）
  + 候选池全文 / 当前草案。skills 索引与 read_skill 通道已移除（规则全部前置内联）。
- **选型在本 Agent 内闭环**：v3 再次重构后聊天主 Agent 不再预取候选，`task_type` /
  `reward_types` 由 TaskRunner 自己按 `PREPARE_SELECTION_GUIDE` 决定。选错类型不是
  不可恢复缺陷——规则卡第 4 条授权它换一个 task_type 重新 prepare（4 轮预算内）。
- **定价卡事后补投**：奖励定价卡依赖 prepare 揭示的真实 `task_type`，launch 时无法预知。
  `_on_round_tool_calls` 在 dispatch 前捕获 prepare 的**调用参数**（工具返回里没有
  task_type），`_post_round_messages` 再把定价卡作为一条 `user` 消息**追加**到对话尾部
  ——不改写 `messages[0]`，system 前缀保持不变，前缀缓存不受影响。
- **情形分流**：draft 模式启动时 `_prepare_done=False` → 轮1 只有 prepare；prepare 回来
  即翻转为 True，轮内**动态收窄**为 [draft]。update 模式 `_prepare_done` 恒 True，全程只有 update。
- 循环：LLM（非流式，tools）→ 执行 tool_calls（**同批多个并行 asyncio.gather**，修 B3 的串行；
  工具本身是 CPU 轻操作 + 内存检索，无需 to_thread）→ 结果回填 → 直至 draft 成功/失败或达轮限。
- 校验反馈形态由 `05` 定义：聚合错误 + 修复指引 + 候选清单；数值类问题已被后端微调
  （auto_repaired 标注），打回只含选择类问题。
- update 模式：只调 `update_task_draft(draft_id, modify_fields)`；讨价还价计数（上限 2 次）保留。
- 事件：启动发 progress（「准备候选中」/「修改方案中」）；prepare 完成发 intermediate
  （方向种子，见 §3）；校验失败发一次 intermediate（「校验修正中」）+ 后续 progress；
  结束发 final（`draft_summary` + `task_type`，供汇合按类型条件拼接说明要点）/failed。
- **逐轮 INFO 日志**：轮次、工具表、结果 status、校验失败逐条 rule/field/message（可观测性）。

## 6. SearchRunner 内部（search_runner.py）

```python
class SearchRunner(SubagentBase):
    tool_names = ["search_knowledge", "search_items", "search_stages"]
```

- 输入：`route_search.query`（缺省回落玩家原话）+ NPC 视角简述。输出 final：`findings`（≤400 字结论，附关键出处类型）。
- 这是「检索 Agent」的落地（用户问题 4）：Tier-1 预检索保底（每回合必有），SearchRunner 负责模型主动发起的深挖。
- 不吃 `recent_dialogue`，因此也不带滚动摘要（§7.1）。

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

### 7.1 精确短期记忆档位（summarize_interval）

前端「精确短期记忆长度」滑条（短/中/长/几乎无限 → 10/30/100/500）按请求传入；
白名单唯一权威是 `services.memory.store.PRECISE_MEMORY_OPTIONS`，非白名单值一律
按「未提供」处理（不静默改写，收敛在 `TurnOrchestrator.__init__`）。

| 请求档位                     | 从库读多少条（history_limit）  | 逐字进 prompt 多少条              | 滚动摘要节奏 |
| ---------------------------- | ------------------------------ | --------------------------------- | ------------ |
| 10 / 30 / 100 / 500          | = 档位                         | = 档位                            | 不变         |
| 不传 / 非白名单              | `deps.history_limit`（20）     | `deps.recent_dialogue_limit`（8） | 不变         |

两条边界，都是刻意的：

- **档位只作用于聊天主 Agent。** 子 Agent 的 `recent_dialogue` 用固定预算
  `SUBAGENT_RECENT_MESSAGES`（8 条），不随档位膨胀——它只需要「刚才谈到哪」，
  灌 500 条历史只会挤占它自己的工具循环预算。任务 Agent 另外拿到滚动摘要
  （`memory_summary`，来自同一份永久模糊记忆），在 prompt 里作为独立的
  「更早的对话印象」块排在最近 8 条之前：摘要体积恒定、不随档位膨胀，
  既补上「这玩家早先谈过什么」的背景，又不吃掉工具循环预算。
  `SearchRunner` 不吃 `recent_dialogue`，因此也不带摘要。
- **档位不影响滚动摘要。** 摘要恒为每 `SUMMARIZE_INTERVAL`（30）条触发、压最近
  30 条。前端设置页对玩家的承诺是「角色已具有单个会话内容的永久模糊记忆，此处
  调整的仅为精确的短期记忆长度」；若让档位连带拉长摘要间隔，选「几乎无限」的
  玩家反而会在 500 条之前彻底失去模糊记忆，与承诺相反。
  （字段名 `summarize_interval` 是 v2 遗留——当年它同时驱动摘要间隔与历史长度，
  v3 恢复时只保留了后者，名字为兼容已发布前端而未改。）

## 8. 新旧行为对照（必须做到的改善）

| 场景      | 旧                                                  | 新                                      |
| --------- | --------------------------------------------------- | --------------------------------------- |
| 纯聊天    | supervisor→(可能 worker)→dialogue，2~3 次串行 LLM | **1 次流式调用**，情绪同步前置    |
| 发任务    | 串行等待 TaskAgent 全程，玩家干等                   | 路由轮静默，TaskRunner 后台并行，prepare 一回来就说过渡话，汇合再详述 |
| 查资料    | 路由→QueryAgent→dialogue 串行                     | 路由轮静默，SearchRunner 后台并行，汇合讲述 |
| 确认/取消 | 重跑整条 agent 链路                                 | 路由声明直发，后端同步执行 <300ms，办完一次收场调用 |
| 情绪      | supervisor 决定+回显约束                            | meta 行自带，解析即发，天然先于正文     |
| 路由判据  | 挤在聊天 system 里，与扮演/选型规则混写             | 写在路由工具的 description 里，工具表即路由提示词；`agent_enabled=false` 时整张表摘掉 |
| 聊天 prompt | 含 1229 字任务选型规则                            | 选型规则整体移交任务 Agent，聊天 system 只留「该不该路由、路由给谁」 |

## 9. 测试要求（P3/P4 验收）

1. Fake LLM（脚本化流）驱动 orchestrator：纯聊天/发任务/确认/取消/讨价还价/岔开话题保留与过期/搜索 七条主路径 e2e（`tests/orchestrator/test_turn_e2e.py`）。
2. 中间结果规则：构造「校验失败 1 次后成功」的 TaskRunner，断言过渡语只说一次、不含数字、种子是 prepare 选出的真实 task_type 对应的方向。
3. 宽限路径：TaskRunner 在宽限内完成 → 无过渡语直接汇合（只有 2 次流式调用）。
4. bridge 排水：prepare 晚于宽限期完成 → 过渡语照说（3 次流式调用），玩家不对着空白等草案。
5. 工具表动态收窄 + 定价卡补投：断言逐轮工具表为 `[[prepare],[draft]]`、定价卡不在轮1 的 system 里、轮2 恰好追加一条含卡片的 user 消息、且轮2 的 system 与轮1 逐字一致（前缀缓存）。
6. confirm 失败补救：mock confirm 抛错 → 断言 `system_notice` 先于补救正文、草案保留。
7. 收场/补救调用空回复 → 静态兜底话术，玩家侧不留空白。
8. 路由轮违规说话 → 正文照发（不吞玩家已看到的内容）+ WARNING 留痕。
9. 路由参数坏 JSON → 路由仍生效，note 回落玩家原话。
10. `agent_enabled` 开关：关闭时不下发任何工具、system 不含路由/发布约束、残留草案不进 prompt 但照常过期、模型硬发 `route_*` 一律忽略（不发布任务、不落草案）。
11. 子 Agent 取消：回合中玩家新消息到达 → 旧子 Agent cancelled，无泄漏（task 数归零）。
12. 并发：同会话两请求不交叉（会话级锁）；NPC 状态更新不丢（06 的锁）。
