# 03 · 对话编排与多 Agent 并行（services/orchestrator/ + services/subagents/）

> 实施阶段 P3/P4。本章定义回合状态机、任务全流程时序、检索路径、讨价还价、HITL 与「中间结果规则」。
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

**LLM 调用预算（替代旧 token 死熔断，B4）**：每回合聊天主 Agent ≤ 3 次调用（#1 + 过渡 #2a + 汇合 #2b；SYNC_ACTION 路径与常规检索路径是 #1 + 收场 #2）；TaskRunner ≤ 4 轮、SearchRunner ≤ 3 轮工具循环（config 可调）。`route_search` 默认不启动子 Agent——预取由编排层代码完成，只有 `deep_search=true` 才起 SearchRunner。任何超限立即走失败话术路径，绝不无限循环。

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
- `long_term_memory_enabled`：跨对话长期记忆开关（前端每请求带），见 §7.2。

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

指令里**不给现成的口吻模板**（2026-10-11 起）：此前写着「口吻是『先跟你打个招呼，容我把
这事安排出来』」，真机走查中模型直接逐字照抄，且轮 1、轮 3 重复同一句，读起来像念稿；
改为「用你自己的口吻说，不要照搬固定句式」。同时补两条：不要许诺「稍后再细讲」（紧接着的
汇合调用 #2b 就会把这件事讲完，许诺了自相矛盾）、不要带动作描写。作者随后把口径收得更死：
「准确的说是不要涉及任何具体内容，此时你脑海中只有个大致方向，还没想好任何任务细节。
不要把关注点放在任务内容上，而是要放在上下过渡上，核心是表述你确实【有】个任务/委托/需求/
想法等，而不是你有【什么】任务。」回归锁见
`tests/orchestrator/test_turn_e2e.py::test_task_draft_with_interim_rule`。

### 4.2 玩家回应（HITL，下一回合）

编排层依据路由声明直接处理，**不经子 Agent**：

| 玩家意图 | 路由 | 动作 |
| -------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 接受     | `route_task_confirm` | 同步 confirm（校验+原子写+分配 ID）→ tool_status(success) → 收场调用 #2 让 NPC 交代 → system_notice「委托已发布」 |
| 拒绝     | `route_task_cancel`  | 同步 cancel → tool_status(success) → 收场调用符合人设地送客 → system_notice「委托已取消」；无草案可撤不算失败，NPC 顺着话头自然回应、不发系统行 |
| 讨价还价 | `route_task_update`  | 启动 TaskRunner（update 模式）→ 完成 → 调用#2 重新说明方案并再次询问。**讨价还价不需要 intermediate 过渡**（调整简单，直接等 final）。玩家明确要求整体换向时由任务 Agent 自己走 prepare→draft 重拟（见 §5），此时 prepare 完成后照常说过渡语；路由错发 update 而实际无草案时降级为 draft 模式重拟 |
| 岔开话题 | 无路由调用 | 草案保留；草案触碰计数器 +1，达到`draft_keep_turns`（默认 3）自动 cancel + system_notice「过期的委托草案已取消」 |

「不愿意让步」不是一种路由：模型不调任何工具，直接以角色口吻拒绝并问玩家到底接还是不接
（判据写在 `route_task_update` / `route_task_cancel` 的 description 里）。

confirm 失败（草案过期/校验回归，极少）：tool_status(failed) → system_notice 给真实原因 →
补救调用让 NPC 解释（"这份委托刚才出了点问题……"）。路由轮本来就没有台词，所以不存在
「丢弃已生成正文」这一步；草案保留不清。收场/补救调用若模型空回复，用静态兜底话术补上。

#### 4.2.1 草案在途时仍可检索（2026-10-17）

工具表按**有无待确认草案**二选一，两张表都在 `services/tools/base.py` 定量给出：

| 状态 | 工具表（顺序固定） |
| ---- | ------------------ |
| 无草案 | `ROUTE_TOOLS_NEW`＝`route_task_draft`、`route_search` |
| 有草案 | `ROUTE_TOOLS_DRAFT`＝`route_task_confirm`、`route_task_cancel`、`route_task_update`、`route_task_draft`、`route_search` |

草案在途时玩家仍可能问设定、人物往事、关卡或物品——那是检索，不是对草案表态。
`route_search` 因此**在两张表里都在**：检索既不取消草案，也不重置/推进
`rounds_without_task`（草案过期只由「连续 `draft_keep_turns` 轮纯聊天」触发，
`route_search` 路径压根不碰这个计数器）。`build_routing_guide(has_pending_draft=True)`
也把这条写成了明文规则（「草案在途时仍可检索，检索不会取消草案，也不会让它提前过期」）。
`ToolRegistry.dispatch_batch` 只按 pipeline 分组、不按本轮工具表过滤，故补表后检索工具照常执行。
回归锁＝`tests/orchestrator/test_draft_route_search.py`。

### 4.3 检索（route_search）

`route_search` 承担检索的第一步：`keywords` + `sources` 交给编排层代码预取（`orchestrator/search_prefetch.py`，不调模型），
常规路径不再经过子 Agent。

```
玩家："你知道安迪以前的事吗？"
├─ 调用 #1（路由轮，无台词）：meta {"emo":"普通","fav":0}
│    + route_search(keywords="安迪 过去", sources=["knowledge"], deep_search=false)
├─ 编排层预取：knowledge 走 Tier-1 管线（与本轮已有结果按节点 id 去重），
│    items/stages 走名称/区域匹配；分类型截断（3600/1200/1200，合计约 6000 字）
└─ 调用 #2（收场轮，不带 tools）：结果块【后台结果：检索到的资料】拼进 user
     → NPC 以人设口吻讲述；查不到就自然带过。
```

`deep_search=true`（且必须填 `query`）时才委派 SearchRunner：预取结果连同聊天 Agent
本轮已有的 Tier-1 材料一起给它，由它决定补查还是直接总结，最后仍由收场轮转述。
`deep_search=true` 但缺 `query` 时记 warning 并按常规检索处理（路由轮已静默发出，不能回头）。

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
MODE_UPDATE: 初表 [update_task_draft, prepare_task_context] →（重拟 prepare 后）[draft_agent_task]
```

- **节点精确注入**（每轮只注入当前必须内容）：system = 协调员基座 + 当前模式规则卡
  （draft：任务选型规则 `PREPARE_SELECTION_GUIDE` + 字段说明；update：讨价还价卡 +
  整体重拟条目 + 同一套选型规则/报酬能力/字段说明——重拟路径要用）；
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
- **情形分流**：两种模式都从 `_prepare_done=False` 起步——draft 轮1 只有 prepare，update
  初表 [update + prepare]；prepare 回来即翻转为 True，轮内**动态收窄**为 [draft]。
  update 的 prepare 是**重拟路径**（仅玩家明确要求整体换向时触发，提示词里写明限制），
  走完等于重走两步式生成、新草案整体替换旧草案；常规讨价还价不碰它。
- 循环：LLM（非流式，tools）→ 执行 tool_calls（**同批多个并行 asyncio.gather**，修 B3 的串行；
  工具本身是 CPU 轻操作 + 内存检索，无需 to_thread）→ 结果回填 → 直至 draft 成功/失败或达轮限。
- 校验反馈形态由 `05` 定义：聚合错误 + 修复指引 + 候选清单；数值类问题已被后端微调
  （auto_repaired 标注），打回只含选择类问题。
- update 模式：常规只调 `update_task_draft(draft_id, modify_fields)`；讨价还价计数（上限 2 次）保留。
  整体重拟（罕见）走 `prepare_task_context → draft_agent_task`，定价卡同样按新类型补投。
- 事件：启动发 progress（「准备候选中」/「修改方案中」）；prepare 完成发 intermediate
  （方向种子，见 §3）；校验失败发一次 intermediate（「校验修正中」）+ 后续 progress；
  结束发 final（`draft_summary` + `task_type`，供汇合按类型条件拼接说明要点）/failed。
- **逐轮 INFO 日志**：轮次、工具表、结果 status、校验失败逐条 rule/field/message（可观测性）。

## 6. SearchRunner 内部（search_runner.py）

```python
class SearchRunner(SubagentBase):
    tool_names = ["search_knowledge", "search_items", "search_stages"]
```

- 输入：`route_search.query`（深度检索时必填）+ NPC 视角简述 + 两批现成材料（聊天 Agent 本轮已有的 Tier-1 文本、编排层按 `keywords` 预取的结果）。输出 final：`findings`（提示词要求 ≤400 字，后端到 1000 字才截断，且无论是否截断都回传给收场轮）。
- 只有 `deep_search=true` 才启动；常规检索由编排层代码预取直接进收场轮，本 Runner 不参与。
- 这是「检索 Agent」的落地（用户问题 4）：Tier-1 预检索保底（每回合必有）+ 关键词预取，SearchRunner 只负责模型主动发起的深挖。
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

### 7.2 跨对话长期记忆（long_term_memory_enabled，2026-10-12 阶段三）

与 §7.1 的「精确短期记忆」和滚动摘要是三套不同的东西：短期记忆是逐字进 prompt 的最近若干条；
摘要按**会话**存、只总结对话内容；长期记忆按 **(作用域, NPC)** 存，跨会话与跨对话窗口保留。
实现集中在 `services/memory/long_term.py`，表在既有的 `tools/memory.db` 里（`long_term_memory`）。

**开关**：`NPCChatRequest.long_term_memory_enabled`。判定在 `resolve_enabled()`——
embedded 口径（`--embedded` 启动或已被 bind）**恒定启用、不受理关闭**，前端在游戏内也不给这个复选框；
standalone 按字段值，不传＝关闭。**不落后端库**：玩家的选择由前端 localStorage 记住，每请求带一次即可。

**作用域**：绑定存档＝槽位键，每存档一份；未绑定＝ `store.STANDALONE_SCOPE`（`__standalone__`），
与各存档互不混淆。摘要 worker 延后执行，所以作用域键在**提交时**就解析好存进
`SummaryRequest.long_term_scope`，执行期间玩家换存档也不会写错地方。

**内容**：一条 `kind='base'`（基础长期记忆＝玩家与 NPC 的整体关系情况＋NPC 对玩家的整体印象，
提示词口径 300 字）＋至多 `LTM_MAX_FACT_ENTRIES`（20）条 `kind='fact'`（关键事件或信息，提示词口径 100 字）。
入库统一截断到 `LTM_MAX_CHARS`（500）。`fact` 的 `entry_no` 是作用域内该 NPC 的直观编号，
取最大值 +1，**永不复用、不重排**（改写不换号，所以提示词里的编号跨轮稳定可引用）。

**注入**：`context.py` 一轮读一次快照进 `TurnContext.long_term`，`prompts.build_user_shared_core`
把它拼成独立块。块序（2026-10-17 按前缀缓存重排，本批再插入两块任务块）＝
已完成委托 → 进行中委托 → 刚发布委托 → 长期记忆 → 早期摘要 → 近期对话 → 会话态（好感压尾）
→ 关系跨档提示 → RAG/提及 NPC（按前缀缓存从稳定到易变：三块任务块都是「任务相关」，
任务一完成/过期就整块消失，改动稀疏，故排在长期记忆之前；长期记忆又放在每轮全变的 RAG 之前）。
作用域内一条都没有＝整块不注入，不留空壳。摘要轮另有 `format_summary_block()`，
块尾写明「长期记忆不参与对话摘要的撰写」——摘要只总结本次给出的对话记录。

**任务三块（2026-10-17，user 侧共享 core 的最前面三块）**：

| 块 | 来源 | 生命周期 |
| -- | ---- | -------- |
| 已完成委托 | 直读 `agent_tasks.json`/`agent_text.json`（共享 ∪ 本存档），按 `progress.tasks_finished` 过滤 | 无状态，每轮实时算，最多 3 条 |
| 进行中委托 | 同上读取，按绑定投影 `progress.tasks_active` 过滤 | 无状态、只读、不记账；最多 3 条，超出在块尾写「另有 N 条进行中的委托未展开」 |
| 刚发布委托 | 同上读取，由**会话库** `session_published_task_highlights` 的 task_id 驱动 | 发布轮起算 `published_task_highlight_turns`（默认 5，含发布轮）轮；任务完成即清 |

三块按 task_id 互相去重：已完成 ∩ 进行中、进行中 ∩ 刚发布都不重复出现；highlight 过期后
若任务仍未完成，由进行中块接管。**发布轮**把块拼进本轮 **result_block**（收场调用可见），
后续轮才进 **user 侧共享 core**——同一轮不会出现两遍（`turn.py` 记本轮已注入的 task_id）。
**记录只在会话库**（`memory.db` 的一张会话级表），不落 `npc_state_db` / status / progress，
轮次也只存在库里、不进提示词（保持文本稳定、减少前缀缓存抖动）。

**未完成两块的字段清单**（已完成块不在此列）：标题 → 任务说明 → 报酬 → 任务达成内容 →
**物品详情** → 完成 NPC（场景）→ 关卡与怪物。后三段查不到就整段省略，绝不写「未知」。
**物品详情**（2026-10-17 补）覆盖任务涉及的**提交物 / 持有获取物 / 奖励**，逐件给
「分类用途、等级、单价、重量、介绍」——口径与装备块共用 `npc_equipment.format_weight`，
避免同一件物品在两处写出不同重量；**不给获取途径**（那是攻略信息，且可能与委托要求互相打架：
委托让玩家去打，这里又说商店有卖），玩家真问到「去哪弄」由 RAG 物品图鉴兜底。
去重两层：① 本轮 RAG 上下文已列过的物品跳过（`parse_rag_game_entity_mentions` 解析，
与草案摘要同一套口径）——所以**这两块的装配位置在检索之后**，构建时把 `rag_context_text` 传进去；
② 同块内前面已补过的物品跳过（进行中块跨条目共享一份 `seen`，同一件物品只给一次详情）。
物品表查不到的跳过，全部查不到＝整个「物品：」节不出现。同一件物品在一条委托里出现多次
（提交＋持有＋奖励）只出一次，数量取**首次出现**的那个。

**工具（关键设计约束）**：聊天侧永远只有一个工具名 `update_long_term_memory`，
schema 与描述按作用域当前状态**三选一**；摘要侧另有 `set_base_long_term_memory`（只在摘要那次调用下发）。
两个工具都**不进 `ToolRegistry`**——注册表的 `description`/`parameters_schema` 是类属性字面量，
做不到同名按状态换形；改由 `turn.py` 手拼 tool dict，并在新方法 `_apply_long_term_calls()` 里就地执行。
写库失败只记警告，**不阻断本轮对话**（记忆是附加能力，玩家这句话必须照常答完）。

| 作用域当前状态 | 参数 | 模型看到的语义 |
| -------------- | ---- | -------------- |
| 尚无基础长期记忆 | `base_content*` + `content*` | 记录时必须同时给出基础印象与一条关键记录 |
| 有基础印象、关键记录未满 20 | `entry_no?` + `content*` | 与既有记录相关就填编号融合改写，无关才留空新记一条 |
| 关键记录满 20 | `entry_no*` + `content*` | **只有改写能力** |

满 20 条那一形态是硬约束：模型视角下**完全感知不到"留空即可新记一条"这个分支**，
所以该形态的工具名、描述、参数表里零处出现「追加」「新记一条」「留空」等字样
（`tests/orchestrator/test_long_term_memory.py` 与 `tests/memory/test_long_term.py` 各有禁词锁）。
业务上仍能记新事＝把不重要的那条**改写**成新的。

**任务发布→完成**：`services/agent_tools/finished_tasks.py` 每轮实时算「本 NPC 发布、
且当前存档已完成」的委托，把完成对话原文拼进 prompt（一轮最多 3 条，按 id 从新到旧）。
只能直读磁盘：`TaskRegistry` 看不到 `data/rag/agent_saves/<slotKey>/` 覆盖层，
玩家台词也不进检索语料。该 NPC 已有关键记录时，块尾再追加一句
`prompts.LTM_TASK_DONE_HINT`，要求把仍记着「已发布」的那条改成「已完成」；
没有关键记录时这句不拼（没有可执行对象）。

## 8. 新旧行为对照（必须做到的改善）

| 场景      | 旧                                                  | 新                                      |
| --------- | --------------------------------------------------- | --------------------------------------- |
| 纯聊天    | supervisor→(可能 worker)→dialogue，2~3 次串行 LLM | **1 次流式调用**，情绪同步前置    |
| 发任务    | 串行等待 TaskAgent 全程，玩家干等                   | 路由轮静默，TaskRunner 后台并行，prepare 一回来就说过渡话，汇合再详述 |
| 查资料    | 路由→QueryAgent→dialogue 串行                     | 路由轮就给出关键词与范围，编排层代码直接预取，收场轮一次讲述；只有深度检索才起 SearchRunner |
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
13. update 整体重拟：update 回合模型走 prepare→draft，断言逐轮工具表为
    `[[update_task_draft, prepare_task_context], [draft_agent_task]]`、update 系统提示含
    「除非玩家明确要求把任务方向整个换掉，否则不要调用 prepare」限制、新草案整体替换旧草案。
14. 工具表完整性：聊天路由表 + 任务/检索子 Agent 的全部工具名都能在默认注册表解析（`tests/tools/test_dispatch.py`）。
