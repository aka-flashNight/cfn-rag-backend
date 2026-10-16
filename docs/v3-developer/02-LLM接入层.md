# 02 · LLM 接入层（services/llm/）

> 实施阶段 P1。本章替代 `services/llm_client.py` 与散落在 5 处的「图/工具回退矩阵」（A1/A2/A3/A5/A6/A7）。
> 依据：`docs/v3-searching/02-联网调研-模型与工具调用.md`、`02-联网调研-思考模式与响应速度.md`。

---

## 1. 模块组成

```
services/llm/
├── client.py     # LLMClient：统一调用入口（流式/非流式），连接复用、超时重试、降级
├── profiles.py   # ModelProfile 注册表：按模型名匹配思考参数/视觉能力/特殊行为
├── meta.py       # meta 行协议：解析器 + 校验 + prompt 说明文本生成
└── errors.py     # 错误分类：可降级参数错误 / 图像不支持 / 工具不支持 / 配额 / 网络
```

## 2. LLMClient

### 2.1 接口

```python
class LLMClient:
    @classmethod
    def for_config(cls, cfg: LLMConfig) -> "LLMClient": ...
        # 按 (api_base, api_key, proxy_url, model) 做客户端缓存（进程内 dict），
        # AsyncOpenAI 实例复用，禁止每次调用新建（修 A5）。
        # proxy 经 httpx 客户端级 proxies 参数传入，禁止改 os.environ（修 E3）。

    async def chat(self, req: ChatRequest) -> ChatResult: ...
        # 非流式。供后台子 Agent 使用。

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]: ...
        # 流式。供聊天主 Agent 使用。StreamEvent 见 §2.3。

class ChatRequest(BaseModel):
    messages: list[dict]
    tools: list[dict] | None = None        # 聊天 Agent 只注册路由声明工具（services/tools/route/，
                                           # 按有无待确认草案二选一）；子 Agent 全量工具循环。
                                           # 流式+tools 无兼容坑（实测）。
    purpose: Literal["chat", "subagent", "summary"] = "chat"
    send_image: bool = True                # purpose != "chat" 时强制 False（图片只进聊天轮）
    max_tokens: int | None = None
    # 聊天轮**不做**「剥 tools 重试」：现网没有不支持流式 tools 的平台，
    # 兼容分支只会白拖慢流程（用户裁定）。tools 不支持只影响子 Agent 路径，见 §3.2。
```

### 2.2 固定行为

- **超时**：连接 10s，读 120s（`timeout=httpx.Timeout(120, connect=10)`）。
- **重试**：网络错误/5xx 重试 1 次（指数退避 1s）；4xx 不重试（走 §3 降级）；同一请求最多 2 次真实 API 调用（首次 + 降级重试），子 Agent 循环外的总调用数受 orchestrator 回合预算约束。
- **usage 回收**：非流式读 `resp.usage`；流式传 `stream_options={"include_usage": True}`，末帧取 usage。全部落 `LatencyTracker`/日志，done 事件带回前端（修 A6 无统计问题；旧 `token_budget_spent` 死熔断删除，用 §5 的回合预算替代）。
- **采样参数**：默认不传 `temperature/top_p/penalty`（思考模型下这些参数失效甚至报错，调研 §3.1；Kimi 思考模式固定 1.0 且官方建议勿显式传）。
- **死代码删除**：`prefix_parts`/`prompt_prefix` 空转逻辑、`_emit_mood_updates`（llm 层硬编码业务工具并解析半截 JSON，A3）整体删除——情绪改由 meta 行承载。

### 2.3 流式事件

```python
@dataclass
class StreamEvent:
    kind: Literal["content", "reasoning", "tool_calls", "usage", "finish"]
    text: str = ""                 # content / reasoning 增量
    tool_calls: list[dict] | None = None   # 聚合完成的 tool_calls（finish 时）
    usage: dict | None = None
```

- `reasoning_content` delta 一律归入 `kind="reasoning"`，**绝不混入 content**（修 A7）。聊天主 Agent 丢弃 reasoning（不进正文、不进历史、不发前端）；其存在只影响 TTFT，靠 §3 思考参数压制。
- 多轮回传：仅当模型 Profile 标记 `retransmit_reasoning=True`（Qwen 系开思考时才需要）且本轮开了思考时，才把 reasoning_content 存进 assistant 消息回传；本项目默认全场景关思考/低思考，该路径正常不会触发，但代码要留。

## 3. 模型 Profile 与降级链（核心）

### 3.1 ModelProfile 注册表（profiles.py）

按 `model_name` 小写子串匹配，先匹配先中；**匹配不到 = `DEFAULT_PROFILE`（不传任何思考参数、vision 探测型）**。

| Profile（匹配子串）              | vision | 思考控制参数（随请求发送）                 | 备注                                                                                                                                                                              |
| -------------------------------- | ------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `glm-5.3-flash`                | ✅     | `thinking={"type":"enabled"}` + 低强度档 | **不能关**，传 disabled 会报错（官方明确）。`reasoning_effort` 是官方参数（枚举含 `low`），2026-10-16 真机非流式/流式各一次均无 400、未触发降级；将来参数名不被接受时由 §3.2.b 剥掉整个 `extra_body` 重试一次 |
| `glm-5.3`（其他档）            | ✅     | 同上                                       | 同上                                                                                                                                                                              |
| `deepseek-v4-flash-vision-exp` | ✅     | `thinking={"type":"disabled"}`           | 视觉需手动声明启用的细节以官方文档为准；按 OpenAI image_url 发送即可                                                                                                              |
| `deepseek-v4-flash`            | ❌     | `thinking={"type":"disabled"}`           | 纯文本                                                                                                                                                                            |
| `qwen3.8-flash`                | ✅     | `extra_body={"enable_thinking": False}`  | 百炼默认即关；显式传确保。另有`/no_think` 软开关可写进 prompt 兜底                                                                                                              |
| `gpt-5.6-luna`                 | ✅     | `reasoning_effort="none"`                | none=真·非思考（官方）                                                                                                                                                           |
| `doubao-seed-2-0-lite`         | ✅     | `thinking={"type":"disabled"}`           | 方舟统一参数                                                                                                                                                                      |
| `kimi-k3`                      | ✅     | 不传（恒开，无法关）                       | 不传 temperature                                                                                                                                                                  |
| `minimax-m3`                   | ✅     | `thinking={"type":"disabled"}`           |                                                                                                                                                                                   |
| `gemini-3`（flash 系）         | ✅     | `extra_body={"thinking_level":"low"}`    | 未见完全关闭档；low 为最低                                                                                                                                                        |
| `DEFAULT_PROFILE`              | 探测   | 不传                                       | 新模型接入零配置可用                                                                                                                                                              |

> 上表参数以调研文档为准，但**所有参数都被视为"可能错"**：§3.2 的降级链是正确性的最终保证。Profile 表是集中配置，后续新增/修正模型只改这一处。

### 3.2 发送与降级链（对每次调用）

```
1. 按 Profile 组装请求（思考参数 + 图片[若 vision 且 send_image] + tools[若有]）
2. 发送。成功 → 完。网络错误/5xx → 退避 1s 重试 1 次（只有首次请求享有，降级重试不再重试）。
3. 捕获 400/422。**降级动作不以错误措辞为闸门**——国产端点回中文且不点名参数（实测智谱
   400「该模型始终思考，不支持关闭思考；请使用 low、high 或 max」），靠措辞判读会把
   「本可自动退化」的轮次报成整轮失败；判读结果只写进日志当线索。
   a. 命中「图像不支持」特征且本次确实带了图 → 去掉图片重试（记 session 级标记：该模型
      后续回合直接不发图，避免每轮白付一次失败）。特征中英两套：image_url/vision/
      multimodal/图片/图像/视觉/多模态 + 否定措辞；另认「content 类型被收窄到 'text'」
      这种整条消息不出现 image 字样的写法（实测纯文本档 400「messages.content.type
      参数非法，取值范围 ['text']」）。
   b. 本次请求发过思考参数（extra_body 非空）→ 剥离思考参数，重试（仅 1 次），无条件执行。
   c. 剥离后仍 400/422：
      - 与首次同码同文 → 刚剥的那个参数不是原因，上抛原始错误，不再裸请求浪费第三次。
      - 否则 → 剥离全部非标准参数（只留 model/messages/stream[/tools]），最后重试 1 次。
   d. 其他 400/422（本次没发过思考参数）→ 直接上抛真实错误（禁止旧式"一律当不支持工具"，修 A1）。
4. 401/403/429/配额 → 上抛，error 事件给前端明确提示（429 不重试；免费/限额池的退避策略待定，见 实施手记 未开工项）。
```

单次调用的最坏代价是 3 次真实请求（原始 + 剥参 + 裸请求）；「同码同文即停」是为了不让一句
与参数无关的 400 白烧掉第三次额度。

**这是全项目唯一的降级位置**。旧的 5 处复制回退矩阵、`_TRUNCATE_PREFIXES`、`strip_trailing_tool_call_text` 全部删除（A2/A4）。子 Agent 的 tools 失败降级：tools 不支持（特征同 a 的关键词探测，中英两套，但仅限子 Agent 路径）→ 该子 Agent 直接失败并回报「当前模型不支持工具调用，任务/检索功能不可用」，由聊天主 Agent 用人设话术告知玩家（魔搭平台即此形态，README 需说明，见 10）。

### 3.3 图片注入规则

- **只有 `purpose="chat"` 的调用允许带图**；子 Agent、摘要等一律不带（用户明确要求）。
- 图片 part 在后、文本 part 在前（沿用现状的缓存对齐注释结论）。
- 图片来源与处理见 `07-立绘与多模态.md`；形象描述文本（appearance）恒在 prompt 中，与是否发图无关。

## 4. meta 行协议 + 路由工具（meta.py / tools/route/）

### 4.1 格式

聊天主 Agent 的响应**第一行**必须是单行紧凑 JSON，其后要么是 NPC 台词正文，
要么是一个路由工具调用——两者只取其一：

```
{"emo":"微笑","fav":0}
诶，是你啊。今天有什么想聊的？
```

meta 行只有 `emo` / `fav` 两个字段。委派意图不在 meta 里，改由五个路由声明工具承载
（`services/tools/route/`，`category="route"`，工具本体零副作用）：

| 路由工具 | 参数 | 何时下发 |
|---|---|---|
| `route_task_draft` | `note`（必填，≤60 字，只写诉求要点、不写数量/金额） | 恒下发 |
| `route_search` | `query`（必填，≤40 字） | 仅无待确认草案时 |
| `route_task_confirm` | 无 | 仅有待确认草案时 |
| `route_task_cancel` | 无 | 仅有待确认草案时 |
| `route_task_update` | `note`（必填，≤60 字，玩家的新条件） | 仅有待确认草案时 |

字段约束：

- `emo`：必须 ∈ 该 NPC 的 emotions 列表；非法 → 回退 `普通`。
- `fav`：整数，clamp 到 [-5, 5]。
- meta 行里的未知键一律忽略（模型偶发多写 `"act"` 之类不会让整行作废）。
- 路由参数是坏 JSON 时**路由本身仍生效**，`note`/`query` 按空处理（回落到玩家原话），
  不让一次解析失败毁掉整轮委派。
- 工具表按「有无待确认草案」二选一（`ROUTE_TOOLS_NEW` / `ROUTE_TOOLS_DRAFT`），
  顺序固定 → system/tools 前缀逐轮稳定，利于前缀缓存与逐轮 diff。无草案时不给
  confirm/cancel/update，摆出来只会诱导模型空调用。
- 任务选型（task_type / reward_types / keywords）完全不在聊天侧：那是任务 Agent
  自己的 `prepare_task_context` 调用，见 `03` §5。

### 4.2 解析器（split_meta / split_meta_events）

```python
async def split_meta_events(stream: AsyncIterator[StreamEvent], npc_emotions: Sequence[str]
                            ) -> tuple[Meta, AsyncIterator[StreamEvent]]:
    """缓冲流的开头，直到凑齐第一行（\n）或超过 512 字符：
    - 首非空字符是 '{' 且首行 json.loads 成功且通过 schema 校验 → Meta + 剩余事件流
    - 否则 → 默认 Meta(emo=普通, fav=0)，已缓冲文本原样还给事件流
    容错：JSON 解析失败、缺字段、类型错误一律按『无 meta』处理，绝不让解析失败弄丢正文。"""
```

meta 解析成功的瞬间：发出 `meta` SSE 事件（情绪/好感）→ 继续转发正文与 `finish`
里的 `tool_calls`。这保证**情绪先于正文到达前端**（B5 开发者注的硬性要求）。

⚠ `client.py` 只在 `finish` 事件里聚合出 `tool_calls`，**不流式吐工具调用**，所以编排层
在正文开始流出时还不知道本轮会不会路由。取舍（用户裁定）：正文无条件直发以保住
token 流式体验，「路由轮不说话」由提示词与工具描述约束，模型违规说话时只在日志留痕
（`路由轮（X）仍输出了台词`）而不吞掉玩家已经看到的内容。**不为「平台不支持流式
tools」做任何兼容分支**——现网没有这种平台，兼容只会白白拖慢流程。

### 4.3 prompt 说明（生成函数 `meta_prompt_block(npc_emotions, *, agent_enabled)`）

写进聊天 system 层2，要点：

1. 第一行只能是一个 JSON、一行、不解释、不用代码块包裹。
2. emo 从给定列表选；fav 范围 -5~5，无变化写 0。
3. 第一行之后要么台词、要么一个路由工具调用，两者只取其一。
4. `agent_enabled=False`（玩家在设置页关掉 Agent 能力）时，第 3 点里「路由工具调用」
   那半句整块不注入——既不下发路由工具、也没有路由说明块，提了就是悬空指代。
   emo/fav 协议不受影响。编排层另有兜底：模型仍硬发 `route_*` 时忽略并按纯聊天处理（03 §2）。

路由判据本体**不在 meta 协议块里**，而在各路由工具的 `description`（工具表就是聊天主
Agent 的路由提示词）与 `build_routing_guide`（跨工具公共规则：路由轮不说话、一轮最多
路由一件事、草案不等于已发布）中。三处分工不重叠，避免同一规则写两遍后自相矛盾。

## 5. 路由的执行语义（orchestrator 侧，本章只定义契约）

| 路由 | 执行方式 | 失败处理 |
| --- | --- | --- |
| `route_task_draft` | 立即启动 TaskRunner（draft 模式，`note` 作为方向注入；无 note 时回落玩家原话） | 见 03 §4 汇合规则 |
| `route_task_update` | 立即启动 TaskRunner（update 模式，携带 pending draft）；无草案可改时降级为 draft 模式重拟 | 同上 |
| `route_search` | 立即启动 SearchRunner（`query` 缺省时回落玩家原话） | 同上 |
| `route_task_confirm` | **同步执行** confirm_agent_task（校验+原子写，<300ms）→ tool_status(success) → 一次收场调用让 NPC 交代；失败 → tool_status(failed) + `system_notice` 给真实原因 + 携带原因做 1 次补救调用让 NPC 解释 | 见 05 §6 |
| `route_task_cancel` | 同步执行 cancel → tool_status(success) → 收场调用；无草案可撤不算失败，让 NPC 顺着话头自然回应 | 极少失败 |

confirm/cancel 不经过子 Agent——它们是纯后端操作，编排层直接执行（用户决策："模型聊天+快速调用对应工具就行了"）。
**路由轮不输出台词**，所以这两条路径上 NPC 的那句话来自事后的收场调用（模型空回复时
用静态兜底话术，玩家侧不留空白）；事件顺序固定为
`meta → tool_status(running) → tool_status(success) → content → system_notice → done`
（失败时 `system_notice` 排在补救 `content` 之前，让真实原因先落地）。

## 6. 测试要求（P1 验收）

1. meta 解析器：正常 meta / 无 meta / 半截 JSON / 非法 emo / fav 越界 / 多行 JSON 冒充 / 未知多余键（如残留的 `"act"`）不影响整行 / 首行 JSON 但既无 emo 也无 fav → 当正文 / meta 之后无正文（路由轮的形态）——`tests/llm/test_meta.py`。
2. meta 协议说明块：只讲 emo/fav 两个字段（不得出现 act / task_* / search / prepare_task_context / route_ 等词汇）、写清「台词或路由工具调用二选一」、`agent_enabled=False` 时不提路由、同 NPC 同开关状态跨轮逐字一致。
3. Profile 匹配：表中每个模型名命中正确 Profile；未知模型落 DEFAULT。
4. 降级链（mock HTTP）：思考参数 400→剥离重试成功；**措辞不可判读（含中文错误体）的 400 且发过思考参数→同样剥一次**；剥后同码同文→上抛原始错误且**不发起第三次**；剥后异文→裸请求最后重试成功；图像 400→去图重试成功且后续不再带图；**没发过思考参数的 400→一次调用后上抛真实错误**；401→不重试不上抛错码混淆。错误分类另测中文例句（智谱实测措辞：关思考被拒、`content` 类型收窄到 `'text'`、不支持工具调用）。
5. reasoning delta 不混入 content。
6. 流式 usage 正确回收。
