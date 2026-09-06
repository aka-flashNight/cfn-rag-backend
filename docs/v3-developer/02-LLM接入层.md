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
    tools: list[dict] | None = None        # 聊天 Agent 按业务注册（当前仅 prepare_task_context）；
                                           # 子 Agent 全量工具循环。流式+tools 无兼容坑（实测），
    purpose: Literal["chat", "subagent", "summary"] = "chat"
    send_image: bool = True                # purpose != "chat" 时强制 False（图片只进聊天轮）
    max_tokens: int | None = None
    # 不支持流式 tools 的平台：chat 轮剥 tools 重试一次（turn.py 唯一降级点）
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
| `glm-5.3-flash`                | ✅     | `thinking={"type":"enabled"}` + 低强度档 | **不能关**，传 disabled 会报错（官方明确）。强度参数名官方文档待复核（调研标注），先按 `reasoning_effort="low"` 发送，靠 §3.2 降级链兜底：报参数错→只发 enabled→裸请求 |
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
2. 发送。成功 → 完。
3. 捕获 400/422：
   a. 错误体命中「未知/非法参数」特征（param 字段点名我们发的思考参数，或消息含
      unknown/unsupported/invalid + 参数名）→ 剥离思考参数，重试（仅 1 次）。
   b. 剥离后仍 400/422 → 剥离全部非标准参数（只留 model/messages/stream[/tools]），最后重试 1 次。
   c. 错误体命中「图像不支持」特征（先查结构化字段，最后才关键词嗅探：
      image_url/vision/multimodal…，S8 记录在案的风险接受）→ 去掉图片重试（记
      session 级标记：该模型后续回合直接不发图，避免每轮白付一次失败）。
   d. 其他 400/422 → 直接上抛真实错误（禁止旧式"一律当不支持工具"，修 A1）。
4. 401/403/429/配额 → 上抛，error 事件给前端明确提示。
```

**这是全项目唯一的降级位置**。旧的 5 处复制回退矩阵、`_TRUNCATE_PREFIXES`、`strip_trailing_tool_call_text` 全部删除（A2/A4）。子 Agent 的 tools 失败降级：tools 不支持（特征同 c 的关键词探测，但仅限子 Agent 路径）→ 该子 Agent 直接失败并回报「当前模型不支持工具调用，任务/检索功能不可用」，由聊天主 Agent 用人设话术告知玩家（魔搭平台即此形态，README 需说明，见 10）。

### 3.3 图片注入规则

- **只有 `purpose="chat"` 的调用允许带图**；子 Agent、摘要等一律不带（用户明确要求）。
- 图片 part 在后、文本 part 在前（沿用现状的缓存对齐注释结论）。
- 图片来源与处理见 `07-立绘与多模态.md`；形象描述文本（appearance）恒在 prompt 中，与是否发图无关。

## 4. meta 行协议（meta.py）

### 4.1 格式

聊天主 Agent 的响应**第一行**必须是单行紧凑 JSON，第二行起为 NPC 台词正文：

```
{"emo":"微笑","fav":1,"act":null}
诶，是你啊。今天有什么想聊的？
```

```jsonc
// act 的全部合法形态（kind 枚举）
null                                                        // 纯聊天
{"kind":"task_draft"}   // 委派发任务：prepare 参数由 prepare_task_context 工具调用承载（台词先行、后调工具）
{"kind":"task_update","note":"玩家嫌奖励少，希望金币加两成"}
{"kind":"task_confirm"}                                     // 玩家确认接受当前草案
{"kind":"task_cancel"}                                      // 玩家拒绝/放弃当前草案
{"kind":"search","query":"安迪·洛的过去"}                     // 需要深入查资料
```

字段约束：

- `emo`：必须 ∈ 该 NPC 的 emotions 列表；非法 → 回退 `普通`。
- `fav`：整数，clamp 到 [-5, 5]。
- `task_update.note`：玩家新条件（≤60 字），任务 Agent 据此修改草案。
- `search.query`：≤40 字。
- task_draft 的候选池参数（task_type/reward_types/keywords）不在 meta 行内——由聊天主 Agent 在流式响应中
  调用 `prepare_task_context`（唯一注册工具）承载；后端执行后直通任务 Agent。prepare 失败/缺失时
  由任务 Agent 以 prepare_then_draft 模式自行重试（继承对话思路，不回炉交流 Agent）。
- meta_prompt_block 按「无草案 / 有草案」双分支互斥注入（无草案讲发起新任务，有草案讲
  confirm/cancel/update/大改重拟路由）。

### 4.2 解析器（parse_meta_stream）

```python
async def split_meta(stream: AsyncIterator[StreamEvent]
                     ) -> tuple[Meta, AsyncIterator[str]]:
    """缓冲流的开头，直到凑齐第一行（\n）或超过 512 字符：
    - 首非空字符是 '{' 且首行 json.loads 成功且通过 schema 校验 → Meta + 正文流
    - 否则 → 默认 Meta(emo=普通, fav=0, act=null)，已缓冲文本原样还给正文流
    容错：JSON 解析失败、缺字段、类型错误一律按『无 meta』处理，绝不让解析失败弄丢正文。"""
```

meta 解析成功的瞬间：发出 `meta` SSE 事件（情绪/好感）→ 处理 act（§5）→ 继续转发明文正文。这保证**情绪先于正文到达前端**（B5 开发者注的硬性要求）。

### 4.3 prompt 说明（生成函数 `meta_prompt_block(npc_emotions)`）

写进聊天尾部指令，要点：

1. 第一行只能是一个 JSON、一行、不解释、不用代码块包裹。
2. emo 从给定列表选；fav 范围 -5~5，无变化写 0。
3. act 决策规则（精简版）：
   - 玩家明确要求委托/工作/任务 → `task_draft`，direction 写清方向。**拟定了方向就必须委派，不要自己编任务细节**。
   - 存在待确认草案时：玩家接受 → `task_confirm`；拒绝 → `task_cancel`；谈条件 → `task_update`；岔开话题 → 保持 null（草案会保留若干回合）。
   - 需要查证设定/物品/关卡信息才能回答 → `search`。
   - **草案只是拟定，不是已发布**。确认前禁止说"任务已发布"。
4. 委派 task_draft/task_update/search 时，第一段的正文只写 1~2 句过渡话（如"我想想给你安排点什么……"），不要写任何具体任务内容/数字。

## 5. act 的执行语义（orchestrator 侧，本章只定义契约）

| act              | 执行方式                                                                                                                                                                  | 失败处理           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `task_draft`   | 立即启动 TaskRunner 后台任务（direction 作为严格指令注入）                                                                                                                | 见 03 §4 汇合规则 |
| `task_update`  | 立即启动 TaskRunner（update 模式，携带 pending draft）                                                                                                                    | 同上               |
| `search`       | 立即启动 SearchRunner 后台任务                                                                                                                                            | 同上               |
| `task_confirm` | **同步执行** confirm_agent_task（校验+原子写，<300ms）成功→放行正文 + tool_status(success)；失败→**丢弃已生成正文**，携带错误原因做 1 次补救调用让 NPC 解释 | 见 05 §6          |
| `task_cancel`  | 同步执行 cancel，同上                                                                                                                                                     | 极少失败           |

confirm/cancel 不经过子 Agent——它们是纯后端操作，聊天主 Agent 直接触发（用户决策："模型聊天+快速调用对应工具就行了"）。

## 6. 测试要求（P1 验收）

1. meta 解析器：正常 meta / 无 meta / 半截 JSON / 非法 emo / fav 越界 / 多行 JSON 冒充 共 6 类用例。
2. Profile 匹配：表中每个模型名命中正确 Profile；未知模型落 DEFAULT。
3. 降级链（mock HTTP）：思考参数 400→剥离重试成功；图像 400→去图重试成功且后续不再带图；两次都 400→上抛原始错误；401→不重试不上抛错码混淆。
4. reasoning delta 不混入 content。
5. 流式 usage 正确回收。
