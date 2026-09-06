# CFN-RAG Backend（cfn-rag-backend）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CFN-RAG 后端是一个面向 [Crazy Flash Night (CFN)](https://github.com/FlashNightModReborn/CrazyFlashNight) 的 **NPC 角色扮演对话 + 游戏数据 RAG + 任务发布/协商 Agent** 服务。它读取 CFN 的 `resources` 资源数据，通过检索增强生成（RAG）让 NPC 理解游戏世界设定、角色背景与任务信息；玩家在对话中即可与 NPC 协商委托任务，确认后由后端校验并原子写入游戏文件（`agent_tasks.json` / `agent_text.json`），游戏内可直接接取。

前端终端界面为独立仓库：[cfn-terminal-web](https://github.com/aka-flashNight/cfn-terminal-web)（v3 后端需配套 v3 前端，见版本说明）。

## 预览

![alt text](image.png)

## 功能特点

- **一次调用出情绪 + 正文**：NPC 每回合先返回情绪/好感度（前端即时切立绘），台词全程流式输出，告别「转圈等待无反馈」
- **后台并行编排**：发任务、查资料在后台子 Agent 并行执行，NPC 前台先说过渡话，不打断对话节奏；玩家视角 NPC 始终是「一个人」
- **混合检索（Hybrid RAG）**：Dense 向量 + BM25 + RRF 融合，按对话/任务/世界观/情报/实体分池召回，覆盖各类话题
- **轻量向量引擎**：bge-small-zh-v1.5（512 维）ONNX int8 量化运行时，单条查询编码约 1ms，索引全量构建 ≤25 秒，仅在语料指纹变化时自动重建
- **任务全流程 HITL**：对话中拟定任务草案 → 玩家接/不接/讨价还价 → 确认发布；后端校验管线（物品/关卡/进度/奖励预算/等级匹配）先自动微调、仅在必要时打回，几乎不再「卡死无草案」
- **NPC 立绘自动跟随游戏资产**：直接读取游戏项目的对话立绘 manifest，游戏更新立绘后立即可用，无需再手动导出/解压立绘包
- **多模态对话**：视觉模型可携带 NPC 立绘增强角色扮演；非视觉模型自动无图运行，互不影响
- **会话管理**：多会话、历史分页、滚动摘要；好感度/关系等级跨会话持久化

## 前置要求

CFN-RAG 后端需要配合 **Crazy Flash Night 游戏资源** 使用。请将游戏项目的 `resources` 文件夹与本项目放在同一目录层级（或通过环境变量 `CFN_RESOURCES_DIR` 指定路径）：

```
父目录/
├── resources/              # Crazy Flash Night 游戏资源文件夹
│   ├── data/
│   └── ...
├── CrazyFlashNight/        # 游戏项目根（可选，对话立绘 manifest 所在）
└── cfn-rag-backend/        # 本项目
    ├── launcher.py
    └── ...
```

Crazy Flash Night 游戏项目地址：`https://github.com/FlashNightModReborn/CrazyFlashNight`

## 架构概览

```
玩家消息 (HTTP /api/game/ask, SSE)
   │
   ▼
TurnOrchestrator（每回合一个实例，纯 asyncio）
   ├─ 1. 上下文装配（并行）：NPC 状态 / 会话记忆 / Tier-1 混合检索 / 立绘（仅视觉模型）
   ├─ 2. 聊天主 Agent（流式 LLM 调用 #1）
   │     首行 = meta JSON（情绪/好感/委派指令），其余 = NPC 台词实时流给前端
   │     ├─ meta 解析 → 立即发 meta 事件（前端先切立绘）
   │     ├─ 委派 = 发任务 / 查资料 → 启动后台子 Agent（asyncio.Task）
   │     ├─ 委派 = 确认 / 取消任务 → 后端同步执行（<300ms）
   │     └─ 无委派 → 正文流完即回合结束
   ├─ 3. 等待/汇合（仅当有后台子 Agent）：完成 → 汇合详说；未完成 → 过渡语/保活等待
   └─ 4. 后处理：写记忆、更新好感、usage 落库、done 事件

后台子 Agent（与 2/3 并行运行）
   ├─ TaskRunner：工具循环（prepare_task_context → draft/update → 校验反馈 + 自动微调 + 兜底）
   └─ SearchRunner：search_knowledge / search_items / search_stages → 结论摘要
```

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI + Uvicorn | 异步 HTTP 服务；SSE（Server-Sent Events）流式回复 |
| **对话编排** | 自研 TurnOrchestrator（asyncio） | 一次流式调用 + fire-and-steer 后台子 Agent + 检查点汇合；任务确认等同步动作直接执行 |
| **LLM 调用** | OpenAI 兼容 REST（`openai` SDK） | `base_url` + `model` 可指向 DeepSeek / 豆包 / GLM / Qwen / Gemini 等；meta 行协议与思考参数降级链与模型能力解耦 |
| **RAG 检索** | 自研混合检索 | Dense + BM25（`rank-bm25`）+ RRF 融合；对话/任务/世界观/情报/实体分池召回，阈值与名额集中配置 |
| **嵌入模型** | BAAI/**bge-small-zh-v1.5**（512 维） | ONNX int8 量化运行时（`onnxruntime` + `tokenizers`，模型约 55MB）；单条查询编码约 1ms，全库构建 ≤25s |
| **存储** | SQLite（WAL）+ 二进制向量库 | 会话记忆/滚动摘要/任务草案/NPC 状态单库；向量索引与 BM25 索引落盘（约 7.5MB），指纹不变直接加载 |
| **Prompt 缓存** | 分层 prompt 设计 | 世界观/扮演约束/共享上下文/尾部指令分层对齐前缀缓存（实测命中率 85%+） |
| **立绘** | 游戏项目对话立绘 manifest 查表 | 情绪回退链 + 角色名归一化；无 manifest 时立绘自动降级，不影响聊天 |
| **打包发布** | PyInstaller | 单文件 exe（实测约 91MB）+ 完整独立版 zip；`launcher.py` 内置前端静态服务与 `/api` 反向代理（SSE 流式转发） |
| **评估** | 自研检索层评估 | recall@k / precision@k / MRR / nDCG，dense / bm25 / hybrid_rrf 三轨对比（不调 LLM，零费用回归） |

### Prompt 缓存命中优化

主流 LLM API 服务会对「与历史请求相同前缀」的输入 token 计为**缓存命中（Cached tokens）**，与未命中部分区分计价。本项目采用分层 prompt 设计（静态 system → 共享 user core → 尾部指令）对齐前缀缓存，工具决策轮命中率多在 95%+，最终生成轮约 85%~95%。

> 完整数据文件格式与任务系统规则见仓库根目录 `data_files_overview.md`。

## 任务生成系统（草案 → 协商 → 写入）

### 硬约束：可写文件

**仅允许**后端修改以下两个文件（原子写入，避免半截 JSON）：

- `resources/data/task/agent_tasks.json`
- `resources/data/task/text/agent_text.json`

其余任务/物品/关卡等数据**只读**，供检索与校验。

### 流程

1. **准备上下文**：`prepare_task_context(task_type, reward_types)` —— 后端按任务类型一次性返回筛选后的关卡/物品/奖励候选与规则说明
2. **生成草案**：`draft_agent_task(...)` —— LLM 输出结构化草案，**聚合校验**（物品/关卡/进度/奖励预算等一次报齐）；数值类问题**先自动微调**（对齐奖励阶梯），选择类问题才带反馈打回
3. **协商**：玩家可接受 / 拒绝 / 讨价还价（最多 2 次）/ 局部修改（`update_task_draft`）；岔开话题时草案保留 3 回合后自动取消
4. **确认发布**：`confirm_agent_task(...)` —— 合并任务说明与对话后二次校验，通过后分配任务 ID 并**原子写入**上述两个文件，游戏内即可接取

`task_type` 枚举（12 种）：`问候`、`传话`、`通关`、`清理`、`挑战`、`切磋`、`资源收集`、`装备缴纳`、`特殊物品获取`、`物品持有`、`通关并收集`、`通关并持有`。

## 快速开始

### 方式一：使用预编译的可执行文件（推荐有游戏项目的用户）

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面下载 `CFN-RAG.exe`
2. 将 `CFN-RAG.exe` 放在与 `resources` 文件夹同一目录
3. 双击运行 `CFN-RAG.exe`（首次运行如弹出 Windows 防火墙授权，请点「允许」）
4. 浏览器自动打开界面，按引导完成模型配置

**注意**：必须配合 GitHub 上最新版本的 `resources` 游戏资源文件夹。

### 方式二：使用完整独立版（推荐无游戏项目的用户）

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面下载 `CFN-RAG-Full.zip`
2. 解压到任意位置，双击运行 `CFN-RAG.exe`
3. 浏览器自动打开界面，按引导完成模型配置

**优点**：无需额外下载游戏资源，独立运行
**注意**：内置资源可能不是最新版本

### 方式三：从源码运行（推荐开发者）

#### 环境要求

- Python 3.10+
- 足够的磁盘空间（约 1GB 用于依赖和模型）

#### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/aka-flashNight/cfn-rag-backend.git
cd cfn-rag-backend
```

2. 创建虚拟环境并安装运行时依赖

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

3. 准备嵌入模型（一次性，二选一）

运行时只依赖 `models/bge-small-zh-v1.5-onnx-int8/`（ONNX int8 权重不入 git）：

- **直接获取**：从 Release 附件或已有部署复制该目录；
- **自行导出**（开发机）：安装开发依赖、下载原始模型并导出 int8：

```bash
pip install -r requirements-dev.txt
python scripts/download_model.py --modelscope   # 或 --mirror / --proxy http://127.0.0.1:10809
python scripts/export_onnx_int8.py              # 导出并校验 int8（余弦 ≥0.99 自动把关）
```

4. 配置 API Key

复制 `.env.example` 为 `.env` 并填入 API Key（获取方式见下方「免费体验 / 低价 API」）：

```bash
cp .env.example .env
```

5. 启动服务

```bash
python launcher.py
```

浏览器将自动打开界面（`launcher.py` 同时启动后端与前端静态服务，并做 `/api` 反向代理与 SSE 流式转发）。

## 配置说明

### 免费体验 / 低价 API（2026-09-06 联网核实）

#### 1. 火山引擎（火山方舟）——首选体验平台

- 每个模型赠送 **50 万 token** 免费推理额度（首次开通语言模型即享，约 30 天内有效）
- 候选模型齐全（`doubao-seed-2-0-lite` 等），**支持工具调用，可体验完整 Agent 功能（任务发布 / 讨价还价）**
- 国内访问稳定，OpenAI 兼容端点
- **注意**：额度有限而非无限免费，用完后转为低价按量计费（如 `doubao-seed-2-0-lite` 约 0.6 元/百万输入 token）

**获取步骤**：访问[火山方舟控制台](https://console.volcengine.com/ark) → 实名注册并开通语言模型 → 「API Key 管理」创建 Key。模型需在「在线推理」中开通接入点（或直接使用模型 ID）。

```env
LLM_API_KEY=your_ark_key_here
LLM_API_BASE=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL_NAME=doubao-seed-2-0-lite
```

#### 2. 各家官方 API——长期使用的正规低价渠道

按 token 计费但单价极低，适合长期使用：

| 平台 | 推荐模型 | 价格（约，元/百万 token，2026-09-06 核实） | 备注 |
|------|---------|------|------|
| [DeepSeek 开放平台](https://platform.deepseek.com) | `deepseek-v4-flash`（多模态版 `deepseek-v4-flash-vision-exp`） | 输入 1 / 输出 2（缓存命中 0.02） | 高峰时段（10:00-24:00 之外的部分时段除外）价格上浮，以[官方价目](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)为准 |
| [智谱开放平台](https://open.bigmodel.cn) | `glm-5.3-flash` | 输入 0.8 / 输出 2.8 | 思考恒开，建议低强度档位；另有免费档 `glm-4.5-flash` |
| [阿里云百炼](https://bailian.console.aliyun.com) | `qwen3.8-flash` | 输入 0.8 / 输出 2.7 | 新用户有免费额度（约 90 天有效），`enable_thinking` 可关 |
| [OpenAI](https://platform.openai.com) | `gpt-5.6-luna` | $0.20 输入 / $1.20 输出 | 2026-07 大幅降价；reasoning 选 none 档；国内需代理 |

#### 3. Google Gemini——免费层额度已缩水，需代理

[Google AI Studio](https://aistudio.google.com/app/apikey) 创建 Key 即可用免费层，但 **2025 年底起免费层每日请求数已大幅下调**（社区实测 Flash 系从 250 次/天降至约 20 次/天，且限额按模型动态变化），仅够轻度体验；使用需配置代理（见下方代理配置）。

#### 4. ModelScope 魔搭社区——零成本试聊，功能受限

每日 2000 次免费调用（单模型另有上限，部分模型每日仅 20~500 次）。

> **重要限制**：魔搭推理 API 的**工具调用支持因模型而异**（仅部分模型支持），因此**任务发布 / 讨价还价等 Agent 功能不保证可用**（后端检测到工具不可用时会以 NPC 话术说明）。仅建议用于零成本快速试聊。官方亦注明免费 API 不适用于高并发在线任务。

**获取步骤**：

1. 注册 [ModelScope](https://www.modelscope.cn/) 并[绑定阿里云实名账户](https://www.modelscope.cn/my/settings/account)
2. 在[访问令牌](https://modelscope.cn/my/access/token)页面「创建新的访问令牌」作为 api_key
3. 在[模型库](https://www.modelscope.cn/models)筛选「支持体验 → 推理 API-Inference」，进入模型详情页查看 `model` 名称（模型清单以魔搭实时筛选为准）

```env
LLM_API_KEY=your_modelscope_token_here
LLM_API_BASE=https://api-inference.modelscope.cn/v1
LLM_MODEL_NAME=Qwen/Qwen3.5-397B-A17B
```

#### 5. Kimi（Moonshot 开放平台）——小额赠送，付费备选

注册并完成个人实名认证后**赠送 15 元体验额度**（另含千万级 token 赠送额度，以[平台账户总览](https://platform.kimi.com)显示为准），非无限免费；用完按量计费，可作为付费备选。

#### 6. 其他渠道参考

- **硅基流动**（siliconflow.cn）：新用户赠送活动持续中（2026-12-31 前），含低价小模型
- **腾讯云混元 / 百度千帆**：各有每模型百万 token 级的限期赠送
- 社区持续维护的免费额度汇总（如 GitHub 项目 `FreeLLM-API-KeyHub`）可作薅羊毛索引；注意免费渠道随时可能调整，以各平台官方页面为准

本项目默认配置示例（`.env`）：

```env
LLM_API_KEY=your_api_key_here
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL_NAME=deepseek-v4-flash-vision-exp
```

### 配置方式

- **前端界面（推荐）**：启动后在设置页填写 api_key / api_base / model / 代理，按会话即时生效
- **`.env` 文件**：同目录创建 `.env`（参考 `.env.example`），作为默认配置

### 代理配置

**如果使用国外模型（如 Gemini、OpenAI）或开启了全局代理，需要配置代理。** 代理仅作用于 LLM 请求（客户端级，不修改进程环境变量）：

- 前端设置页填写代理地址，按会话生效；
- 或 `.env` 中设置 `LLM_PROXY_URL=http://127.0.0.1:7890`。

常见端口：`http://127.0.0.1:7890`（Clash）、`http://127.0.0.1:10809`（v2rayN）。

## 立绘（自动跟随游戏资产）

v3 起**不再需要立绘包 / SWF 导出 / 手动放置图片**。NPC 立绘由游戏项目的**对话立绘 manifest** 体系自动提供：

- 后端按 `CFN_GAME_PROJECT_DIR`（默认取 `resources` 上级的游戏项目根）读取对话立绘 manifest，按「角色名 + 情绪」查表定位立绘文件；
- 角色名支持别名/空白归一化；情绪缺失时按「请求情绪 → 普通 → defaultExpression → 首个表情」回退；查无角色自动无图降级，**绝不影响聊天主功能**；
- 游戏更新立绘后无需任何操作，下次请求即读到最新资产；
- NPC 可选的 `appearance` 形象描述字段（`npc_state_db.json` 的 `appearance`）会拼入对话 prompt，无立绘/无视觉模型时也能维持形象一致；
- 头像仍取自 `resources/flashswf/portraits/profiles/`（游戏自带，未配置立绘时的兜底）。

立绘相关接口：`GET /api/assets/illustration/{npc_name}/{emotion}`（源图直出）、`GET /api/assets/avatar/{npc_name}`。前端展示裁剪由前端处理；视觉模型对话时后端会自动取立绘并压缩后随请求发送（长边 ≤480）。

## 评估（开发者）

检索层评估资产位于 `evals/`（不调 LLM，零费用，适合回归）：

```bash
python -m evals.runners.build_golden_set --tiny    # 5 条微型集 -> evals/datasets/tiny_golden.jsonl
python -m evals.runners.build_golden_set --full    # 130 条 -> evals/datasets/golden_v1.jsonl
python -m evals.runners.run_all --suite retriever --dataset evals/datasets/tiny_golden.jsonl --sample 0
```

- golden 集从当前语料分层采样（台词 50 / 世界观 30 / 任务 30 / 情报 20），排除「彩蛋」「成员」阵营样本；语料变化后需重建
- 指标：recall@k / precision@k / MRR / nDCG，dense / bm25 / hybrid_rrf 三轨对比；报告输出到 `evals/reports/`
- 当前基线：v3 索引上 full 集（130 条）三轨 recall@10 = 1.000（详见 `evals/reports/` 与 docs/v3-developer/验收报告.md）

详见 [evals/README.md](evals/README.md)。

## 项目结构

```
cfn-rag-backend/
├── api/                         # API 路由层
│   ├── game_api.py              # /api/game/* 会话与对话接口（SSE）
│   └── assets_api.py            # 头像 / 立绘接口
├── core/
│   ├── config.py                # 全局配置（仅本地形态）
│   ├── exceptions.py
│   └── startup.py               # 启动初始化（并行预载数据/索引/模型）
├── schemas/                     # Pydantic 请求/响应模型
├── services/
│   ├── llm/                     # LLM 接入层（client / profiles / meta 行协议 / errors 降级链）
│   ├── orchestrator/            # 回合编排器（turn / context / events / merge / prompts）
│   ├── subagents/               # 后台子 Agent（task_runner / search_runner）
│   ├── retrieval/               # 混合检索（embedder / store / hybrid / pools / loader）
│   ├── agent_tools/             # 任务纯函数（prepare/draft/校验/自动微调/兜底草案）
│   ├── tools/                   # 子 Agent 原子工具（query / task / system）
│   ├── skills/                  # Skills 渐进披露（list_skills / read_skill / read_skill_file）
│   ├── npc/                     # NPC 状态（内存单例 + 锁 + 防抖落盘）
│   ├── memory/                  # 会话记忆 + 摘要 + 任务草案存储（SQLite）
│   ├── portraits/               # 对话立绘 manifest 查表 + provider + 缓存
│   └── game_data/               # 游戏静态数据 Registry（items/stages/tasks/shops/...）
├── models/
│   └── bge-small-zh-v1.5-onnx-int8/   # 嵌入模型（ONNX int8，512 维，权重不入 git）
├── scripts/                     # build_exe.py / download_model.py / export_onnx_int8.py
├── tests/                       # pytest 测试套件
├── evals/                       # 检索层评估（golden 集 / 三轨指标 / 报告）
├── dist/                        # 前端构建产物（静态）
├── launcher.py                  # 启动器（后端 + 前端静态服务 + /api 反向代理）
├── main.py                      # FastAPI 入口
└── requirements.txt / requirements-dev.txt
```

## 打包可执行文件

如果你想自己打包可执行文件（需先 `pip install -r requirements-dev.txt` 安装 PyInstaller）：

```bash
python scripts/build_exe.py
```

打包完成后会在项目根目录生成 `CFN-RAG.exe`（单文件 onefile）。

## 常见问题

### Q: 启动时提示找不到 resources 文件夹？

A: 确保 `resources` 文件夹与项目在同一目录层级（参考【前置要求】目录结构），或在 `.env` 中用 `CFN_RESOURCES_DIR` 指定路径。

### Q: 立绘在哪里配置？需要下载立绘包吗？

A: 不需要。v3 立绘由游戏项目的对话立绘 manifest 自动提供，游戏更新资产后自动生效。若游戏项目目录不在 `resources` 上级，请设置 `CFN_GAME_PROJECT_DIR`。没有立绘时对话功能完全正常，仅视觉模型不带图。

### Q: 第一次启动后对话加载很慢吗？

A: 不会。向量索引首次全量构建约 **25 秒以内**（后台进行，不阻塞服务），之后启动直接加载落盘索引（约 1 秒）。只有当游戏语料变化（指纹不同）时才会自动重建。

### Q: 游戏数据更新了，如何让索引重新构建？

A: 自动处理：后端按语料指纹判断，数据变化后下次启动会在后台重建索引。无需手动删除。

### Q: 模型下载失败或很慢？

A: 模型下载（`scripts/download_model.py`）是开发者导出 ONNX 时的步骤，普通用户直接使用 Release 包即可。开发者可用国内镜像：

```bash
python scripts/download_model.py --modelscope
```

### Q: API 调用报错/无响应？

A: 检查以下几点：
1. 模型名称、API Base、API Key 是否正确配置
2. 如使用国外模型，是否配置了代理
3. 前端聊天界面如显示「模型/网络不可用」错误提示（error 事件），按 `message` 内容检查配置；`retryable` 为 true 可直接重试

### Q: 如何更换其他 LLM 模型？

A: 在前端设置中修改（优先级最高），或修改 `.env` 中的 `LLM_API_BASE` / `LLM_MODEL_NAME`。只要 API 兼容 OpenAI 格式即可。注意：**任务发布/讨价还价功能需要平台支持工具调用（function calling）**，不支持工具调用的平台仅可体验聊天。

### Q: 为什么要同步升级前端？

A: v3 后端精简了 SSE 事件契约（meta/content/tool_status/agent_status/system_notice/done/error），与 v2 前端不兼容（事件名与 data 字段有变化，见仓库 docs/v3-developer/09-前端适配说明.md）。请前后端配套升级。

## 发布版本说明

我们在 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面提供两种发布包：

### 1. CFN-RAG-Full.zip（完整独立版）

**面向人群**：首次下载、想独立体验、不想下载完整游戏项目的用户

| 特点 | 说明 |
|------|------|
| 文件大小 | 约 200MB（v3 瘦身后，含必要资源文件；以 Release 实包为准） |
| 使用方式 | 解压到任意位置，运行 `CFN-RAG.exe` |
| 依赖 | 无需外部 `resources` 文件夹，无需 Python 环境 |
| 缺点 | 无法随游戏更新获取最新数据，仅包含基础资源 |

### 2. CFN-RAG.exe（单文件版）

**面向人群**：有完整游戏项目的用户

| 特点 | 说明 |
|------|------|
| 文件大小 | 约 91MB（v3 由 357MB 瘦身而来，预算 ≤160MB） |
| 使用方式 | 放到与 `resources` 文件夹同一目录，双击运行 |
| 依赖 | 需要游戏项目 `resources` 文件夹（GitHub 最新版本），无需 Python 环境 |
| 注意 | 首次运行弹 Windows 防火墙授权请点「允许」 |

### 版本选择建议

| 你的情况 | 推荐版本 |
|---------|---------|
| 首次体验，没有游戏项目 | **CFN-RAG-Full.zip** |
| 有游戏项目，想体验完整功能 | **CFN-RAG.exe** |
| 开发者，需要修改代码 | **源码克隆** |

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 致谢

- [Crazy Flash Night](https://github.com/FlashNightModReborn/CrazyFlashNight) - 游戏项目
- [cfn-terminal-web](https://github.com/aka-flashNight/cfn-terminal-web) - 前端终端界面（Vue 3）
- [BAAI](https://github.com/FlagOpen/FlagEmbedding) - BGE 嵌入模型
- [onnxruntime](https://onnxruntime.ai/) / [rank-bm25](https://github.com/dorianbrown/rank_bm25) - 轻量检索运行时
- [FastAPI](https://fastapi.tiangolo.com/) / [Pydantic](https://docs.pydantic.dev/) - Web 与数据校验

## 联系方式

如有问题或建议，欢迎提交 [Issue](https://github.com/aka-flashNight/cfn-rag-backend/issues) 或 [Pull Request](https://github.com/aka-flashNight/cfn-rag-backend/pulls)。
