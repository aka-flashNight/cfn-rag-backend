# CFN-RAG Backend（cfn-rag-backend）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CFN-RAG 后端是面向 [Crazy Flash Night (CFN)](https://github.com/FlashNightModReborn/CrazyFlashNight) 的**本地 AI 对话终端**：负责 NPC 角色扮演对话、游戏数据检索增强（RAG）、任务发布/协商 Agent，以及游戏内嵌面板所需的存档绑定、长期记忆、对话插话与任务联动。

服务只监听 `127.0.0.1`，单用户本地运行；游戏数据只读，运行期写入严格限制在独立的增量文件与本地存储中。前端终端界面为独立仓库：[cfn-terminal-web](https://github.com/aka-flashNight/cfn-terminal-web)。V3.0.0 起前后端需配套升级。

## 预览

![alt text](image.png)

## 功能特点

### 对话与角色扮演

- **一次调用出情绪 + 正文**：NPC 每回合先返回情绪/好感度（前端即时切立绘、更新好感条），台词全程流式输出，不再「转圈等待无反馈」
- **后台并行编排**：发任务、查资料、改对话由后台子 Agent 并行执行；NPC 前台先说过渡话，办完再汇合详说，玩家视角 NPC 始终是「一个人」
- **多模态对话**：视觉模型可按当前情绪携带 NPC 立绘；非视觉模型自动无图运行，互不影响
- **会话管理**：多会话、历史分页、滚动摘要；好感度/关系等级跨会话持久化
- **长期记忆**：按「存档 + NPC」保存基础印象与关键记录，跨会话、跨对话窗口保留；与按会话的滚动摘要相互独立
- **对话修改**：玩家可通过对话让 NPC 改写自己的日常台词，或改写已发布委托的文本；修改写入独立覆盖层，基础 XML 永不改坏

### 游戏内嵌终端（V3）

- **游戏内直接打开**：游戏宿主以 `--embedded` 拉起后端，7077 单端口同时伺服 API 与前端页面，面板以同源 iframe 嵌入，无浏览器地址栏、无独立窗口
- **存档绑定**：宿主推送玩家、进度、场景、存档槽位等只读投影；后端不主动读取游戏存档文件，所有存档信息以宿主推送为准
- **按存档隔离**：任务、任务文本、对话覆盖层、会话、NPC 好感/关系、长期记忆均按存档槽位隔离；静态身份（阵营、头衔、形象描述等）保持全存档共享
- **自动启动与退出**：游戏设置可选择启动游戏时自动拉起终端；游戏退出时通过 `/api/integration/shutdown` 让终端优雅退出
- **插话输入**：NPC 台词播放过程中，玩家可以直接插话；后端流式返回新增、改写、删除、结束语等结果，支持选项直连与物品赠送/索取
- **任务联动**：任务完成事件由宿主推送给后端，挂有对话重写标记的委托会自动委派后台子 Agent 改写后续对话；「我接了」可走终态 `task_publish` 完成自动接取交接

### 检索与知识库

- **混合检索（Hybrid RAG）**：Dense 向量 + BM25 + RRF 融合，按对话、任务、世界观、情报、实体、loading 文案分池召回，覆盖游戏设定、角色台词、任务与物品/关卡等话题
- **轻量向量引擎**：bge-small-zh-v1.5（512 维）ONNX int8 量化运行时；单条查询编码约 1ms，混合检索单轮 ≤150ms 量级
- **语料指纹与自动重建**：按语料文件集合的路径、大小、修改时间计算指纹；语料不变直接加载落盘索引，语料变化后在后台自动重建，不阻塞服务
- **查询期覆盖**：玩家用终端改过的对话/任务文本在检索时按覆盖层替换，不进入向量索引本体；改对话不会触发全量重建
- **知识库重置**：设置页可手动重建向量库，适合大批量数据更新后使用

### 任务系统与游戏写入

- **任务全流程 HITL**：对话中拟定任务草案 → 玩家接受/拒绝/讨价还价/局部修改 → 明确确认后发布；草案拟定不等于发布
- **校验管线**：物品、关卡、进度、奖励预算、等级匹配等一次聚合报错；数值类问题先自动微调，选择类问题才带原因和修正方向打回，另有后端兜底草案，尽量避免「卡死无草案」
- **两种任务模式**：新草案走 `prepare → draft` 两步式；讨价还价/局部修改走 `update`，整体换方向时才重新拟稿
- **原子写入**：确认发布时分配任务 ID 并原子写入任务与任务文本文件；其余任务、物品、关卡数据只读
- **物品交互**：插话/对话中支持 NPC 赠送物品与向玩家索取物品，带有效池、价值计算、数量校验、领取状态机与跨组账本；请求入口与失败理由都会落日志
- **任务完成联动**：任务完成事件会触发发布 NPC 的后续对话修改；同一存档同一任务只受理一次

### 工程与部署

- **单文件 exe**：约 92MB（onefile）；内置前端、int8 嵌入模型与启动种子，可放在游戏目录旁边直接运行
- **独立体验 zip**：内置聊天所需的精简游戏数据、NPC 立绘与头像，解压后双击 exe 即可独立体验聊天，不需要游戏本体、Python、额外模型或索引
- **本地存储**：SQLite（WAL）单连接管理会话/摘要/草案/长期记忆；NPC 状态进程内单例 + 防抖原子落盘；日志写在 exe 同目录
- **模型兼容**：任何 OpenAI 兼容 API 均可接入；内置模型 Profile（思考参数、视觉能力）与参数报错自动降级链，未知模型零配置可用
- **检索评估**：仓库自带 golden 集与 recall@k / precision@k / MRR / nDCG 三轨评估，不调 LLM，零费用回归

## 运行形态与目录要求

后端支持三种运行方式，按你的情况选一种即可。

### 形态一：游戏内嵌（推荐已有游戏）

把版本化的 exe 放在**游戏安装目录中 `CrazyFlashNight` 文件夹的同一级**。常见位置：

```text
游戏安装目录/
├── CrazyFlashNight/          # 游戏项目根（含 data/、docs/story/、flashswf/、launcher/）
└── CFN-RAG-v3.0.0.exe        # 放在这里，不要改文件名
```

Steam 独立版中，游戏项目根通常是 `project\CrazyFlashNight`，因此 exe 放在 `project\` 下：

```text
.../CRAZYFLASHER7StandAloneStarter/
└── project/
    ├── CrazyFlashNight/
    └── CFN-RAG-v3.0.0.exe
```

启动游戏后，从游戏内终端入口打开，或在游戏设置中开启「启动游戏时自动启动终端」。

**不要重命名 exe**：游戏宿主按 `CFN-RAG-v*.exe` 规则自动发现终端程序，改名后可能无法被拉起。

后端会在 exe 同级的 `resources` 或 `CrazyFlashNight` 目录中自动定位游戏数据根（也可用环境变量 `CFN_RESOURCES_DIR` 显式指定）。如果 exe 放在其它位置，只要其同级存在 `CrazyFlashNight` 文件夹，也能自动找到。

### 形态二：独立体验 zip（没有游戏，只想聊天）

从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 下载 `CFN-RAG-Full-v3.0.0.zip`，解压到任意位置，双击 `CFN-RAG-v3.0.0.exe`。

```text
CFN-RAG-Full-v3.0.0/
├── CFN-RAG-v3.0.0.exe
└── CrazyFlashNight/          # 精简游戏数据 + 立绘 + 头像（随包提供）
```

zip 已包含聊天所需的数据、立绘、头像、嵌入模型与前端页面，**无任何外部文件依赖**；不包含游戏本体，也不包含预生成的向量索引（首次启动会在后台自动构建）。独立模式适合体验角色对话与任务草案；游戏内接取任务、存档联动等能力需要游戏本体。

### 形态三：源码运行（推荐开发者）

```text
父目录/
├── CrazyFlashNight/          # 游戏项目根（或改名为 resources）
└── cfn-rag-backend/          # 本项目
```

也可以直接设置 `CFN_RESOURCES_DIR` 指向游戏项目根，不要求与项目同级。

游戏项目地址：`https://github.com/FlashNightModReborn/CrazyFlashNight`。建议使用 **2026-10-26 及之后**的游戏数据；更早版本可能缺少 V3 使用的字段，部分功能会降级。

## 快速开始

### 方式一：exe + 已有游戏

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 下载 `CFN-RAG-v3.0.0.exe`
2. 按上面的目录说明放到 `CrazyFlashNight` 文件夹的同一级，保持文件名不变
3. 启动游戏，从终端入口打开面板；或在游戏设置中开启自动启动
4. 首次使用在终端设置页填写 API Key / API Base / 模型（打包版只从页面设置读取配置，不需要 `.env`）

> 只要游戏整包是 **2026-10-26 之后的每日包或公开整包**，或使用 **GitHub 最新游戏项目**，就只需要下载这个 exe，不需要再下载数据、立绘、头像或其它资源包。

### 方式二：Full zip + 独立体验

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 下载 `CFN-RAG-Full-v3.0.0.zip`
2. 解压到任意位置，双击 `CFN-RAG-v3.0.0.exe`
3. 浏览器自动打开终端页面，在设置页填写 API Key / API Base / 模型

首次启动会弹出 Windows 防火墙授权，请点「允许」（只监听本机回环地址）。启动器窗口保持打开时服务才会运行，关闭窗口即退出。

### 方式三：从源码运行

#### 环境要求

- Python 3.10+
- 约 1GB 磁盘空间（依赖与嵌入模型）

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

复制 `.env.example` 为 `.env` 并填入 API Key（获取方式见下方「模型推荐」）：

```bash
cp .env.example .env
```

5. 启动服务

```bash
python launcher.py
```

浏览器会自动打开界面；`launcher.py` 同时启动后端（7077）与前端静态服务（7080 起），并对 `/api` 与 SSE 做反向代理。开发时也可以只用 `python -m uvicorn main:app --port 7077` 启动后端，前端另行 `npm run dev`。

## 游戏内嵌集成说明

本节只描述后端支持的能力与契约；游戏侧的拉起、面板与投影实现属于游戏项目。

### 启动模式

| 模式 | 启动方式 | 端口与页面 |
|------|----------|------------|
| standalone | 用户双击 exe / `python launcher.py` | 后端 7077；前端静态服务 7080 起，浏览器访问 |
| embedded | 游戏宿主以 `--embedded` 拉起 | 7077 单端口同时伺服 API 与前端；面板 iframe 使用 `http://127.0.0.1:7077` |

embedded 模式无浏览器跳转、无启动提示音、无独立状态窗，只写 `cfn-rag.log`。同一时间只允许一个实例占用 7077；已有实例在服务时，新拉起的 embedded 进程会静默退出，由宿主复用既有实例。

### 存档绑定

宿主通过 `POST /api/integration/bind` 推送当前存档投影，后端只保留内存快照：

```json
{
  "slot_key": "存档槽位键",
  "character": { "name": "玩家名", "sex": "男", "faction": "阵营", "...": "..." },
  "progress": { "stage": 3, "...": "..." },
  "scene": { "area": "区域", "stage": "场景", "...": "..." },
  "player": { "items": ["物品名"], "item_counts": { "物品名": 2 } }
}
```

- 同槽位重复 bind＝投影刷新；缺省段按空处理；换槽位等价于重新绑定
- `GET /api/integration/status` 返回模式、绑定槽位、投影快照与推导出的玩家阶段
- `POST /api/integration/unbind` 回到无存档状态
- 后端**不主动读取游戏存档文件**，也不在后台轮询存档；所有存档信息以宿主推送为准

### 按存档隔离的落点

绑定后，任务与覆盖层写入 `data/rag/agent_saves/<slotKey>/`：

```text
data/rag/agent_saves/<slotKey>/
├── agent_tasks.json              # 本存档的 Agent 任务
├── agent_text.json               # 本存档的任务文本
├── dialogue_overrides.json       # 本存档的日常对话覆盖层
└── task_text_overrides.json      # 本存档的任务文本覆盖层
```

未绑定存档时退回共享落点 `data/task/agent_tasks.json` 与 `data/task/text/agent_text.json`。会话表记录 `slot_key`，列表/历史按当前存档过滤；NPC 状态按 `global` 静态身份 + `saves/<slotKey>` 存档数值存储；长期记忆按 `(槽位, NPC)` 作用域存储。槽位键只接受 `A-Za-z0-9_-`，拒绝路径分隔符。

### 插话与动作回执

- `POST /api/game/dialogue-interject`：流式插话，返回新增/改写/删除/结束语等结果；请求携带当前对话实例、已播台词、剩余台词与参考续写
- `POST /api/game/dialogue-interject/finalize`：宿主已播对账，幂等
- `POST /api/game/dialogue-interject/action-result`：宿主动作回执（选项直连、物品领取/拒绝），幂等
- 支持在台词播放中插入玩家发言；选项轮可直接执行任务接受/拒绝、物品赠送/赠送拒绝等动作
- 插话链路与终端聊天共用 NPC 资料、场景、同阵营、说话方式与最近一次 LLM 配置；长文本解析失败会降级，不阻塞已播台词

### 其他联动接口

- `POST /api/integration/task-completed`：任务完成事件。挂有对话重写标记的委托会触发后台对话修改子 Agent；同一存档同一任务只受理一次
- `GET/POST /api/game/llm-config`：与终端页面同步当前 LLM 连接配置（只保留进程内内存副本，不写配置文件）
- `POST /api/integration/shutdown`：仅 embedded 模式接受，用于游戏退出链优雅关闭终端

## 配置说明

### 配置方式

- **页面设置（打包版 exe 唯一方式）**：启动后在前端设置页填写 API Key / API Base / 模型 / 代理，按会话即时生效。打包后的 exe 只使用页面设置里的配置，不读取 `.env`。
- **`.env` 文件（仅源码开发与测试）**：源码运行时可在项目根创建 `.env`（参考 `.env.example`），作为开发/测试默认配置；不用于打包版 exe。
- **进程内最近配置**：终端与游戏内配置小窗共享最近一次有效配置；后端只在内存中保存，供插话等没有前端设置可带的链路复用。

### 模型推荐

> **信息时间：2026-10-26。** 各平台的试用额度、价格与模型名可能随时调整，请以官方控制台、价目页和模型列表为准；下面只作为选型参考。

#### 首推：智谱 GLM —— 聊天角色扮演

控制台：[https://open.bigmodel.cn/apikey/platform](https://open.bigmodel.cn/apikey/platform)  
API Base：`https://open.bigmodel.cn/api/paas/v4`

| 模型 | 特点 | 适合 |
|------|------|------|
| `glm-5.3-flash` | 有一个月 **1500 万 token** 的试用资源包，足以支持上千次对话或数百次任务发布；角色扮演能力不错、模型能力优秀、API 价格偏低；缺点是生成速度较慢 | **聊天（角色还原度较高）** |
| `glm-5.3-flashx` | `glm-5.3-flash` 的更快版本；试用资源包额度较少（**300 万 token**），API 价格也更高一些 | 对速度更敏感、额度够用的聊天 |

```env
LLM_API_KEY=your_zhipu_key_here
LLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL_NAME=glm-5.3-flash
```

#### 次推：DeepSeek 官方 API —— 发任务

平台：[https://platform.deepseek.com](https://platform.deepseek.com)  
API Base：`https://api.deepseek.com/v1`

- **优点**：回复速度极快，API 价格偏低
- **缺点**：角色扮演相对弱一些；没有赠送额度
- **适合**：**任务发布（速度快、效率高）**

| 模型 | 说明 |
|------|------|
| `deepseek-v4-flash` | 纯文本，最快、最省，适合以任务发布为主的使用 |
| `deepseek-v4-flash-vision-exp` | 多模态版本，可带 NPC 立绘，适合兼顾聊天与识图 |

```env
LLM_API_KEY=your_deepseek_key_here
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL_NAME=deepseek-v4-flash
```

#### 其他平台

| 平台 | 推荐模型 | 优点 | 缺点 | 适合 |
|------|----------|------|------|------|
| Google Gemini | `gemini-3.8-flash` | 速度快，角色扮演不错 | 国内需要代理 | 聊天 |
| Kimi（Moonshot） | `kimi-k2.8` | 角色扮演能力强 | 价格较贵 | 角色扮演 |
| ModelScope 魔搭 | 以模型库实时显示为准（如 `Qwen/Qwen3.5-397B-A17B`） | 免费试用大量模型 | 可能无法调用工具，只能聊天；速度偏慢 | 零成本试聊 |

**Gemini**：API Base `https://generativelanguage.googleapis.com/v1beta/openai`，需在设置页或 `.env` 配置代理。

**Kimi**：API Base `https://api.moonshot.cn/v1`。

**ModelScope 魔搭**：

1. 注册 [ModelScope](https://www.modelscope.cn/) 并[绑定阿里云实名账户](https://www.modelscope.cn/my/settings/account)
2. 在[访问令牌](https://modelscope.cn/my/access/token)页面创建访问令牌，作为 API Key
3. 在[模型库](https://www.modelscope.cn/models)筛选支持「推理 API-Inference」的模型，进入模型详情页确认 `model` 名称

```env
LLM_API_KEY=your_modelscope_token_here
LLM_API_BASE=https://api-inference.modelscope.cn/v1
LLM_MODEL_NAME=Qwen/Qwen3.5-397B-A17B
```

> **重要限制**：魔搭的**工具调用支持因模型而异**，因此任务发布/讨价还价/物品交互等 Agent 功能不保证可用；仅建议用于零成本试聊。

任何兼容 OpenAI 接口的平台都可以接入；**需要工具调用（function calling）才能使用任务发布、讨价还价、物品交互等 Agent 功能**，不支持工具调用的平台只能聊天。

### 代理配置

如果使用国外模型（如 Gemini）或开启了全局代理，需要配置代理。代理仅作用于 LLM 请求（客户端级，不修改进程环境变量）：

- 打包版 exe 在前端设置页填写代理地址，按会话生效；
- 源码开发/测试还可在 `.env` 中设置 `LLM_PROXY_URL=http://127.0.0.1:7890`。

常见端口：`http://127.0.0.1:7890`（Clash）、`http://127.0.0.1:10809`（v2rayN）。

### 模型兼容与降级

- 按模型名匹配内置 Profile，决定是否携带思考参数、是否发送图片；匹配不到时使用默认 Profile（不传任何思考参数、视觉能力探测型）
- 平台不识别某个思考参数时会自动剥掉整个扩展参数重试一次；图片发送失败会在本进程内标记该模型不支持视觉，后续自动无图，不影响聊天主功能
- 所有模型统一走 OpenAI 兼容客户端，支持流式输出、工具调用与用量统计

## 数据、写入与索引

### 写入边界

后端对游戏数据**只读**，运行期只写以下位置：

| 路径 | 内容 |
|------|------|
| `data/rag/agent_saves/<slotKey>/agent_tasks.json` | 该存档的 Agent 任务 |
| `data/rag/agent_saves/<slotKey>/agent_text.json` | 该存档的任务文本 |
| `data/rag/agent_saves/<slotKey>/dialogue_overrides.json` | 该存档的日常对话覆盖层 |
| `data/rag/agent_saves/<slotKey>/task_text_overrides.json` | 该存档的任务文本覆盖层 |
| `data/task/agent_tasks.json`、`data/task/text/agent_text.json` | 未绑定存档时的共享任务落点 |
| `data/rag/npc_state_db.json` | NPC 好感/关系/存档内状态 |
| `tools/memory.db` | 会话、消息、摘要、任务草案、长期记忆（SQLite WAL） |
| `tools/vector_index_v3/` | 向量索引与 BM25 索引（自动生成） |

基础对话 XML、任务 XML、物品/关卡/地图等数据一律不修改；对话与任务文本的玩家侧改动全部落在覆盖层，避免污染游戏原始数据。所有任务文件采用原子写入，避免半截 JSON。

### 向量索引

- 索引目录：`<游戏数据根>/tools/vector_index_v3/`（`vectors.npy`、`meta.json`、`fingerprint.json`、`bm25.pkl.gz`）
- 指纹口径：语料文件的相对路径、大小、修改时间（纳秒）集合哈希；**游戏数据每次更新都会在下次启动时自动重建**
- 首次启动或指纹失配时，后台自动构建，不阻塞服务；构建期间检索静默降级（不报错，只是暂时查不到新内容）
- 本机实测全量冷构建约 1～2 分钟（视机器与语料规模）；索引已就绪时启动直接加载，通常 1 秒级
- 索引不能随分发包预置：指纹包含文件修改时间，换机器必然失配，首次启动仍会重建
- 手动重建：设置页「重置知识库」或 `POST /api/game/knowledge-base/reset`

### 日志与端口

- 日志：exe 同目录 `cfn-rag.log`（源码运行写在项目根）
- 端口：后端固定 `127.0.0.1:7077`；standalone 前端静态服务从 `7080` 起找空闲端口
- 单实例：打包版有进程互斥；embedded 模式检测到 7077 已在监听时静默退出，复用既有实例

## API 概览

所有接口均在 `http://127.0.0.1:7077/api` 下；SSE 响应头带 `X-CFN-Version: 3.0.0`。

### 对话与会话

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查、当前模式、绑定槽位、前端地址 |
| POST | `/api/game/ask` | NPC 对话；`?stream=true` 走 SSE，否则聚合为整包响应 |
| GET | `/api/game/sessions` | 当前存档作用域内的会话列表与可通讯 NPC |
| POST | `/api/game/sessions` | 创建会话 |
| PUT | `/api/game/sessions/{session_id}/title` | 重命名会话 |
| DELETE | `/api/game/sessions/{session_id}` | 删除会话 |
| GET | `/api/game/history/{session_id}` | 分页获取历史消息 |
| GET | `/api/game/npc/{npc_name}/favorability` | 好感度与关系等级 |
| GET/POST | `/api/game/llm-config` | 读取/同步当前 LLM 连接配置（内存） |
| POST | `/api/game/knowledge-base/reset` | 重建向量知识库 |

### 插话与游戏集成

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/game/dialogue-interject` | 对话内嵌插话（SSE） |
| POST | `/api/game/dialogue-interject/finalize` | 已播对账（幂等） |
| POST | `/api/game/dialogue-interject/action-result` | 选项/物品动作回执（幂等） |
| POST | `/api/integration/bind` | 绑定/刷新存档投影 |
| POST | `/api/integration/unbind` | 解除绑定 |
| GET | `/api/integration/status` | 集成状态快照 |
| POST | `/api/integration/task-completed` | 任务完成事件 |
| POST | `/api/integration/shutdown` | embedded 实例优雅退出 |

### 静态资源

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/assets/avatar/{npc_name}` | NPC 头像 PNG |
| GET | `/api/assets/illustration/{npc_name}/{emotion}` | NPC 情绪立绘 PNG（查表直出，裁剪由前端处理） |

### SSE 事件契约

`/api/game/ask?stream=true` 返回以下事件（实现权威：`services/orchestrator/events.py`）：

- `meta`：`{emotion, favorability_change, favorability, relationship_level}`，每回合一次且必先于正文
- `content`：`{delta}`，纯台词增量
- `tool_status`：`{tool, status, ui_hint}`，同步工具进度
- `agent_status`：`{agent, phase, ui_hint}`，后台子 Agent 进度
- `system_notice`：`{text}`，系统消息（草案拟定、委托发布、错误说明等）
- `done`：`{session_id, usage}`，回合结束
- `error`：`{code, message, retryable}`，不可恢复错误
- `: keep-alive`：等待后台子 Agent 时的 SSE 注释帧

## 立绘与头像

V3 不再需要立绘包、SWF 导出或手动放置图片：

- **立绘**：后端按游戏项目的对话立绘 manifest（`launcher/web/assets/dialogue-portraits/manifest.json`）做「角色名 + 情绪」查表，命中后原始 PNG 直出；角色名支持别名与空白归一化，情绪缺失时按「请求情绪 → 普通 → defaultExpression → 首个表情」回退
- **头像**：取自 `flashswf/portraits/profiles/{npc}.png`，与立绘体系解耦
- 查无角色、manifest 缺失或视觉模型不可用时自动无图降级，**绝不影响聊天主功能**
- NPC 的 `appearance` 形象描述会拼入对话 prompt，无立绘/无视觉模型时也能维持形象一致
- Full zip 已内置上述立绘与头像；使用游戏整包的用户直接读取游戏项目中的最新资产

相关接口：`GET /api/assets/illustration/{npc_name}/{emotion}`（源图直出）、`GET /api/assets/avatar/{npc_name}`。视觉模型对话时后端会自动取立绘并压缩后随请求发送（长边 ≤480）。

## 项目结构

```text
cfn-rag-backend/
├── api/                         # API 路由
│   ├── game_api.py              # 会话、对话、LLM 配置、知识库重置
│   ├── assets_api.py            # 头像 / 立绘
│   └── integration_api.py       # 游戏内嵌集成
├── core/
│   ├── config.py                # 全局配置（本地单用户）
│   ├── exceptions.py
│   └── startup.py               # 启动初始化（并行预载数据/索引/模型/任务 ensure）
├── schemas/                     # Pydantic 请求/响应模型
├── services/
│   ├── llm/                     # LLM 接入层（client / profiles / meta 行协议 / errors 降级链）
│   ├── orchestrator/            # 回合编排（turn / context / events / merge / prompts / 检索预取）
│   ├── subagents/               # 后台子 Agent（task / search / dialogue）
│   ├── retrieval/               # 混合检索（embedder / store / hybrid / pools / loader / refresh）
│   ├── agent_tools/             # 任务纯函数（prepare / draft / 校验 / 自动微调 / 兜底 / 完成态）
│   ├── tools/                   # 子 Agent 原子工具（route / query / task / system）
│   ├── skills/                  # Skills 渐进披露（元工具按需读取）
│   ├── npc/                     # NPC 状态、场景上下文、任务发布策略
│   ├── memory/                  # 会话记忆、滚动摘要、跨会话长期记忆（SQLite）
│   ├── dialogue/                # 对话覆盖层、对话修改子 Agent、任务文本覆盖
│   ├── dialogue_interject/      # 游戏内嵌插话、选项直连、物品赠送/索取
│   ├── integration/             # 启动模式、存档绑定、按存档路径约定
│   ├── portraits/               # 立绘 manifest 查表 + provider + 缓存
│   └── game_data/               # 游戏静态数据 Registry（items/stages/tasks/shops/map/...）
├── models/
│   └── bge-small-zh-v1.5-onnx-int8/   # 嵌入模型（ONNX int8，512 维，权重不入 git）
├── scripts/                     # build_exe.py / build_full_zip.py / download_model.py / export_onnx_int8.py
├── tests/                       # pytest 测试套件
├── evals/                       # 检索层评估（golden 集 / 三轨指标 / 报告）
├── dist/                        # 前端构建产物（静态）
├── launcher.py                  # 启动器（后端 + 前端静态服务 + /api 反向代理）
├── main.py                      # FastAPI 入口
└── requirements.txt / requirements-dev.txt
```

## 打包与发布

### 构建 exe

```bash
pip install -r requirements-dev.txt
python scripts/build_exe.py
```

产物为项目根目录的 `CFN-RAG.exe`（onefile，约 92MB）。发布时请复制为 `CFN-RAG-v<版本>.exe`（例如 `CFN-RAG-v3.0.0.exe`）；游戏宿主按 `CFN-RAG-v*.exe` 发现程序，**版本号可以改，前缀不能改**。

### 构建 Full zip

```bash
python scripts/build_full_zip.py \
  --game-dir /path/to/CrazyFlashNight \
  --exe CFN-RAG.exe \
  --version 3.0.0
```

脚本只从游戏项目复制运行所需内容：

- `data/` 中实际读取的子目录（对话、任务、情报、术语表、关卡、物品、商店、合成、单位、怪物属性、地图等）
- `docs/story/` 世界观设定文档
- `launcher/web/assets/dialogue-portraits/` 对话立绘与 manifest
- `flashswf/portraits/profiles/` NPC 头像

不会打包：向量索引、运行期存档、`data/rag/` 增量、无关的美术/音频/开发文档。zip 会内置空的 Agent 任务种子与版本化 exe，解压后独立运行。

### Release 资产

| 资产 | 面向人群 | 使用方式 |
|------|----------|----------|
| `CFN-RAG-v3.0.0.exe` | 已有 2026-10-26 之后游戏整包或 GitHub 最新游戏项目 | 放到 `CrazyFlashNight` 文件夹的同一级（Steam 版通常是 `project\`） |
| `CFN-RAG-Full-v3.0.0.zip` | 没有对应版本游戏、只想独立体验聊天 | 解压任意位置，双击 `CFN-RAG-v3.0.0.exe` |

立绘与头像已随 Full zip 提供，无需单独下载。

## 评估（开发者）

检索层评估资产位于 `evals/`（不调 LLM，零费用，适合回归）：

```bash
python -m evals.runners.build_golden_set --tiny    # 5 条微型集 -> evals/datasets/tiny_golden.jsonl
python -m evals.runners.build_golden_set --full    # 130 条 -> evals/datasets/golden_v1.jsonl
python -m evals.runners.run_all --suite retriever --dataset evals/datasets/tiny_golden.jsonl --sample 0
```

- golden 集从当前语料分层采样，排除「彩蛋」「成员」阵营样本；语料变化后需重建
- 指标：recall@k / precision@k / MRR / nDCG，dense / bm25 / hybrid_rrf 三轨对比；报告输出到 `evals/reports/`
- 详见 [evals/README.md](evals/README.md)

## 常见问题

### Q: exe 应该放在哪里？为什么不能被游戏自动拉起？

A: 放在游戏项目根 `CrazyFlashNight` 的同一级。Steam 独立版通常是 `project\`；完整整包通常是解压包根目录。**不要改文件名**，游戏宿主只识别 `CFN-RAG-v*.exe`。

### Q: 启动时提示找不到游戏资源？

A: 确认 exe 同级存在 `CrazyFlashNight` 或 `resources` 文件夹，且其中有 `data/`。也可以设置环境变量 `CFN_RESOURCES_DIR` 指向游戏项目根。独立体验请使用 Full zip，不要只下载 exe。

### Q: 第一次启动很久没有反应，是卡住了吗？

A: 不是。首次启动或游戏数据更新后会在后台重建向量索引，构建期间服务已经可用，只是检索暂时降级。本机实测约 1～2 分钟；索引就绪后再次启动通常 1 秒级加载。不要重复点击「重置知识库」。

### Q: 游戏数据更新了，需要手动删索引吗？

A: 不需要。后端按语料指纹判断，数据变化后下次启动自动重建。设置页的「重置知识库」只在需要强制全量重建时使用。

### Q: 需要下载立绘包吗？

A: 不需要。V3 立绘由游戏项目的对话立绘 manifest 自动提供，头像直接取游戏自带 `profiles` 目录；Full zip 已内置这两类资源。没有立绘时对话功能完全正常，仅视觉模型不带图。

### Q: 任务发布功能不可用 / NPC 只会聊天？

A: 检查两点：① 所用平台和模型是否支持工具调用（function calling）；② 终端设置里的「Agent 能力」是否开启。魔搭等平台的部分模型只支持聊天。

### Q: 为什么前后端必须一起升级？

A: V3 使用新的 SSE 事件契约（meta/content/tool_status/agent_status/system_notice/done/error），与旧前端不兼容。请使用配套的 v3 前端构建产物；后端 exe 已内置前端，源码运行请同步更新 [cfn-terminal-web](https://github.com/aka-flashNight/cfn-terminal-web)。

### Q: 多个存档的数据会混在一起吗？

A: 不会。游戏内嵌模式下后端按宿主推送的 `slot_key` 隔离任务、覆盖层、会话、NPC 状态与长期记忆；未绑定存档的独立模式使用单独的共享作用域。

### Q: API 调用报错 / 没有响应？

A: 检查模型名、API Base、API Key 与代理配置；查看 exe 同目录 `cfn-rag.log` 中的错误详情。前端显示 `error` 事件时，`retryable=true` 可直接重试，`false` 请先修正配置。

### Q: 数据会上传到外部服务器吗？

A: 后端只监听 `127.0.0.1`，不向本项目之外的服务器上传游戏数据；只有你配置的 LLM API 会收到对话所需的提示词与上下文（包括按需携带的立绘）。请自行确认所用模型平台的隐私政策。

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
