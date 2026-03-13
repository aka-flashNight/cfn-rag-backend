# CFN-RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CFN-RAG 是一个基于 RAG（检索增强生成）技术的智能问答系统，专为 [Crazy Flash Night (CFN)](https://github.com/FlashNightModReborn/CrazyFlashNight) 游戏项目设计。通过读取游戏资源文件，系统能够理解游戏世界设定、角色背景、任务信息等内容，并以对话形式回答玩家的问题。

## 功能特点

- **智能问答**：基于游戏数据回答关于角色、任务、物品等各类问题
- **NPC 记忆系统**：记录与每个 NPC 的对话历史，保持上下文连贯性
- **好感度系统**：NPC 对玩家有独立的好感度、关系等级和情绪状态，影响对话风格
- **多模态对话**：支持传入 NPC 立绘/头像，利用模型的视觉理解能力增强角色扮演体验
- **向量索引持久化**：知识库索引保存到本地，二次启动秒开无需等待
- **对话管理**：支持删除、重命名对话会话，方便整理聊天记录
- **离线运行**：支持本地部署，保护数据隐私
- **嵌入模型本地运行**：使用 BGE 中文嵌入模型，无需联网即可进行文本向量化
- **简单易用**：提供单文件可执行版本，双击即可运行

## 前置要求

CFN-RAG 需要配合 **Crazy Flash Night 游戏资源** 使用。请将游戏项目的 `resources` 文件夹与本项目放在同一目录层级：

```
父目录/
├── resources/              # Crazy Flash Night 游戏资源文件夹
│   ├── data/
│   └── ...
└── cfn-rag-backend/        # 本项目
    ├── launcher.py
    └── ...
```

Crazy Flash Night 游戏项目地址：https://github.com/FlashNightModReborn/CrazyFlashNight

## 快速开始

### 方式一：使用预编译的可执行文件（推荐普通用户）

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面下载 `CFN-RAG.exe`
2. 确保 `resources` 文件夹与 `CFN-RAG.exe` 在同一目录
3. 双击运行 `CFN-RAG.exe`
4. 浏览器将自动打开界面

**注意**：必须配合`resources`游戏项目文件夹，且是github上的最新版本


### 方式二：使用完整独立版（推荐无游戏项目的用户）

适合没有下载游戏项目，但想体验功能的用户。

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面下载 `CFN-RAG-Full.zip`
2. 解压到任意位置
3. 双击运行 `CFN-RAG.exe`
4. 浏览器将自动打开界面

**优点**：无需额外下载游戏资源，独立运行
**注意**：内置资源可能不是最新版本

### 方式三：从源码运行（推荐开发者）

#### 环境要求

- Python 3.8+
- 足够的磁盘空间（约 500MB 用于依赖和模型）

#### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/aka-flashNight/cfn-rag-backend.git
cd cfn-rag-backend
```

2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 下载嵌入模型（可选，首次运行会自动下载）

```bash
# 使用国内镜像（推荐）
python scripts/download_model.py --modelscope

# 或使用 HuggingFace 镜像
python scripts/download_model.py --mirror

# 或使用代理
python scripts/download_model.py --proxy http://127.0.0.1:10809
```

5. 配置 API Key

复制 `.env` 文件并配置你的 API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

6. 启动服务

```bash
python launcher.py
```

## 发布版本说明

我们在 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面提供以下两种发布包，请根据你的需求选择：

### 1. CFN-RAG-Full.zip（完整独立版）

**面向人群**：想独立体验功能，不想下载完整游戏项目的用户

| 特点 | 说明 |
|------|------|
| 文件大小 | 约 350MB（含必要的资源文件） |
| 使用方式 | 解压到任意位置，进入文件夹运行 `CFN-RAG.exe` |
| 依赖 | 无需外部 `resources` 文件夹，无需 Python 环境 |
| 优点 | 完全独立运行，不依赖游戏项目 |
| 缺点 | 无法随游戏更新获取最新数据，仅包含基础资源 |
| 缺点 | 启动时需要解压到临时目录，启动速度稍慢 |

**目录结构**：
```
任意位置/
├── resources/                  # 包含必要的游戏数据文件
└── CFN-RAG.exe                 # 单文件可执行程序
```

---

### 2. CFN-RAG.exe（单文件版）

**面向人群**：有完整游戏项目的用户

| 特点 | 说明 |
|------|------|
| 文件大小 | 约 300MB |
| 使用方式 | 将 `CFN-RAG.exe` 放到与 `resources` 文件夹同一目录，双击运行 |
| 依赖 | 需要完整的游戏项目 `resources` 文件夹，无需 Python 环境 |
| 优点 | 单个文件，下载即用，移动方便 |
| 缺点 | 必须配合 `resources` 文件夹，且是github上的最新版本 |
| 缺点 | 启动时需要解压到临时目录，启动速度稍慢 |

**目录结构**：
```
你的游戏目录/
├── resources/              # 游戏资源文件夹
└── CFN-RAG.exe            # 单文件可执行程序
```

---

### 版本选择建议

| 你的情况 | 推荐版本 |
|---------|---------|
| 没有游戏项目，想独立体验功能 | **CFN-RAG-Full.zip** |
| 有游戏项目，想体验完整功能 | **CFN-RAG.exe** |
| 开发者，需要修改代码 | **源码克隆** |

## 配置说明

### 获取 API Key

本项目需要配置 LLM API Key 才能使用。以下是几种获取免费 API Key 的方式：

#### ModelScope 魔搭社区（首推，国内访问稳定，免费额度充足）

ModelScope 提供每日刷新的免费额度，单模型20~500 次，总共 2000 次，足以支持日常聊天使用。单个模型达到额度后可切换其他模型名称。

**获取步骤**：

1. **注册并绑定阿里云实名账户**
   - 访问 [ModelScope 官网](https://www.modelscope.cn/) 注册账号
   - 进入[账号绑定页面](https://www.modelscope.cn/my/settings/account)，绑定阿里云实名认证的账号（必须先完成阿里云实名认证）

2. **获取 API Key（访问令牌）**
   - 进入 [访问控制 - 访问令牌](https://modelscope.cn/my/access/token) 页面
   - 点击 "创建新的访问令牌"作为api_key

3. **选择模型并获取配置信息**
   - 进入 [模型库](https://www.modelscope.cn/models)
   - 在筛选条件中勾选 **"支持体验" → "推理 API-Inference"**，筛选出支持免费 API 调用的模型
   - 点击感兴趣的模型进入详情页
   - 在"推理 API" 或 "代码范例" 标签页中查看：
     - `model`：模型名称（如 `moonshotai/Kimi-K2.5`）
     - `base_url`：API Base地址（固定为 `https://api-inference.modelscope.cn/v1`）

**推荐模型**：
- `moonshotai/Kimi-K2.5`：Moonshot 的 Kimi K2.5 多模态模型，性能优秀，每日约50次免费调用次数（2026.3.11测试）
- `Qwen/Qwen3.5-397B-A17B`：阿里 Qwen3.5 多模态moe大模型，每日约100次免费调用次数（2026.3.11测试）
- `MiniMax/MiniMax-M2.5`：纯文本生成模型，每日约100次免费调用次数（2026.3.11测试）
- `ZhipuAI/GLM-5`：智谱文本生成模型，参数最大（但可能稍慢），每日约100次免费调用次数（2026.3.11测试）
- `deepseek-ai/DeepSeek-V3.2`：DeepSeek文本生成模型，参数大，每日约20次免费调用次数（2026.3.11测试）
- `Qwen/Qwen3.5-27B`：阿里 Qwen3.5 多模态模型，参数较小的版本，每日约**500**次免费调用次数（2026.3.11测试）
- `Qwen/Qwen3.5-122B-A10B`：阿里 Qwen3.5 多模态模型，参数中等的moe版本，每日约200次免费调用次数（2026.3.11测试）

**免费额度**：绑定阿里云实名账户后，每日 2000 次免费调用（单模型上限 500 次，但部分模型可能更少，达到上限后可更换模型）。

**配置示例**：
```env
LLM_API_KEY=your_modelscope_token_here
LLM_API_BASE=https://api-inference.modelscope.cn/v1
LLM_MODEL_NAME=moonshotai/Kimi-K2.5
```

#### Google Gemini（免费额度充足，需代理）

1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 使用 Google 账号登录
3. 点击 "Create API Key"
4. 复制生成的 Key 到 `.env` 文件

**免费额度**：每分钟 60 次请求，完全满足个人使用需求。

**注意**：使用 Gemini 可能需要配置代理，请参考下方代理配置部分。

#### 其他推荐平台

- **[Moonshot AI](https://platform.moonshot.cn/)**：月之暗面 Kimi API，注册有15元免费额度
- **[QWEN](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key)**：阿里云百炼 QWEN API，每个模型百万token免费额度
- **[DeepSeek](https://platform.deepseek.com/)**：国产大模型，价格便宜且有免费额度

### 配置文件说明

创建 `.env` 文件，参考以下配置：

```env
# LLM 配置（默认使用 Gemini）
LLM_API_KEY=your_api_key_here
LLM_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL_NAME=gemini-2.5-flash

# 或使用其他 OpenAI 兼容的 API
# LLM_API_BASE=https://api.siliconflow.cn/v1
# LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 代理配置

**如果你使用国外模型（如 Gemini、OpenAI）或开启了全局代理，需要在前端界面中配置代理。**

代理配置已集成到前端界面中，启动服务后，在前端界面的设置区域填写代理地址即可，例如：
- `http://127.0.0.1:7890`（Clash 默认端口）
- `http://127.0.0.1:10809`（v2rayN 默认端口）
- `http://127.0.0.1:1080`（Shadowsocks 默认端口）

配置后，代理会立即生效，对后续所有 LLM API 调用及网络请求生效。

### 立绘文件格式（可选）

为获得更好的多模态对话体验，你可以在 `resources/flashswf/portraits/illustration` 目录下放置 NPC 立绘图片。

**文件格式要求**：

| 项目 | 说明 |
|------|------|
| 文件名格式 | `{NPC名称}#{情绪}.png`，例如：`凯特#普通.png`、`凯特#开心.png` |
| 分辨率 | 建议 **300×600** 像素（宽高比 1:2）|
| 文件格式 | PNG |
| 文件大小 | 建议控制在 200KB 以内，过大的图片会消耗大量 Token |

**情绪标签**：系统支持以下情绪（会自动匹配，无需全部制作）：
- `普通`（默认情绪，每个 NPC 至少有一张）
- `微笑`、`严肃`、`悲伤`、`愤怒` 等（需要和npc拥有的情绪匹配）

**目录结构示例**：
```
resources/
└── flashswf/
    └── portraits/
        ├── illustration/           # 立绘目录
        │   ├── Andy Law#普通.png
        │   ├── Andy Law#微笑.png
        │   └── Andy Law#严肃.png
        └── profiles/               # 头像目录（游戏自带）
            └── Andy Law.png
```

**说明**：
- 如果找不到对应情绪的立绘，会自动回退到 `普通` 情绪
- 如果找不到任何立绘，会自动尝试使用 `profiles` 目录下的头像
- 立绘不是必须的，没有时对话功能仍可正常使用


## 项目结构

```
cfn-rag-backend/
├── api/                    # API 路由层
│   ├── assets_api.py       # 静态资源 API
│   └── game_api.py         # 游戏相关 API
├── ai_engine/              # AI 引擎核心
│   └── game_data_loader.py # 游戏数据加载器
├── core/                   # 核心配置
│   ├── config.py           # 全局配置
│   └── exceptions.py       # 异常处理
├── dist/                   # 前端构建产物
├── models/                 # 本地模型存储
│   └── bge-small-zh-v1.5/  # 中文嵌入模型
├── schemas/                # Pydantic 数据模型
├── scripts/                # 工具脚本
│   ├── build_exe.py        # 打包脚本
│   └── download_model.py   # 模型下载脚本
├── services/               # 业务逻辑层
│   ├── game_rag_service.py # RAG 服务
│   ├── memory_manager.py   # 记忆管理
│   └── npc_manager.py      # NPC 管理
├── launcher.py             # 启动器（同时启动前后端）
├── main.py                 # FastAPI 应用入口
└── requirements.txt        # Python 依赖
```

## 打包可执行文件

如果你想自己打包可执行文件：

```bash
python scripts/build_exe.py
```

打包完成后会在项目根目录生成 `CFN-RAG.exe`。

## 常见问题

### Q: 启动时提示找不到 resources 文件夹？

A: 确保 `resources` 文件夹与项目在同一目录层级，参考上方【前置要求】部分的目录结构说明，项目目录在最上方。

### Q: 模型下载失败或很慢？

A: 使用国内镜像下载：
```bash
python scripts/download_model.py --modelscope
```

### Q: API 调用报错/无响应？

A: 检查以下几点：
1. 模型名称、API Base、API Key 是否正确配置
2. 如使用国外模型，是否配置了代理
3. 代理地址和端口是否正确

### Q: 如何更换其他 LLM 模型？

A: 在前端配置中修改（优先级最高），或者修改 `.env` 文件中的 `LLM_API_BASE` 和 `LLM_MODEL_NAME` 配置项。只要 API 兼容 OpenAI 格式即可使用。

### Q: 第一次启动后对话加载很慢，第二次就很快？

A: 这是正常现象。第一次启动时需要构建向量索引（读取所有游戏数据并计算向量），这个过程可能需要 10-30 秒。索引构建完成后会自动保存到 `resources/tools/vector_index` 目录，下次启动时会直接加载。

### Q: 游戏数据更新了，如何让索引重新构建？

A: 删除 `resources/tools/vector_index` 文件夹，下次启动时会自动重新构建索引。

### Q: 立绘图片如何获取？

A: 你可以：
1. 从游戏资源（fla或swf）中提取
2. 自己绘制或委托画师创作
3. 使用 AI 绘画工具生成（注意版权问题）

立绘文件放置在 `resources/flashswf/portraits/illustration/` 目录下，具体格式请参考上方【立绘文件格式】章节。

## 技术栈

- **后端**：FastAPI + Uvicorn
- **RAG 框架**：LlamaIndex
- **嵌入模型**：BAAI/bge-small-zh-v1.5（本地运行）
- **LLM**：支持任何 OpenAI 兼容的 API（Gemini、GPT、Claude 等）
- **数据库**：SQLite（会话记忆）
- **打包**：PyInstaller

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 致谢

- [Crazy Flash Night](https://github.com/FlashNightModReborn/CrazyFlashNight) - 游戏项目
- [LlamaIndex](https://www.llamaindex.ai/) - RAG 框架
- [BAAI](https://github.com/FlagOpen/FlagEmbedding) - BGE 嵌入模型

## 联系方式

如有问题或建议，欢迎提交 [Issue](https://github.com/aka-flashNight/cfn-rag-backend/issues) 或 [Pull Request](https://github.com/aka-flashNight/cfn-rag-backend/pulls)。
