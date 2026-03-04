# CFN-RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CFN-RAG 是一个基于 RAG（检索增强生成）技术的智能问答系统，专为 [Crazy Flash Night (CFN)](https://github.com/FlashNightModReborn/CrazyFlashNight) 游戏项目设计。通过读取游戏资源文件，系统能够理解游戏世界设定、角色背景、任务信息等内容，并以对话形式回答玩家的问题。

## 功能特点

- **智能问答**：基于游戏数据回答关于角色、任务、物品等各类问题
- **NPC 记忆系统**：记录与每个 NPC 的对话历史，保持上下文连贯性
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

游戏项目地址：https://github.com/FlashNightModReborn/CrazyFlashNight

## 快速开始

### 方式一：使用预编译的可执行文件（推荐普通用户）

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面下载 `CFN-RAG.exe`
2. 确保 `resources` 文件夹与 `CFN-RAG.exe` 在同一目录
3. 双击运行 `CFN-RAG.exe`
4. 按提示配置代理（如有需要）
5. 浏览器将自动打开界面

**注意**：该方式数据库文件位于临时文件夹，停止运行exe后，聊天记录可能不会保存。

### 方式二：使用便携版（推荐有基础的用户）

适合想要快速运行但不需要修改代码的用户。

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面下载 `CFN-RAG-Portable.zip`
2. 解压到与 `resources` 文件夹同一层级：
   ```
   你的目录/
   ├── resources/              # 游戏资源文件夹
   └── cfn-rag-backend/        # 解压后的项目文件夹
       ├── start.exe
       └── ...
   ```
3. 进入 `cfn-rag-backend` 文件夹，双击运行 `start.exe`
4. 按提示配置代理（如有需要）和 API Key

**优点**：启动速度快，文件结构清晰
**注意**：需要配合外部的 `resources` 文件夹使用

### 方式三：使用完整独立版（推荐无游戏项目的用户）

适合没有下载游戏项目，但想体验功能的用户。

1. 从 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面下载 `CFN-RAG-Full.zip`
2. 解压到任意位置
3. 进入 `cfn-rag-backend` 文件夹，双击运行 `start.exe`
4. 按提示配置代理（如有需要）和 API Key

**优点**：无需额外下载游戏资源，独立运行
**注意**：内置资源可能不是最新版本

### 方式四：从源码运行（推荐开发者）

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

我们在 [Releases](https://github.com/aka-flashNight/cfn-rag-backend/releases) 页面提供以下三种发布包，请根据你的需求选择：

### 1. CFN-RAG.exe（单文件版）

**面向人群**：普通用户，追求最简单使用方式

| 特点 | 说明 |
|------|------|
| 文件大小 | 约 350MB |
| 使用方式 | 将 `CFN-RAG.exe` 放到与 `resources` 文件夹同一目录，双击运行 |
| 依赖 | 无需 Python 环境，无需安装依赖 |
| 优点 | 单个文件，下载即用，移动方便 |
| 缺点 | 启动时需要解压到临时目录，启动速度稍慢，且关闭后聊天历史可能无法保存 |

**目录结构**：
```
你的游戏目录/
├── resources/              # 游戏资源文件夹
└── CFN-RAG.exe            # 单文件可执行程序
```

---

### 2. CFN-RAG-Portable.zip（便携版）

**面向人群**：有一定基础的用户

| 特点 | 说明 |
|------|------|
| 文件大小 | 约 100MB（不含模型） |
| 使用方式 | 在解压到 `cfn-rag-backend` 文件夹，运行文件夹内的 `start.exe` |
| 依赖 | 无需 Python 环境，但可能需要下载嵌入模型（首次运行自动下载） |
| 优点 | 启动速度快，文件体积小，可查看/修改配置 |
| 缺点 | 需要配合外部的 `resources` 文件夹使用 |

**目录结构**：
```
你的目录/
├── resources/                  # 游戏资源文件夹
└── cfn-rag-backend/            # 解压后的项目文件夹
    ├── start.exe              # 启动程序
    ├── .env                   # 配置文件（可手动编辑）
    ├── data/                  # 数据库目录
    └── ...
```

---

### 3. CFN-RAG-Full.zip（完整独立版）

**面向人群**：想独立体验功能，不想下载完整游戏项目的用户

| 特点 | 说明 |
|------|------|
| 文件大小 | 约 150MB+（含必要的资源文件） |
| 使用方式 | 解压到任意位置，进入项目文件夹运行 `start.exe` |
| 依赖 | 无需外部 `resources` 文件夹，无需 Python 环境 |
| 优点 | 完全独立运行，不依赖游戏项目 |
| 缺点 | 无法随游戏更新获取最新数据，仅包含基础资源 |

**目录结构**：
```
任意位置/
├── resources/                  # 包含必要的游戏数据文件
└── cfn-rag-backend/            # 解压后的项目文件夹
    ├── start.exe              # 启动程序
    ├── .env                   # 配置文件（可手动编辑）
    ├── data/                  # 数据库目录
    └── ...
```

---

### 版本选择建议

| 你的情况 | 推荐版本 |
|---------|---------|
| 只是想快速体验，不想折腾 | **CFN-RAG.exe** |
| 有游戏项目，想体验完整功能 | **CFN-RAG-Portable.zip** |
| 没有游戏项目，想独立体验 | **CFN-RAG-Full.zip** |
| 开发者，需要修改代码 | **源码克隆** |

## 配置说明

### 获取 API Key

本项目需要配置 LLM API Key 才能使用。以下是几种获取免费 API Key 的方式：

#### Google Gemini（推荐，免费额度充足）

1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 使用 Google 账号登录
3. 点击 "Create API Key"
4. 复制生成的 Key 到 `.env` 文件

**免费额度**：每分钟 60 次请求，完全满足个人使用需求。

**注意**：使用 Gemini 可能需要配置代理，请参考下方代理配置部分。

#### 其他推荐平台

- **[SiliconFlow](https://siliconflow.cn/)**：国内友好的大模型 API 平台，注册即送免费额度
- **[DeepSeek](https://platform.deepseek.com/)**：国产大模型，价格便宜且有免费额度
- **[Moonshot AI](https://platform.moonshot.cn/)**：月之暗面 Kimi API

### 配置文件说明

创建 `.env` 文件，参考以下配置：

```env
# Gemini 配置（默认）
GEMINI_API_KEY=your_gemini_api_key_here
LLM_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL_NAME=gemini-2.5-flash

# 或使用其他 OpenAI 兼容的 API
# LLM_API_BASE=https://api.siliconflow.cn/v1
# LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 代理配置

**如果你使用国外模型（如 Gemini、OpenAI）或开启了全局代理，需要在启动时配置代理。**

运行 `launcher.py` 或 `CFN-RAG.exe` 时，第一步会询问是否需要配置代理：

```
==================================================
CFN-RAG 启动器
==================================================

是否需要为 HuggingFace/LLM 配置 HTTP 代理？如果开启了全局代理，也请配置。(y/N):
```

- 输入 `y` 启用代理配置
- 默认代理地址为 `http://127.0.0.1:10809`
- 可根据你的代理软件实际端口进行修改

常见代理软件默认端口：
- Clash: `http://127.0.0.1:7890`
- v2rayN: `http://127.0.0.1:10809`
- Shadowsocks: `http://127.0.0.1:1080`

## 项目结构

```
cfn-rag-backend/
├── api/                    # API 路由层
│   ├── assets_api.py       # 静态资源 API
│   ├── game_api.py         # 游戏相关 API
│   └── knowledge_api.py    # 知识库 API
├── ai_engine/              # AI 引擎核心
│   └── game_data_loader.py # 游戏数据加载器
├── core/                   # 核心配置
│   ├── config.py           # 全局配置
│   └── exceptions.py       # 异常处理
├── data/                   # 数据存储
│   └── memory.db           # SQLite 数据库（会话记录）
├── dist/                   # 前端构建产物
├── models/                 # 本地模型存储
│   └── bge-small-zh-v1.5/  # 中文嵌入模型
├── schemas/                # Pydantic 数据模型
├── scripts/                # 工具脚本
│   ├── build_exe.py        # 打包脚本
│   └── download_model.py   # 模型下载脚本
├── services/               # 业务逻辑层
│   ├── game_rag_service.py # RAG 服务
│   ├── knowledge_service.py# 知识库服务
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

A: 确保 `resources` 文件夹与项目在同一目录层级，参考上方【前置要求】部分的目录结构说明。

### Q: 模型下载失败或很慢？

A: 使用国内镜像下载：
```bash
python scripts/download_model.py --modelscope
```

### Q: API 调用报错/无响应？

A: 检查以下几点：
1. API Key 是否正确配置
2. 如使用国外模型，是否配置了代理
3. 代理地址和端口是否正确

### Q: 如何更换其他 LLM 模型？

A: 修改 `.env` 文件中的 `LLM_API_BASE` 和 `LLM_MODEL_NAME` 配置项。只要 API 兼容 OpenAI 格式即可使用。

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
