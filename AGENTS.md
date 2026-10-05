
# 项目基础规范
- 已创立虚拟环境，执行python 前先激活虚拟环境 .\venv\Scripts\Activate.ps1
- 使用简体中文和我进行交流以及编写代码注释。思考等其他情况下中英文均可。
- 激活 venv 后用 `python -m pytest tests/` 跑测试套件（pytest 属 requirements-dev，不在运行时依赖中）。
- 运行时依赖只装 `requirements.txt`；torch/transformers/pyinstaller 等开发依赖装 `requirements-dev.txt`，永不进 exe 打包清单。

# 硬性约束

## 命名冻结：本项目生成的文件 / 文件夹名一律固定，禁止改名
- 适用范围：本项目代码、脚本、打包流程**输出或依赖**的全部文件与文件夹名（索引、运行时状态、日志、种子等），不限于索引。
- **尤其禁止因版本号变动而改名**（加 `_v2` / `_v3` 后缀、把年份或版本写进名字等）。版本信息应落在文件**内容**里（例：`fingerprint.json` 的 `format` 字段），不落进名字。
- 当前已固定、不得更改的名字（括号内为定义位置）：
  - 向量索引目录 `vector_index_v3`；索引文件 `vectors.npy` / `meta.json` / `fingerprint.json` / `bm25.pkl.gz`（`services/retrieval/store.py`、`services/retrieval/hybrid.py`）
  - 运行时状态 `tools/memory.db`（`services/memory/store.py`）、`data/rag/npc_state_db.json`（`services/npc/manager.py`）、`data/task/agent_tasks.json` 与 `data/task/text/agent_text.json`（`services/agent_tools/task_tools.py`）
  - 启动种子 `backup_resources/` 下的 `npc_state_db.json` / `agent_tasks.json` / `agent_text.json`
  - 同级旧格式索引 `vector_index`（LlamaIndex 遗留）与 `vector_index_v3` 并存，两者都不得改名或增删版本后缀。
- 改名的代价：用户与游戏侧已有文件立即失效并被重复生成，且必须同轮修改游戏仓库 `CrazyFlashNight/.gitignore`，否则生成物会被误提交。
- **唯一例外：打包产物 exe 的文件名允许带版本号**（形如 `CFN-RAG-v3.0.0.exe`）——它是分发物，不是被程序按路径读取的对象。
- 如确需换代：属独立变更，先评估迁移与兼容方案并同轮更新游戏仓库忽略规则，不得只改常量了事。