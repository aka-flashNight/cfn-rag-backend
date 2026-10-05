
# 项目基础规范
- 已创立虚拟环境，执行python 前先激活虚拟环境 .\venv\Scripts\Activate.ps1
- 使用简体中文和我进行交流以及编写代码注释。思考等其他情况下中英文均可。
- 激活 venv 后用 `python -m pytest tests/` 跑测试套件（pytest 属 requirements-dev，不在运行时依赖中）。
- 运行时依赖只装 `requirements.txt`；torch/transformers/pyinstaller 等开发依赖装 `requirements-dev.txt`，永不进 exe 打包清单。

# 硬性约束
- **禁止重命名向量索引目录，文件夹名必须定死。**
  - 新格式索引恒为 `resources/tools/vector_index_v3`（`services/retrieval/store.py` 的 `INDEX_DIR_NAME`）；
    旧版 LlamaIndex 索引为其同级的 `resources/tools/vector_index`。两者并存，不得改名、不得增删版本后缀。
  - 改名会使所有用户已有索引失效并被重复重建，且必须同步修改游戏仓库 `CrazyFlashNight/.gitignore`，
    否则索引文件会被误提交。
  - 如确需换代，属独立变更：先评估迁移与兼容方案，并同轮更新游戏仓库忽略规则，不得只改常量了事。