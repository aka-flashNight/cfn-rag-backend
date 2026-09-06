
# 项目基础规范
- 已创立虚拟环境，执行python 前先激活虚拟环境 .\venv\Scripts\Activate.ps1
- 使用简体中文和我进行交流以及编写代码注释。思考等其他情况下中英文均可。
- 激活 venv 后用 `python -m pytest tests/` 跑测试套件（pytest 属 requirements-dev，不在运行时依赖中）。
- 运行时依赖只装 `requirements.txt`；torch/transformers/pyinstaller 等开发依赖装 `requirements-dev.txt`，永不进 exe 打包清单。