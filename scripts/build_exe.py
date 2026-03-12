#!/usr/bin/env python3
"""
打包脚本：将launcher.py和后端代码打包成单个exe文件
"""

import os
import sys
import shutil
import subprocess


def collect_all_py_files():
    """收集所有需要打包的Python文件目录"""
    dirs_to_include = []

    # 获取项目根目录（脚本所在目录的父目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    # 需要包含的目录
    target_dirs = ['api', 'core', 'services', 'ai_engine', 'schemas']

    for dir_name in target_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            dirs_to_include.append(dir_name)
            print(f"  包含目录: {dir_name}/")

    # 包含根目录下的py文件
    root_py_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'launcher.py' and f != 'build_exe.py']
    print(f"  包含根目录Python文件: {root_py_files}")

    return dirs_to_include, root_py_files


def create_spec_file(dirs_to_include, root_py_files, script_dir):
    """创建PyInstaller的spec文件"""

    # 构建add-data参数
    add_data_lines = []

    # 添加前端dist目录
    dist_dir = os.path.abspath('dist')
    add_data_lines.append(f"             (r'{dist_dir}', 'dist'),")

    # 添加模型目录（如果存在）
    models_dir = os.path.abspath('models')
    if os.path.exists(models_dir):
        add_data_lines.append(f"             (r'{models_dir}', 'models'),")
        print(f"  包含模型目录: models/")


    # 添加后端Python目录
    for dir_name in dirs_to_include:
        abs_path = os.path.abspath(dir_name)
        add_data_lines.append(f"             (r'{abs_path}', '{dir_name}'),")

    # 添加根目录的py文件
    for py_file in root_py_files:
        abs_path = os.path.abspath(py_file)
        add_data_lines.append(f"             (r'{abs_path}', '.'),")

    add_data_str = '\n'.join(add_data_lines)

    launcher_path = os.path.abspath('launcher.py')

    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(5000)

a = Analysis(
    [r'{launcher_path}'],
    pathex=[r'{os.path.abspath('.')}'],
    binaries=[],
    datas=[
{add_data_str}
    ],
    hiddenimports=[
        'main',
        'api',
        'api.game_api',
        'api.assets_api',
        'core',
        'core.config',
        'core.exceptions',
        'services',
        'services.memory_manager',
        'services.npc_manager',
        'services.game_rag_service',
        'ai_engine',
        'ai_engine.game_data_loader',
        'fastapi',
        'fastapi.middleware.cors',
        'uvicorn',
        'uvicorn.loops.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan.on',
        'pydantic',
        'pydantic_settings',
        'llama_index',
        'llama_index.core',
        'llama_index.embeddings.huggingface',
        'openai',
        'httpx',
        'aiofiles',
        'pypdf',
        'docx',
        'sqlalchemy',
        'sqlite3',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CFN-RAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r'{os.path.join(script_dir, 'icon.ico')}',
)
'''

    return spec_content


def main():
    """打包主函数"""
    # 切换到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print("=" * 50)
    print("CFN-RAG 完整打包工具")
    print(f"项目根目录: {project_root}")
    print("=" * 50)

    # 检查是否安装了 PyInstaller
    try:
        import PyInstaller
        print("PyInstaller 已安装")
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller 安装完成")

    # 清理之前的构建文件
    print("\n清理之前的构建文件...")
    for folder in ["build_temp", "build", "__pycache__"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"  已删除 {folder}/")

    # 删除旧的exe
    exe_path = os.path.join(project_root, 'CFN-RAG.exe')
    if os.path.exists(exe_path):
        try:
            os.remove(exe_path)
            print(f"  已删除旧的 CFN-RAG.exe")
        except PermissionError:
            print("  [警告] 无法删除旧的 CFN-RAG.exe，可能正在运行")
            print("  请关闭正在运行的CFN-RAG.exe后再试")
            input("  按回车键退出...")
            sys.exit(1)

    # 收集文件
    print("\n收集后端代码文件...")
    dirs_to_include, root_py_files = collect_all_py_files()

    if not dirs_to_include and not root_py_files:
        print("[错误] 没有找到后端代码文件！")
        sys.exit(1)

    # 创建spec文件
    print("\n生成打包配置文件...")
    spec_content = create_spec_file(dirs_to_include, root_py_files, script_dir)
    spec_path = os.path.join('build_temp', 'CFN-RAG.spec')

    os.makedirs('build_temp', exist_ok=True)
    with open(spec_path, 'w', encoding='utf-8') as f:
        f.write(spec_content)

    print(f"  配置文件: {spec_path}")

    # 执行打包
    print("\n开始打包（这可能需要几分钟）...")
    print("-" * 50)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        spec_path,
        '--clean',
        '--noconfirm'
    ]

    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 50)
        print("打包成功!")
        print("=" * 50)

        # 移动exe到根目录
        dist_exe_path = os.path.join(project_root, 'dist', 'CFN-RAG.exe')
        target_exe_path = os.path.join(project_root, 'CFN-RAG.exe')
        if os.path.exists(dist_exe_path):
            # 如果存在旧的exe，先删除
            if os.path.exists(target_exe_path):
                os.remove(target_exe_path)
            shutil.move(dist_exe_path, target_exe_path)
            # 不删除dist文件夹，保留前端文件
            print("  保留dist/文件夹（包含前端文件）")

        exe_size = os.path.getsize(target_exe_path) / (1024 * 1024)
        print(f"输出文件: CFN-RAG.exe ({exe_size:.1f} MB)")

        # 清理打包临时文件
        print("\n清理临时文件...")
        for folder in ["build_temp", "build", "__pycache__"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"  已删除 {folder}/")

        print("\n使用方法:")
        print("  双击 CFN-RAG.exe 即可启动服务")
        print("  - 后端API: http://127.0.0.1:7077")
        print("  - 前端页面: http://127.0.0.1:7080")
        print("\n注意事项:")
        print("  1. 首次运行需要安装依赖，请确保网络畅通")
        print("  2. 需要Node.js环境来启动前端服务")

        # 检查模型是否已打包
        models_dir = os.path.join(project_root, 'models')
        if os.path.exists(models_dir) and any(os.listdir(models_dir)):
            print("  4. 模型已打包到 exe 中，无需额外下载")
        else:
            print("  4. [警告] 未检测到模型文件，首次运行时需联网下载")
            print("     或手动运行: python scripts/download_model.py")
        print("=" * 50)

    except subprocess.CalledProcessError as e:
        print(f"\n打包失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
