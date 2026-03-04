#!/usr/bin/env python3
"""
打包脚本：将 start.py 打包成轻量级的 start.exe
这个 exe 只负责配置代理并调用 launcher.py 启动项目
"""

import os
import sys
import shutil
import subprocess


def main():
    """打包主函数"""
    # 切换到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print("=" * 50)
    print("CFN-RAG Start.exe 打包工具")
    print(f"项目根目录: {project_root}")
    print("=" * 50)

    # 检查 start.py 是否存在
    start_py_path = os.path.join(project_root, 'start.py')
    if not os.path.exists(start_py_path):
        print(f"[错误] 未找到 start.py: {start_py_path}")
        print("请确保 start.py 位于项目根目录")
        sys.exit(1)

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
    for folder in ["build_start", "build"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"  已删除 {folder}/")

    # 删除旧的 start.exe
    start_exe_path = os.path.join(project_root, 'start.exe')
    if os.path.exists(start_exe_path):
        try:
            os.remove(start_exe_path)
            print("  已删除旧的 start.exe")
        except PermissionError:
            print("  [警告] 无法删除旧的 start.exe，可能正在运行")
            print("  请关闭正在运行的 start.exe 后再试")
            input("  按回车键退出...")
            sys.exit(1)

    # 创建 spec 文件内容
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(5000)

a = Analysis(
    [r'{start_py_path}'],
    pathex=[r'{project_root}'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'subprocess',
        'threading',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'numpy',
        'pandas',
        'matplotlib',
        'PIL',
        'cv2',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
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
    name='start',
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
)
'''

    # 保存 spec 文件
    spec_path = os.path.join(project_root, 'build_start', 'start.spec')
    os.makedirs('build_start', exist_ok=True)
    with open(spec_path, 'w', encoding='utf-8') as f:
        f.write(spec_content)

    print(f"\n生成打包配置文件: {spec_path}")

    # 执行打包
    print("\n开始打包 start.exe（这将是一个轻量级启动器）...")
    print("-" * 50)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        spec_path,
        '--clean',
        '--noconfirm',
        '--distpath', project_root,  # 直接输出到根目录
        '--workpath', os.path.join(project_root, 'build_start', 'build'),
    ]

    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 50)
        print("打包成功!")
        print("=" * 50)

        # 显示文件大小
        if os.path.exists(start_exe_path):
            exe_size = os.path.getsize(start_exe_path) / (1024 * 1024)
            print(f"输出文件: start.exe ({exe_size:.1f} MB)")

        # 清理临时文件
        print("\n清理临时文件...")
        if os.path.exists('build_start'):
            shutil.rmtree('build_start')
            print("  已删除 build_start/")
        if os.path.exists('build'):
            shutil.rmtree('build')
            print("  已删除 build/")

        print("\n使用方法:")
        print("  1. 确保 start.exe 位于项目根目录（与 launcher.py 同级）")
        print("  2. 双击 start.exe 即可启动服务")
        print("  3. 启动时会询问是否需要配置代理")
        print("  4. 然后会自动调用 launcher.py 启动前后端服务")
        print("\n注意事项:")
        print("  - start.exe 只是一个启动器，需要配合项目文件使用")
        print("  - 确保项目已安装依赖: pip install -r requirements.txt")
        print("  - 确保 .env 文件已配置 API Key")
        print("=" * 50)

    except subprocess.CalledProcessError as e:
        print(f"\n打包失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
