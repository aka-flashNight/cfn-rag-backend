#!/usr/bin/env python3
"""
CFN-RAG 项目启动器 (start.exe)
用于在项目目录中启动服务，支持代理配置

这个脚本会被打包成 start.exe，放在项目根目录中
"""

import os
import sys
import subprocess
import threading
import time


def ask_yes_no_with_timeout(question, timeout=10, default='n'):
    """询问用户是/否，带超时功能"""
    print(question, end='', flush=True)
    print(f" ({timeout}秒后默认: {default})")

    result = {"value": None}

    def input_thread():
        try:
            user_input = input().strip().lower()
            if user_input in ['y', 'yes']:
                result["value"] = True
            elif user_input in ['n', 'no', '']:
                result["value"] = False
            else:
                result["value"] = default == 'y'
        except:
            result["value"] = default == 'y'

    thread = threading.Thread(target=input_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if result["value"] is None:
        print(f"超时，使用默认值: {default}")
        return default == 'y'

    return result["value"]


def ask_proxy_config():
    """配置代理"""
    print("=" * 50)
    print("CFN-RAG 启动器")
    print("=" * 50)

    # 第一步：询问是否需要代理（10秒超时）
    need_proxy = ask_yes_no_with_timeout(
        "\n是否需要为 HuggingFace/LLM 配置 HTTP 代理？如果开启了全局代理，也请配置。(y/N): ",
        timeout=10,
        default='n'
    )

    if not need_proxy:
        print("跳过代理配置\n")
        return

    # 第二步：询问代理地址（不限时，有默认值）
    default_proxy = "http://127.0.0.1:10809"
    print(f"\n请输入代理地址（默认为 {default_proxy}）: ", end='', flush=True)

    try:
        proxy = input().strip()
        if not proxy:
            proxy = default_proxy
        else:
            # 确保代理地址有 http:// 前缀
            if not proxy.startswith("http://") and not proxy.startswith("https://"):
                proxy = "http://" + proxy
    except:
        proxy = default_proxy

    # 设置代理环境变量
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    print(f"已设置代理: {proxy}\n")


def check_python():
    """检查 Python 是否可用"""
    try:
        result = subprocess.run(
            ["python", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"[检查] Python 版本: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"[错误] 无法运行 Python: {e}")

    # 尝试 python3
    try:
        result = subprocess.run(
            ["python3", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"[检查] Python 版本: {result.stdout.strip()}")
            return True
    except:
        pass

    print("[错误] 未找到 Python，请确保 Python 3.8+ 已安装并添加到 PATH")
    return False


def find_project_root():
    """
    查找项目根目录
    start.exe 应该位于项目根目录中
    """
    # 获取 start.exe 所在目录
    if getattr(sys, 'frozen', False):
        # 打包后的 exe
        exe_dir = os.path.dirname(sys.executable)
    else:
        # 开发环境
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    # 检查当前目录是否有 launcher.py
    if os.path.exists(os.path.join(exe_dir, "launcher.py")):
        return exe_dir

    # 如果当前目录没有，尝试查找子目录
    for item in os.listdir(exe_dir):
        item_path = os.path.join(exe_dir, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, "launcher.py")):
                return item_path

    print(f"[错误] 未找到 launcher.py，请确保 start.exe 位于项目根目录")
    print(f"当前目录: {exe_dir}")
    print(f"目录内容: {os.listdir(exe_dir)[:10]}...")
    return None


def main():
    """主函数"""
    # 配置代理
    ask_proxy_config()

    # 检查 Python 环境
    print("\n[检查] 检查 Python 环境...")
    if not check_python():
        input("\n按回车键退出...")
        sys.exit(1)

    # 查找项目根目录
    project_root = find_project_root()
    if not project_root:
        input("\n按回车键退出...")
        sys.exit(1)

    print(f"\n[启动] 项目目录: {project_root}")

    # 切换到项目目录
    os.chdir(project_root)

    # 设置环境变量
    os.environ["LLAMA_INDEX_CACHE_DIR"] = ".llamaindex_cache"

    # 启动 launcher.py
    print("\n" + "=" * 50)
    print("正在启动 CFN-RAG 服务...")
    print("=" * 50 + "\n")

    try:
        # 使用 sys.executable 确保使用相同的 Python 解释器
        python_cmd = sys.executable if not getattr(sys, 'frozen', False) else "python"

        # 如果启动器本身是用 Python 运行的，直接用 python 启动 launcher.py
        if python_cmd == sys.executable and not getattr(sys, 'frozen', False):
            subprocess.run([python_cmd, "launcher.py"])
        else:
            # 打包后的情况，使用系统 python
            subprocess.run(["python", "launcher.py"])

    except KeyboardInterrupt:
        print("\n\n服务已停止")
    except Exception as e:
        print(f"\n[错误] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")


if __name__ == "__main__":
    main()
