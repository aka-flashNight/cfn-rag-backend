#!/usr/bin/env python3
"""
CFN-RAG 启动器
同时启动后端FastAPI服务和前端静态文件服务
"""

import os
import sys

# 在导入其他模块之前设置，避免 tiktoken 编码问题
os.environ["LLAMA_INDEX_CACHE_DIR"] = ".llamaindex_cache"

import time
import threading
import webbrowser
import socketserver
import signal
import http.server




def check_python_environment():
    """检查Python环境是否满足要求"""
    print("[检查] 检查Python环境...", flush=True)

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("[错误] Python版本过低，需要Python 3.8或更高版本", flush=True)
        print(f"当前版本: {sys.version}", flush=True)
        return False

    print(f"[检查] Python版本: {sys.version.split()[0]} ✓", flush=True)

    # 检查必要的包
    required_packages = ['fastapi', 'uvicorn', 'pydantic']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"[错误] 缺少必要的Python包: {', '.join(missing_packages)}", flush=True)
        print("[提示] 请运行: pip install -r requirements.txt", flush=True)
        return False

    print("[检查] 必要的Python包已安装 ✓", flush=True)
    return True




def is_packaged_environment():
    """检查是否在PyInstaller打包环境中运行"""
    return hasattr(sys, '_MEIPASS') or getattr(sys, 'frozen', False)


def find_resources_directory():
    """
    查找resources数据目录。
    resources是外部项目文件夹，和本项目放在同一目录下。

    目录结构：
        父目录/
        ├── resources/          # 外部游戏数据
        └── cfn-rag-backend/    # 本项目（开发环境）
            └── ...

        或打包后：
        部署目录/
        ├── resources/          # 外部游戏数据
        └── CFN-RAG.exe         # 打包后的exe
    """
    # 获取基础目录
    if is_packaged_environment():
        # 打包环境：exe所在目录
        base_dir = os.path.dirname(sys.executable)
    else:
        # 开发环境：脚本所在目录的父目录（因为代码在项目文件夹内，resources在父目录）
        # launcher.py 位置: cfn-rag-backend/launcher.py
        # resources 位置: cfn-rag-backend/../resources
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)

    # 情况1: base_dir/resources（打包后或resources和项目同级）
    resources_path = os.path.join(base_dir, "resources")
    if os.path.exists(resources_path) and os.path.isdir(resources_path):
        print(f"[定位] 找到resources: {resources_path}")
        return os.path.abspath(resources_path)

    # 情况2: 开发环境特殊情况，resources在当前工作目录
    cwd = os.path.abspath(".")
    cwd_resources = os.path.join(cwd, "resources")
    if os.path.exists(cwd_resources) and os.path.isdir(cwd_resources):
        print(f"[定位] 找到resources (当前目录): {cwd_resources}")
        return os.path.abspath(cwd_resources)

    # 情况3: 让用户指定
    print(f"\n[定位] 未自动找到resources文件夹")
    print(f"查找位置: {resources_path}")
    print(f"当前工作目录: {cwd}")
    print(f"当前目录内容: {os.listdir(cwd)[:10]}...")

    while True:
        folder_path = input("\n请输入resources文件夹路径（或直接回车退出）: ").strip()
        if not folder_path:
            print("未指定resources路径，退出")
            sys.exit(1)

        # 支持相对路径和绝对路径
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(cwd, folder_path)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"[定位] 使用resources: {folder_path}")
            return os.path.abspath(folder_path)
        else:
            print(f"[错误] 路径不存在: {folder_path}")


def find_project_directory():
    """
    查找项目目录。
    - 打包环境：返回exe所在目录（代码已打包在exe内）
    - 开发环境：查找包含main.py的目录
    """
    # 获取exe/脚本所在目录
    if is_packaged_environment():
        exe_dir = os.path.dirname(sys.executable)
        print(f"[定位] 打包环境，exe目录: {exe_dir}")
        return exe_dir
    else:
        # 开发环境：查找包含main.py的目录
        exe_dir = os.path.abspath(".")

        # 情况1: 当前目录有main.py
        if os.path.exists(os.path.join(exe_dir, "main.py")):
            print(f"[定位] 开发环境，项目目录: {exe_dir}")
            return exe_dir

        # 情况2: 查找子目录
        print(f"[定位] 正在查找项目文件夹...")
        for item in os.listdir(exe_dir):
            item_path = os.path.join(exe_dir, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "main.py")):
                    print(f"[定位] 找到项目: {item}")
                    return item_path

        print(f"[错误] 未找到包含main.py的项目目录")
        sys.exit(1)


def get_resource_path(relative_path):
    """获取前端dist目录路径"""
    # 打包环境：dist已打包在exe内，从_MEIPASS获取
    if is_packaged_environment():
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        else:
            # 有些PyInstaller版本没有_MEIPASS
            return os.path.join(os.path.dirname(sys.executable), relative_path)
    else:
        # 开发环境
        project_root = find_project_directory()
        return os.path.join(project_root, relative_path)


def setup_environment():
    """设置运行环境，确保Python能找到后端模块和资源"""
    # 获取项目/exe所在目录
    base_path = find_project_directory()
    resources_path = find_resources_directory()

    print(f"[环境] 基础路径: {base_path}")
    print(f"[环境] 资源路径: {resources_path}")

    # 设置资源目录环境变量（供game_data_loader使用）
    os.environ['CFN_RESOURCES_DIR'] = resources_path

    # 在打包环境中，代码已在sys.path中，不需要额外添加
    if not is_packaged_environment():
        # 开发环境：添加到sys.path
        if base_path not in sys.path:
            sys.path.insert(0, base_path)
        for subdir in ['api', 'core', 'services', 'ai_engine', 'schemas']:
            subdir_path = os.path.join(base_path, subdir)
            if os.path.exists(subdir_path) and subdir_path not in sys.path:
                sys.path.insert(0, subdir_path)
        os.environ['PYTHONPATH'] = os.pathsep.join(sys.path[:5])

    return base_path


def start_backend():
    """启动后端FastAPI服务"""
    try:
        import uvicorn

        print("\n[后端] 正在启动 FastAPI 服务...")
        print("[后端] 服务地址: http://127.0.0.1:7077")
        print("-" * 50)

        # 设置环境
        base_path = setup_environment()

        print(f"[后端] 工作目录: {os.getcwd()}")

        # 清除可能的模块缓存（开发环境需要）
        if not is_packaged_environment():
            modules_to_clear = [key for key in sys.modules.keys() if key in ['main', 'api', 'core', 'services', 'ai_engine']]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]

        # 尝试导入main模块
        try:
            import main
            print("[后端] 成功导入main模块 ✓")
        except Exception as import_error:
            print(f"[后端] 导入main模块失败: {import_error}")
            print("\n[调试信息]")
            print(f"sys.path前5项: {sys.path[:5]}")
            import traceback
            traceback.print_exc()
            return

        # 启动uvicorn服务
        print("[后端] 正在启动Uvicorn服务器...")
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=7077,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"[后端] 启动失败: {e}")
        import traceback
        traceback.print_exc()


def start_frontend():
    """启动前端静态文件服务"""
    try:
        # 等待后端服务真正就绪（通过检查端口是否可连接）
        print("\n[前端] 等待后端服务就绪...")
        import socket
        backend_ready = False
        for _ in range(30):  # 最多等待30秒
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                result = sock.connect_ex(('127.0.0.1', 7077))
                sock.close()
                if result == 0:
                    backend_ready = True
                    print("[前端] 后端服务已就绪 ✓")
                    break
            except:
                pass
            time.sleep(1)

        if not backend_ready:
            print("[前端] 警告: 后端服务可能未完全就绪，继续启动前端...")

        time.sleep(0.5)  # 额外缓冲时间
        print("\n[前端] 正在启动静态文件服务...")

        # 获取dist目录路径（从打包资源或开发目录）
        dist_path = get_resource_path("dist")

        if not os.path.exists(dist_path):
            print(f"[前端] 错误: 未找到 dist 目录: {dist_path}")
            print(f"[前端] 请确保dist文件夹存在（前端构建产物）")
            return

        print(f"[前端] 静态文件目录: {dist_path}")

        # 直接使用Python内置服务器（简单可靠）
        start_builtin_server(dist_path)

    except Exception as e:
        print(f"[前端] 启动失败: {e}")
        import traceback
        traceback.print_exc()




class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """自定义请求处理器，正确处理MIME类型，并抑制连接错误输出"""

    extensions_map = {
        '': 'application/octet-stream',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.svg': 'image/svg+xml',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.ico': 'image/x-icon',
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.otf': 'font/otf',
        '.eot': 'application/vnd.ms-fontobject',
    }

    def end_headers(self):
        # 添加CORS头，允许跨域
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def guess_type(self, path):
        """根据扩展名返回正确的MIME类型"""
        import mimetypes

        # 先使用父类的方法
        mime_type, _ = mimetypes.guess_type(path)

        # 如果父类返回None或text/plain（Windows上的常见问题），使用自定义映射
        ext = os.path.splitext(path)[1].lower()

        if ext in self.extensions_map:
            return self.extensions_map[ext]

        return mime_type or 'application/octet-stream'

    def log_message(self, format, *args):
        """重写日志方法，过滤掉常见的连接错误日志"""
        # 过滤掉连接重置/中断的错误信息，避免污染控制台输出
        message = format % args
        if any(err in message for err in ['ConnectionAborted', 'ConnectionReset', 'Broken pipe']):
            return
        print(f"[前端] {message}")

    def handle_one_request(self):
        """重写请求处理方法，捕获连接错误"""
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # 客户端主动关闭连接，这是正常的浏览器行为，忽略错误
            pass


def start_builtin_server(dist_path):
    """使用Python内置服务器，带正确的MIME类型处理（多线程版本）"""
    import socket

    os.chdir(dist_path)
    handler = CustomHTTPRequestHandler

    for port in range(7080, 7090):
        try:
            # 检查端口是否被占用
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()

            if result == 0:
                print(f"[前端] 端口 {port} 被占用，尝试下一个...")
                continue

            # 使用ThreadingTCPServer替代TCPServer，支持并发处理多个请求
            with socketserver.ThreadingTCPServer(("", port), handler) as httpd:
                print(f"[前端] 服务地址: http://127.0.0.1:{port}")
                print("-" * 50)

                url = f"http://127.0.0.1:{port}"
                print(f"[系统] 正在打开浏览器: {url}")
                # 增加等待时间，确保服务器完全启动
                time.sleep(1.5)
                webbrowser.open(url)

                print("\n" + "=" * 50)
                print("服务启动完成！")
                print("后端API: http://127.0.0.1:7077")
                print(f"前端页面: {url}")
                print("=" * 50 + "\n")

                httpd.serve_forever()
                break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"[前端] 端口 {port} 被占用，尝试下一个...")
                continue
            print(f"[前端] 启动失败: {e}")
            raise


def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print("\n\n正在关闭服务...")
    sys.exit(0)


def main():
    """主函数"""
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 禁用输出缓冲，确保print立即显示（解决Windows下黑屏问题）
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 50, flush=True)
    print("CFN-RAG 启动器", flush=True)
    print("=" * 50, flush=True)

    # 检查Python环境
    if not check_python_environment():
        print("\nPython环境检查未通过，请修复后重试", flush=True)
        input("按回车键退出...")
        sys.exit(1)

    print("\n" + "=" * 50, flush=True)
    print("正在启动服务...", flush=True)
    print("=" * 50, flush=True)

    # 创建并启动后端线程
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()

    # 创建并启动前端线程
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    frontend_thread.start()

    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
