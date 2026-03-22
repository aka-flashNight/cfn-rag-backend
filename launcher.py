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
import urllib.request
import urllib.error

# 打包模式下由 tk 状态窗使用；关闭窗口时 shutdown 前端 HTTP 服务
_httpd_instance = None
_splash_root = None
_splash_main_label = None
_splash_sub_label = None




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


# 打包 exe 单实例：进程持有期间不关闭，退出时由系统回收
_single_instance_mutex_handle = None


def _show_packaged_notice(message: str, title: str = "CFN-RAG"):
    """无控制台 exe 下用系统对话框提示"""
    if os.name == "nt":
        try:
            import ctypes

            MB_ICONINFORMATION = 0x40
            ctypes.windll.user32.MessageBoxW(None, message, title, MB_ICONINFORMATION)
        except Exception:
            print(message, flush=True)
    else:
        print(message, flush=True)


def _try_acquire_packaged_single_instance() -> bool:
    """
    打包环境下确保全局单实例（Windows 命名互斥体）。
    返回 True 表示可继续启动；False 表示已有同程序实例在运行。
    """
    global _single_instance_mutex_handle
    if not is_packaged_environment():
        return True
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetLastError(0)
        kernel32.CreateMutexW.argtypes = [
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        ERROR_ALREADY_EXISTS = 183
        name = "Global\\CFN-RAG-Launcher-SingleInstance-v1"
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            return True
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        _single_instance_mutex_handle = handle
        return True
    except Exception:
        return True


def _tcp_local_port_open(port: int) -> bool:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.25)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def _packaged_ports_suggest_already_running() -> bool:
    """7077（API）与 7080（默认前端）均可连时，高度疑似本程序栈已在运行（含旧版无互斥体）。"""
    if not is_packaged_environment():
        return False
    return _tcp_local_port_open(7077) and _tcp_local_port_open(7080)


def _configure_stdio_line_buffering():
    """尽量行缓冲；无控制台（windowed exe）时忽略失败"""
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass


def _stream_is_usable(stream):
    if stream is None:
        return False
    try:
        stream.write("")
        stream.flush()
        return True
    except Exception:
        return False


def _ensure_stdio_for_windowed():
    """
    PyInstaller windowed 模式下 stdout/stderr 常为 None，print 与 uvicorn 日志会立即抛错，
    后端线程静默失败，表现为一直「等待后端就绪」。重定向到 NUL，不写入磁盘日志。
    """
    if not is_packaged_environment():
        return
    if _stream_is_usable(sys.stdout) and _stream_is_usable(sys.stderr):
        return
    try:
        dn = open(os.devnull, "w", encoding="utf-8", errors="replace")
        sys.stdout = dn
        sys.stderr = dn
    except Exception:
        pass


def _find_launcher_icon_path():
    """scripts/icon.ico：开发目录或打包 _MEIPASS / exe 旁"""
    candidates = []
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "scripts", "icon.ico"))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "scripts", "icon.ico"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "scripts", "icon.ico"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _apply_launcher_window_chrome(tk_root, width, height):
    """固定窗口大小、居中、任务栏图标"""
    tk_root.resizable(False, False)
    tk_root.geometry(f"{width}x{height}")
    tk_root.minsize(width, height)
    tk_root.maxsize(width, height)
    tk_root.update_idletasks()
    sw = tk_root.winfo_screenwidth()
    sh = tk_root.winfo_screenheight()
    x = max(0, (sw - width) // 2)
    y = max(0, (sh - height) // 2)
    tk_root.geometry(f"{width}x{height}+{x}+{y}")
    icon_path = _find_launcher_icon_path()
    if icon_path and os.name == "nt":
        try:
            tk_root.iconbitmap(default=icon_path)
        except Exception:
            pass


def _close_pyi_splash_if_any():
    """关闭 PyInstaller onefile 解压阶段闪屏（若存在）"""
    try:
        import pyi_splash  # type: ignore

        pyi_splash.close()
    except Exception:
        pass


def _exit_packaged_early_notice(message: str) -> None:
    """重复启动等早退：先关闪屏再弹窗，避免 Splash 挡住 MessageBox。"""
    _close_pyi_splash_if_any()
    _show_packaged_notice(message)
    sys.exit(1)


def splash_update(main_text, sub_text=""):
    """从任意线程更新打包模式下的状态窗（通过 tk after 切回主线程）"""
    global _splash_root, _splash_main_label, _splash_sub_label
    root = _splash_root
    main_lbl = _splash_main_label
    sub_lbl = _splash_sub_label
    if not root or not main_lbl:
        return

    def apply():
        main_lbl.config(text=main_text)
        if sub_lbl is not None:
            sub_lbl.config(text=sub_text)

    try:
        root.after(0, apply)
    except Exception:
        pass


def shutdown_launcher():
    """停止前端静态服务并退出进程（后端线程为 daemon，随进程结束）"""
    global _httpd_instance
    h = _httpd_instance
    if h is not None:
        try:
            h.shutdown()
        except Exception:
            pass
    os._exit(0)


def find_resources_directory():
    """
    查找游戏资源数据目录（原 resources 文件夹；若无则尝试 CrazyFlashNight）。
    优先使用与 services.game_data.paths.find_resources_directory 相同的自动推断；
    失败时进入交互式输入。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not is_packaged_environment() and project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from services.game_data.paths import find_resources_directory as _auto_find_resources

        p = _auto_find_resources()
        print(f"[定位] 找到游戏资源目录: {p}")
        return os.path.abspath(str(p))
    except FileNotFoundError:
        pass

    # 自动推断失败：让用户指定
    cwd = os.path.abspath(".")
    print("\n[定位] 未自动找到游戏资源目录（已尝试 resources 与 CrazyFlashNight）")
    print(f"当前工作目录: {cwd}")
    try:
        print(f"当前目录内容: {os.listdir(cwd)[:10]}...")
    except OSError:
        pass

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

    # 设置资源目录环境变量（供 game_data_loader 使用；向量索引在 resources/tools/vector_index）
    # 打包时默认不重建向量库以节省时间，需要刷新索引时: python scripts/build_exe.py --rebuild-vector
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

        # 清除可能的模块缓存（开发环境需要）
        if not is_packaged_environment():
            modules_to_clear = [key for key in sys.modules.keys() if key in ['main', 'api', 'core', 'services', 'ai_engine']]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]

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
    """启动前端静态文件服务（不等待后端，立即启动）"""
    try:
        print("\n[前端] 正在启动静态文件服务...")

        dist_path = get_resource_path("dist")

        if not os.path.exists(dist_path):
            print(f"[前端] 错误: 未找到 dist 目录: {dist_path}")
            print(f"[前端] 请确保dist文件夹存在（前端构建产物）")
            return

        print(f"[前端] 静态文件目录: {dist_path}")

        start_builtin_server(dist_path)

    except Exception as e:
        print(f"[前端] 启动失败: {e}")
        import traceback
        traceback.print_exc()




# 后端 API 地址，前端收到的 /api 请求会转发到此地址以规避跨域
BACKEND_PROXY_TARGET = "http://127.0.0.1:7077"
API_PREFIX = "/api"
# 普通接口代理超时（秒），至少 2 分钟
PROXY_TIMEOUT = 120
# 长时间任务接口（如立绘导出）代理超时（秒），15 分钟，满足 10min 以上
PROXY_TIMEOUT_LONG = 900
# 需要 15 分钟长超时的路径（不含 query），立绘导出、重置向量库等
LONG_TIMEOUT_PATHS = (
    API_PREFIX + "/assets/export-illustrations",
    API_PREFIX + "/game/knowledge-base/reset",
)


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """自定义请求处理器：静态文件 + /api 请求转发到后端 7077，正确处理 MIME，抑制连接错误"""

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
        '.webp': 'image/webp',
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

    def _should_proxy_to_backend(self):
        """请求路径是否应转发到后端（避免前端跨域）"""
        return self.path.startswith(API_PREFIX)

    def _proxy_to_backend(self):
        """将当前请求转发到后端 7077，并把响应写回客户端"""
        # 构建后端 URL（path + query）
        path = self.path.split("?", 1)
        url = BACKEND_PROXY_TARGET + path[0]
        if len(path) == 2:
            url += "?" + path[1]

        try:
            # 读取请求体（POST/PUT/PATCH）
            body = None
            if self.command in ("POST", "PUT", "PATCH"):
                content_length = self.headers.get("Content-Length")
                if content_length:
                    body = self.rfile.read(int(content_length))

            # 构造转发请求
            req_headers = {}
            for key, value in self.headers.items():
                key_lower = key.lower()
                if key_lower in ("host", "connection"):  # 不转发 Host，避免后端拒绝
                    continue
                req_headers[key] = value

            req = urllib.request.Request(
                url,
                data=body,
                headers=req_headers,
                method=self.command,
            )
            path_normalized = path[0].rstrip("/")
            timeout = PROXY_TIMEOUT_LONG if path_normalized in {p.rstrip("/") for p in LONG_TIMEOUT_PATHS} else PROXY_TIMEOUT
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            resp = e  # 4xx/5xx 也有 read() 和 headers
        except (urllib.error.URLError, OSError) as e:
            # HTTP status phrase 必须为 latin-1，中文会导致 UnicodeEncodeError
            detail = e.reason if hasattr(e, "reason") and e.reason else e
            print(f"[前端] 代理失败 /api -> 7077: {detail}", flush=True)
            self.send_error(502, "Bad Gateway")
            return

        # 判断是否为流式响应（如 SSE），需边收边转不能整段缓冲
        content_type = (resp.headers.get("Content-Type") or "").lower()
        is_streaming = "text/event-stream" in content_type

        self.send_response(resp.status)
        # 转发响应头（去掉 hop-by-hop 等）；流式时不转发 Content-Length，让客户端按流读取
        skip_headers = {"transfer-encoding", "connection", "content-encoding"}
        if is_streaming:
            skip_headers = skip_headers | {"content-length"}
        for name, value in resp.headers.items():
            if name.lower() in skip_headers:
                continue
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            if is_streaming:
                # 不缓冲：每次读 1 字节并立即写回，后端来多少就转多少（仅 OS/内核仍有缓冲，应用层零缓冲）
                while True:
                    chunk = resp.read(1)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                self.wfile.write(resp.read())

    def do_GET(self):
        if self._should_proxy_to_backend():
            self._proxy_to_backend()
        else:
            super().do_GET()

    def do_HEAD(self):
        if self._should_proxy_to_backend():
            self._proxy_to_backend()
        else:
            super().do_HEAD()

    def do_POST(self):
        if self._should_proxy_to_backend():
            self._proxy_to_backend()
        else:
            self.send_error(405, "Method Not Allowed")

    def do_PUT(self):
        if self._should_proxy_to_backend():
            self._proxy_to_backend()
        else:
            self.send_error(405, "Method Not Allowed")

    def do_DELETE(self):
        if self._should_proxy_to_backend():
            self._proxy_to_backend()
        else:
            self.send_error(405, "Method Not Allowed")

    def do_PATCH(self):
        if self._should_proxy_to_backend():
            self._proxy_to_backend()
        else:
            self.send_error(405, "Method Not Allowed")


def open_browser_when_ready(frontend_port):
    """在独立线程中等待后端就绪后再打开浏览器（打包后 import 较慢，需足够等待）"""
    import socket

    url = f"http://127.0.0.1:{frontend_port}"
    # 端口可连上后再稍等，降低首请求撞上 lifespan 尚未完成的概率
    BROWSER_OPEN_DELAY = 1.0
    poll_interval = 0.5
    # 约 30 秒上限（与界面文案一致）；后端正常时通常数秒内就绪
    max_attempts = 60

    splash_update(
        "正在等待后端就绪…",
        "请勿关闭启动器窗口。若超过约 30 秒仍无响应，可关闭后重新运行或改用源码启动查看控制台输出。",
    )

    ready = False
    for attempt in range(max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", 7077))
            sock.close()
            if result == 0:
                print("[系统] 后端服务已就绪 ✓")
                splash_update("后端已就绪，即将打开浏览器…", "")
                time.sleep(BROWSER_OPEN_DELAY)
                ready = True
                break
        except Exception:
            pass
        time.sleep(poll_interval)
        if attempt > 0 and attempt % 10 == 0:
            elapsed = attempt * poll_interval
            splash_update(
                "正在等待后端就绪…",
                f"已等待约 {elapsed:.0f} 秒（最多约 30 秒）…",
            )
            print(
                f"[系统] 仍在等待后端监听 7077... ({elapsed:.0f}s)",
                flush=True,
            )

    if not ready:
        print(
            "[系统] 警告: 在超时内未检测到后端 7077 端口，仍将打开浏览器；"
            "若页面 API 失败请刷新或稍后重试。",
            flush=True,
        )
        splash_update(
            "后端在预期时间内未就绪",
            "仍将尝试打开浏览器。若页面异常请刷新浏览器页面或重新运行本程序。",
        )

    print(f"[系统] 正在打开浏览器: {url}")
    webbrowser.open(url)

    splash_update(
        "服务运行中……",
        f"浏览器已打开页面，也可手动输入网址 127.0.0.1:{frontend_port} 访问。\n如需停止服务，可关闭本窗口。",
    )

    print("\n" + "=" * 50)
    print("服务启动完成！")
    print("后端API: http://127.0.0.1:7077")
    print(f"前端页面: {url}")
    print("=" * 50 + "\n")


def start_builtin_server(dist_path):
    """使用Python内置服务器，带正确的MIME类型处理（多线程版本）"""
    import socket
    from functools import partial

    handler = partial(CustomHTTPRequestHandler, directory=dist_path)

    for port in range(7080, 7090):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()

            if result == 0:
                print(f"[前端] 端口 {port} 被占用，尝试下一个...")
                continue

            global _httpd_instance
            httpd = socketserver.ThreadingTCPServer(("", port), handler)
            _httpd_instance = httpd
            try:
                print(f"[前端] 服务地址: http://127.0.0.1:{port}")
                print("-" * 50)

                browser_thread = threading.Thread(
                    target=open_browser_when_ready,
                    args=(port,),
                    daemon=True,
                )
                browser_thread.start()

                httpd.serve_forever()
            finally:
                _httpd_instance = None
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


def main_console():
    """开发/源码运行：保留控制台日志"""
    signal.signal(signal.SIGINT, signal_handler)
    _configure_stdio_line_buffering()

    print("=" * 50, flush=True)
    print("CFN-RAG 启动器", flush=True)
    print("=" * 50, flush=True)

    if not check_python_environment():
        print("\nPython环境检查未通过，请修复后重试", flush=True)
        input("按回车键退出...")
        sys.exit(1)

    print("\n[环境] 正在设置运行环境...", flush=True)
    setup_environment()

    print("\n" + "=" * 50, flush=True)
    print("正在启动服务...", flush=True)
    print("=" * 50, flush=True)

    backend_thread = threading.Thread(target=start_backend, daemon=True)
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    backend_thread.start()
    frontend_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


def main_packaged_gui():
    """打包 exe（无控制台）：轻量状态窗 + 关闭即退出"""
    import tkinter as tk
    from tkinter import messagebox

    global _splash_root, _splash_main_label, _splash_sub_label

    signal.signal(signal.SIGINT, lambda s, f: shutdown_launcher())
    _configure_stdio_line_buffering()

    win_w, win_h = 440, 200
    text_wrap = win_w - 48

    root = tk.Tk()
    root.title("CFN-RAG")
    _apply_launcher_window_chrome(root, win_w, win_h)

    frame = tk.Frame(root, padx=24, pady=18)
    frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(frame, text="CFN-RAG", font=("Segoe UI", 13, "bold")).pack()
    main_lbl = tk.Label(
        frame,
        text="正在启动本地服务…",
        justify=tk.CENTER,
        fg="#222",
        wraplength=text_wrap,
    )
    main_lbl.pack(pady=(12, 0))
    sub_lbl = tk.Label(
        frame,
        text="请勿关闭本窗口。",
        justify=tk.CENTER,
        fg="#555",
        font=("Segoe UI", 9),
        wraplength=text_wrap,
    )
    sub_lbl.pack(pady=(8, 0))

    _splash_root = root
    _splash_main_label = main_lbl
    _splash_sub_label = sub_lbl

    root.update_idletasks()
    root.update()
    _close_pyi_splash_if_any()

    if not check_python_environment():
        _splash_root = None
        _splash_main_label = None
        _splash_sub_label = None
        messagebox.showerror(
            "CFN-RAG",
            "运行环境检查未通过。\n若使用源码，请安装 requirements.txt 后重试。",
        )
        sys.exit(1)

    splash_update("正在配置运行环境…", "请稍候")
    root.update()

    try:
        setup_environment()
    except SystemExit:
        raise
    except Exception as e:
        _splash_root = None
        _splash_main_label = None
        _splash_sub_label = None
        messagebox.showerror("CFN-RAG", f"环境配置失败：\n{e}")
        sys.exit(1)

    def on_quit():
        if messagebox.askokcancel(
            "退出 CFN-RAG",
            "关闭本窗口将停止本地服务，已打开的网页将无法继续访问。\n\n确定退出吗？",
        ):
            shutdown_launcher()

    root.protocol("WM_DELETE_WINDOW", on_quit)

    splash_update("正在启动后端与网页服务…", "首次启动可能较慢，请耐心等待")
    root.update()

    backend_thread = threading.Thread(target=start_backend, daemon=True)
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    backend_thread.start()
    frontend_thread.start()

    root.mainloop()
    shutdown_launcher()


def main():
    if is_packaged_environment():
        _ensure_stdio_for_windowed()
        if not _try_acquire_packaged_single_instance():
            _exit_packaged_early_notice(
                "检测到 CFN-RAG 已在运行。\n\n请勿重复启动；"
                "若需重启请先关闭已打开的启动器窗口。",
            )
        if _packaged_ports_suggest_already_running():
            _exit_packaged_early_notice(
                "本机 7077 与 7080 端口均已被占用，"
                "可能已有 CFN-RAG 或其它程序在使用相同端口。\n\n"
                "请先关闭已运行的实例或释放端口后再启动。",
            )
        main_packaged_gui()
    else:
        main_console()


if __name__ == "__main__":
    main()
