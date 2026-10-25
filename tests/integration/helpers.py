"""集成测试辅助：临时 MemoryStore 与真实 Uvicorn 线程。"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from typing import Iterator

import uvicorn


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def run_server(app, *, startup_timeout: float = 10.0) -> Iterator[str]:
    """在后台线程跑真实 Uvicorn，返回 base URL；退出时优雅关闭。"""
    port = free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="off",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + startup_timeout
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("Uvicorn 未能在超时内启动")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
