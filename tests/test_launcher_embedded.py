"""launcher --embedded 分支测试（B1-3）：静默约束与端口占用静默退出。"""

from __future__ import annotations

import os
import sys

import pytest

import launcher


def test_embedded_launch_is_silent(monkeypatch):
    calls: dict[str, object] = {}
    logs: list[str] = []

    monkeypatch.setattr(launcher, "_ensure_stdio_for_windowed", lambda: calls.setdefault("stdio", True))
    monkeypatch.setattr(launcher, "_configure_stdio_line_buffering", lambda: None)
    monkeypatch.setattr(launcher, "_embedded_log", logs.append)
    monkeypatch.setattr(launcher, "_close_pyi_splash_if_any", lambda: calls.setdefault("splash", True))
    monkeypatch.setattr(launcher, "_tcp_local_port_open", lambda port: False)
    monkeypatch.setattr(launcher, "check_python_environment", lambda: True)
    monkeypatch.setattr(launcher, "setup_environment", lambda: calls.setdefault("setup", True))
    monkeypatch.setattr(launcher, "start_backend", lambda: calls.setdefault("backend", True))
    monkeypatch.setattr(launcher, "start_frontend", lambda: calls.setdefault("frontend", True))
    monkeypatch.setattr(launcher, "main_packaged_gui", lambda: calls.setdefault("gui", True))
    monkeypatch.setattr(
        launcher, "_notify_browser_opened_user_attention", lambda: calls.setdefault("notify", True)
    )
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: calls.setdefault("browser", True))
    monkeypatch.setattr(launcher, "_LAUNCH_MODE", "standalone")
    monkeypatch.setattr(launcher, "_backend_shutdown_requested", False)
    monkeypatch.setenv("CFN_RAG_LAUNCH_MODE", "placeholder")
    monkeypatch.setattr(sys, "argv", ["CFN-RAG-v3.0.0.exe", "--embedded"])

    # start_backend 被打桩立即返回（真实实现阻塞至进程结束）→ 走「后端未能启动」退出路径
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 1

    assert calls.get("backend") is True
    assert "setup" in calls and "splash" in calls
    for silent_key in ("frontend", "gui", "browser", "notify"):
        assert silent_key not in calls, f"embedded 模式不应触发 {silent_key}"
    assert os.environ["CFN_RAG_LAUNCH_MODE"] == "embedded"
    assert launcher._LAUNCH_MODE == "embedded"
    assert any("后端未能启动" in msg for msg in logs)


def test_embedded_exits_quietly_when_port_busy(monkeypatch):
    calls: dict[str, object] = {}
    logs: list[str] = []

    monkeypatch.setattr(launcher, "_ensure_stdio_for_windowed", lambda: None)
    monkeypatch.setattr(launcher, "_configure_stdio_line_buffering", lambda: None)
    monkeypatch.setattr(launcher, "_embedded_log", logs.append)
    monkeypatch.setattr(launcher, "_tcp_local_port_open", lambda port: True)
    monkeypatch.setattr(launcher, "check_python_environment", lambda: calls.setdefault("check", True))
    monkeypatch.setattr(launcher, "start_backend", lambda: calls.setdefault("backend", True))
    monkeypatch.setattr(launcher, "_LAUNCH_MODE", "standalone")
    monkeypatch.setenv("CFN_RAG_LAUNCH_MODE", "placeholder")
    monkeypatch.setattr(sys, "argv", ["CFN-RAG-v3.0.0.exe", "--embedded"])

    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 0

    assert "backend" not in calls and "check" not in calls
    assert any("已在监听" in msg for msg in logs)


def test_embedded_graceful_shutdown_exits_zero(monkeypatch):
    logs: list[str] = []

    monkeypatch.setattr(launcher, "_ensure_stdio_for_windowed", lambda: None)
    monkeypatch.setattr(launcher, "_configure_stdio_line_buffering", lambda: None)
    monkeypatch.setattr(launcher, "_embedded_log", logs.append)
    monkeypatch.setattr(launcher, "_close_pyi_splash_if_any", lambda: None)
    monkeypatch.setattr(launcher, "_tcp_local_port_open", lambda port: False)
    monkeypatch.setattr(launcher, "check_python_environment", lambda: True)
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(launcher, "_LAUNCH_MODE", "standalone")
    monkeypatch.setattr(launcher, "_backend_shutdown_requested", False)
    monkeypatch.setenv("CFN_RAG_LAUNCH_MODE", "placeholder")
    monkeypatch.setattr(sys, "argv", ["CFN-RAG-v3.0.0.exe", "--embedded"])

    def fake_start_backend():
        # 模拟 /api/integration/shutdown 回调置位后 uvicorn 事件循环返回
        launcher._backend_shutdown_requested = True

    monkeypatch.setattr(launcher, "start_backend", fake_start_backend)

    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 0
    assert any("游戏侧退出请求" in msg for msg in logs)


def test_standalone_launch_untouched(monkeypatch):
    monkeypatch.setattr(launcher, "_LAUNCH_MODE", "standalone")
    monkeypatch.setattr(launcher, "is_packaged_environment", lambda: False)
    monkeypatch.setattr(launcher, "main_console", lambda: None)
    monkeypatch.setattr(sys, "argv", ["CFN-RAG-v3.0.0.exe"])

    launcher.main()

    assert launcher._LAUNCH_MODE == "standalone"
