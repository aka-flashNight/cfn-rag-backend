"""B2-7 刷新模块：指纹失配→后台重建；未就绪/无差异 no-op；折叠触发与失败复位。

用假引擎 + 假指纹/语料（monkeypatch loader 模块属性），不触碰全局单例与真实资源。
"""

from __future__ import annotations

import threading
import time

import pytest

from services.retrieval import loader as loader_mod
from services.retrieval import refresh as refresh_mod


class _Store:
    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint


class _FakeEngine:
    """store=None 未就绪；build_store 记录并更新 store，可注入阻塞/抛错回调。"""

    def __init__(self, fingerprint: str | None = None) -> None:
        self.store = _Store(fingerprint) if fingerprint is not None else None
        self.builds: list[str] = []
        self.on_build = None  # Callable[[], None]，可阻塞或抛错
        self._build_calls = 0

    def build_store(self, nodes, expected_fingerprint: str) -> None:
        self._build_calls += 1
        if self.on_build is not None:
            self.on_build()
        self.builds.append(expected_fingerprint)
        self.store = _Store(expected_fingerprint)


def _wait_until(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.fixture(autouse=True)
def _reset_refresh_state():
    def _reset():
        with refresh_mod._refresh_lock:
            refresh_mod._refreshing = False
            refresh_mod._folded_trigger = False

    _reset()
    yield
    _reset()


@pytest.fixture
def engine_holder(monkeypatch):
    """安装假引擎与假指纹：holder["fp"] 控制当前语料指纹。"""
    holder = {"fp": "fp1", "calls": 0}

    def _compute() -> str:
        holder["calls"] += 1
        return holder["fp"]

    monkeypatch.setattr(loader_mod, "compute_corpus_fingerprint", _compute)
    monkeypatch.setattr(loader_mod, "load_corpus", lambda: ["node"])

    engines: dict[str, _FakeEngine] = {}

    def _install(engine: _FakeEngine) -> _FakeEngine:
        engines["engine"] = engine
        from services.retrieval import hybrid as hybrid_mod

        monkeypatch.setattr(hybrid_mod, "get_retrieval_engine", lambda: engine)
        return engine

    holder["install"] = _install
    return holder


def test_not_ready_is_noop(engine_holder):
    engine = engine_holder["install"](_FakeEngine(fingerprint=None))
    assert refresh_mod.refresh_index_if_stale() is False
    assert engine.builds == []
    assert refresh_mod._refreshing is False


def test_matching_fingerprint_no_rebuild(engine_holder):
    engine = engine_holder["install"](_FakeEngine(fingerprint="fp1"))
    assert refresh_mod.refresh_index_if_stale() is False
    assert engine.builds == []


def test_stale_fingerprint_rebuilds_in_background(engine_holder):
    engine = engine_holder["install"](_FakeEngine(fingerprint="fp0"))
    assert refresh_mod.refresh_index_if_stale() is True
    assert _wait_until(lambda: engine.builds == ["fp1"])
    assert _wait_until(lambda: refresh_mod._refreshing is False)
    assert engine.store.fingerprint == "fp1"


def test_write_during_build_rebuilds_to_latest(engine_holder):
    """构建阻塞期间语料再变（触发被折叠）：循环重建至追平最新指纹。"""
    engine = _FakeEngine(fingerprint="fp0")
    gate = threading.Event()
    engine.on_build = lambda: gate.wait(timeout=5)
    engine_holder["install"](engine)

    assert refresh_mod.refresh_index_if_stale() is True
    assert _wait_until(lambda: engine._build_calls == 1)  # 首轮构建已进入（阻塞中）

    engine_holder["fp"] = "fp2"  # 构建期间写盘
    assert refresh_mod.refresh_index_if_stale() is False  # 折叠，不并发重建
    assert refresh_mod._folded_trigger is True

    engine.on_build = None
    gate.set()
    assert _wait_until(lambda: engine.builds == ["fp1", "fp2"])
    assert _wait_until(lambda: refresh_mod._refreshing is False)
    assert engine.store.fingerprint == "fp2"
    # 收尾复核（折叠触发）：指纹已一致，不再多余重建
    assert _wait_until(lambda: engine_holder["calls"] >= 6)
    assert engine.builds == ["fp1", "fp2"]


def test_build_failure_resets_and_can_retry(engine_holder):
    engine = _FakeEngine(fingerprint="fp0")
    engine.on_build = lambda: (_ for _ in ()).throw(RuntimeError("embed boom"))
    engine_holder["install"](engine)

    assert refresh_mod.refresh_index_if_stale() is True
    assert _wait_until(lambda: refresh_mod._refreshing is False)
    assert engine.builds == []  # 失败无产物，store 保持旧值
    assert engine.store.fingerprint == "fp0"

    engine.on_build = None
    assert refresh_mod.refresh_index_if_stale() is True
    assert _wait_until(lambda: engine.builds == ["fp1"])
    assert _wait_until(lambda: refresh_mod._refreshing is False)


def test_check_failure_never_raises(engine_holder, monkeypatch):
    engine_holder["install"](_FakeEngine(fingerprint="fp0"))

    def _boom() -> str:
        raise RuntimeError("fingerprint boom")

    monkeypatch.setattr(loader_mod, "compute_corpus_fingerprint", _boom)
    assert refresh_mod.refresh_index_if_stale() is False
    assert refresh_mod._refreshing is False
