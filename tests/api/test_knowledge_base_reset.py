"""知识库重置端点：`/api/game/knowledge-base/reset` 的导入路径与建库接线。

端点在请求体内取 `load_corpus` / `compute_corpus_fingerprint`，取的是子模块
`services.retrieval.loader`；包根 `services.retrieval` 从未再导出这两个名字，早先按包根取
会让按钮一点就把 ImportError 当失败信息回给前端（2026-10-25 修）。
语料装配与建库全部打桩，不碰真实索引。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.game_api as game_api
import services.retrieval as retrieval_pkg
import services.retrieval.loader as retrieval_loader


class _StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list, str]] = []

    def build_store(self, nodes, expected_fingerprint):
        self.calls.append((nodes, expected_fingerprint))
        return object()


@pytest.fixture
def engine() -> _StubEngine:
    return _StubEngine()


@pytest.fixture
def client(monkeypatch, engine) -> TestClient:
    monkeypatch.setattr(retrieval_pkg, "get_retrieval_engine", lambda: engine)
    monkeypatch.setattr(retrieval_loader, "compute_corpus_fingerprint", lambda: "fp-42")
    monkeypatch.setattr(retrieval_loader, "load_corpus", lambda: ["n1", "n2", "n3"])
    app = FastAPI()
    app.include_router(game_api.router, prefix="/api/game")
    return TestClient(app)


def test_reset_uses_loader_names_and_passes_fingerprint(client, engine):
    resp = client.post("/api/game/knowledge-base/reset")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "3" in (body["message"] or "")
    assert engine.calls == [(["n1", "n2", "n3"], "fp-42")]


def test_reset_reports_build_failure_reason(client, engine, monkeypatch):
    def _boom(nodes, expected_fingerprint):
        raise RuntimeError("语料解析失败")

    monkeypatch.setattr(engine, "build_store", _boom)

    body = client.post("/api/game/knowledge-base/reset").json()
    assert body["success"] is False
    assert "语料解析失败" in (body["message"] or "")
