"""SPA 伺服测试（B1-1）：7077 同源伺服前端，/api 路由优先，未知路径回退 index.html。"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from main import create_app

    return TestClient(create_app())


def test_root_serves_spa_index():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert '<div id="app">' in resp.text


def test_spa_asset_served():
    client = _client()
    index = client.get("/").text
    match = re.search(r'src="\./(assets/[^"]+\.js)"', index)
    assert match, "index.html 应包含构建产物入口脚本"
    resp = client.get("/" + match.group(1))
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]


def test_spa_route_fallback_to_index():
    resp = _client().get("/some/spa/route")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert '<div id="app">' in resp.text


def test_api_route_takes_precedence():
    resp = _client().get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_unknown_api_path_not_fallback_to_index():
    resp = _client().get("/api/definitely-not-exist")
    assert resp.status_code == 404
    assert "text/html" not in resp.headers.get("content-type", "")
