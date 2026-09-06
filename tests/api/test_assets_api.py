"""assets_api 立绘接口测试（07 §6）：manifest 查表返回裁剪 PNG，旧导出/解压接口已删。"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.portraits import manifest_lookup
from services.portraits.cache import get_cache
from tests.portraits.conftest import build_portrait_dir


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    portrait_dir = build_portrait_dir(tmp_path)
    monkeypatch.setattr(manifest_lookup, "discover_portrait_dir", lambda: portrait_dir)
    manifest_lookup.reset_portrait_lookup()
    get_cache().clear()

    from api.assets_api import router

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/assets")
    yield TestClient(app)

    manifest_lookup.reset_portrait_lookup()
    get_cache().clear()


def test_avatar_returns_cropped_png(client):
    resp = client.get("/assets/avatar/Andy Law")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(resp.content))
    assert im.size == (60, 40)  # bounds(10,20,60,40) 裁剪


def test_illustration_with_emotion_and_fallback(client):
    resp = client.get("/assets/illustration/Andy Law/微笑")
    assert resp.status_code == 200
    assert Image.open(io.BytesIO(resp.content)).size == (40, 30)
    # 缺失情绪 → manifest 回退链兜底到「普通」，仍 200
    resp2 = client.get("/assets/illustration/Andy Law/愤怒")
    assert resp2.status_code == 200
    assert Image.open(io.BytesIO(resp2.content)).size == (60, 40)


def test_unknown_character_404(client):
    assert client.get("/assets/avatar/不存在的角色").status_code == 404
    assert client.get("/assets/illustration/不存在的角色/微笑").status_code == 404


def test_hero_key_404(client):
    """主角无静态立绘（heroKeys 特例）→ 404。"""
    assert client.get("/assets/avatar/玩家").status_code == 404


def test_no_manifest_404_not_500(monkeypatch):
    """无图模式：接口 404 而不是 500（不报错原则，07 §1）。"""
    monkeypatch.setattr(manifest_lookup, "discover_portrait_dir", lambda: None)
    manifest_lookup.reset_portrait_lookup()
    get_cache().clear()
    try:
        from api.assets_api import router

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/assets")
        client = TestClient(app)
        assert client.get("/assets/avatar/Andy Law").status_code == 404
    finally:
        manifest_lookup.reset_portrait_lookup()


def test_legacy_export_endpoint_removed():
    """旧「立绘生成/解压」接口已按 07 §7 删除。"""
    import api.assets_api as mod

    assert not hasattr(mod, "export_illustrations")
    assert not hasattr(mod, "_extract_illustration_zip")
    routes = {r.path for r in mod.router.routes}
    assert all("export" not in p for p in routes)
