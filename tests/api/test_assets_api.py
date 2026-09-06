"""assets_api 测试：头像走原 profiles 目录；立绘 manifest 查表后原始文件直出（07 §6）。"""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from services.portraits import manifest_lookup
from services.portraits.cache import get_cache
from tests.portraits.conftest import build_portrait_dir


def _make_app() -> FastAPI:
    from api.assets_api import router

    app = FastAPI()
    app.include_router(router, prefix="/assets")
    return app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    portrait_dir = build_portrait_dir(tmp_path)
    monkeypatch.setattr(manifest_lookup, "discover_portrait_dir", lambda: portrait_dir)
    manifest_lookup.reset_portrait_lookup()
    get_cache().clear()
    # 头像目录：resources/flashswf/portraits/profiles/
    import services.game_data.paths as paths_mod

    profiles = tmp_path / "resources" / "flashswf" / "portraits" / "profiles"
    profiles.mkdir(parents=True)
    Image.new("RGBA", (64, 64), (128, 128, 128, 255)).save(profiles / "Andy Law.png")
    monkeypatch.setattr(paths_mod, "find_resources_directory", lambda: tmp_path / "resources")

    yield TestClient(_make_app())

    manifest_lookup.reset_portrait_lookup()
    get_cache().clear()


def test_avatar_uses_legacy_profiles_dir(client):
    """头像仍在原位置（profiles/{npc}.png 原文件直出），不经 manifest 查表。"""
    resp = client.get("/assets/avatar/Andy Law")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(resp.content))
    assert im.size == (64, 64)  # 原文件原尺寸，无裁剪


def test_avatar_missing_404(client):
    assert client.get("/assets/avatar/不存在的角色").status_code == 404


def test_illustration_serves_original_file_and_fallback(client):
    """立绘原始文件直出（无裁剪/重编码）：返回尺寸 = 源 PNG 画布尺寸。"""
    resp = client.get("/assets/illustration/Andy Law/微笑")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # 微笑源图 40×30 全幅直出（若裁剪过 bounds 也恰为全幅，用「普通」区分更严）
    assert Image.open(io.BytesIO(resp.content)).size == (40, 30)
    # 普通源图 100×80 —— 若走了裁剪会变 60×40，原始直出才是 100×80
    resp2 = client.get("/assets/illustration/Andy Law/普通")
    assert resp2.status_code == 200
    assert Image.open(io.BytesIO(resp2.content)).size == (100, 80)
    # 缺失情绪 → manifest 回退链兜底到「普通」，仍 200
    resp3 = client.get("/assets/illustration/Andy Law/愤怒")
    assert resp3.status_code == 200
    assert Image.open(io.BytesIO(resp3.content)).size == (100, 80)


def test_illustration_unknown_character_404(client):
    assert client.get("/assets/illustration/不存在的角色/微笑").status_code == 404


def test_illustration_hero_key_404(client):
    """主角无静态立绘（heroKeys 特例）→ 404。"""
    assert client.get("/assets/illustration/玩家").status_code in (404, 422)


def test_no_manifest_illustration_404_not_500(monkeypatch, tmp_path):
    """无图模式：立绘接口 404 而不是 500（不报错原则，07 §1）；头像不受影响。"""
    import services.game_data.paths as paths_mod

    monkeypatch.setattr(manifest_lookup, "discover_portrait_dir", lambda: None)
    manifest_lookup.reset_portrait_lookup()
    get_cache().clear()
    profiles = tmp_path / "resources" / "flashswf" / "portraits" / "profiles"
    profiles.mkdir(parents=True)
    Image.new("RGBA", (64, 64), (128, 128, 128, 255)).save(profiles / "Andy Law.png")
    monkeypatch.setattr(paths_mod, "find_resources_directory", lambda: tmp_path / "resources")
    try:
        client = TestClient(_make_app())
        assert client.get("/assets/illustration/Andy Law/普通").status_code == 404
        assert client.get("/assets/avatar/Andy Law").status_code == 200  # 头像不受无图模式影响
    finally:
        manifest_lookup.reset_portrait_lookup()


def test_legacy_export_endpoint_removed():
    """旧「立绘生成/解压」接口已按 07 §7 删除。"""
    import api.assets_api as mod

    assert not hasattr(mod, "export_illustrations")
    assert not hasattr(mod, "_extract_illustration_zip")
    routes = {r.path for r in mod.router.routes}
    assert all("export" not in p for p in routes)
