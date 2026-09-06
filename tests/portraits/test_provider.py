"""图片处理测试（07 §8.2）：bounds 裁剪、长边 ≤512、缓存命中不再调 PIL、降级。"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from services.portraits import provider
from services.portraits.cache import get_cache


def _open_webp(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _decode_data_url(url: str) -> bytes:
    assert url.startswith("data:image/webp;base64,")
    return base64.b64decode(url.split(",", 1)[1], validate=True)


def test_data_url_format_and_bounds_crop(portrait_env):
    url = provider.get_portrait_data_url("Andy Law", "普通")
    assert url is not None
    im = _open_webp(_decode_data_url(url))
    # 100×80 源图按 bounds(10,20,60,40) 裁剪 → 60×40，长边 ≤512 不缩放
    assert im.size == (60, 40)


def test_long_edge_scaled_to_480(portrait_env):
    url = provider.get_portrait_data_url("清水结衣", "普通")
    assert url is not None
    im = _open_webp(_decode_data_url(url))
    # 1200×900 全幅 bounds → 长边 1200 缩到 512（512×384）
    assert im.size == (480, 360)


def test_cache_hit_skips_pil_processing(portrait_env, monkeypatch):
    calls = {"n": 0}
    real = provider._process_image

    def counting(asset, *, resize):
        calls["n"] += 1
        return real(asset, resize=resize)

    monkeypatch.setattr(provider, "_process_image", counting)
    first = provider.get_portrait_data_url("Andy Law", "微笑")
    second = provider.get_portrait_data_url("Andy Law", "微笑")
    assert first == second
    assert calls["n"] == 1  # 命中缓存不重跑 PIL（修 F2）
    provider.get_portrait_data_url("Andy Law", "普通")
    assert calls["n"] == 2  # 情绪不同 → 缓存 miss


def test_cache_key_includes_mtime(portrait_env, monkeypatch):
    calls = {"n": 0}
    real = provider._process_image

    def counting(asset, *, resize):
        calls["n"] += 1
        return real(asset, resize=resize)

    monkeypatch.setattr(provider, "_process_image", counting)
    assert provider.get_portrait_data_url("Andy Law", "微笑") is not None
    assert calls["n"] == 1

    # 模拟 manifest 烘焙再生成后源文件更新：mtime 变化 → 缓存键失效 → 重新处理
    png = portrait_env / "external" / "p_a" / "e_smile.png"
    import os

    st = png.stat()
    os.utime(png, (st.st_atime + 5, st.st_mtime + 5))
    assert provider.get_portrait_data_url("Andy Law", "微笑") is not None
    assert calls["n"] == 2  # 内容相同编码结果一致，但确实重算了（不缓存跨版本路径）


def test_hero_and_unknown_character_no_image(portrait_env):
    assert provider.get_portrait_data_url("玩家", "微笑") is None  # 主角特例
    assert provider.get_portrait_data_url("不存在的角色", "微笑") is None


def test_manifest_missing_degrades(monkeypatch):
    from services.portraits import manifest_lookup

    monkeypatch.setattr(manifest_lookup, "discover_portrait_dir", lambda: None)
    manifest_lookup.reset_portrait_lookup()
    try:
        assert provider.get_portrait_data_url("Andy Law", "普通") is None
        assert provider.get_portrait_png("Andy Law", "普通") is None
        assert provider.is_portrait_available() is False
    finally:
        manifest_lookup.reset_portrait_lookup()


def test_corrupt_source_file_degrades(portrait_env):
    # 源 PNG 损坏 → 处理异常被吞掉，返回 None（无图模式不报错，07 §1）
    png = portrait_env / "external" / "p_a" / "e_normal.png"
    png.write_bytes(b"not a png")
    get_cache().clear()
    assert provider.get_portrait_data_url("Andy Law", "普通") is None


def test_portrait_png_for_display(portrait_env):
    data = provider.get_portrait_png("Andy Law", "普通")
    assert data is not None
    im = Image.open(io.BytesIO(data))
    assert im.format == "PNG"
    assert im.size == (60, 40)  # bounds 裁剪后原分辨率（展示不缩放）


def test_build_image_message_content_layout():
    text = "玩家：你好"
    assert provider.build_image_message_content(text, None) == text
    parts = provider.build_image_message_content(text, "data:image/webp;base64,AAA")
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": text}
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAA"}}
