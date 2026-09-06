"""portraits 测试夹具：临时 manifest + 小 PNG（07 §8 测试资产）。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterator

import pytest
from PIL import Image

from services.portraits import cache as portraits_cache
from services.portraits import manifest_lookup


def make_png_bytes(size: tuple[int, int] = (100, 80), color=(255, 0, 0, 255)) -> bytes:
    """生成测试用 RGBA PNG 字节（纯色）。"""
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, "PNG")
    return buf.getvalue()


def _write_png(path: Path, size: tuple[int, int], color) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, "PNG")
    path.write_bytes(buf.getvalue())
    return path.stat().st_mtime


def build_portrait_dir(root: Path) -> Path:
    """构造与真实布局一致的假游戏项目（manifest 位于
    <项目根>/launcher/web/assets/dialogue-portraits/）。

    - Andy Law（external 风格）：普通 100×80 bounds(10,20,60,40)；微笑 40×30；
    - 清水结衣（internal 风格）：仅「普通」大图 1200×900（测长边缩放）；
    - 顶层 aliases 与条目级 aliases 重合，另测大小写/空格变体命中。
    """
    pdir = root / "CrazyFlashNight" / "launcher" / "web" / "assets" / "dialogue-portraits"
    e_normal = pdir / "external" / "p_a" / "e_normal.png"
    e_smile = pdir / "external" / "p_a" / "e_smile.png"
    i_normal = pdir / "internal" / "p_b" / "e_normal.png"

    manifest = {
        "schema": "cf7-dialogue-portraits-v2",
        "generatedAt": "2026-09-05T00:00:00+0800",
        "heroKeys": ["$PC_CHAR", "主角模板", "玩家"],
        "entries": {
            "Andy Law": {
                "key": "Andy Law",
                "aliases": ["andy law", "AndyLaw"],
                "source": "external-swf",
                "defaultExpression": "普通",
                "expressions": {
                    "普通": {
                        "uri": "external/p_a/e_normal.png",
                        "width": 100, "height": 80,
                        "bounds": {"x": 10, "y": 20, "width": 60, "height": 40},
                    },
                    "微笑": {
                        "uri": "external/p_a/e_smile.png",
                        "width": 40, "height": 30,
                        "bounds": {"x": 0, "y": 0, "width": 40, "height": 30},
                    },
                },
            },
            "清水结衣": {
                "key": "清水结衣",
                "aliases": [],
                "source": "dialogue-ui-sprite",
                "defaultExpression": "普通",
                "expressions": {
                    "普通": {
                        "uri": "internal/p_b/e_normal.png",
                        "width": 1200, "height": 900,
                        "bounds": {"x": 0, "y": 0, "width": 1200, "height": 900},
                    },
                },
            },
        },
        "aliases": {"andy law": "Andy Law", "shimizu": "清水结衣"},
    }
    (pdir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (pdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _write_png(e_normal, (100, 80), (255, 0, 0, 255))
    _write_png(e_smile, (40, 30), (0, 255, 0, 255))
    _write_png(i_normal, (1200, 900), (0, 0, 255, 255))
    return pdir


@pytest.fixture(autouse=True)
def _reset_lookup_singleton(monkeypatch):
    """每个测试前后重置 manifest 查表单例与缓存，并隔离全局 Settings：
    用空值环境变量覆盖 .env 的 CFN_GAME_PROJECT_DIR（环境变量优先级高于
    env_file，单测不允许读到本机真实游戏项目路径）。"""
    monkeypatch.setenv("CFN_GAME_PROJECT_DIR", "")
    from core.config import get_settings

    get_settings.cache_clear()
    manifest_lookup.reset_portrait_lookup()
    portraits_cache.get_cache().clear()
    yield
    get_settings.cache_clear()
    manifest_lookup.reset_portrait_lookup()
    portraits_cache.get_cache().clear()


@pytest.fixture
def portrait_dir(tmp_path: Path) -> Path:
    return build_portrait_dir(tmp_path)


@pytest.fixture
def portrait_env(portrait_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把单例探测指向假目录（不触碰全局 Settings），并隔离缓存。"""
    monkeypatch.setattr(manifest_lookup, "discover_portrait_dir", lambda: portrait_dir)
    manifest_lookup.reset_portrait_lookup()
    cache = portraits_cache.get_cache()
    cache.clear()
    yield portrait_dir
    manifest_lookup.reset_portrait_lookup()
    cache.clear()
