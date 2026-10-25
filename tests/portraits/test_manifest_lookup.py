"""manifest 查表测试（07 §8.1，协议逐条对齐游戏项目指南 §3）。"""

from __future__ import annotations

from services.portraits.manifest_lookup import (
    DialoguePortraitLookup,
    discover_portrait_dir,
    get_portrait_lookup,
)


def test_lookup_by_original_name_and_alias_and_variants(portrait_dir):
    lookup = DialoguePortraitLookup(portrait_dir)
    # 原名 / 别名 / 大小写 / 去空白变体（协议 §3.1 三注册 + 三探测）
    for probe in ("Andy Law", "andy law", "AndyLaw", "ANDY LAW", "  Andy Law  "):
        entry = lookup.find_entry(probe)
        assert entry is not None and entry["key"] == "Andy Law", probe
    # 顶层 aliases 注册（shimizu → 清水结衣）
    entry = lookup.find_entry("shimizu")
    assert entry is not None and entry["key"] == "清水结衣"


def test_emotion_fallback_chain(portrait_dir):
    lookup = DialoguePortraitLookup(portrait_dir)
    entry = lookup.find_entry("Andy Law")
    # 请求情绪命中 / 缺情绪回退「普通」/ 空串视为「普通」（协议 §3.2）
    assert lookup.find_expression(entry, "微笑")["uri"].endswith("e_smile.png")
    assert lookup.find_expression(entry, "愤怒")["uri"].endswith("e_normal.png")
    assert lookup.find_expression(entry, "")["uri"].endswith("e_normal.png")
    assert lookup.find_expression(entry, "普通")["uri"].endswith("e_normal.png")


def test_resolve_paths_and_bounds(portrait_dir):
    lookup = DialoguePortraitLookup(portrait_dir)
    r = lookup.resolve("Andy Law", "普通")
    assert isinstance(r, dict)
    assert r["png_path"] == portrait_dir / "external" / "p_a" / "e_normal.png"
    assert r["bounds"] == {"x": 10, "y": 20, "width": 60, "height": 40}
    assert r["key"] == "Andy Law"
    assert r["expression_used"] == "普通"


def test_hero_returns_hero_marker(portrait_dir):
    lookup = DialoguePortraitLookup(portrait_dir)
    for name in ("$PC_CHAR", "主角模板", "玩家"):
        assert lookup.resolve(name, "微笑") == "hero"


def test_unknown_character_returns_none(portrait_dir):
    lookup = DialoguePortraitLookup(portrait_dir)
    assert lookup.resolve("不存在的角色", "微笑") is None
    # 主角特例优先于「查不到」
    assert lookup.resolve("玩家", "") == "hero"


def test_singleton_loads_once_and_dedupes(portrait_dir, monkeypatch):
    calls = {"n": 0}

    def fake_discover():
        calls["n"] += 1
        return portrait_dir

    monkeypatch.setattr(
        "services.portraits.manifest_lookup.discover_portrait_dir", fake_discover
    )
    assert get_portrait_lookup() is not None
    assert get_portrait_lookup() is get_portrait_lookup()
    assert calls["n"] == 1  # 进程内单例：manifest 只现读一次（协议 §6.7 的「每次进程启动」粒度）


def test_manifest_missing_degrades_to_none(monkeypatch):
    monkeypatch.setattr(
        "services.portraits.manifest_lookup.discover_portrait_dir", lambda: None
    )
    assert get_portrait_lookup() is None  # 无图模式，不报错


def test_discover_prefers_explicit_config(monkeypatch, portrait_dir):
    """CFN_GAME_PROJECT_DIR 显式指定游戏项目根时优先于自动探测（07 §1）。"""
    from core.config import get_settings

    monkeypatch.setenv("CFN_GAME_PROJECT_DIR", str(portrait_dir.parents[3]))
    get_settings.cache_clear()
    try:
        assert discover_portrait_dir() == portrait_dir
    finally:
        get_settings.cache_clear()


def test_discover_scans_resources_siblings(monkeypatch, portrait_dir, tmp_path):
    """resources 的同级兄弟目录（游戏项目根）可被自动探测（07 §1）。"""
    import services.game_data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "find_resources_directory", lambda: tmp_path / "resources")
    # portrait_dir = tmp_path/CrazyFlashNight/...，与 tmp_path/resources 同级 → 应命中
    assert discover_portrait_dir() == portrait_dir
