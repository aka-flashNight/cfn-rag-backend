"""route_search 编排层预取（2026-10-17 检索链结构优化）。

覆盖：来源归一化、knowledge 相对下限与已有结果去重、分类型预算与单条上限、
items/stages 的名称/区域匹配与排序、结果块兜底。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from services.game_data.models import Item, StageInfo
from services.retrieval.hybrid import RetrievalBundle, ScoredNode
from services.retrieval.store import Node
from services.orchestrator import search_prefetch as sp


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

def _node(node_id: str, text: str, **kwargs: Any) -> Node:
    return Node(id=node_id, text=text, type=kwargs.pop("type", "lore"), **kwargs)


def _scored(node: Node, dense: float, fused: float) -> ScoredNode:
    return ScoredNode(node=node, dense_score=dense, fused_score=fused)


@dataclass
class FakeItems:
    items: list[Item] = field(default_factory=list)

    def search(self, term: str, limit: int = 10) -> list[Item]:
        needle = (term or "").strip().lower()
        if not needle:
            return []
        hits = [
            item for item in self.items
            if needle in (item.name or "").lower()
            or needle in (item.displayname or "").lower()
        ]
        return hits[:limit]


@dataclass
class FakeStages:
    _stage_infos: dict[tuple[str, str], StageInfo] = field(default_factory=dict)


@dataclass
class FakeGameData:
    items: FakeItems = field(default_factory=FakeItems)
    stages: FakeStages = field(default_factory=FakeStages)
    equipment_mods: Any = None


def _patch_tier1(monkeypatch, bundle: RetrievalBundle | None) -> None:
    import services.orchestrator.context as context_module

    monkeypatch.setattr(
        context_module, "retrieve_tier1", lambda **kwargs: bundle,
    )


# ---------------------------------------------------------------------------
# 来源归一化
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, sp.ALL_SOURCES),
    ("all", sp.ALL_SOURCES),
    ("ALL", sp.ALL_SOURCES),
    ("items", ("items",)),
    (["knowledge"], ("knowledge",)),
    (["stages", "items"], ("stages", "items")),
    (["items", "items"], ("items",)),
    (["bogus"], sp.ALL_SOURCES),
    ([], sp.ALL_SOURCES),
    ("all items", sp.ALL_SOURCES),  # 非枚举字符串按未知处理 → 回落全部
])
def test_normalize_sources(raw, expected):
    assert sp.normalize_sources(raw) == expected


# ---------------------------------------------------------------------------
# knowledge：相对下限 / 排序 / 去重 / 预算
# ---------------------------------------------------------------------------

def test_knowledge_drops_low_relevance_before_ranking(monkeypatch):
    bundle = RetrievalBundle(world_lore=[
        _scored(_node("w-high", "高相关世界观"), dense=0.9, fused=0.5),
        _scored(_node("w-low", "低相关世界观"), dense=0.1, fused=0.9),
    ])
    _patch_tier1(monkeypatch, bundle)

    result = sp.run_search_prefetch(
        engine=object(), game_data=FakeGameData(), keywords="安迪", sources=["knowledge"],
    )

    assert "高相关世界观" in result.text
    assert "低相关世界观" not in result.text  # dense 低于最高分 35%，先筛掉再排序
    assert result.knowledge_count == 1
    assert result.node_ids == frozenset({"w-high"})


def test_knowledge_filters_short_list_dedupe_and_labels(monkeypatch):
    bundle = RetrievalBundle(
        npc_dialogue=[
            _scored(_node("d1", "我已经不干那行了", character="安迪"), 0.9, 0.6),
        ],
        supp_intel=[
            _scored(_node("s1", "安迪离开A兵团后下落不明"), 0.8, 0.5),
        ],
    )
    _patch_tier1(monkeypatch, bundle)

    result = sp.run_search_prefetch(
        engine=object(), game_data=FakeGameData(), keywords="安迪", sources=["knowledge"],
    )

    assert "[你的过往台词·安迪] 我已经不干那行了" in result.text
    assert "[补充设定/情报] 安迪离开A兵团后下落不明" in result.text


def test_knowledge_dedupes_against_existing_bundle(monkeypatch):
    existing = RetrievalBundle(world_lore=[
        _scored(_node("w1", "已有的世界观"), 0.9, 0.5),
    ])
    fetched = RetrievalBundle(world_lore=[
        _scored(_node("w1", "已有的世界观"), 0.9, 0.5),
        _scored(_node("w2", "新查到的世界观"), 0.9, 0.4),
    ])
    _patch_tier1(monkeypatch, fetched)

    result = sp.run_search_prefetch(
        engine=object(), game_data=FakeGameData(), keywords="安迪",
        sources=["knowledge"], existing_bundle=existing,
    )

    assert "已有的世界观" not in result.text
    assert "新查到的世界观" in result.text
    assert result.node_ids == frozenset({"w2"})


def test_knowledge_respects_char_budget_and_node_cap(monkeypatch):
    long_text = "甲" * 2000
    bundle = RetrievalBundle(world_lore=[
        _scored(_node(f"w{i}", long_text), dense=0.9, fused=0.9 - i * 0.01)
        for i in range(10)
    ])
    _patch_tier1(monkeypatch, bundle)

    result = sp.run_search_prefetch(
        engine=object(), game_data=FakeGameData(), keywords="安迪", sources=["knowledge"],
    )

    assert 0 < len(result.text) <= sp.KNOWLEDGE_MAX_CHARS
    # 单条按 800 字截断，只取前几条；不会把 10 条全塞进来
    assert result.knowledge_count <= 5
    assert result.text.endswith("…")


# ---------------------------------------------------------------------------
# items：完全匹配 > 前缀 > 子串；已有物品排除
# ---------------------------------------------------------------------------

def _item(name: str, **kwargs: Any) -> Item:
    return Item(name=name, **kwargs)


def test_items_ranked_by_match_and_excludes_existing(monkeypatch):
    game_data = FakeGameData(items=FakeItems(items=[
        _item("旧枪"),
        _item("安迪的旧枪", type="武器", use="枪"),
        _item("安迪", type="收集品"),
        _item("毫不相干的木箱", type="收集品"),
    ]))
    existing = RetrievalBundle(entity_items=[
        _scored(_node("e1", "安迪", item_name="安迪"), 0.9, 0.9),
    ])

    result = sp.run_search_prefetch(
        engine=object(), game_data=game_data, keywords="安迪 旧枪",
        sources=["items"], existing_bundle=existing,
    )

    assert result.text.startswith("【物品资料】")
    assert "安迪的旧枪" in result.text
    assert "旧枪" in result.text
    assert "安迪" not in result.text.replace("安迪的旧枪", "")  # 已有物品不再重复给
    assert "毫不相干" not in result.text
    # 完全相等 > 前缀 > 子串：旧枪（exact）排在安迪的旧枪（prefix）前
    assert result.text.index("旧枪") < result.text.index("安迪的旧枪")


def test_items_budget_cap(monkeypatch):
    # 描述写足长度，让单条逼近 320 字上限：截断必须由 1200 字预算产生，
    # 而不是碰巧卡在注册表的返回条数上。
    game_data = FakeGameData(items=FakeItems(items=[
        _item(f"安迪的装备{i}", type="武器", use="枪", description="装备说明" * 150)
        for i in range(30)
    ]))

    result = sp.run_search_prefetch(
        engine=object(), game_data=game_data, keywords="安迪", sources=["items"],
    )

    assert len(result.text) <= sp.ITEMS_MAX_CHARS
    assert result.text.count("名称：") == result.item_count
    assert 0 < result.item_count < 30


# ---------------------------------------------------------------------------
# stages：关卡名匹配优先于区域匹配；已有结果排除
# ---------------------------------------------------------------------------

def test_stages_name_match_before_area_match(monkeypatch):
    game_data = FakeGameData(stages=FakeStages(_stage_infos={
        ("基地门口", "黑龙堂"): StageInfo(area="基地门口", name="黑龙堂", description="黑龙堂入口"),
        ("废城", "旧仓库"): StageInfo(area="废城", name="旧仓库", description="废弃仓库"),
    }))

    result = sp.run_search_prefetch(
        engine=object(), game_data=game_data, keywords="废城 黑龙", sources=["stages"],
    )

    assert result.text.startswith("【关卡资料】")
    assert result.text.index("黑龙堂") < result.text.index("旧仓库")
    assert result.stage_count == 2


def test_stages_exclude_existing(monkeypatch):
    game_data = FakeGameData(stages=FakeStages(_stage_infos={
        ("基地门口", "黑龙堂"): StageInfo(area="基地门口", name="黑龙堂"),
        ("废城", "旧仓库"): StageInfo(area="废城", name="旧仓库"),
    }))
    existing = RetrievalBundle(entity_stages=[
        _scored(
            _node("s1", "旧仓库", stage_area="废城", stage_name="旧仓库"),
            0.9, 0.9,
        ),
    ])

    result = sp.run_search_prefetch(
        engine=object(), game_data=game_data, keywords="废城 黑龙",
        sources=["stages"], existing_bundle=existing,
    )

    assert "黑龙堂" in result.text
    assert "旧仓库" not in result.text
    assert result.stage_names == frozenset({"黑龙堂"})


# ---------------------------------------------------------------------------
# 来源选择与总预算
# ---------------------------------------------------------------------------

def test_sources_limit_skips_other_sides(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("sources=items 时不应触发知识检索")

    import services.orchestrator.context as context_module
    monkeypatch.setattr(context_module, "retrieve_tier1", boom)

    game_data = FakeGameData(items=FakeItems(items=[_item("安迪的旧枪")]))
    result = sp.run_search_prefetch(
        engine=object(), game_data=game_data, keywords="安迪", sources=["items"],
    )

    assert "【物品资料】" in result.text
    assert "【关卡资料】" not in result.text
    assert result.knowledge_count == 0


def test_total_budget_across_all_sources(monkeypatch):
    bundle = RetrievalBundle(world_lore=[
        _scored(_node(f"w{i}", "甲" * 2000), dense=0.9, fused=0.9 - i * 0.01)
        for i in range(10)
    ])
    _patch_tier1(monkeypatch, bundle)
    game_data = FakeGameData(
        items=FakeItems(items=[_item(f"安迪的装备{i}", type="武器", use="枪") for i in range(30)]),
        stages=FakeStages(_stage_infos={
            ("基地门口", f"安迪关卡{i}"): StageInfo(area="基地门口", name=f"安迪关卡{i}")
            for i in range(30)
        }),
    )

    result = sp.run_search_prefetch(
        engine=object(), game_data=game_data, keywords="安迪", sources=None,
    )

    # 三段之间各有一个空行分隔
    assert len(result.text) <= sp.TOTAL_MAX_CHARS + 4
    assert result.knowledge_count > 0 and result.item_count > 0 and result.stage_count > 0


def test_empty_prefetch_result_block():
    block = sp.build_prefetch_result_block("")
    assert block == "【后台结果：检索到的资料】\n（本次没有检索到新的相关资料）"
    assert sp.build_prefetch_result_block("【世界观】资料") == "【后台结果：检索到的资料】\n【世界观】资料"
    assert sp.SearchPrefetch().is_empty
    assert not sp.SearchPrefetch(text="有内容").is_empty
