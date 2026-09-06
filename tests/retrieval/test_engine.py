"""检索引擎测试（04 §7.3）：池过滤 self/others/any、阈值生效、去重、q3 变体仅在有上一条发言时加入。"""

from __future__ import annotations

import numpy as np
import pytest

from services.retrieval.hybrid import BM25Index, RetrievalEngine, RetrievalInput, build_queries
from services.retrieval.store import Node, VectorStore

from tests.conftest import FakeEmbedder

DIM = FakeEmbedder.dim


def _basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def _make_nodes() -> list[Node]:
    """构造带受控向量的语料：文本 → registry 向量（与 query 相同向量 = 相似度 1.0）。"""
    spec = [
        # (id, text, type, character, basis_index, task_source)
        ("d-self-1", "我们的初次相遇", "dialogue", "smith", 1, None),
        ("d-self-2", "今天锻造一把新剑", "dialogue", "smith", 1, None),  # 同向量但分低→由融合排序决定
        ("d-other-1", "铁匠的其他角色对话", "dialogue", "andy", 2, None),
        ("d-pc", "玩家台词不进其他池", "dialogue", "$PC", 2, None),
        ("lore-1", "世界观的起源", "world_lore", None, 3, None),
        ("loading-1", "loading 提示短句", "loading_lore", None, 3, None),
        ("task-self", "铁匠的任务台词", "task", "smith", 1, None),
        ("task-other", "其他角色的任务台词", "task", "andy", 2, None),
        ("task-guide", "教学引导任务台词", "task", "smith", 1, "guide"),
        ("item-1", "精炼石", "game_item", None, 4, None),
        ("stage-1", "废弃矿坑", "game_stage", None, 4, None),
    ]
    nodes = []
    for nid, text, ntype, char, bi, source in spec:
        nodes.append(Node(id=nid, text=text, type=ntype, character=char, task_source=source))
    return nodes


@pytest.fixture
def engine(tmp_path):
    embedder = FakeEmbedder()
    nodes = _make_nodes()
    # 注册文本向量：query 文本与目标文本同向量 → 相似度 1.0；无关 query → 正交向量
    for i, node in enumerate(nodes):
        embedder.register(node.text, _basis(i))
    store = VectorStore.build(nodes, embedder, expected_fingerprint="fp")
    store.save(tmp_path)
    engine = RetrievalEngine(embedder=embedder, index_dir=tmp_path)
    engine.set_store(store)
    return engine


def _input(**kw) -> RetrievalInput:
    base = dict(user_query="我们的初次相遇", npc_name="smith")
    base.update(kw)
    return RetrievalInput(**base)


def test_self_pool_only_returns_current_npc(engine):
    bundle = engine.retrieve(_input())
    ids = {s.node.id for s in bundle.npc_dialogue}
    assert ids <= {"d-self-1", "d-self-2"}  # character=self 过滤
    assert "d-other-1" not in ids and "d-pc" not in ids


def test_others_pool_excludes_self_and_pc(engine):
    bundle = engine.retrieve(_input(user_query="铁匠的其他角色对话"))
    ids = {s.node.id for s in bundle.other_npc}
    assert "smith" not in [s.node.character for s in bundle.other_npc]
    assert "d-pc" not in ids  # $PC 占位角色排除
    assert "d-other-1" in ids or "task-other" in ids


def test_threshold_filters_low_scores(engine):
    """query 与语料正交（相似度 0）→ 过阈值池全空。"""
    bundle = engine.retrieve(_input(user_query="完全无关的查询xyz"))
    assert bundle.world_lore == []
    assert bundle.loading == []
    assert bundle.supp_intel == []


def test_guide_task_excluded_by_higher_threshold(engine):
    """guide 类任务阈值 0.38：与 query 完全同向（1.0）时可进；正交时被滤掉。"""
    bundle = engine.retrieve(_input(user_query="教学引导任务台词"))
    assert any(s.node.id == "task-guide" for s in bundle.npc_task)


def test_entity_pools_return_top1(engine):
    bundle = engine.retrieve(_input(user_query="精炼石"))
    assert [s.node.id for s in bundle.entity_items] == ["item-1"]
    assert [s.node.id for s in bundle.entity_stages] == []  # 关卡池对物品 query 不命中（正交向量）


def test_q3_variant_only_with_last_message(engine):
    """q3 变体仅在存在 NPC 上一条发言时加入（encode 批次 2 条 vs 3 条）。"""
    embedder = engine._embedder
    embedder.encode_calls.clear()
    engine.retrieve(_input())
    assert len(embedder.encode_calls[-1]) == 2  # q1 + q2

    embedder.encode_calls.clear()
    engine.retrieve(_input(npc_last_message="上次我们聊到锻造"))
    assert len(embedder.encode_calls[-1]) == 3  # q1 + q2 + q3


def test_queries_composition():
    q = build_queries(RetrievalInput(user_query="你好", npc_name="铁匠", npc_titles=["大师"], npc_faction="王国"))
    assert q["q1"] == "你好"
    assert q["q2"] == "你好 铁匠 大师 王国"
    assert "q3" not in q

    q2 = build_queries(RetrievalInput(user_query="你好", npc_name="铁匠", npc_last_message="上次说的话"))
    assert q2["q3"] == "你好 上次说的话"


def test_dedup_across_pools(engine):
    """同一节点不重复出现在两个池（如 npc_task 与 other_npc 的重叠）。"""
    bundle = engine.retrieve(_input(user_query="铁匠的任务台词", npc_last_message="继续聊任务"))
    seen = {}
    for pool_name in ("npc_dialogue", "world_lore", "loading", "npc_task", "supp_intel", "other_npc"):
        for sn in bundle.pool_result(pool_name):
            assert sn.node.id not in seen, f"{sn.node.id} 同时出现在 {seen.get(sn.node.id)} 与 {pool_name}"
            seen[sn.node.id] = pool_name


def test_engine_not_ready_returns_empty_bundle(tmp_path):
    engine = RetrievalEngine(embedder=FakeEmbedder(), index_dir=tmp_path)
    bundle = engine.retrieve(_input())
    assert bundle.is_empty


def test_bm25_contributes_to_fusion(tmp_path):
    """BM25 与 dense 同池 RRF 融合（生产启用 hybrid，修 C2 的回归点）。"""
    embedder = FakeEmbedder()
    texts = [f"任务台词{i}：收集材料{i}号" for i in range(20)]
    nodes = [Node(id=f"t{i}", text=t, type="task", character="smith") for i, t in enumerate(texts)]
    target = Node(id="hit", text="收集猫爪交给铁匠", type="task", character="smith")
    nodes.append(target)
    embedder.register(target.text, _basis(0))
    # 所有任务台词共享一个与 query 正交的向量 → dense 全靠 BM25 把目标顶上来
    for n in nodes[:-1]:
        embedder.register(n.text, _basis(1))

    store = VectorStore.build(nodes, embedder, expected_fingerprint="fp")
    engine = RetrievalEngine(embedder=embedder, index_dir=tmp_path)
    engine.set_store(store)
    assert engine._bm25 is not None  # 语料 ≥ BM25_MIN_CORPUS 时必须构建 BM25

    bundle = engine.retrieve(_input(user_query="收集猫爪交给铁匠"))
    task_ids = [s.node.id for s in bundle.npc_task]
    assert task_ids[0] == "hit"  # dense 正交，融合靠 BM25 命中
