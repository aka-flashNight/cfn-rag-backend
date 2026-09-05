"""hybrid 检索执行：单次嵌入 + 全库矩阵乘 + BM25 RRF 融合 + 池内过滤（修 C1/C2/C5/C8）。

对应 docs/v3-developer/04-检索与向量模型.md §3。查询计划：
  q1 = user_query；q2 = user_query + NPC名+称号+阵营；q3 = user_query + npc_last_message（有才加）
  vecs = embedder.encode(queries)  # ≤3 条一次批量编码
  dense = matrix @ vecs.T          # 全库一次矩阵乘
  bm25  = BM25 全库一次打分
  fused = RRF(dense, bm25)         # k=60
  per_pool_select(fused, pools)    # 池内过滤/阈值/topk/名额
旧的「NPC 上一条发言触发第二轮全池重查」改为 q3 变体同批融合（语义不变，成本 x2 → +1 次编码）。
"""

from __future__ import annotations

import gzip
import logging
import pickle
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from services.retrieval import config as rcfg
from services.retrieval.pools import POOLS, Pool
from services.retrieval.store import Node, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 统一的中英文分词：英文按词 / 数字单独 / 中文按字 / 其余符号按单字符
# （沿用 ai_engine 旧实现，BM25 与设定文档切块共用）
# ---------------------------------------------------------------------------

_CJK_TOKEN_RE = re.compile(
    r"[a-zA-Z]+"
    r"|\d+"
    r"|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]"
    r"|[^\s]"
)


def cjk_tokenizer(text: str) -> list[str]:
    tokens = _CJK_TOKEN_RE.findall(text or "")
    return tokens if tokens else [""]


BM25_FILE = "bm25.pkl.gz"


class BM25Index:
    """全库 BM25（一次 O(N) 打分替代旧 filter_nodes_by_metadata 全量线性扫描，修 C8）。"""

    def __init__(self, nodes: list[Node]) -> None:
        corpus_tokens = [cjk_tokenizer(n.text) for n in nodes]
        corpus_tokens = [t if t else [""] for t in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens)
        self._corpus_tokens = corpus_tokens  # 分词缓存

    def scores(self, query: str) -> np.ndarray:
        q_tokens = cjk_tokenizer(query or "")
        return np.asarray(self._bm25.get_scores(q_tokens if q_tokens else [""]), dtype=np.float32)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based 排名（分数高→排名靠前；同分按索引稳定）。"""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks


def rrf_fuse(dense_scores: np.ndarray, bm25_scores: np.ndarray, k: int = rcfg.RRF_K) -> np.ndarray:
    """RRF: score(d) = Σ 1/(k + rank_i(d) + 1)。"""
    return 1.0 / (k + _ranks(dense_scores) + 1.0) + 1.0 / (k + _ranks(bm25_scores) + 1.0)


# ---------------------------------------------------------------------------
# 输入 / 输出结构
# ---------------------------------------------------------------------------

_PC_CHAR_SUBSTR = "$pc"  # 玩家占位角色（$PC / $PC_TITLE / $PC_CHAR 及小写变体）


@dataclass
class RetrievalInput:
    """单轮检索的输入（编排器上下文装配的 Tier-1 部分）。"""

    user_query: str
    npc_name: str
    npc_titles: list[str] = field(default_factory=list)
    npc_faction: str | None = None
    npc_last_message: str | None = None
    forbidden_other_chars: set[str] = field(default_factory=set)  # 彩蛋/成员阵营角色名（小写）


@dataclass
class ScoredNode:
    node: Node
    dense_score: float
    fused_score: float


@dataclass
class RetrievalBundle:
    npc_dialogue: list[ScoredNode] = field(default_factory=list)
    world_lore: list[ScoredNode] = field(default_factory=list)
    loading: list[ScoredNode] = field(default_factory=list)
    npc_task: list[ScoredNode] = field(default_factory=list)
    supp_intel: list[ScoredNode] = field(default_factory=list)
    other_npc: list[ScoredNode] = field(default_factory=list)
    entity_items: list[ScoredNode] = field(default_factory=list)
    entity_stages: list[ScoredNode] = field(default_factory=list)

    def pool_result(self, name: str) -> list[ScoredNode]:
        return getattr(self, name)

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.npc_dialogue, self.world_lore, self.loading, self.npc_task,
             self.supp_intel, self.other_npc, self.entity_items, self.entity_stages)
        )


def build_queries(inp: RetrievalInput) -> dict[str, str]:
    """构造 ≤3 个查询变体（04 §3.3）。"""
    q1 = (inp.user_query or "").strip()
    q2_parts = [q1, (inp.npc_name or "").strip()]
    q2_parts += [t.strip() for t in inp.npc_titles if t and t.strip()]
    if inp.npc_faction and inp.npc_faction.strip():
        q2_parts.append(inp.npc_faction.strip())
    q2 = " ".join(p for p in q2_parts if p)
    queries = {"q1": q1, "q2": q2}
    npc_last = (inp.npc_last_message or "").strip()
    if npc_last:
        queries["q3"] = " ".join(p for p in (q1, npc_last) if p)
    return queries


# ---------------------------------------------------------------------------
# 检索引擎
# ---------------------------------------------------------------------------


class RetrievalEngine:
    """进程内单例。store 未就绪时 retrieve 返回空 bundle（就绪前检索降级）。"""

    def __init__(self, embedder=None, index_dir: Path | None = None) -> None:
        self._embedder = embedder  # 惰性解析默认 embedder
        self._index_dir = Path(index_dir) if index_dir else None
        self._store: VectorStore | None = None
        self._bm25: BM25Index | None = None
        # 预计算过滤掩码（store 就绪时生成）
        self._type_masks: dict[str, np.ndarray] = {}
        self._char_lower: list[str] = []
        self._guide_mask: np.ndarray | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # store 生命周期
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._store is not None

    @property
    def store(self) -> VectorStore | None:
        return self._store

    def _dir(self) -> Path:
        if self._index_dir is None:
            from services.retrieval.store import get_index_dir

            self._index_dir = get_index_dir()
        return self._index_dir

    def try_load(self, expected_fingerprint: str) -> bool:
        """指纹匹配则直接加载（<1s）；不匹配/损坏返回 False（由调用方决定重建）。"""
        try:
            store = VectorStore.load(self._dir(), expected_fingerprint)
        except Exception as exc:
            logger.info("向量库不可用（%s），将重建", exc)
            return False
        self.set_store(store)
        return True

    def set_store(self, store: VectorStore) -> None:
        """装载 store 并就绪 BM25（存在 bm25.pkl 且指纹匹配则反序列化，否则重建并落盘）。"""
        with self._lock:
            self._store = store
            self._build_masks(store)
            self._bm25 = self._load_or_build_bm25(store)

    def build_store(self, nodes: list[Node], expected_fingerprint: str) -> VectorStore:
        """全量构建 + 持久化 + 装载（重活，调用方放线程里跑）。"""
        embedder = self._resolve_embedder()
        store = VectorStore.build(nodes, embedder, expected_fingerprint)
        store.save(self._dir())
        self.set_store(store)
        return store

    def _build_masks(self, store: VectorStore) -> None:
        types = [n.type for n in store.nodes]
        self._type_masks = {
            t: np.asarray([x == t for x in types], dtype=bool)
            for t in set(types)
        }
        self._char_lower = [(n.character or "").strip().lower() for n in store.nodes]
        self._guide_mask = np.asarray(
            [n.task_source == "guide" for n in store.nodes], dtype=bool
        )

    def _load_or_build_bm25(self, store: VectorStore) -> BM25Index | None:
        """优先反序列化 gzip 压缩的 bm25.pkl.gz（省 ~4MB 落盘体积），失败则重建并落盘。"""
        bm25_path = self._dir() / BM25_FILE
        if bm25_path.exists():
            try:
                with gzip.open(bm25_path, "rb") as f:
                    payload = pickle.load(f)
                if payload.get("fingerprint") == store.fingerprint and payload.get("count") == len(store.nodes):
                    index = payload["bm25"]
                    if len(index._corpus_tokens) == len(store.nodes):
                        return index
            except Exception as exc:
                logger.info("bm25.pkl 加载失败，改为重建: %s", exc)
        if len(store.nodes) < rcfg.BM25_MIN_CORPUS:
            return None
        index = BM25Index(store.nodes)
        try:
            with gzip.open(bm25_path, "wb") as f:
                pickle.dump({"fingerprint": store.fingerprint, "count": len(store.nodes), "bm25": index}, f)
        except Exception as exc:
            logger.warning("bm25.pkl.gz 写盘失败（不影响检索）: %s", exc)
        return index

    def _resolve_embedder(self):
        if self._embedder is None:
            from services.retrieval.embedder import get_default_embedder

            self._embedder = get_default_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # 检索主流程
    # ------------------------------------------------------------------

    def retrieve(self, inp: RetrievalInput) -> RetrievalBundle:
        store = self._store
        if store is None or store.matrix.shape[0] == 0:
            return RetrievalBundle()

        queries = build_queries(inp)
        query_list = list(queries.values())
        vecs = self._resolve_embedder().encode(query_list)  # ≤3 条一次批量编码
        # 全库一次矩阵乘：[N, Q]
        dense_all = store.matrix @ vecs.T
        bm25_all = {
            key: (self._bm25.scores(q) if self._bm25 is not None else np.zeros(store.matrix.shape[0], dtype=np.float32))
            for key, q in queries.items()
        }
        fused_all = {
            key: rrf_fuse(dense_all[:, i], bm25_all[key])
            for i, key in enumerate(queries)
        }

        npc_char = (inp.npc_name or "").strip().lower()
        bundle = RetrievalBundle()
        picked_ids: set[str] = set()

        for pool in POOLS:
            main_key = f"q{pool.query_variant}"
            picked = self._select_pool(
                pool, dense_all[:, queries_key_index(queries, main_key)], fused_all[main_key],
                npc_char=npc_char, npc_name=inp.npc_name,
                special_chars=inp.forbidden_other_chars,
            )
            # q3 变体补充（仅存在 npc_last_message 时）
            if pool.npc_extra_top_k > 0 and "q3" in queries:
                extra = self._select_pool(
                    pool, dense_all[:, queries_key_index(queries, "q3")], fused_all["q3"],
                    npc_char=npc_char, npc_name=inp.npc_name,
                    special_chars=inp.forbidden_other_chars,
                    use_extra_thresholds=True,
                )
                extra = [s for s in extra if s.node.id not in {p.node.id for p in picked}]
                picked = picked + extra[: pool.npc_extra_top_k]

            picked = [s for s in picked if s.node.id not in picked_ids]
            picked_ids.update(s.node.id for s in picked)
            setattr(bundle, _BUNDLE_FIELD_BY_POOL.get(pool.name, pool.name), picked)

        return bundle

    # ------------------------------------------------------------------
    # 池内选择
    # ------------------------------------------------------------------

    def _pool_mask(self, pool: Pool, npc_char: str) -> np.ndarray:
        mask = np.zeros(len(self._char_lower), dtype=bool)
        for t in pool.type_filter:
            m = self._type_masks.get(t)
            if m is not None:
                mask |= m
        if pool.character == "self":
            mask &= np.asarray([c == npc_char for c in self._char_lower], dtype=bool)
        elif pool.character == "others":
            mask &= np.asarray(
                [c != npc_char and _PC_CHAR_SUBSTR not in c for c in self._char_lower],
                dtype=bool,
            )
        return mask

    def _select_pool(
        self,
        pool: Pool,
        dense: np.ndarray,
        fused: np.ndarray,
        *,
        npc_char: str,
        npc_name: str,
        special_chars: set[str],
        use_extra_thresholds: bool = False,
    ) -> list[ScoredNode]:
        if self._store is None or self._guide_mask is None:
            return []
        mask = self._pool_mask(pool, npc_char)
        idxs = np.nonzero(mask)[0]
        if len(idxs) == 0:
            return []

        base_th = (
            pool.npc_extra_threshold
            if use_extra_thresholds and pool.npc_extra_threshold is not None
            else pool.dense_threshold
        )
        strict_th = (
            pool.npc_extra_guide_threshold
            if use_extra_thresholds and pool.npc_extra_guide_threshold is not None
            else pool.guide_threshold
        )
        th = np.full(len(idxs), float(base_th), dtype=np.float32)
        if strict_th is not None:
            special = np.asarray(
                [self._char_lower[i] in special_chars for i in idxs], dtype=bool
            ) if special_chars else np.zeros(len(idxs), dtype=bool)
            th[self._guide_mask[idxs] | special] = float(strict_th)

        cand = idxs[dense[idxs] >= th]
        if len(cand) == 0:
            return []

        if pool.quota_task or pool.quota_free:
            picked_idx = self._quota_select(
                pool, cand, fused, npc_name,
                task_quota=1 if use_extra_thresholds else None,  # extra 路径沿用旧「最多 1 条任务」
            )
        else:
            order = np.argsort(-fused[cand], kind="stable")[: pool.top_k]
            picked_idx = cand[order]
        return [
            ScoredNode(self._store.nodes[int(i)], float(dense[i]), float(fused[i]))
            for i in picked_idx
        ]

    def _quota_select(
        self,
        pool: Pool,
        cand: np.ndarray,
        fused: np.ndarray,
        npc_name: str,
        *,
        task_quota: int | None = None,
    ) -> np.ndarray:
        """other_npc 名额策略：先保 quota_task 条非引导任务，再自由竞争补满（任务优先、
        提及当前 NPC 优先，沿用旧 2+3 策略）。"""
        assert self._store is not None
        guide = self._guide_mask
        quota_task = pool.quota_task if task_quota is None else task_quota

        def _key(i: int) -> tuple[int, float]:
            mentions = 1 if (npc_name and npc_name in self._store.nodes[int(i)].text) else 0
            return (mentions, float(fused[i]))

        strict = [int(i) for i in cand if self._store.nodes[int(i)].type == "task" and not guide[int(i)]]
        strict.sort(key=_key, reverse=True)
        picked: list[int] = strict[:quota_task]
        picked_set = set(picked)

        free = [int(i) for i in cand]
        free.sort(key=_key, reverse=True)
        for i in free:
            if len(picked) >= pool.top_k:
                break
            if i in picked_set:
                continue
            picked.append(i)
            picked_set.add(i)
        return np.asarray(picked, dtype=np.int64)


def queries_key_index(queries: dict[str, str], key: str) -> int:
    return list(queries.keys()).index(key)


# 池名 → RetrievalBundle 字段名（实体池不进 section，单独承载给实体提示拼装）
_BUNDLE_FIELD_BY_POOL = {
    "game_item": "entity_items",
    "game_stage": "entity_stages",
}


# ---------------------------------------------------------------------------
# section 拼装（沿用旧 prompt 行为：行级去重、空行压缩、max_chars 截断）
# ---------------------------------------------------------------------------

def _node_text(sn: ScoredNode) -> str:
    return (sn.node.text or "").strip()


def _nodes_to_text(nodes: list[ScoredNode], max_chars: int | None = None) -> str:
    seen_lines: set[str] = set()
    output_lines: list[str] = []
    last_blank = False
    for sn in nodes:
        text = _node_text(sn)
        if not text:
            continue
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "…"
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if output_lines and not last_blank:
                    output_lines.append("")
                    last_blank = True
                continue
            last_blank = False
            if line in seen_lines:
                continue
            seen_lines.add(line)
            output_lines.append(line)
    return "\n".join(output_lines)


def _nodes_to_other_npc_text(
    nodes: list[ScoredNode],
    max_chars: int | None,
    forbidden_other_chars: set[str],
) -> str:
    """每条台词标注说话人；彩蛋/成员阵营附加「非正式角色」标注（保留业务行为）。"""
    seen_lines: set[str] = set()
    output_lines: list[str] = []
    last_blank = False
    for sn in nodes:
        text = _node_text(sn)
        if not text:
            continue
        speaker = (sn.node.character or "").strip() or "未知角色"
        is_special = bool(forbidden_other_chars and speaker.lower() in forbidden_other_chars)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "…"
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if output_lines and not last_blank:
                    output_lines.append("")
                    last_blank = True
                continue
            last_blank = False
            if line in seen_lines:
                continue
            seen_lines.add(line)
            if is_special:
                output_lines.append(f"{speaker}: {line}{rcfg.NON_CANONICAL_ANNOTATION}")
            else:
                output_lines.append(f"{speaker}: {line}")
    return "\n".join(output_lines)


def format_sections(
    bundle: RetrievalBundle, forbidden_other_chars: set[str] | None = None
) -> list[tuple[str, str]]:
    """按配置顺序把池结果拼成 (标题, 正文) sections；空池跳过。"""
    from services.retrieval.pools import SECTION_POOLS

    forbidden = forbidden_other_chars or set()
    sections: list[tuple[str, str]] = []
    for name in SECTION_POOLS:
        nodes = bundle.pool_result(name)
        if not nodes:
            continue
        if name == "other_npc":
            body = _nodes_to_other_npc_text(nodes, cfg_max_chars(name), forbidden)
        else:
            body = _nodes_to_text(nodes, cfg_max_chars(name))
        if body.strip():
            sections.append((rcfg.SECTION_HEADERS[name], body))
    return sections


def cfg_max_chars(name: str) -> int | None:
    return rcfg.SECTION_MAX_CHARS.get(name)


def format_retrieval_context(
    bundle: RetrievalBundle, forbidden_other_chars: set[str] | None = None
) -> str:
    """六类 section 合并为一段检索上下文文本（旧 _retrieve_context 的输出形态）。"""
    return "\n\n".join(f"{header}\n{body}" for header, body in format_sections(bundle, forbidden_other_chars))


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_DEFAULT_ENGINE: RetrievalEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_retrieval_engine() -> RetrievalEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is not None:
        return _DEFAULT_ENGINE
    with _ENGINE_LOCK:
        if _DEFAULT_ENGINE is None:
            _DEFAULT_ENGINE = RetrievalEngine()
        return _DEFAULT_ENGINE


def set_retrieval_engine(engine: RetrievalEngine | None) -> None:
    global _DEFAULT_ENGINE
    with _ENGINE_LOCK:
        _DEFAULT_ENGINE = engine
