"""池定义（配置化，修 C4）：每类语料一个池，全部池子集中注册。

对应 docs/v3-developer/04-检索与向量模型.md §3.2。池 = 类型过滤 + 角色维度 +
top_k + 阈值 + q3 补充名额 + 截断；特殊业务规则（guide/彩蛋上浮阈值、
其他 NPC 名额策略 2+3）从旧 500 行函数体抽成池级配置项。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from services.retrieval import config as cfg


@dataclass(frozen=True)
class Pool:
    name: str
    type_filter: frozenset[str]          # metadata.type 白名单
    character: Literal["self", "others", "any"]
    query_variant: int = 1               # 1=user_query；2=user_query+NPC名+称号+阵营
    top_k: int = 5
    dense_threshold: float = 0.0
    guide_threshold: float | None = None  # task_source=guide（及彩蛋/成员）上浮阈值
    rrf_k: int = cfg.RRF_K
    max_chars: int | None = None         # section 截断
    # q3（NPC 上一条发言）变体补充召回
    npc_extra_top_k: int = 0
    npc_extra_threshold: float | None = None
    npc_extra_guide_threshold: float | None = None
    # other_npc 名额策略（2+3）；0 = 不启用
    quota_task: int = 0
    quota_free: int = 0


POOLS: tuple[Pool, ...] = (
    Pool(
        name="npc_dialogue",
        type_filter=frozenset({"dialogue"}),
        character="self",
        top_k=cfg.TOP_K["npc_dialogue"],
        dense_threshold=cfg.THRESHOLDS["npc_dialogue"],
    ),
    Pool(
        name="world_lore",
        type_filter=frozenset({"world_lore"}),
        character="any",
        query_variant=2,
        top_k=cfg.TOP_K["world_lore"],
        dense_threshold=cfg.THRESHOLDS["world_lore"],
        npc_extra_top_k=cfg.NPC_EXTRA["world_lore"][0],
        npc_extra_threshold=cfg.NPC_EXTRA["world_lore"][1],
        npc_extra_guide_threshold=cfg.NPC_EXTRA["world_lore"][2],
    ),
    Pool(
        name="loading",
        type_filter=frozenset({"loading_lore"}),
        character="any",
        query_variant=2,
        top_k=cfg.TOP_K["loading"],
        dense_threshold=cfg.THRESHOLDS["loading"],
        max_chars=cfg.SECTION_MAX_CHARS["loading"],
        npc_extra_top_k=cfg.NPC_EXTRA["loading"][0],
        npc_extra_threshold=cfg.NPC_EXTRA["loading"][1],
        npc_extra_guide_threshold=cfg.NPC_EXTRA["loading"][2],
    ),
    Pool(
        name="npc_task",
        type_filter=frozenset({"task"}),
        character="self",
        top_k=cfg.TOP_K["npc_task"],
        dense_threshold=cfg.THRESHOLDS["npc_task"],
        guide_threshold=cfg.STRICT_THRESHOLDS["npc_task"],
        max_chars=cfg.SECTION_MAX_CHARS["npc_task"],
        npc_extra_top_k=cfg.NPC_EXTRA["npc_task"][0],
        npc_extra_threshold=cfg.NPC_EXTRA["npc_task"][1],
        npc_extra_guide_threshold=cfg.NPC_EXTRA["npc_task"][2],
    ),
    Pool(
        name="supp_intel",
        type_filter=frozenset({"supplementary_lore", "intelligence"}),
        character="any",
        query_variant=2,
        top_k=cfg.TOP_K["supp_intel"],
        dense_threshold=cfg.THRESHOLDS["supp_intel"],
        max_chars=cfg.SECTION_MAX_CHARS["supp_intel"],
        npc_extra_top_k=cfg.NPC_EXTRA["supp_intel"][0],
        npc_extra_threshold=cfg.NPC_EXTRA["supp_intel"][1],
        npc_extra_guide_threshold=cfg.NPC_EXTRA["supp_intel"][2],
    ),
    Pool(
        name="other_npc",
        type_filter=frozenset({"dialogue", "task"}),
        character="others",
        top_k=cfg.TOP_K["other_npc"],
        dense_threshold=cfg.THRESHOLDS["other_npc"],
        guide_threshold=cfg.STRICT_THRESHOLDS["other_npc"],
        max_chars=cfg.SECTION_MAX_CHARS["other_npc"],
        npc_extra_top_k=cfg.NPC_EXTRA["other_npc"][0],
        npc_extra_threshold=cfg.NPC_EXTRA["other_npc"][1],
        npc_extra_guide_threshold=cfg.NPC_EXTRA["other_npc"][2],
        quota_task=cfg.OTHER_NPC_QUOTA_TASK,
        quota_free=cfg.OTHER_NPC_QUOTA_FREE,
    ),
    # 实体提示池（只跑 user_query 一遍，修 C6 的每轮两遍全表）
    Pool(
        name="game_item",
        type_filter=frozenset({"game_item"}),
        character="any",
        top_k=cfg.TOP_K["game_item"],
        dense_threshold=cfg.THRESHOLDS["game_item"],
    ),
    Pool(
        name="game_stage",
        type_filter=frozenset({"game_stage"}),
        character="any",
        top_k=cfg.TOP_K["game_stage"],
        dense_threshold=cfg.THRESHOLDS["game_stage"],
    ),
)

POOL_BY_NAME: dict[str, Pool] = {p.name: p for p in POOLS}

# section 输出顺序中的池名（实体池不进 section，走实体提示拼装）
SECTION_POOLS: tuple[str, ...] = cfg.SECTION_ORDER
