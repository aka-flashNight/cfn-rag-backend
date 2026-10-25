"""检索配置：全部池子的阈值 / 名额 / 截断集中在此（修 C4 魔数散落）。

对应 docs/v3-developer/04-检索与向量模型.md §3.2。每个数的来源：
- dense_threshold 一档：沿用旧 game_rag_service._retrieve_context 的手调魔数初值；
- npc_extra_*（NPC 上一条发言触发的补充召回）阈值更高，同样沿用旧值；
- other_npc 名额策略（2 条非引导任务 + 3 条自由竞争）沿用旧「其他 NPC 名额策略（2+3）」。
调参只改本文件，不改检索执行代码。
"""

from __future__ import annotations

# 嵌入批量大小（构建索引与查询共用；本机实测 32 最快 216 条/s vs 64 的 198）
EMBED_BATCH_SIZE = 32

# RRF 融合常数 k（调研 §4.2 社区共识 k=60；语料 4426 条不在「极小语料 RRF 反噬」区间）
RRF_K = 60

# 请求级编码缓存上限（同回合相同 query 只编码一次，修 C1）
EMBED_CACHE_LIMIT = 256

# 池级 dense 阈值（cosine，阈值下不进池）
THRESHOLDS: dict[str, float] = {
    "npc_dialogue": 0.00,   # 旧版无阈值，仅按 top_k
    "world_lore": 0.22,     # 世界观稍宽松，避免完全无关片段混入
    "loading": 0.28,
    "npc_task": 0.28,       # 玩家检索主导
    "supp_intel": 0.30,     # 补充设定 + 情报合并池
    "other_npc": 0.36,      # 其他 NPC 对话参考：仅明显相关才出现
    "game_item": 0.48,      # 实体提示：强相关才补充
    "game_stage": 0.48,
}

# guide 类任务 / 彩蛋·成员阵营台词与 NPC 形象弱关联，仅强相关时才采用
STRICT_THRESHOLDS: dict[str, float] = {
    "npc_task": 0.38,       # 教学引导任务
    "other_npc": 0.44,      # 引导任务 / 彩蛋 / 成员
}

# NPC 上一条发言（q3 变体）补充召回：名额与更高阈值
NPC_EXTRA: dict[str, tuple[int, float, float]] = {
    # name: (top_k, threshold, guide/special threshold)
    "world_lore": (1, 0.28, 0.28),
    "loading": (2, 0.32, 0.32),
    "npc_task": (2, 0.28, 0.38),
    "supp_intel": (1, 0.36, 0.36),
    "other_npc": (2, 0.40, 0.48),
}

# other_npc 名额策略：先保 2 条非引导任务，再 3 条自由竞争（任务优先 + 提及当前 NPC 优先）
OTHER_NPC_QUOTA_TASK = 2
OTHER_NPC_QUOTA_FREE = 3
OTHER_NPC_TOTAL = 5

# 池 top_k
TOP_K: dict[str, int] = {
    "npc_dialogue": 8,
    "world_lore": 3,
    "loading": 5,           # 旧版 raw[:7] 过滤后取 5
    "npc_task": 8,          # 玩家检索主导（约 80%）
    "supp_intel": 3,
    "other_npc": OTHER_NPC_TOTAL,
    "game_item": 1,         # 旧版 top4 取第一条过阈值的（等价于过阈值取 top1）
    "game_stage": 1,
}

# section 拼装：标题原文与 max_chars 截断（沿用旧 prompt 行为，勿随意改动文案）
SECTION_ORDER: tuple[str, ...] = (
    "npc_dialogue",
    "world_lore",
    "loading",
    "npc_task",
    "other_npc",
    "supp_intel",
)

SECTION_HEADERS: dict[str, str] = {
    "npc_dialogue": "【你的过往台词示例】",
    "world_lore": "【世界观设定摘取片段（用户输入相似度检索结果，可能与你无关，无关时忽略）】",
    "loading": "tips节选：",
    "npc_task": "【你的参考任务对话(任务可能超过玩家当前进度，仅参考语气和设定，忽略具体剧情，避免剧透，且勿重复原文语句)】",
    "other_npc": "【其他NPC相关对话参考（不是你的台词，只参考设定，忽略语气，并忽略剧情以避免剧透）】",
    "supp_intel": "【补充设定与情报参考（用户输入相似度检索结果，可能与你无关，无关时忽略）】",
}

SECTION_MAX_CHARS: dict[str, int | None] = {
    "npc_dialogue": None,
    "world_lore": None,
    "loading": 300,
    "npc_task": 350,
    "other_npc": 350,
    "supp_intel": 350,
}

# 彩蛋/成员阵营台词的附加标注（保留业务行为：不排除而是标注 + 上浮阈值）
NON_CANONICAL_ANNOTATION = "（该角色为非正式角色，相关信息可能不属于世界观正式内容，你的角色可能并不知情，仅作参考）"

# BM25 全库打分时的最小候选规模（构建 BM25 的语料下限，低于此直接跳过 BM25 路径）
BM25_MIN_CORPUS = 16
