"""
每个 Worker Agent 可见的原子工具白名单。

与 Anthropic 2026 Skills 规范配合：Worker 的 **原子工具** 在这里静态配置；
**流程/领域知识** 通过 ``list_skills / read_skill / read_skill_file`` 按需加载。
之所以把 system 元工具 **无条件** 加到每个 worker，是因为 worker 可能遇到不熟悉的场景，
需要临时拉 skill body 才能继续工作（渐进式披露）。

- QueryAgent：只做知识检索，不触碰任务流水线、不上报情绪（情绪由 DialogueAgent 最终回复时上报）。
- TaskAgent：任务流水线全家桶；不能做最终 NPC 对白生成（那个阶段仅走 Dialogue）。
- DialogueAgent：纯对话生成 + mood 上报；不能调用任务/检索工具（避免 worker 越权）。
"""

from __future__ import annotations

from typing import FrozenSet

# 系统元工具（所有 worker 共用，方便按需 read_skill）
SYSTEM_TOOLS: FrozenSet[str] = frozenset({
    "list_skills",
    "read_skill",
    "read_skill_file",
})

# 注意：下面的三个 scope 是 **可见集合**；实际发给 LLM 的 tools schema 还会进一步裁剪。
QUERY_WORKER_TOOLS: FrozenSet[str] = frozenset({
    "search_knowledge",
    "search_stages",
    "search_items",
}) | SYSTEM_TOOLS

TASK_WORKER_TOOLS: FrozenSet[str] = frozenset({
    "prepare_task_context",
    "draft_agent_task",
    "update_task_draft",
    "confirm_agent_task",
    "cancel_agent_task",
    # Task 过程中 NPC 可能顺带查一下关卡 / 物品细节，因此保留 query 类工具
    "search_stages",
    "search_items",
}) | SYSTEM_TOOLS

DIALOGUE_WORKER_TOOLS: FrozenSet[str] = frozenset({
    "update_npc_mood",
}) | SYSTEM_TOOLS


def tools_for_worker(worker_name: str) -> FrozenSet[str]:
    if worker_name == "query":
        return QUERY_WORKER_TOOLS
    if worker_name == "task":
        return TASK_WORKER_TOOLS
    if worker_name == "dialogue":
        return DIALOGUE_WORKER_TOOLS
    return SYSTEM_TOOLS


__all__ = [
    "SYSTEM_TOOLS",
    "QUERY_WORKER_TOOLS",
    "TASK_WORKER_TOOLS",
    "DIALOGUE_WORKER_TOOLS",
    "tools_for_worker",
]
