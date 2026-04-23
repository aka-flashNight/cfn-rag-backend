"""
Supervisor 主图 + AsyncSqliteSaver checkpointer + HITL v2 挂起点。

编排形态（路线三 v2 修订，2026-04）：

    START
      │
      ▼
    prepare_context
      │
      ▼
    supervisor  ──► [routing_decision]
      ├── "query"    ─► query_worker   ──┐
      ├── "task"     ─► task_worker    ──┤ ★ 静态边：worker 完成后**直接**进 dialogue_worker
      ├── "dialogue" ─► dialogue_worker ◄─┘   不再回 supervisor，避免二次 LLM 决策与死循环。
      └── "end"      ─► finalize
    dialogue_worker ─► finalize ─► END

关键设计：
- 单次用户消息的 LLM 调用数：1 (supervisor) + 0~2 (worker 决策/工具环) + 1 (dialogue)。
- "一轮内连环 draft→confirm" 由 task_worker 内部 guard 阻断：任一成功状态
  (draft_created / draft_updated / confirmed / cancelled) 出现即立刻退出决策环。
- HITL：draft_created 后 awaiting_confirmation=True，dialogue 介绍完草案正常 END。
  下一条玩家消息（checkpointer 带着 pending_draft 恢复 state）进来时，supervisor 根据玩家
  意图决定是否再进 task_worker；无需 ``interrupt_before``，HTTP 轮次天然充当 pause/resume。
- ``thread_id = session_id``，所有 state 由 ``AsyncSqliteSaver`` 自动持久化。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from services.agent_graph.nodes import (
    post_process_node,
    prepare_context_node,
)
from services.agents.state import SupervisorState
from services.agents.supervisor import supervisor_node
from services.agents.workers import (
    build_dialogue_worker,
    build_query_worker,
    build_task_worker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpointer resolution
# ---------------------------------------------------------------------------

def _default_checkpoint_path() -> Path:
    from services.memory_manager import DB_PATH
    return DB_PATH.parent / "langgraph_checkpoint.sqlite"


def resolve_checkpoint_db_path() -> Path:
    """
    决定本地 checkpoint 的 sqlite 路径：
    - 优先读 env ``CFN_LANGGRAPH_CHECKPOINT_DB``
    - 否则放在 memory.db 的同级目录 ``langgraph_checkpoint.sqlite``

    （后续 Route 4.2.2b 会替换为 AsyncPostgresSaver / AsyncRedisSaver；本函数只处理 local sqlite。）
    """
    override = os.environ.get("CFN_LANGGRAPH_CHECKPOINT_DB")
    if override:
        p = Path(override)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return _default_checkpoint_path()


# ---------------------------------------------------------------------------
# Routing edge
# ---------------------------------------------------------------------------

def _route_from_supervisor(state: dict[str, Any]) -> str:
    """根据 supervisor_node 的 routing_decision 字段决定下一跳。"""
    r = (state.get("routing_decision") or "").strip().lower()
    if r in ("query", "task", "dialogue"):
        return r
    return "finalize"


async def _finalize_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """最终节点：等价于原单 agent 的 post_process。"""
    return await post_process_node(state, config)


# ---------------------------------------------------------------------------
# Build uncompiled graph（便于测试/替换 checkpointer）
# ---------------------------------------------------------------------------

def build_supervisor_graph() -> StateGraph:
    """构建未编译的主图（supervisor + 3 worker）。

    拓扑要点：
    - 只有 supervisor 做条件路由；query/task 完成后**静态**衔接 dialogue_worker，
      不再回到 supervisor（避免多次 LLM 决策和连环动作）。
    - supervisor 可以直接路由到 dialogue_worker（普通闲聊）或 finalize（hard-guard 熔断 / end）。
    """
    graph = StateGraph(SupervisorState)

    graph.add_node("prepare_context", prepare_context_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("query_worker", build_query_worker().compile())
    graph.add_node("task_worker", build_task_worker().compile())
    graph.add_node("dialogue_worker", build_dialogue_worker().compile())
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "query": "query_worker",
            "task": "task_worker",
            "dialogue": "dialogue_worker",
            "finalize": "finalize",
        },
    )

    # ★ 静态边：query / task 完成后**直接**进对话生成，不再让 supervisor 再判一次。
    # dialogue 本身产出 final_reply 后走 finalize → END。
    graph.add_edge("query_worker", "dialogue_worker")
    graph.add_edge("task_worker", "dialogue_worker")
    graph.add_edge("dialogue_worker", "finalize")

    graph.add_edge("finalize", END)
    return graph


# ---------------------------------------------------------------------------
# Compile w/ AsyncSqliteSaver
# ---------------------------------------------------------------------------

_COMPILED_GRAPH = None
_CHECKPOINTER_CM = None  # async context manager 的 __aenter__ 结果，进程内长期持有
_CHECKPOINTER = None


async def get_supervisor_graph() -> Any:
    """
    懒初始化带 checkpointer 的编译图。
    进程级单例；第一次 await 完成后，后续复用。
    """
    global _COMPILED_GRAPH, _CHECKPOINTER_CM, _CHECKPOINTER
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少 langgraph-checkpoint-sqlite 依赖。请在当前 venv 里运行：\n"
            "    pip install langgraph-checkpoint-sqlite\n"
            "若 requirements.txt 已更新过，重跑 `pip install -r requirements.txt` 也可；"
            "注意确认激活的是 `.\\venv\\Scripts\\Activate.ps1` 而不是系统解释器。"
        ) from e

    db_path = resolve_checkpoint_db_path()
    logger.info("LangGraph Checkpointer DB: %s", db_path)
    # AsyncSqliteSaver.from_conn_string 返回 async context manager
    _CHECKPOINTER_CM = AsyncSqliteSaver.from_conn_string(str(db_path))
    _CHECKPOINTER = await _CHECKPOINTER_CM.__aenter__()

    graph = build_supervisor_graph()
    _COMPILED_GRAPH = graph.compile(checkpointer=_CHECKPOINTER)
    return _COMPILED_GRAPH


async def reset_supervisor_graph() -> None:
    """仅测试 / 热重载使用；关闭 checkpointer 并清空缓存。"""
    global _COMPILED_GRAPH, _CHECKPOINTER_CM, _CHECKPOINTER
    if _CHECKPOINTER_CM is not None:
        try:
            await _CHECKPOINTER_CM.__aexit__(None, None, None)
        except Exception:  # pragma: no cover
            logger.warning("关闭 checkpointer 失败", exc_info=True)
    _COMPILED_GRAPH = None
    _CHECKPOINTER_CM = None
    _CHECKPOINTER = None


__all__ = [
    "build_supervisor_graph",
    "get_supervisor_graph",
    "reset_supervisor_graph",
    "resolve_checkpoint_db_path",
]
