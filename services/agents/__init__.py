"""
Supervisor + Worker 多 Agent 编排包（路线三，对齐 2026.04 业界实践）。

架构：

    START ─► prepare_context ─► supervisor ─┐
                                             │  (路由 + 可选 interim_reply)
                                             ├─► query_worker  ─┐
                                             ├─► task_worker   ─┼─► supervisor 回环（最多 MAX_HOPS）
                                             ├─► dialogue_worker┘
                                             └─► finalize ─► END

- supervisor 是「纯路由 + 可选短伪流式 interim_reply」形态（用户确认方案：supervisor_pure_with_interim）。
- 每个 Worker 是一个子图，复用 ``services.agent_graph.nodes`` 的决策/工具/生成节点，
  但 **tools 过滤** 后只看到与自身职责相关的原子工具 + system 元工具。
- Checkpointer（``AsyncSqliteSaver``）在 Route 3 启用，
  ``thread_id = session_id``；``interrupt_before=["await_confirm"]`` 支持 HITL v2。
- task_draft 不再写 ``services/task_draft_store.py``，而是随 checkpoint 一起持久化。
"""

from __future__ import annotations

from services.agents.state import SupervisorState

__all__ = ["SupervisorState"]
