"""
SupervisorState：多 Agent 编排共享状态（扩展 AgentState）。

字段布局与 AgentState 尽量**向后兼容**：已有节点读到 SupervisorState 时，除了多看到几个新字段以外，行为不应变化。
新增字段分两类：

1. Supervisor 路由与防护 ——
   - ``routing_decision``：本轮 supervisor 决定的 worker 名字（"query" / "task" / "dialogue" / "end"）。
   - ``routing_reason``：路由解释（短中文，仅用于 debug / SSE agent_status）。
   - ``interim_reply``：supervisor 可选产出的短伪流式回复（<= 80 字，用于"任务准备中"穿插对白）。
   - ``worker_hops``：主图回环次数（每次 supervisor -> worker -> supervisor 视为一跳）。
   - ``agent_call_counts``：各 worker 的调用次数（用于 per-agent 调用上限防护）。
   - ``agent_consecutive_failures``：连续失败计数（同一 worker 连续 N 次失败将进入黑名单）。
   - ``agent_blacklist``：被熔断的 worker 名单（本请求内不再调用）。
   - ``token_budget_spent``：粗略累计 token 开销（超过阈值触发熔断）。

2. HITL v2 —— 
   - ``awaiting_confirmation``：是否处于"待玩家确认任务草案"的挂起状态。
   - ``confirmation_draft_id``：挂起时待确认的 draft_id。
   - ``confirmation_payload``：前端/玩家回复后传入的 confirm 参数（title / description / dialogues）。
"""

from __future__ import annotations

from typing import Any, Optional

from services.agent_graph.state import AgentState


# Sentinel sub-dict keys used across nodes
RoutingDecision = str  # Literal["query", "task", "dialogue", "end", "await_confirm"]


class SupervisorState(AgentState, total=False):
    # Supervisor 路由
    routing_decision: RoutingDecision
    routing_reason: str
    interim_reply: str
    interim_reply_emitted: bool  # 避免同一请求反复发 interim

    # 编排防护
    worker_hops: int
    agent_call_counts: dict[str, int]
    agent_consecutive_failures: dict[str, int]
    agent_blacklist: list[str]
    token_budget_spent: int

    # 每个 worker 完成后写入的子结果
    last_worker_name: str
    last_worker_ok: bool
    last_worker_summary: str  # 供 supervisor 下一轮决策参考的 worker 输出摘要

    # HITL v2
    awaiting_confirmation: bool
    confirmation_draft_id: Optional[str]
    confirmation_payload: Optional[dict[str, Any]]

    # Per-worker tool scope（由 supervisor 动态注入；在 tool_executor_node 之外被忽略）
    _active_worker: str
    _active_worker_tool_names: list[str]

    # 累积 SSE 事件（沿用 AgentState._ui_events 之外，额外记录 agent_status / interim）
    # NOTE: 新事件统一走 _ui_events（tool_executor_node 已使用），
    # 以 event_type 区分：agent_status / interim_content / interim_done / pending_confirmation。


__all__ = ["SupervisorState", "RoutingDecision"]
