"""TurnOrchestrator：单一路径的回合编排（替代 LangGraph 双路径，决策 D1）。

对外入口：
- TurnOrchestrator(...).run() → AsyncIterator[SSEEvent]（events.py 契约）
- OrchestratorDeps / default_deps（依赖注入）
"""

from services.orchestrator.context import TurnContext, assemble_context
from services.orchestrator.events import (
    SSEEvent,
    accumulate_usage,
    agent_status_event,
    content_event,
    done_event,
    error_event,
    meta_event,
    system_notice_event,
    tool_status_event,
)
from services.orchestrator.merge import MergeCoordinator, MergeOutcome
from services.orchestrator.turn import (
    OrchestratorDeps,
    TurnOrchestrator,
    default_deps,
    get_session_lock,
)

__all__ = [
    "MergeCoordinator",
    "MergeOutcome",
    "OrchestratorDeps",
    "SSEEvent",
    "TurnContext",
    "TurnOrchestrator",
    "accumulate_usage",
    "agent_status_event",
    "assemble_context",
    "content_event",
    "default_deps",
    "done_event",
    "error_event",
    "get_session_lock",
    "meta_event",
    "system_notice_event",
    "tool_status_event",
]
