"""后台子 Agent（fire-and-steer）：TaskRunner / SearchRunner / DialogueRunner。

对外入口：
- SubagentHandle / SubagentEvent / SessionSubagents（base）
- TaskRunner.launch(kind="task_draft"|"task_update", ...)（task_runner）
- SearchRunner.launch(...)（search_runner）
- DialogueRunner.launch(...)（dialogue_runner；由全局串行 worker 驱动，
  不注册进会话级 SessionSubagents——独立链路，见 base 模块说明）
"""

from services.subagents.base import (
    AgentLabel,
    SessionSubagents,
    SubagentBase,
    SubagentEvent,
    SubagentHandle,
    SubagentKind,
    count_pending_subagent_tasks,
    get_session_subagents,
    reset_session_subagents,
)
from services.subagents.dialogue_runner import DialogueRunner
from services.subagents.search_runner import SearchRunner
from services.subagents.task_runner import TaskRunner

__all__ = [
    "AgentLabel",
    "DialogueRunner",
    "SearchRunner",
    "SessionSubagents",
    "SubagentBase",
    "SubagentEvent",
    "SubagentHandle",
    "SubagentKind",
    "TaskRunner",
    "count_pending_subagent_tasks",
    "get_session_subagents",
    "reset_session_subagents",
]
