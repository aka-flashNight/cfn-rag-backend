"""
Agent 工具分发：对外 ``dispatch_tool_call`` 入口，并再导出草案格式化与各 skill 执行实现（兼容旧 import 路径）。
"""

from __future__ import annotations

from typing import Any, Optional

from services.game_data.registry import GameDataRegistry

from services.agent_tools.draft_formatting import (
    _build_draft_entity_context_lines,
    _detailed_draft_summary,
    _format_items_with_price,
)
from services.agent_tools.skill_tool_executors import (
    execute_cancel_agent_task,
    execute_confirm_agent_task,
    execute_draft_agent_task,
    execute_prepare_task_context,
    execute_search_knowledge,
    execute_update_npc_mood,
    execute_update_task_draft,
)


def dispatch_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    npc_name: str = "",
    npc_faction: str = "",
    npc_challenge: Optional[str] = None,
    player_progress: int = 1,
    npc_affinity: int = 0,
    npc_states: Optional[dict[str, Any]] = None,
    game_data: Optional[GameDataRegistry] = None,
    pending_draft: Optional[dict[str, Any]] = None,
    retrieve_fn: Any = None,
    rag_context_text: Optional[str] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
    """
    统一工具分发入口（委托 services.skills.SkillRegistry）。

    返回: (tool_result_str, updated_pending_draft, task_write_result)
    """
    from services.skills import get_skill_registry
    from services.skills.base import SkillDispatchContext

    ctx = SkillDispatchContext(
        npc_name=npc_name,
        npc_faction=npc_faction,
        npc_challenge=npc_challenge,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
        npc_states=npc_states,
        game_data=game_data,
        pending_draft=pending_draft,
        retrieve_fn=retrieve_fn,
        rag_context_text=rag_context_text,
    )
    return get_skill_registry().dispatch(tool_name, tool_args, ctx)


__all__ = [
    "dispatch_tool_call",
    "_detailed_draft_summary",
    "_format_items_with_price",
    "_build_draft_entity_context_lines",
    "execute_prepare_task_context",
    "execute_draft_agent_task",
    "execute_update_task_draft",
    "execute_confirm_agent_task",
    "execute_cancel_agent_task",
    "execute_update_npc_mood",
    "execute_search_knowledge",
]
