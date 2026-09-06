"""SearchRunner：检索后台子 Agent（对应 docs/v3-developer/03 §6）。

输入 query + NPC 视角简述；工具循环 ≤3 轮；输出 final：findings（≤400 字结论，
附关键出处类型）。Tier-1 预检索保底由 orchestrator 上下文装配承担，本 Runner
负责模型主动发起的深挖。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from services.llm import LLMClient
from services.skills import SkillRegistry, get_skill_registry
from services.subagents.base import (
    AgentLabel,
    SubagentBase,
    SubagentHandle,
    SubagentKind,
)
from services.subagents.prompts import SEARCH_RUNNER_SYSTEM, search_runner_user_prompt
from services.tools.base import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

SEARCH_TOOLS = [
    "search_knowledge",
    "search_items",
    "search_stages",
]

_FINDINGS_MAX_CHARS = 400


class SearchRunner(SubagentBase):
    kind: SubagentKind = "search"
    agent: AgentLabel = "search"
    default_ui_hint = "查资料中"

    def _final_from_text(self, text: str) -> dict[str, Any]:
        findings = (text or "").strip()
        if len(findings) > _FINDINGS_MAX_CHARS:
            findings = findings[:_FINDINGS_MAX_CHARS].rstrip() + "…"
        return {"findings": findings}

    @classmethod
    def launch(
        cls,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        query: str,
        player_query: str = "",
        npc_name: str,
        npc_faction: str = "",
        npc_titles: Optional[list[str]] = None,
        npc_challenge: Optional[str] = None,
        player_progress: int = 1,
        npc_affinity: int = 0,
        npc_states: Optional[dict] = None,
        game_data: Any = None,
        retrieve_fn: Optional[Callable[[str], str]] = None,
        max_rounds: int = 3,
        skill_registry: Optional[SkillRegistry] = None,
    ) -> SubagentHandle:
        ctx = ToolContext(
            npc_name=npc_name,
            npc_faction=npc_faction,
            npc_challenge=npc_challenge,
            player_progress=max(1, min(7, player_progress or 1)),
            npc_affinity=npc_affinity,
            npc_states=npc_states,
            game_data=game_data,
            retrieve_fn=retrieve_fn,
            skill_registry=skill_registry if skill_registry is not None else get_skill_registry(),
        )
        user_prompt = search_runner_user_prompt(
            query=query,
            npc_name=npc_name,
            player_query=player_query,
        )
        runner = cls(
            llm=llm,
            registry=registry,
            tool_names=SEARCH_TOOLS,
            system_prompt=SEARCH_RUNNER_SYSTEM,
            user_prompt=user_prompt,
            ctx=ctx,
            max_rounds=max(1, max_rounds),
            queue=asyncio.Queue(),
        )
        return SubagentHandle(kind="search", agent="search", coro=runner.run(), events=runner.queue)
