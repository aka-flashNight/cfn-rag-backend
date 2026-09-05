"""TaskRunner：任务拟定/修改后台子 Agent（对应 docs/v3-developer/03 §5）。

工具循环：LLM（非流式 + tools）→ 同批 tool_calls 并行执行 → 结果回填，
直至 draft 成功/失败或达轮限。轮限耗尽仍无草案 → 后端兜底（05 §5）。

事件：
- 启动/每轮：progress（ui_hint「拟定委托中」）；
- 校验失败：首次 intermediate（vague_note 只说类型不说数字，03 §3 硬性规则），
  之后降级为 progress「校验修正中」；
- 终态：final（draft/draft_summary/bargain_count/deviation_note/fallback 等）；
- LLM/工具失败：failed。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from services.agent_tools.handlers import execute_fallback_draft
from services.llm import LLMClient
from services.skills import SkillRegistry, get_skill_registry
from services.subagents.base import (
    AgentLabel,
    SubagentBase,
    SubagentEvent,
    SubagentHandle,
    SubagentKind,
)
from services.subagents.prompts import TASK_RUNNER_SYSTEM, task_runner_user_prompt
from services.tools.base import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

TASK_TOOLS = [
    "prepare_task_context",
    "draft_agent_task",
    "update_task_draft",
    "search_items",
    "search_stages",
    "list_skills",
    "read_skill",
    "read_skill_file",
]

_TERMINAL_STATUSES = ("draft_created", "draft_updated")


class TaskRunner(SubagentBase):
    kind: SubagentKind = "task_draft"
    agent: AgentLabel = "task"
    default_ui_hint = "拟定委托中"

    def __init__(self, *, direction: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.direction = direction
        self._intermediate_sent = False

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def _on_round_results(self, results: list[tuple[str, Any, ToolContext]]) -> None:
        saw_failure = False
        for _call_id, tr, _ctx in results:
            try:
                payload = json.loads(tr.result_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get("status") == "validation_failed":
                saw_failure = True
        if not saw_failure:
            return
        if not self._intermediate_sent:
            # 中间结果硬性规则：只说任务类型/方向，禁止数量/物品名/金额（03 §3）
            self._intermediate_sent = True
            self._push(SubagentEvent(
                kind="intermediate",
                ui_hint="校验修正中",
                vague_note="委托内容还在调整，稍等一下。",
            ))
        else:
            self._push(SubagentEvent(kind="progress", ui_hint="校验修正中"))

    def _terminal_payload(self, results: list[tuple[str, Any, ToolContext]]) -> Optional[dict[str, Any]]:
        for _call_id, tr, ctx in results:
            try:
                payload = json.loads(tr.result_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict) or payload.get("status") not in _TERMINAL_STATUSES:
                continue
            draft = ctx.pending_draft
            return {
                "status": payload.get("status"),
                "draft_id": payload.get("draft_id") or (draft or {}).get("draft_id", ""),
                "draft_summary": payload.get("draft_summary", ""),
                "draft": draft,
                "bargain_count": ctx.bargain_count,
                "repaired_notes": payload.get("repaired_notes") or [],
                "deviation_note": self._extract_deviation_note(),
                "fallback": bool((draft or {}).get("fallback")),
            }
        return None

    def _extract_deviation_note(self) -> str:
        """模型最终回复中「偏离说明：」开头的内容（03 §5 规则 1）。"""
        text = (getattr(self, "_last_model_text", "") or "").strip()
        marker = "偏离说明："
        idx = text.find(marker)
        if idx < 0:
            return ""
        return text[idx + len(marker):].strip()[:80]

    async def _on_exhausted(self) -> Optional[dict[str, Any]]:
        """轮限耗尽 → 后端兜底草案（05 §5）。"""
        self._push(SubagentEvent(kind="progress", ui_hint="生成保底方案"))
        outcome = execute_fallback_draft(
            direction=self.direction,
            reward_hint=getattr(self, "_reward_hint", "") or "",
            npc_name=self.ctx.npc_name,
            npc_faction=self.ctx.npc_faction,
            npc_challenge=self.ctx.npc_challenge,
            player_progress=self.ctx.player_progress,
            npc_affinity=self.ctx.npc_affinity,
            npc_states=self.ctx.npc_states,
            game_data=self.ctx.game_data,
            rag_context_text=self.ctx.rag_context_text,
        )
        try:
            payload = json.loads(outcome.result_json)
        except json.JSONDecodeError:
            return None
        if payload.get("status") != "draft_created":
            return None
        return {
            "status": "draft_created",
            "draft_id": payload.get("draft_id", ""),
            "draft_summary": payload.get("draft_summary", ""),
            "draft": outcome.draft,
            "bargain_count": outcome.bargain_count,
            "repaired_notes": [],
            "deviation_note": "",
            "fallback": True,
            "fallback_note": payload.get("fallback_note", ""),
        }

    async def run(self) -> None:
        await super().run()

    # ------------------------------------------------------------------

    @classmethod
    def launch(
        cls,
        *,
        kind: SubagentKind,
        llm: LLMClient,
        registry: ToolRegistry,
        direction: str,
        reward_hint: str = "",
        note: str = "",
        player_query: str = "",
        npc_name: str,
        npc_faction: str = "",
        npc_titles: Optional[list[str]] = None,
        npc_challenge: Optional[str] = None,
        player_progress: int = 1,
        progress_desc: str = "",
        npc_affinity: int = 0,
        npc_states: Optional[dict[str, Any]] = None,
        pending_draft: Optional[dict[str, Any]] = None,
        bargain_count: int = 0,
        draft_commit_valid: bool = False,
        game_data: Any = None,
        rag_context_text: Optional[str] = None,
        retrieve_fn: Optional[Callable[[str], str]] = None,
        max_rounds: int = 4,
        skill_registry: Optional[SkillRegistry] = None,
    ) -> SubagentHandle:
        """构造并启动后台任务，返回 Handle。"""
        ctx = ToolContext(
            npc_name=npc_name,
            npc_faction=npc_faction,
            npc_challenge=npc_challenge,
            player_progress=max(1, min(7, player_progress or 1)),
            npc_affinity=npc_affinity,
            npc_states=npc_states,
            game_data=game_data,
            pending_draft=pending_draft,
            bargain_count=bargain_count,
            draft_commit_valid=draft_commit_valid,
            retrieve_fn=retrieve_fn,
            rag_context_text=rag_context_text,
            skill_registry=skill_registry if skill_registry is not None else get_skill_registry(),
        )
        user_prompt = task_runner_user_prompt(
            direction=direction,
            reward_hint=reward_hint,
            note=note,
            player_query=player_query,
            npc_name=npc_name,
            npc_faction=npc_faction,
            npc_titles=npc_titles,
            npc_challenge=npc_challenge,
            player_progress=ctx.player_progress,
            progress_desc=progress_desc,
            pending_draft_summary=pending_draft_summary_text(pending_draft, game_data),
            skill_registry=ctx.skill_registry,
        )
        runner = cls(
            direction=direction,
            llm=llm,
            registry=registry,
            tool_names=TASK_TOOLS,
            system_prompt=TASK_RUNNER_SYSTEM,
            user_prompt=user_prompt,
            ctx=ctx,
            max_rounds=max(1, max_rounds),
            queue=asyncio.Queue(),
        )
        runner.kind = kind
        runner._reward_hint = reward_hint
        return SubagentHandle(kind=kind, agent="task", coro=runner.run(), events=runner.queue)


def pending_draft_summary_text(
    pending_draft: Optional[dict[str, Any]], game_data: Any,
) -> str:
    """update 模式注入用的草案摘要（轻量版，避免循环依赖 draft_formatting 的 rag 去重逻辑）。"""
    if not pending_draft:
        return ""
    from services.agent_tools.draft_formatting import _detailed_draft_summary

    return _detailed_draft_summary(pending_draft, game_data)
