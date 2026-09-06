"""TaskRunner：任务拟定/修改后台子 Agent（对应 docs/v3-developer/03 §5）。

两种情形（按交流阶段 prepare 的成败分流）：
- 情形 1（prepare 成功）：tools=[draft_agent_task]，候选池已在首轮 user 注入，直接 draft；
- 情形 2（prepare 失败/参数非法/候选池空）：**不回炉交流 Agent**——先 tools=[prepare_task_context]
  重新准备候选（必须继承 direction 思路），拿到候选后轮内动态收窄为 [draft_agent_task]。

事件：
- 启动：progress（ui_hint「拟定委托中」/「准备候选中」）；
- 校验失败：首次 intermediate（vague_note 只说类型不说数字，03 §3 硬性规则）；
- 终态：final（draft/draft_summary/bargain_count/deviation_note/fallback 等）；
- LLM/工具失败：failed。

每一轮记 INFO 日志：轮次、工具表、结果 status、校验失败逐条摘要（可观测性）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from services.agent_tools.handlers import execute_fallback_draft
from services.llm import LLMClient
from services.subagents.base import (
    AgentLabel,
    SubagentBase,
    SubagentEvent,
    SubagentHandle,
    SubagentKind,
)
from services.subagents.prompts import (
    build_direction_for_task,
    build_task_system,
    build_task_user,
)
from services.tools.base import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_DRAFT_TOOLS = ["draft_agent_task"]
_PREPARE_TOOLS = ["prepare_task_context"]
_UPDATE_TOOLS = ["update_task_draft"]

_TERMINAL_STATUSES = ("draft_created", "draft_updated")

# 拟定模式（情形 1 直接 draft；情形 2 先 prepare 再 draft）
MODE_DRAFT = "draft"
MODE_PREPARE_THEN_DRAFT = "prepare_then_draft"
MODE_UPDATE = "update"


class TaskRunner(SubagentBase):
    kind: SubagentKind = "task_draft"
    agent: AgentLabel = "task"
    default_ui_hint = "拟定委托中"

    def __init__(self, *, mode: str = MODE_DRAFT, **kwargs: Any) -> None:
        self.direction = str(kwargs.pop("direction", "") or "")
        self._reward_hint = str(kwargs.pop("reward_hint", "") or "")
        super().__init__(**kwargs)
        self.mode = mode
        self._intermediate_sent = False
        self._prepare_done = mode != MODE_PREPARE_THEN_DRAFT

    # ------------------------------------------------------------------
    # 动态工具表（prepare 未完成时只给 prepare；完成后收窄为 draft/update）
    # ------------------------------------------------------------------

    def _tools_schema(self) -> list[dict[str, Any]]:
        if self.mode == MODE_UPDATE:
            names = _UPDATE_TOOLS
        elif not self._prepare_done:
            names = _PREPARE_TOOLS
        else:
            names = _DRAFT_TOOLS
        out: list[dict[str, Any]] = []
        for name in names:
            tool = self.registry.get(name)
            if tool is not None:
                out.append(tool.to_openai_tool())
        return out

    # ------------------------------------------------------------------
    # 事件与日志
    # ------------------------------------------------------------------

    def _on_round_results(self, results: list[tuple[str, Any, ToolContext]]) -> None:
        statuses: list[str] = []
        saw_failure = False
        saw_prepare = False
        for _call_id, tr, ctx in results:
            try:
                payload = json.loads(tr.result_json)
            except (json.JSONDecodeError, TypeError):
                statuses.append("unreadable")
                continue
            payload = payload if isinstance(payload, dict) else {}
            status = payload.get("status")
            statuses.append(str(status))
            # prepare 结果无 status 键：以候选字段的存在判定完成
            if not self._prepare_done and (_PREPARE_KEYS & set(payload.keys())):
                saw_prepare = True
            if status == "validation_failed":
                saw_failure = True
                for issue in payload.get("issues") or []:
                    logger.info(
                        "TaskRunner 校验失败 [%s][%s] %s",
                        issue.get("rule"), issue.get("field"), issue.get("message"),
                    )
            if status in _TERMINAL_STATUSES and payload.get("repaired_notes"):
                logger.info("TaskRunner 后端自动微调: %s", payload["repaired_notes"])

        logger.info("TaskRunner R%d tools=%s statuses=%s", self.rounds_used,
                    "prepare" if not self._prepare_done else ("update" if self.mode == MODE_UPDATE else "draft"),
                    statuses)

        if saw_prepare and not self._prepare_done:
            # prepare 完成 → 下一轮收窄为 draft
            self._prepare_done = True
            logger.info("TaskRunner 候选池已就绪，切换至 draft 节点")

        if saw_failure:
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
        """轮限耗尽 → 后端兜底草案（仅当模型始终未交出结构完整草案时才会走到）。"""
        self._push(SubagentEvent(kind="progress", ui_hint="生成保底方案"))
        logger.warning("TaskRunner 轮限耗尽（draft 未成功），触发后端兜底草案: direction=%s", self.direction)
        outcome = execute_fallback_draft(
            direction=self.direction,
            reward_hint=self._reward_hint,
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
            logger.warning("TaskRunner 兜底草案构造失败: %s", payload.get("message"))
            return None
        logger.info("TaskRunner 兜底草案已生成（fallback=True）")
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
        prepared_context: Optional[str] = None,
        prepare_error: str = "",
        recent_dialogue: str = "",
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
    ) -> SubagentHandle:
        """构造并启动后台任务，返回 Handle。

        - kind=task_update：update 模式（讨价还价），tools=[update_task_draft]；
        - kind=task_draft 且 prepared_context 非空：情形 1，tools=[draft]；
        - kind=task_draft 且 prepared_context 为空：情形 2，先 prepare 后 draft（动态收窄）。
        """
        if kind == "task_update":
            mode = MODE_UPDATE
            task_type = str((pending_draft or {}).get("task_type") or "")
        else:
            prepared_ok = bool((prepared_context or "").strip())
            mode = MODE_DRAFT if prepared_ok else MODE_PREPARE_THEN_DRAFT
            # task_type 用于定价卡：从 prepared_context 解析；情形 2 由后端兜底解析
            task_type = _task_type_from_prepared(prepared_context) or _guess_task_type(direction)

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
            skill_registry=None,  # v3：子 Agent 不再持有 skills 通道
        )
        system = build_task_system(
            mode=mode,
            task_type=task_type,
            stage=ctx.player_progress,
            affinity=npc_affinity,
            prepare_error=prepare_error,
        )
        if mode == MODE_UPDATE:
            from services.agent_tools.draft_formatting import _detailed_draft_summary

            user = build_task_user(
                direction_block=build_direction_for_task(direction="", note=note or ""),
                recent_dialogue=recent_dialogue,
                npc_name=npc_name,
                npc_faction=npc_faction,
                npc_titles=npc_titles,
                npc_challenge=npc_challenge,
                player_progress=ctx.player_progress,
                progress_desc=progress_desc,
                draft_summary=_detailed_draft_summary(pending_draft or {}, game_data),
                player_note=note or "",
            )
        else:
            candidates = prepared_context or ""
            user = build_task_user(
                direction_block=build_direction_for_task(
                    direction=direction, reward_hint=reward_hint, note=note,
                ),
                recent_dialogue=recent_dialogue,
                candidates_block=candidates,
                npc_name=npc_name,
                npc_faction=npc_faction,
                npc_titles=npc_titles,
                npc_challenge=npc_challenge,
                player_progress=ctx.player_progress,
                progress_desc=progress_desc,
            )
        runner = cls(
            mode=mode,
            direction=direction,
            reward_hint=reward_hint,
            llm=llm,
            registry=registry,
            tool_names=[],  # 动态表见 _tools_schema
            system_prompt=system,
            user_prompt=user,
            ctx=ctx,
            max_rounds=max(1, max_rounds),
            queue=asyncio.Queue(),
        )
        runner.kind = kind
        runner.default_ui_hint = "准备候选中" if mode == MODE_PREPARE_THEN_DRAFT else (
            "修改方案中" if mode == MODE_UPDATE else "拟定委托中"
        )
        logger.info(
            "TaskRunner 启动: kind=%s mode=%s task_type=%s prepared=%s rounds=%d",
            kind, mode, task_type, bool(prepared_context), max_rounds,
        )
        return SubagentHandle(kind=kind, agent="task", coro=runner.run(), events=runner.queue)


_PREPARE_KEYS = {"collectable_items", "stage_list", "reward_item_candidates", "holdable_items",
                 "equipment_items", "special_items", "npc_list", "challenge_targets", "stage_loot_list"}


def _task_type_from_prepared(prepared_context: Optional[str]) -> str:
    """从 prepare 结果 JSON 推断 task_type（定价卡用；解析失败返回空串）。"""
    if not prepared_context:
        return ""
    try:
        data = json.loads(prepared_context)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    if data.get("stage_list") or data.get("challenge_targets") or data.get("stage_loot_list"):
        return "通关"
    if data.get("npc_list"):
        return "问候"
    if data.get("equipment_items"):
        return "装备缴纳"
    if data.get("special_items"):
        return "特殊物品获取"
    if data.get("holdable_items"):
        return "物品持有"
    return "资源收集"


def _guess_task_type(direction: str) -> str:
    """prepare 失败时的类型兜底解析（与 fallback.detect_task_type 同规则）。"""
    from services.agent_tools.fallback import detect_task_type

    return detect_task_type(direction)
