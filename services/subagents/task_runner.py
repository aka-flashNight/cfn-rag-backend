"""TaskRunner：任务拟定/修改后台子 Agent（对应 docs/v3-developer/03 §5）。

v3 再次重构（路由工具化）后只有两种模式：
- draft：新草案。聊天主 Agent 只发路由声明，**不再预取候选**，所以本 Agent 每轮
  新草案都自己走 tools=[prepare_task_context] → 拿到候选后轮内动态收窄为
  [draft_agent_task]。task_type 由本 Agent 按选型规则自行判断，选错落进自己的
  重试循环（校验打回后可换类型重新 prepare），不再像旧设计那样一旦交流阶段选错
  就无法挽回。
- update：讨价还价/小幅修改，初表 tools=[update_task_draft, prepare_task_context]。
  仅当玩家明确要求整体改变任务方向时才走重拟：调用 prepare_task_context（拿到候选后
  轮内动态收窄为 [draft_agent_task]）提交新草案整体替换，等于重走两步式生成。

事件：
- 启动：progress（ui_hint「准备候选中」/「修改方案中」）；
- prepare 完成：intermediate（vague_note 只给方向、不给具体内容）——聊天主 Agent
  据此说一句模糊过渡话，填住拟定草案的静默期（03 §3）；
- 校验失败：首次 intermediate（只说类型不说数字），之后降为 progress；
- 终态：final（draft/draft_summary/bargain_count/deviation_note/fallback 等）；
- LLM/工具失败：failed。

定价卡依赖 task_type，而 task_type 要等 prepare 回来才知道，故 launch 时不注入，
改由 _post_round_messages 在 prepare 完成后按真实类型补投（追加消息，system 前缀不变）。

每一轮记 INFO 日志：轮次、工具表、结果 status、校验失败逐条摘要（可观测性）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from services.agent_tools.handlers import execute_fallback_draft
from services.agent_tools.pricing_rules import build_pricing_card
from services.agent_tools.schemas import TASK_TYPES
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
    build_direction_vague_note,
    build_task_system,
    build_task_user,
)
from services.tools.base import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_DRAFT_TOOLS = ["draft_agent_task"]
_PREPARE_TOOLS = ["prepare_task_context"]
# update 初表：常规修改走 update；玩家明确要求整体换向时才用 prepare 重拟（罕见路径）
_UPDATE_TOOLS = ["update_task_draft", "prepare_task_context"]

_TERMINAL_STATUSES = ("draft_created", "draft_updated")

MODE_DRAFT = "draft"
MODE_UPDATE = "update"

# prepare 结果里出现任一类型专属字段即视为候选池已就绪
_PREPARE_KEYS = {"collectable_items", "stage_list", "reward_item_candidates", "holdable_items",
                 "equipment_items", "special_items", "npc_list", "challenge_targets", "stage_loot_list"}


class TaskRunner(SubagentBase):
    kind: SubagentKind = "task_draft"
    agent: AgentLabel = "task"
    default_ui_hint = "拟定委托中"

    def __init__(self, *, mode: str = MODE_DRAFT, **kwargs: Any) -> None:
        self.direction = str(kwargs.pop("direction", "") or "")
        self.player_query = str(kwargs.pop("player_query", "") or "")
        super().__init__(**kwargs)
        self.mode = mode
        self._intermediate_sent = False
        # 两种模式都从「prepare 未完成」起步：draft 先只给 prepare；update 给
        # [update + prepare]，prepare 完成（重拟路径触发）后才收窄为 draft
        self._prepare_done = False
        # prepare 的 task_type 只存在于调用参数里（工具返回不带），故单独记录
        self._prepare_task_type = ""
        self._pricing_injected = False
        # prepare 返回里可能带「候选池被关键词全覆盖」的范围说明（否则为空）
        self._pool_note = ""

    # ------------------------------------------------------------------
    # 动态工具表（prepare 前：draft 只给 prepare，update 给 update+prepare；
    # prepare 完成后统一收窄为 draft）
    # ------------------------------------------------------------------

    def _tools_schema(self) -> list[dict[str, Any]]:
        if self._prepare_done:
            names = _DRAFT_TOOLS
        elif self.mode == MODE_UPDATE:
            names = _UPDATE_TOOLS
        else:
            names = _PREPARE_TOOLS
        out: list[dict[str, Any]] = []
        for name in names:
            tool = self.registry.get(name)
            if tool is not None:
                out.append(tool.to_openai_tool())
        return out

    # ------------------------------------------------------------------
    # prepare 参数捕获 → 定价卡补投 + 过渡方向
    # ------------------------------------------------------------------

    def _on_round_tool_calls(self, tool_calls: Optional[list[dict[str, Any]]]) -> None:
        for tc in tool_calls or []:
            func = tc.get("function", tc) if isinstance(tc, dict) else {}
            if str(func.get("name", "") or "") != "prepare_task_context":
                continue
            raw = func.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("TaskRunner: prepare 参数解析失败: %.200s", str(raw))
                continue
            task_type = str((args or {}).get("task_type") or "").strip()
            if task_type in TASK_TYPES:
                self._prepare_task_type = task_type
            else:
                logger.warning("TaskRunner: prepare 的 task_type 非法: %r", task_type)

    def _post_round_messages(self, results: list[tuple[str, Any, ToolContext]]) -> list[dict[str, Any]]:
        if self._pricing_injected or not self._prepare_done or not self._prepare_task_type:
            return []
        self._pricing_injected = True
        card = build_pricing_card(
            task_type=self._prepare_task_type,
            stage=self.ctx.player_progress,
            affinity=self.ctx.npc_affinity,
            game_data=self.ctx.game_data,
        )
        logger.info("TaskRunner 按 prepare 选型补投定价卡: task_type=%s", self._prepare_task_type)
        return [{
            "role": "user",
            "content": (
                f"【后台补充规则（候选已按 task_type={self._prepare_task_type} 取回，"
                "拟定草案时按此计算奖励）】\n" + card
            ),
        }]

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
                self._pool_note = str(payload.get("pool_note") or "")
            if status == "validation_failed":
                saw_failure = True
                for issue in payload.get("issues") or []:
                    logger.info(
                        "TaskRunner 校验失败 [%s][%s] %s",
                        issue.get("rule"), issue.get("field"), issue.get("message"),
                    )
            if status in _TERMINAL_STATUSES and payload.get("repaired_notes"):
                logger.info("TaskRunner 后端自动微调: %s", payload["repaired_notes"])

        if not self._prepare_done:
            table_label = "update+prepare" if self.mode == MODE_UPDATE else "prepare"
        else:
            table_label = "draft"
        logger.info("TaskRunner R%d tools=%s statuses=%s", self.rounds_used, table_label, statuses)

        if saw_prepare and not self._prepare_done:
            # prepare 完成 → 下一轮收窄为 draft，并请聊天主 Agent 说一句过渡话。
            # vague_note 只给方向：过渡语的职责是填住静默期，说具体了会和最终草案打架。
            self._prepare_done = True
            note = build_direction_vague_note(self._prepare_task_type)
            self._push(SubagentEvent(
                kind="intermediate",
                ui_hint="拟定委托中",
                vague_note=note,
                pool_note=self._pool_note,
            ))
            logger.info(
                "TaskRunner 候选池已就绪（task_type=%s），切换至 draft 节点并请求过渡话",
                self._prepare_task_type or "未知",
            )

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
                # 聊天主 Agent 按任务类型条件拼接说明要点（orchestrator.prompts.task_merge_hint）
                "task_type": self._prepare_task_type or (draft or {}).get("task_type", ""),
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
        direction = self.direction or self.player_query
        logger.warning("TaskRunner 轮限耗尽（draft 未成功），触发后端兜底草案: direction=%s", direction)
        outcome = execute_fallback_draft(
            direction=direction,
            reward_hint="",
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
            "task_type": (outcome.draft or {}).get("task_type", ""),
        }

    # ------------------------------------------------------------------

    @classmethod
    def launch(
        cls,
        *,
        kind: SubagentKind,
        llm: LLMClient,
        registry: ToolRegistry,
        direction: str = "",
        player_query: str = "",
        note: str = "",
        recent_dialogue: str = "",
        memory_summary: str = "",
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
        has_shop: bool = False,
        shop_reward_types: Optional[list[str]] = None,
        player_can_challenge: bool = False,
        game_data: Any = None,
        rag_context_text: Optional[str] = None,
        retrieve_fn: Optional[Callable[[str], str]] = None,
        max_rounds: int = 4,
    ) -> SubagentHandle:
        """构造并启动后台任务，返回 Handle。

        - kind=task_update：update 模式（讨价还价），初表 [update_task_draft, prepare_task_context]，
          玩家明确要求整体换向时才走 prepare→draft 重拟（工具表轮内动态收窄）；
        - kind=task_draft：draft 模式，先 prepare 后 draft（工具表轮内动态收窄）。
        """
        mode = MODE_UPDATE if kind == "task_update" else MODE_DRAFT
        # update 的定价/讨价卡在 launch 时就能算（task_type 来自既有草案）；
        # draft 的 task_type 要等 prepare，故此处为空串，定价卡走 _post_round_messages
        task_type = str((pending_draft or {}).get("task_type") or "") if mode == MODE_UPDATE else ""

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
            has_shop=has_shop,
            shop_reward_types=shop_reward_types,
            has_challenge=bool((npc_challenge or "").strip()),
            player_can_challenge=player_can_challenge,
            npc_faction=npc_faction,
        )
        if mode == MODE_UPDATE:
            from services.agent_tools.draft_formatting import _detailed_draft_summary

            user = build_task_user(
                direction_block=build_direction_for_task(player_query=player_query, note=note or ""),
                recent_dialogue=recent_dialogue,
                memory_summary=memory_summary,
                npc_name=npc_name,
                npc_faction=npc_faction,
                npc_titles=npc_titles,
                npc_challenge=npc_challenge,
                player_progress=ctx.player_progress,
                progress_desc=progress_desc,
                draft_summary=_detailed_draft_summary(
                    pending_draft or {}, game_data, npc_states=npc_states,
                ),
                player_note=note or "",
            )
        else:
            user = build_task_user(
                direction_block=build_direction_for_task(
                    direction=direction, player_query=player_query,
                ),
                recent_dialogue=recent_dialogue,
                memory_summary=memory_summary,
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
            player_query=player_query,
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
        runner.default_ui_hint = "修改方案中" if mode == MODE_UPDATE else "准备候选中"
        logger.info(
            "TaskRunner 启动: kind=%s mode=%s task_type=%s rounds=%d",
            kind, mode, task_type or "待 prepare 决定", max_rounds,
        )
        return SubagentHandle(kind=kind, agent="task", coro=runner.run(), events=runner.queue)
