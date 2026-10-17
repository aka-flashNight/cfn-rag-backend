"""DialogueRunner：对话修改子 Agent（阶段四 B4-2，主体三轮＋一次打回重试）。

轮数上限由流程定死，不靠提示词自觉（用户口径 §4.0.3）：

- 第 1 轮：只给 search_npc_dialogues。模型若判断不需要改，直接输出文本＝本次结束
  （记「不动」，final payload changed=False）；
- 第 2 轮：查询未满两次时可再查一次，也可直接调用 edit_npc_dialogues 改；
- 查询用满两次后：查询工具与它的全部提示词一并摘除——system 整体换头成无查询
  表述的版本（``_before_round``），工具表只剩 edit_npc_dialogues；
- 第 3 轮：只能改；
- 第 4 轮：前一轮有组被打回时才会用到，按打回原因重交一次。

终态：
- 一批修改全部生效（无 rejected）即 final（changed=True）；
- 一批里部分生效、部分打回时不终态——已生效的组保留，模型拿打回原因重交；
- 打回只给一次重试机会：重试后仍有 rejected，则以「已落盘的组＋最终打回原因」
  收尾，不再继续烧轮次；
- 轮限耗尽仍无落盘 → final（changed=False, exhausted=True）。

本 Runner 由全局串行 worker 驱动（services/dialogue/worker.py），
不注册进会话级 SessionSubagents——它是完全派出去的独立链路（§4.0.1）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from services.llm import LLMClient
from services.subagents.base import (
    AgentLabel,
    SubagentBase,
    SubagentHandle,
    SubagentKind,
)
from services.subagents.prompts import build_dialogue_system, build_dialogue_user
from services.tools.base import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_SEARCH_TOOL = "search_npc_dialogues"
_EDIT_TOOL = "edit_npc_dialogues"

#: 查询次数硬上限（第 1 轮一次＋第 2 轮一次）
MAX_SEARCH_CALLS = 2
#: 主体三轮（查 → 可查/改 → 只改）＋至多一次打回重试轮
DIALOGUE_MAX_ROUNDS = 4


class DialogueRunner(SubagentBase):
    kind: SubagentKind = "dialogue"
    agent: AgentLabel = "dialogue"
    default_ui_hint = "改写对话中"

    def __init__(self, *, edit_only_system: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._edit_only_system = edit_only_system
        self._search_used = 0
        self._system_swapped = False
        # 跨轮累计：已生效的组不能因为下一轮只重交被打回的组而丢失
        self._applied: list[str] = []
        self._latest_rejected: list[str] = []
        self._last_edit: Optional[dict[str, Any]] = None
        self._rejection_retry_used = False

    # ------------------------------------------------------------------
    # 动态工具表与查询摘除（流程定死轮数，不靠提示词自觉）
    # ------------------------------------------------------------------

    def _tools_schema(self) -> list[dict[str, Any]]:
        if self.rounds_used <= 1:
            names = [_SEARCH_TOOL]
        elif self.rounds_used == 2:
            names = ([_SEARCH_TOOL] if self._search_used < MAX_SEARCH_CALLS else []) + [_EDIT_TOOL]
        else:
            names = [_EDIT_TOOL]
        out: list[dict[str, Any]] = []
        for name in names:
            tool = self.registry.get(name)
            if tool is not None:
                out.append(tool.to_openai_tool())
        return out

    def _before_round(self, messages: list[dict[str, Any]]) -> None:
        """查询工具不再下发的轮次整体换头：system 里不再出现任何查询/范围相关表述。

        触发时机与工具表严格同步（注册了才提、不注册一个字都不提）：
        查询用满两次后的第 2 轮，以及无论如何都只给修改工具的第 3 轮。
        """
        if self._system_swapped:
            return
        search_offered = self.rounds_used <= 1 or (
            self.rounds_used == 2 and self._search_used < MAX_SEARCH_CALLS
        )
        if search_offered:
            return
        self._system_swapped = True
        messages[0] = {"role": "system", "content": self._edit_only_system}
        logger.info(
            "DialogueRunner 第 %d 轮起摘除查询工具并换头（查询已用 %d 次）",
            self.rounds_used, self._search_used,
        )

    def _on_round_tool_calls(self, tool_calls: Optional[list[dict[str, Any]]]) -> None:
        for tc in tool_calls or []:
            func = tc.get("function", tc) if isinstance(tc, dict) else {}
            if str(func.get("name", "") or "") == _SEARCH_TOOL:
                self._search_used += 1

    # ------------------------------------------------------------------
    # 终态与收尾
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_edit_payload(
        results: list[tuple[str, Any, ToolContext]],
    ) -> Optional[dict[str, list[str]]]:
        """合并本轮所有 edit 工具结果；同批多个 edit 调用不丢任何一条。

        查询工具的结果是纯文本，不是 JSON，解析失败直接跳过；只有带 status 的
        edit 结果才计入。
        """
        applied: list[str] = []
        rejected: list[str] = []
        found = False
        for _call_id, tr, _ctx in results:
            try:
                payload = json.loads(tr.result_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict) or "status" not in payload:
                continue
            found = True
            applied.extend(str(item) for item in (payload.get("applied") or []))
            rejected.extend(str(item) for item in (payload.get("rejected") or []))
        if not found:
            return None
        return {"applied": applied, "rejected": rejected}

    def _edit_final_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "changed": bool(self._applied),
            "applied": list(self._applied),
            "rejected": list(self._latest_rejected),
        }
        if not self._applied and self._latest_rejected:
            payload["note"] = "修改被打回，未落盘。"
        return payload

    def _terminal_payload(self, results: list[tuple[str, Any, ToolContext]]) -> Optional[dict[str, Any]]:
        payload = self._collect_edit_payload(results)
        if payload is None:
            return None

        if payload["applied"] or payload["rejected"]:
            self._last_edit = payload
        for item in payload["applied"]:
            if item not in self._applied:
                self._applied.append(item)
        if payload["rejected"]:
            self._latest_rejected = payload["rejected"]
        elif payload["applied"]:
            # 重试的组已经全部生效，旧的打回原因随之清掉
            self._latest_rejected = []

        if payload["rejected"]:
            if not self._rejection_retry_used and self.rounds_used < self.max_rounds:
                self._rejection_retry_used = True
                logger.info(
                    "DialogueRunner 修改打回（第 %d 轮）: applied=%d rejected=%d，按原因重试一次",
                    self.rounds_used, len(self._applied), len(self._latest_rejected),
                )
                return None
            logger.info(
                "DialogueRunner 修改打回且不再重试: applied=%d rejected=%d（第 %d 轮）",
                len(self._applied), len(self._latest_rejected), self.rounds_used,
            )
            return self._edit_final_payload()

        if payload["applied"]:
            logger.info(
                "DialogueRunner 修改落盘: applied=%d（第 %d 轮）",
                len(self._applied), self.rounds_used,
            )
            return self._edit_final_payload()

        # status=noop：本次调用没有产生修改，也没有打回原因；继续等模型动作
        return None

    def _final_from_text(self, text: str) -> dict[str, Any]:
        """文本收笔＝「不动」；轮次用尽仍无落盘＝exhausted。

        模型只要输出过文本就会 break 出工具循环。若此前已经有过修改尝试，
        以修改结果收尾（已落盘的组不能因为模型最后只输出一句话而丢失）。
        """
        if self._last_edit is not None:
            return self._edit_final_payload()
        note = (text or "").strip()
        if not note:
            logger.warning("DialogueRunner 轮次用尽仍无落盘")
            return {"changed": False, "exhausted": True}
        logger.info("DialogueRunner 判断不动: %.80s", note)
        return {"changed": False, "note": note[:200]}

    # ------------------------------------------------------------------

    @classmethod
    def launch(
        cls,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        npc_name: str,
        profile_block: str = "",
        relationship_block: str = "",
        setting_block: str = "",
        source_trigger: str = "",
        reason: str = "",
        instruction: str = "",
        source_task_id: Optional[int] = None,
        task_block: str = "",
        publish_block: str = "",
        game_data: Any = None,
        slot_key: str = "",
        max_rounds: int = DIALOGUE_MAX_ROUNDS,
    ) -> SubagentHandle:
        """构造并启动后台任务，返回 Handle。

        材料块（setting/profile/relationship/task/publish）由委派方装配；触发来源
        （reason/source_task_id/source_trigger）同时进 ctx 供 edit 工具透传给覆盖层；
        slot_key 把整条链路（查询候选＋覆盖层读写）钉在委派受理当刻的存档上。
        """
        ctx = ToolContext(
            npc_name=npc_name,
            game_data=game_data,
            dialogue_reason=reason,
            dialogue_source_task_id=source_task_id,
            dialogue_source_trigger=source_trigger,
            slot_key=slot_key,
            skill_registry=None,  # 子 Agent 不持有 skills 通道（v3 口径）
        )
        system = build_dialogue_system(
            npc_name=npc_name,
            profile_block=profile_block,
            relationship_block=relationship_block,
            setting_block=setting_block,
            allow_search=True,
        )
        edit_only_system = build_dialogue_system(
            npc_name=npc_name,
            profile_block=profile_block,
            relationship_block=relationship_block,
            setting_block=setting_block,
            allow_search=False,
        )
        user = build_dialogue_user(
            source_trigger=source_trigger,
            reason=reason,
            instruction=instruction,
            task_block=task_block,
            publish_block=publish_block,
        )
        runner = cls(
            edit_only_system=edit_only_system,
            llm=llm,
            registry=registry,
            tool_names=[],  # 动态表见 _tools_schema
            system_prompt=system,
            user_prompt=user,
            ctx=ctx,
            max_rounds=max(1, min(DIALOGUE_MAX_ROUNDS, max_rounds)),
            queue=asyncio.Queue(),
        )
        logger.info(
            "DialogueRunner 启动: npc=%s trigger=%s task_id=%s",
            npc_name, source_trigger or "未标注", source_task_id,
        )
        return SubagentHandle(kind="dialogue", agent="dialogue", coro=runner.run(), events=runner.queue)
