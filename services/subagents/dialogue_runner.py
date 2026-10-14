"""DialogueRunner：对话修改子 Agent（阶段四 B4-2，三轮固定流程）。

轮数上限由流程定死，不靠提示词自觉（用户口径 §4.0.3）：

- 第 1 轮：只给 search_npc_dialogues。模型若判断不需要改，直接输出文本＝本次结束
  （记「不动」，final payload changed=False）；
- 第 2 轮：查询未满两次时可再查一次，也可直接调用 edit_npc_dialogues 改；
- 查询用满两次后：查询工具与它的全部提示词一并摘除——system 整体换头成无查询
  表述的版本（``_before_round``），工具表只剩 edit_npc_dialogues；
- 第 3 轮：只能改。

终态：edit 返回 status=ok（有内容落盘）即 final（changed=True）；
edit 被打回（rejected）不终态，模型按打回原因修正后下一轮重交；
轮限耗尽仍无落盘 → final（changed=False, exhausted=True）。

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
#: 固定三轮
DIALOGUE_MAX_ROUNDS = 3


class DialogueRunner(SubagentBase):
    kind: SubagentKind = "dialogue"
    agent: AgentLabel = "dialogue"
    default_ui_hint = "改写对话中"

    def __init__(self, *, edit_only_system: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._edit_only_system = edit_only_system
        self._search_used = 0
        self._system_swapped = False

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

    def _terminal_payload(self, results: list[tuple[str, Any, ToolContext]]) -> Optional[dict[str, Any]]:
        for _call_id, tr, _ctx in results:
            try:
                payload = json.loads(tr.result_json)
            except (json.JSONDecodeError, TypeError):
                continue  # 查询工具的结果是纯文本，不是 JSON
            if not isinstance(payload, dict):
                continue
            if payload.get("status") != "ok":
                continue
            logger.info(
                "DialogueRunner 修改落盘: applied=%d rejected=%d（第 %d 轮）",
                len(payload.get("applied") or []), len(payload.get("rejected") or []),
                self.rounds_used,
            )
            return {
                "changed": True,
                "applied": payload.get("applied") or [],
                "rejected": payload.get("rejected") or [],
            }
        return None

    def _final_from_text(self, text: str) -> dict[str, Any]:
        """文本收笔＝「不动」；三轮工具调用用尽仍无落盘＝exhausted。

        模型只要输出过文本就会 break 出工具循环，故走到这里时 text 非空即「不动」、
        为空即最后一轮仍是（被打回的）工具调用、轮限耗尽。
        """
        note = (text or "").strip()
        if not note:
            logger.warning("DialogueRunner 三轮用尽仍无落盘")
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
