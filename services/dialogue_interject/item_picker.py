"""物品选择子 Agent：两轮 pick / none / search。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional, Sequence

from services.dialogue_interject.item_pool import (
    GOLD,
    remaining_pool_counts,
    sample_names,
    search_names,
)
from services.dialogue_interject.item_value import format_item_info
from services.llm import ChatRequest, LLMClient

logger = logging.getLogger(__name__)

_PICK_TIMEOUT_S = 30.0
_TOTAL_TIMEOUT_S = 60.0
_CANDIDATE_COUNT = 14

PICKER_SYSTEM = """你是物品选择器。根据玩家的指示和候选列表，只返回一个 JSON 对象，不要 Markdown、不要解释。
可选动作：
{"action":"pick","name":"候选中的精确物品名"}
{"action":"none"}
{"action":"search","keyword":"新的检索关键词"}
规则：
1. 只能 pick 候选列表中出现的精确物品名。
2. 只有认为换一个关键词可能找到更合适的物品时，才使用 search；最多一次。
3. 第二轮只能 pick 或 none，不能再 search。
4. 没有合适的就 none，不要编造物品名。"""


PickerCaller = Callable[[list[dict[str, str]]], Awaitable[str]]


class ItemPicker:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        caller: PickerCaller | None = None,
    ) -> None:
        self.llm = llm
        self.caller = caller

    async def pick(
        self,
        *,
        direction: str,
        name: str,
        keyword: str,
        instruction: str,
        pool: Sequence[str],
        game_data: Any,
    ) -> Optional[str]:
        exact = (name or "").strip()
        if exact and exact in pool:
            return exact
        if not pool:
            return None

        keyword = (keyword or "").strip()
        instruction = (instruction or "").strip()
        candidates = search_names(game_data, pool, keyword, limit=10) if keyword else []
        if not candidates:
            candidates = sample_names(
                pool,
                count=_CANDIDATE_COUNT,
                gold_first=(direction == "take"),
            )
        if not candidates:
            return None

        try:
            return await asyncio.wait_for(
                self._run(
                    direction=direction,
                    pool=pool,
                    game_data=game_data,
                    candidates=candidates,
                    instruction=instruction,
                ),
                timeout=_TOTAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[interject] 物品选择子 Agent 超时 direction=%s name=%r keyword=%r instruction=%r",
                direction, name, keyword, instruction,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - 选择失败只影响本次工具
            logger.warning(
                "[interject] 物品选择子 Agent 失败 direction=%s name=%r keyword=%r instruction=%r: %s",
                direction, name, keyword, instruction, exc,
            )
            return None

    async def _run(
        self,
        *,
        direction: str,
        pool: Sequence[str],
        game_data: Any,
        candidates: list[str],
        instruction: str,
    ) -> Optional[str]:
        first = await self._call(
            self._messages(
                direction=direction,
                pool=pool,
                game_data=game_data,
                candidates=candidates,
                instruction=instruction,
                second_round=False,
            )
        )
        action, value = _parse_picker_output(first)
        if action == "pick":
            picked = _validate(value, pool)
            logger.info(
                "[interject] 物品选择子 Agent pick=%r pool=%d candidates=%d",
                picked, len(pool), len(candidates),
            )
            return picked
        if action == "none":
            logger.info("[interject] 物品选择子 Agent none pool=%d candidates=%d", len(pool), len(candidates))
            return None
        if action != "search":
            logger.info("[interject] 物品选择子 Agent 非法输出=%r", first[:120])
            return None

        second_candidates = search_names(game_data, pool, value, limit=10)
        if not second_candidates:
            logger.info(
                "[interject] 物品选择子 Agent 再查无结果 keyword=%r pool=%d",
                value, len(pool),
            )
            return None
        second = await self._call(
            self._messages(
                direction=direction,
                pool=pool,
                game_data=game_data,
                candidates=second_candidates,
                instruction=instruction,
                second_round=True,
            )
        )
        action, value = _parse_picker_output(second)
        if action == "pick":
            picked = _validate(value, pool)
            logger.info(
                "[interject] 物品选择子 Agent 二轮 pick=%r pool=%d candidates=%d",
                picked, len(pool), len(second_candidates),
            )
            return picked
        logger.info(
            "[interject] 物品选择子 Agent 二轮 %s pool=%d candidates=%d",
            action or "非法输出", len(pool), len(second_candidates),
        )
        return None

    @staticmethod
    def _messages(
        *,
        direction: str,
        pool: Sequence[str],
        game_data: Any,
        candidates: list[str],
        instruction: str,
        second_round: bool,
    ) -> list[dict[str, str]]:
        lines = [format_item_info(game_data, name) for name in candidates]
        counts, total = remaining_pool_counts(game_data, pool, candidates)
        count_text = "、".join(f"{key}{value}" for key, value in counts.items() if value)
        direction_text = "送给玩家" if direction == "give" else "向玩家索取"
        header = (
            f"方向：{direction_text}。\n"
            f"有效池共 {total} 件；本轮候选之外剩余：{count_text or '无'}。\n"
            f"候选列表（{len(candidates)} 件）："
        )
        body = "\n".join(f"- {line}" for line in lines)
        if second_round:
            tail = "这是第二轮，只能返回 pick 或 none，不能再 search。"
        else:
            tail = "请返回 pick / none / search。"
        content = "\n".join([
            header,
            body,
            f"玩家指示：{instruction or '（无）'}",
            tail,
        ])
        return [
            {"role": "system", "content": PICKER_SYSTEM},
            {"role": "user", "content": content},
        ]

    async def _call(self, messages: list[dict[str, str]]) -> str:
        async def _invoke() -> str:
            if self.caller is not None:
                return await self.caller(messages)
            if self.llm is None:
                raise RuntimeError("物品选择器没有 LLM")
            result = await self.llm.chat(ChatRequest(
                messages=messages,
                purpose="chat",
                send_image=False,
            ))
            return result.content or ""
        return await asyncio.wait_for(_invoke(), timeout=_PICK_TIMEOUT_S)


def _parse_picker_output(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    action = str(data.get("action") or "").strip().lower()
    if action == "pick":
        return action, str(data.get("name") or "").strip()
    if action == "none":
        return action, ""
    if action == "search":
        return action, str(data.get("keyword") or "").strip()
    return "", ""


def _validate(name: str, pool: Sequence[str]) -> Optional[str]:
    value = (name or "").strip()
    return value if value and value in pool else None
