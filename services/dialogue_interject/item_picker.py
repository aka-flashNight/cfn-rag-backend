"""物品选择子 Agent：两轮 pick / fail / search。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence

from services.dialogue_interject.item_pool import (
    GOLD,
    item_price,
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
_RECENT_DIALOGUE_MAX = 1200

#: 失败理由代码 → 由调用方映射成固定中文说明
REASON_NOT_FOUND = "not_found"
REASON_VALUE_EXCEEDED = "value_exceeded"
REASON_NONE = "none"

PICKER_SYSTEM = """你是物品选择器。根据玩家的原话、近期对话和候选列表，只返回一个 JSON 对象，不要 Markdown、不要解释。
可选动作：
{"action":"pick","name":"候选中的精确物品名"}
{"action":"fail","reason":"not_found"}
{"action":"fail","reason":"value_exceeded"}
{"action":"fail","reason":"none"}
{"action":"search","keyword":"新的检索关键词"}
规则：
1. 只能 pick 候选列表中出现的精确物品名；标注“不可赠送”的候选永远不能 pick。
2. 玩家明确指定了某件物品（名称或唯一可辨识的描述）时：
   - 候选和关键词检索都没有这件物品：返回 fail/not_found，不得用别的物品替代。
   - 候选中这件物品被标注“不可赠送”：返回 fail/value_exceeded，不得用别的物品替代。
3. 玩家没有明确指定物品时，才可以根据指示从候选中自行选择可赠送的物品；没有合适的返回 fail/none。
4. 只有认为换一个关键词可能找到更合适的物品时，才使用 search；最多一次。
5. 第二轮只能 pick 或 fail，不能再 search。
6. 没有合适的就 fail，不要编造物品名。"""


@dataclass(frozen=True)
class PickerResult:
    """选择结果：name 非空＝选中；否则 reason 说明失败类型。"""

    name: Optional[str] = None
    reason: str = ""
    target: str = ""


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
        """兼容旧调用：只返回选中的物品名。"""
        result = await self.pick_result(
            direction=direction,
            name=name,
            keyword=keyword,
            instruction=instruction,
            pool=pool,
            game_data=game_data,
        )
        return result.name

    async def pick_result(
        self,
        *,
        direction: str,
        name: str,
        keyword: str,
        instruction: str,
        pool: Sequence[str],
        game_data: Any,
        random_pool: Sequence[str] | None = None,
        cap: int | None = None,
        player_text: str = "",
        recent_dialogue: str = "",
    ) -> PickerResult:
        """选物品或给出结构化失败理由。

        - 精确 name 不在池内＝not_found，不再回落到随机候选；
        - 精确 name 单价超过 cap＝value_exceeded；
        - random_pool 是可用于随机抽取的池（赠送方向＝已按价值上限过滤）；
        - 关键词命中保留原池结果，超上限项由候选标注 + 模型判断。
        """
        exact = (name or "").strip()
        target = exact or (keyword or "").strip()
        if exact:
            if exact in pool:
                if self._over_cap(game_data, exact, cap):
                    return PickerResult(reason=REASON_VALUE_EXCEEDED, target=exact)
                return PickerResult(name=exact, target=exact)
            return PickerResult(reason=REASON_NOT_FOUND, target=exact)

        if not pool:
            return PickerResult(reason=REASON_NONE, target=target)

        keyword = (keyword or "").strip()
        instruction = (instruction or "").strip()
        sample_pool = list(random_pool) if random_pool is not None else list(pool)
        hits = search_names(game_data, pool, keyword, limit=10) if keyword else []
        if hits:
            candidates = hits
        else:
            candidates = sample_names(
                sample_pool,
                count=_CANDIDATE_COUNT,
                gold_first=(direction == "take"),
            )
        if not candidates:
            return PickerResult(reason=REASON_NONE, target=target)

        over_cap = {
            candidate for candidate in candidates
            if self._over_cap(game_data, candidate, cap)
        }
        try:
            return await asyncio.wait_for(
                self._run(
                    direction=direction,
                    pool=pool,
                    sample_pool=sample_pool,
                    game_data=game_data,
                    candidates=candidates,
                    over_cap=over_cap,
                    cap=cap,
                    keyword=keyword,
                    keyword_hits=len(hits),
                    instruction=instruction,
                    player_text=(player_text or "").strip(),
                    recent_dialogue=(recent_dialogue or "").strip(),
                    target=target,
                ),
                timeout=_TOTAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[interject] 物品选择子 Agent 超时 direction=%s name=%r keyword=%r instruction=%r",
                direction, name, keyword, instruction,
            )
            return PickerResult(reason=REASON_NONE, target=target)
        except Exception as exc:  # noqa: BLE001 - 选择失败只影响本次工具
            logger.warning(
                "[interject] 物品选择子 Agent 失败 direction=%s name=%r keyword=%r instruction=%r: %s",
                direction, name, keyword, instruction, exc,
            )
            return PickerResult(reason=REASON_NONE, target=target)

    async def _run(
        self,
        *,
        direction: str,
        pool: Sequence[str],
        sample_pool: Sequence[str],
        game_data: Any,
        candidates: list[str],
        over_cap: set[str],
        cap: int | None,
        keyword: str,
        keyword_hits: int,
        instruction: str,
        player_text: str,
        recent_dialogue: str,
        target: str,
    ) -> PickerResult:
        first = await self._call(
            self._messages(
                direction=direction,
                pool=pool,
                sample_pool=sample_pool,
                game_data=game_data,
                candidates=candidates,
                over_cap=over_cap,
                cap=cap,
                keyword=keyword,
                keyword_hits=keyword_hits,
                instruction=instruction,
                player_text=player_text,
                recent_dialogue=recent_dialogue,
                second_round=False,
            )
        )
        action, value = _parse_picker_output(first)
        if action == "pick":
            return self._picked(value, candidates, over_cap, len(candidates), target)
        if action == "fail":
            logger.info(
                "[interject] 物品选择子 Agent fail=%s pool=%d candidates=%d",
                value or REASON_NONE, len(pool), len(candidates),
            )
            return PickerResult(reason=_normalize_reason(value), target=target)
        if action == "none":
            logger.info("[interject] 物品选择子 Agent none pool=%d candidates=%d", len(pool), len(candidates))
            return PickerResult(reason=REASON_NONE, target=target)
        if action != "search":
            logger.info("[interject] 物品选择子 Agent 非法输出=%r", first[:120])
            return PickerResult(reason=REASON_NONE, target=target)

        second_candidates = search_names(game_data, pool, value, limit=10)
        if not second_candidates:
            logger.info(
                "[interject] 物品选择子 Agent 再查无结果 keyword=%r pool=%d",
                value, len(pool),
            )
            return PickerResult(reason=REASON_NONE, target=value or target)
        second_over_cap = {
            candidate for candidate in second_candidates
            if self._over_cap(game_data, candidate, cap)
        }
        second = await self._call(
            self._messages(
                direction=direction,
                pool=pool,
                sample_pool=sample_pool,
                game_data=game_data,
                candidates=second_candidates,
                over_cap=second_over_cap,
                cap=cap,
                keyword=value,
                keyword_hits=len(second_candidates),
                instruction=instruction,
                player_text=player_text,
                recent_dialogue=recent_dialogue,
                second_round=True,
            )
        )
        action, value = _parse_picker_output(second)
        if action == "pick":
            return self._picked(value, second_candidates, second_over_cap,
                                len(second_candidates), target)
        if action == "fail":
            logger.info(
                "[interject] 物品选择子 Agent 二轮 fail=%s pool=%d candidates=%d",
                value or REASON_NONE, len(pool), len(second_candidates),
            )
            return PickerResult(reason=_normalize_reason(value), target=target)
        logger.info(
            "[interject] 物品选择子 Agent 二轮 %s pool=%d candidates=%d",
            action or "非法输出", len(pool), len(second_candidates),
        )
        return PickerResult(reason=REASON_NONE, target=target)

    def _picked(
        self,
        name: str,
        candidates: Sequence[str],
        over_cap: set[str],
        candidate_count: int,
        target: str,
    ) -> PickerResult:
        picked = _validate(name, candidates)
        if picked is None:
            logger.info(
                "[interject] 物品选择子 Agent pick 不在候选内 name=%r candidates=%d",
                name, candidate_count,
            )
            return PickerResult(reason=REASON_NONE, target=target)
        if picked in over_cap:
            logger.info("[interject] 物品选择子 Agent 选中超上限物品 name=%r", picked)
            return PickerResult(reason=REASON_VALUE_EXCEEDED, target=picked)
        logger.info(
            "[interject] 物品选择子 Agent pick=%r candidates=%d",
            picked, candidate_count,
        )
        return PickerResult(name=picked, target=picked)

    @staticmethod
    def _over_cap(game_data: Any, name: str, cap: int | None) -> bool:
        if cap is None:
            return False
        return item_price(game_data, name) > int(cap)

    @staticmethod
    def _messages(
        *,
        direction: str,
        pool: Sequence[str],
        sample_pool: Sequence[str],
        game_data: Any,
        candidates: list[str],
        over_cap: set[str],
        cap: int | None,
        keyword: str,
        keyword_hits: int,
        instruction: str,
        player_text: str,
        recent_dialogue: str,
        second_round: bool,
    ) -> list[dict[str, str]]:
        lines = []
        for name in candidates:
            line = format_item_info(game_data, name)
            if name in over_cap:
                line += "（该物品价值超出可赠送范围，不可赠送）"
            lines.append(line)
        counts, total = remaining_pool_counts(game_data, sample_pool, candidates)
        count_text = "、".join(f"{key}{value}" for key, value in counts.items() if value)
        direction_text = "送给玩家" if direction == "give" else "向玩家索取"
        header_lines = [
            f"方向：{direction_text}。",
            f"有效池共 {total} 件；本轮候选之外剩余：{count_text or '无'}。",
        ]
        if cap is not None:
            header_lines.append(f"可赠送价值上限：{cap}（单价超过上限的物品不可赠送）。")
        header_lines.append(f"候选列表（{len(candidates)} 件）：")
        body = "\n".join(f"- {line}" for line in lines)
        dialogue = recent_dialogue
        if len(dialogue) > _RECENT_DIALOGUE_MAX:
            dialogue = dialogue[-_RECENT_DIALOGUE_MAX:]
        tail = "这是第二轮，只能返回 pick 或 fail，不能再 search。" if second_round else "请返回 pick / fail / search。"
        content = "\n".join([
            "\n".join(header_lines),
            body,
            f"玩家原话：{player_text or '（无）'}",
            f"玩家近期对话（供判断是否明确指定了物品）：\n{dialogue or '（无）'}",
            f"本次关键词：{keyword or '（无）'}；池内匹配 {keyword_hits} 件。",
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


def _normalize_reason(value: str) -> str:
    reason = (value or "").strip().lower()
    if reason in (REASON_NOT_FOUND, REASON_VALUE_EXCEEDED, REASON_NONE):
        return reason
    return REASON_NONE


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
    if action == "fail":
        return action, str(data.get("reason") or "").strip()
    if action == "none":
        return action, ""
    if action == "search":
        return action, str(data.get("keyword") or "").strip()
    return "", ""


def _validate(name: str, pool: Sequence[str]) -> Optional[str]:
    value = (name or "").strip()
    return value if value and value in pool else None
