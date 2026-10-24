"""选项直连：任务接受/拒绝与索取赠送/拒绝。

Host 点击输入条选项时带 `action` 直接进入结果轮，省掉一次路由 LLM 调用；
确认/取消文案生成与最终台词轮仍保留。action_id 落 `dialogue_actions` 幂等，
重复请求不重复发布、不重复记账。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.item_claims import ITEM_CLAIMS
from services.dialogue_interject.item_ledger import RECEIVE, record_item_ledger
from services.dialogue_interject.item_tools import is_equipment_item
from services.dialogue_interject.item_value import (
    enhancement_description,
    format_item_info,
    format_recent_take,
    format_take_completion,
)
from services.dialogue_interject.llm import resolve_llm_config, stream_final_llm
from services.dialogue_interject.schemas import (
    InterjectRequest,
    ItemClaimPayload,
    StreamItemClaim,
    StreamTaskPublish,
    TaskPublishPayload,
)
from services.dialogue_interject.tools import DailyToolRuntime
from services.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)

_ITEM_TAKE_KINDS = ("item_take_accept", "item_take_decline")


async def stream_action_direct(
    request: InterjectRequest,
    *,
    blocks: DailyBlocks,
    llm_config: LLMConfig | None = None,
    client: Any = None,
) -> Any:
    """选项直连流：先处理动作，再用 tools=None 的最终轮生成台词。"""
    llm = client or LLMClient.for_config(resolve_llm_config(llm_config))
    action = request.action
    publish = None
    if action is None:
        result_block = "【办理结果】没有可执行的动作。"
        instruction = "以角色口吻正常回应，不要提工具。"
        initial: tuple[StreamItemClaim, ...] = ()
    elif action.kind in ("task_accept", "task_decline"):
        result_block, instruction, publish = await _run_task_action(request, blocks, llm)
        initial = ()
    elif action.kind in _ITEM_TAKE_KINDS:
        result_block, instruction, closed = await _run_item_take_action(request, blocks)
        initial = (closed,) if closed is not None else ()
    else:
        result_block = "【办理结果】未能识别的动作，已忽略。"
        instruction = "以角色口吻正常回应，不要提工具。"
        initial = ()

    if publish is not None:
        # 自动接取：不再生成收尾台词，游戏侧拿 done.task_publish 直接接取
        yield StreamTaskPublish(publish)
        return

    async for chunk in stream_final_llm(
        request,
        blocks=blocks,
        result_block=result_block,
        instruction=instruction,
        initial_events=initial,
        llm_config=llm_config,
        client=llm,
    ):
        yield chunk


async def _run_task_action(
    request: InterjectRequest,
    blocks: DailyBlocks,
    llm: Any,
) -> tuple[str, str, TaskPublishPayload | None]:
    action = request.action
    assert action is not None
    memory = blocks.memory
    saved = await _load_saved(memory, action.action_id)
    if saved is not None:
        result_block, instruction = _saved_text(saved)
        return result_block, instruction, _saved_publish(saved)

    # 任务装配不读草案（工具轮也不需要），选项直连必须在这里补读
    get_draft = getattr(memory, "get_draft", None)
    if blocks.pending_draft_row is None and get_draft is not None:
        try:
            draft_row = await get_draft(blocks.session_id)
        except Exception as exc:
            logger.warning("[interject] 直连动作读取草案失败: %s", exc)
            draft_row = None
        if draft_row is not None:
            blocks = replace(blocks, pending_draft_row=draft_row)

    runtime = DailyToolRuntime(request, blocks, llm=llm)
    publish: TaskPublishPayload | None = None
    if action.kind == "task_accept":
        result_block, instruction, publish = await runtime.run_task_accept()
    else:
        result_block, instruction = await runtime.run_task_decline()
    await _save_action(
        memory,
        action.action_id,
        blocks.session_id,
        action.kind,
        {
            "result_block": result_block,
            "instruction": instruction,
            "task_publish": publish.to_wire() if publish is not None else None,
        },
    )
    return result_block, instruction, publish


async def _run_item_take_action(
    request: InterjectRequest,
    blocks: DailyBlocks,
) -> tuple[str, str, StreamItemClaim | None]:
    action = request.action
    assert action is not None
    memory = blocks.memory
    key = request.dialogue_instance_id
    saved = await _load_saved(memory, action.action_id)
    if saved is not None:
        result_block, instruction = _saved_text(saved)
        closed_data = saved.get("closed")
        closed = (
            StreamItemClaim(ItemClaimPayload(**closed_data))
            if isinstance(closed_data, dict) and closed_data
            else None
        )
        return result_block, instruction, closed

    claim = ITEM_CLAIMS.pending(key)
    if claim is not None and not _claim_matches(claim, action):
        logger.info(
            "[interject] %s 与 pending claim 不一致，拒绝执行 action_id=%s claim_id=%s",
            action.kind, action.action_id, action.claim_id,
        )
        return (
            "【办理结果·玩家给你】这次交接与当前待处理的东西对不上，没有接收物品。",
            "玩家刚才是要把东西交给你。以角色口吻说明这次没收成；"
            "不要声称收到物品，不要提工具，也不要解释办理过程。",
            None,
        )

    item_name = action.item_name.strip()
    quantity = int(action.quantity or 0)
    if action.kind == "item_take_accept":
        if is_equipment_item(blocks.game_data, item_name) and quantity != 1:
            logger.warning(
                "[interject] 装备索取数量固定为 1，收到 %s 按 1 处理 item=%s",
                quantity, item_name,
            )
            quantity = 1
        result_block, instruction = await _accept_item(
            request, blocks, item_name=item_name, quantity=quantity,
        )
        closed_payload = _closed_payload(request, action.claim_id, item_name, quantity)
    else:
        if claim is not None:
            ITEM_CLAIMS.close(key, claim_id=claim.claim_id)
            ITEM_CLAIMS.record_failure(key, f"玩家拒绝向你赠送{item_name}")
        result_block = f"【办理结果·玩家给你】玩家拒绝把{item_name}交给你。"
        instruction = (
            "玩家刚才是要把东西交给你，但这次拒绝了。"
            "以角色口吻回应，承认玩家开过口，不要再索要，不要提工具，也不要解释办理过程。"
        )
        closed_payload = _closed_payload(request, action.claim_id, item_name, quantity)

    closed = StreamItemClaim(closed_payload)
    await _save_action(
        memory,
        action.action_id,
        blocks.session_id,
        action.kind,
        {
            "result_block": result_block,
            "instruction": instruction,
            "closed": closed_payload.model_dump(),
        },
    )
    return result_block, instruction, closed


async def _accept_item(
    request: InterjectRequest,
    blocks: DailyBlocks,
    *,
    item_name: str,
    quantity: int,
) -> tuple[str, str]:
    key = request.dialogue_instance_id
    action = request.action
    assert action is not None
    claim = ITEM_CLAIMS.pending(key)
    if claim is not None:
        ITEM_CLAIMS.close(key, claim_id=claim.claim_id)
    else:
        # 迟到/跨进程回执：Host 已真实扣除，按动作字段补记，不再要求 claim 存在
        logger.info(
            "[interject] item_take_accept 无 pending claim，按迟到回执补记 action_id=%s",
            action.action_id,
        )

    level = action.item_level
    if not is_equipment_item(blocks.game_data, item_name):
        level = None
    elif level is not None and level <= 1:
        level = None

    ITEM_CLAIMS.record_success(
        key, format_recent_take(item_name, quantity, item_level=level),
    )
    await record_item_ledger(
        blocks.memory,
        slot_key_value=blocks.slot_key,
        npc=request.npc,
        direction=RECEIVE,
        item_name=item_name,
        quantity=quantity,
    )

    text = "【办理结果·玩家给你】" + format_take_completion(
        item_name, quantity, item_level=level,
    )
    info = format_item_info(blocks.game_data, item_name)
    if level is not None and level > 1:
        info = (
            f"{info}\n强化度：{level}（{enhancement_description(level)}）"
            if info else f"强化度：{level}（{enhancement_description(level)}）"
        )
    if info:
        text += "\n物品信息：\n" + info
    instruction = (
        "用 JSONL 回应玩家：玩家刚才是要把东西交给你，你确认已经收下这件物品；"
        "不要提工具，也不要解释办理过程，不要输出动作描写。"
    )
    return text, instruction


def _claim_matches(claim: Any, action: Any) -> bool:
    return (
        claim.claim_id == action.claim_id
        and claim.item_name == action.item_name.strip()
        and claim.quantity == int(action.quantity or 0)
    )


def _closed_payload(
    request: InterjectRequest,
    claim_id: str | None,
    item_name: str,
    quantity: int,
) -> ItemClaimPayload:
    return ItemClaimPayload(
        state="closed",
        claim_id=(claim_id or "").strip() or "closed",
        npc=request.npc,
        item_name=item_name,
        quantity=max(1, int(quantity or 0)),
    )


async def _load_saved(memory: Any, action_id: str) -> dict[str, Any] | None:
    if memory is None:
        return None
    try:
        return await memory.dialogue_action_get(action_id)
    except Exception as exc:  # 读失败按未处理继续，不让幂等表拖住本轮
        logger.warning("[interject] dialogue_actions 读取失败 action_id=%s: %s", action_id, exc)
        return None


async def _save_action(
    memory: Any,
    action_id: str,
    session_id: str,
    kind: str,
    result: dict[str, Any],
) -> None:
    if memory is None:
        return
    try:
        await memory.dialogue_action_save(action_id, session_id, kind, result)
    except Exception as exc:  # 写失败只影响重试幂等，不反向破坏已执行动作
        logger.warning("[interject] dialogue_actions 写入失败 action_id=%s: %s", action_id, exc)


def _saved_text(saved: dict[str, Any]) -> tuple[str, str]:
    result = saved.get("result") if isinstance(saved, dict) else None
    if not isinstance(result, dict):
        result = {}
    return (
        str(result.get("result_block") or "【办理结果】这个动作之前已经处理过了。"),
        str(result.get("instruction") or "以角色口吻正常回应，不要重复办理，不要提工具。"),
    )


def _saved_publish(saved: dict[str, Any]) -> TaskPublishPayload | None:
    """重试同一 action_id 时补发 `done.task_publish`，别再走一次自动接取流程。"""
    result = saved.get("result") if isinstance(saved, dict) else None
    data = result.get("task_publish") if isinstance(result, dict) else None
    if not isinstance(data, dict) or not data:
        return None
    try:
        return TaskPublishPayload(**data)
    except Exception as exc:
        logger.warning("[interject] 历史 task_publish 解析失败: %s", exc)
        return None
