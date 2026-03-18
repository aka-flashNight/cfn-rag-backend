"""
Agent 工具分发执行器。

接收 LLM 返回的 tool_calls 列表，逐个执行并返回工具结果。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from services.game_data.registry import GameDataRegistry, get_game_data_registry
from services.game_progress import get_progress_stage_config
from services.agent_tools.context_builder import prepare_task_context
from services.agent_tools.validator import validate_task_draft, DraftValidationContext

logger = logging.getLogger(__name__)


def _safe_json_loads(s: str) -> dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


def _reward_field_value_changed(cur: Any, new: Any) -> bool:
    """奖励类字段是否发生实际变化，用于讨价还价计数去重（同一轮内重复调用不重复计数）。"""
    if cur is None and (new is None or new == []):
        return False
    if (cur is None or cur == []) and new is None:
        return False
    if cur == new:
        return False
    return True


def _build_validation_ctx(
    *,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
) -> DraftValidationContext:
    """根据玩家进度构建校验上下文，字段映射到 DraftValidationContext。"""
    cfg = get_progress_stage_config(player_progress)
    main_task_max_id = cfg.main_task_max_id if cfg else 0
    max_level = cfg.max_level if cfg else 50
    return DraftValidationContext(
        main_task_max_id=main_task_max_id or 0,
        max_level=max_level or 50,
        stage=player_progress,
        affinity=npc_affinity,
        npc_name=npc_name or None,
    )


# ---------------------------------------------------------------------------
# 各工具执行函数
# ---------------------------------------------------------------------------

def execute_prepare_task_context(
    args: dict[str, Any],
    *,
    npc_name: str,
    npc_faction: str = "",
    npc_challenge: Optional[str] = None,
    player_progress: int = 1,
    npc_affinity: int = 0,
    npc_states: Optional[dict[str, Any]] = None,
    game_data: Optional[GameDataRegistry] = None,
) -> str:
    task_type = args.get("task_type", "问候")
    reward_types = args.get("reward_types", {"regular": ["金币", "经验"], "optional": []})
    return prepare_task_context(
        task_type=task_type,
        reward_types=reward_types,
        npc_name=npc_name,
        npc_faction=npc_faction,
        npc_challenge=npc_challenge,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
        npc_states=npc_states,
        game_data=game_data,
    )


def execute_draft_agent_task(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    game_data: Optional[GameDataRegistry] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    """
    创建任务草案并校验。
    返回 (结果JSON字符串, 更新后的草案dict或None)。
    """
    if game_data is None:
        game_data = get_game_data_registry()

    draft_id = str(uuid.uuid4())[:8]
    draft: dict[str, Any] = {
        "draft_id": draft_id,
        "npc_name": npc_name,
        "bargain_count": 0,  # 讨价还价次数，上限 2 次（仅调整奖励时计数）
        **args,
    }

    validation_ctx = _build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )

    result = validate_task_draft(draft, context=validation_ctx, game_data=game_data)
    detailed = _detailed_draft_summary(draft, game_data)
    if not result.success:
        return json.dumps({
            "status": "validation_failed",
            "draft_id": draft_id,
            "errors": result.validation_errors,
            "draft_summary": detailed,
        }, ensure_ascii=False), draft
    payload: dict[str, Any] = {
        "status": "draft_created",
        "draft_id": draft_id,
        "message": "任务草案已创建，等待玩家确认。",
        "draft_summary": detailed,
        "bargain_remaining": 2,  # 讨价还价最多 2 次，提醒 LLM
    }
    if result.validation_warnings:
        payload["warnings"] = result.validation_warnings
    return json.dumps(payload, ensure_ascii=False), draft


def execute_update_task_draft(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    game_data: Optional[GameDataRegistry] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    """
    局部修改已有草案并重新校验。
    返回 (结果JSON字符串, 更新后的草案dict或None)。
    """
    if pending_draft is None:
        return json.dumps({
            "status": "error",
            "message": "当前没有待修改的草案，请先调用 draft_agent_task 创建草案。",
        }, ensure_ascii=False), None

    if game_data is None:
        game_data = get_game_data_registry()

    modify_fields = args.get("modify_fields", {})
    if not isinstance(modify_fields, dict):
        modify_fields = {}

    # 讨价还价上限 2 次：仅当本次修改涉及奖励字段且内容实际发生变化时计数（同一轮中 decision_node 与 generate_response_stream 可能各执行一次，避免重复计数）
    BARGAIN_KEYS = {"rewards", "finish_submit_items", "finish_contain_items"}
    bargain_keys_touched = BARGAIN_KEYS & set(modify_fields)
    is_bargain = bool(bargain_keys_touched)
    reward_actually_changed = False
    if is_bargain:
        for k in bargain_keys_touched:
            new_val = modify_fields[k]
            cur_val = pending_draft.get(k)
            if _reward_field_value_changed(cur_val, new_val):
                reward_actually_changed = True
                break
        if reward_actually_changed:
            bargain_count = int(pending_draft.get("bargain_count", 0))
            if bargain_count >= 2:
                return json.dumps({
                    "status": "error",
                    "message": "最多允许讨价还价2次，已达上限。请让玩家接受或拒绝任务，或取消/拒绝发布。",
                    "draft_id": pending_draft.get("draft_id", ""),
                }, ensure_ascii=False), pending_draft

    for k, v in modify_fields.items():
        pending_draft[k] = v

    validation_ctx = _build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )
    changed = set(modify_fields.keys())
    result = validate_task_draft(
        pending_draft, context=validation_ctx,
        changed_fields=changed, game_data=game_data,
    )
    detailed = _detailed_draft_summary(pending_draft, game_data)
    if not result.success:
        return json.dumps({
            "status": "validation_failed",
            "draft_id": pending_draft.get("draft_id", ""),
            "errors": result.validation_errors,
            "draft_summary": detailed,
        }, ensure_ascii=False), pending_draft

    # 仅在校验通过后增加讨价还价次数，避免失败或重放时误计数
    if is_bargain and reward_actually_changed:
        pending_draft["bargain_count"] = int(pending_draft.get("bargain_count", 0)) + 1

    payload = {
        "status": "draft_updated",
        "draft_id": pending_draft.get("draft_id", ""),
        "message": "草案已更新，等待玩家确认。",
        "draft_summary": detailed,
        "bargain_remaining": max(0, 2 - int(pending_draft.get("bargain_count", 0))),
    }
    if result.validation_warnings:
        payload["warnings"] = result.validation_warnings
    return json.dumps(payload, ensure_ascii=False), pending_draft


def execute_confirm_agent_task(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    game_data: Optional[GameDataRegistry] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
    """
    确认任务草案并写入。
    返回 (结果JSON字符串, 清空后的草案(None), 写入结果描述)。
    """
    if pending_draft is None:
        return json.dumps({
            "status": "error",
            "message": "当前没有待确认的草案。",
        }, ensure_ascii=False), None, None

    if game_data is None:
        game_data = get_game_data_registry()

    # 最终校验
    validation_ctx = _build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )
    result = validate_task_draft(pending_draft, context=validation_ctx, game_data=game_data)
    if not result.success:
        return json.dumps({
            "status": "validation_failed",
            "errors": result.validation_errors,
            "message": "草案校验未通过，无法确认。",
        }, ensure_ascii=False), pending_draft, None

    # 分配正式任务 ID
    task_id = game_data.tasks.get_max_agent_task_id() + 1
    pending_draft["id"] = task_id

    # 写入任务文件（原子写入：先写临时文件再替换）
    try:
        from services.agent_tools.task_tools import write_confirmed_agent_task_files

        write_desc = write_confirmed_agent_task_files(
            draft=pending_draft,
            npc_name_fallback=npc_name or str(pending_draft.get("npc_name") or ""),
            game_data=game_data,
        )
    except Exception as e:
        detailed = _detailed_draft_summary(pending_draft, game_data)
        return json.dumps(
            {
                "status": "error",
                "message": f"任务写入失败：{str(e)}",
                "draft_summary": detailed,
            },
            ensure_ascii=False,
        ), pending_draft, None

    detailed = _detailed_draft_summary(pending_draft, game_data)
    confirm_payload: dict[str, Any] = {
        "status": "confirmed",
        "task_id": task_id,
        "message": write_desc,
        "draft_summary": detailed,
    }
    if result.validation_warnings:
        confirm_payload["warnings"] = result.validation_warnings
    return json.dumps(confirm_payload, ensure_ascii=False), None, write_desc


def execute_cancel_agent_task(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    """取消当前草案。"""
    if pending_draft is None:
        return json.dumps({
            "status": "error",
            "message": "当前没有待取消的草案。",
        }, ensure_ascii=False), None

    draft_id = pending_draft.get("draft_id", "")
    return json.dumps({
        "status": "cancelled",
        "draft_id": draft_id,
        "message": "任务草案已取消。",
    }, ensure_ascii=False), None


def execute_update_npc_mood(args: dict[str, Any]) -> str:
    """update_npc_mood 不需要真正执行，数据在 post_process 中处理。"""
    return json.dumps({
        "status": "ok",
        "message": "情绪与好感度变化已记录。",
    }, ensure_ascii=False)


def execute_search_knowledge(
    args: dict[str, Any],
    *,
    retrieve_fn: Any = None,
) -> str:
    """
    search_knowledge 工具：复用 RAG 检索。
    retrieve_fn 需要由调用方注入（来自 GameRAGService._retrieve_context）。
    """
    keyword = args.get("keyword", "")
    if not keyword:
        return json.dumps({"status": "error", "message": "keyword 不能为空"}, ensure_ascii=False)

    if retrieve_fn is None:
        return json.dumps({
            "status": "error",
            "message": "检索功能暂不可用。",
        }, ensure_ascii=False)

    try:
        result = retrieve_fn(keyword)
        if not result:
            return json.dumps({
                "status": "ok",
                "result": "未找到相关信息。",
            }, ensure_ascii=False)
        truncated = result[:2000]
        return json.dumps({
            "status": "ok",
            "result": truncated,
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"search_knowledge 执行失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"检索失败: {str(e)}",
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _summarize_draft(draft: dict[str, Any]) -> str:
    parts: list[str] = []
    if draft.get("title"):
        parts.append(f"标题: {draft['title']}")
    if draft.get("task_type"):
        parts.append(f"类型: {draft['task_type']}")
    if draft.get("rewards"):
        reward_strs = [
            f"{r.get('item_name', '?')}x{r.get('count', '?')}"
            for r in draft["rewards"][:5]
        ]
        parts.append(f"奖励: {', '.join(reward_strs)}")
    return " | ".join(parts) if parts else "(空草案)"


def _detailed_draft_summary(
    draft: dict[str, Any],
    game_data: Optional[GameDataRegistry] = None,
) -> str:
    """
    生成包含完整字段和物品单价的草案摘要，
    用于注入 prompt 让 LLM 了解当前任务的全部关键信息。
    """
    lines: list[str] = []
    did = draft.get("draft_id", "?")
    lines.append(f"草案ID: {did}")
    lines.append(f"发布NPC: {draft.get('npc_name', '?')}")
    lines.append(f"类型: {draft.get('task_type', '?')}")
    lines.append(f"标题: {draft.get('title', '?')}")
    lines.append(f"描述: {draft.get('description', '?')}")
    # get/finish NPC：缺省时用 npc_name 兜底（后端写入逻辑一致）
    npc_name_fallback = draft.get("npc_name") or "?"
    lines.append(f"接取NPC: {draft.get('get_npc') or npc_name_fallback}")
    lines.append(f"完成NPC: {draft.get('finish_npc') or npc_name_fallback}")

    get_reqs = draft.get("get_requirements") or []
    if get_reqs:
        lines.append(f"前置主线任务ID: {get_reqs}")

    finish_reqs = draft.get("finish_requirements") or []
    if finish_reqs:
        fr_strs = [
            f"{fr.get('stage_name', '?')}({fr.get('difficulty', '?')})"
            for fr in finish_reqs
        ]
        lines.append(f"通关要求: {', '.join(fr_strs)}")

    finish_submit = draft.get("finish_submit_items") or []
    if finish_submit:
        fs_strs = _format_items_with_price(finish_submit, game_data)
        lines.append(f"提交物品: {', '.join(fs_strs)}")

    finish_contain = draft.get("finish_contain_items") or []
    if finish_contain:
        fc_strs = _format_items_with_price(finish_contain, game_data)
        lines.append(f"持有物品: {', '.join(fc_strs)}")

    rewards = draft.get("rewards") or []
    if rewards:
        rw_strs = _format_items_with_price(rewards, game_data)
        lines.append(f"奖励: {', '.join(rw_strs)}")

    def _summarize_dialogue(dialogue: Any) -> str:
        # 新结构：数组[{name,title,emotion?,text}, ...]
        if isinstance(dialogue, list):
            parts: list[str] = []
            for it in dialogue[:8]:
                if not isinstance(it, dict):
                    continue
                n = str(it.get("name") or "").strip()
                emo = str(it.get("emotion") or "").strip()
                t = str(it.get("text") or "").strip()
                if not n and not t:
                    continue
                label = n or "?"
                if emo:
                    label = f"{label}#{emo}"
                t_short = t[:60] + ("…" if len(t) > 60 else "")
                parts.append(f"{label}:{t_short}")
            return "；".join(parts)

        # 旧结构：字符串
        if isinstance(dialogue, str):
            s = dialogue.strip()
            if not s:
                return ""
            return s[:120] + ("…" if len(s) > 120 else "")
        return ""

    get_dialogue = draft.get("get_dialogue")
    get_summary = _summarize_dialogue(get_dialogue)
    if get_summary:
        lines.append(f"接取对话: {get_summary}")
    else:
        get_text = draft.get("get_conversation_text", "")
        if isinstance(get_text, str) and get_text.strip():
            lines.append(f"接取对话(旧字段): {get_text.strip()[:120]}{'…' if len(get_text.strip()) > 120 else ''}")

    finish_dialogue = draft.get("finish_dialogue")
    finish_summary = _summarize_dialogue(finish_dialogue)
    if finish_summary:
        lines.append(f"完成对话: {finish_summary}")
    else:
        finish_text = draft.get("finish_conversation_text", "")
        if isinstance(finish_text, str) and finish_text.strip():
            lines.append(f"完成对话(旧字段): {finish_text.strip()[:120]}{'…' if len(finish_text.strip()) > 120 else ''}")

    return "\n".join(lines)


def _format_items_with_price(
    items: list[dict[str, Any]],
    game_data: Optional[GameDataRegistry] = None,
) -> list[str]:
    parts: list[str] = []
    for it in items[:10]:
        name = it.get("item_name", "?")
        count = it.get("count", "?")
        price = 0
        if game_data:
            try:
                price = game_data.items.get_price(name)
            except Exception:
                pass
        if price and name != "金币":
            parts.append(f"{name}x{count}(单价{price})")
        else:
            parts.append(f"{name}x{count}")
    return parts


# ---------------------------------------------------------------------------
# 统一分发入口
# ---------------------------------------------------------------------------

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
) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
    """
    统一工具分发入口。

    返回: (tool_result_str, updated_pending_draft, task_write_result)
    """
    # Debug: 仅打印任务发布相关工具的入参，便于排查协商/落库逻辑。
    # 注意：这里打印的是 LLM 传入的结构化参数（一般不包含敏感信息）。
    _task_tool_names = {
        "prepare_task_context",
        "draft_agent_task",
        "update_task_draft",
        "confirm_agent_task",
        "cancel_agent_task",
    }
    if tool_name in _task_tool_names:
        try:
            pretty_args = json.dumps(tool_args, ensure_ascii=False)
        except Exception:
            pretty_args = str(tool_args)
        print('——————【工具调用】——————')
        print(f"[agent_tool_call] 工具名称： {tool_name} args={pretty_args}")

    updated_draft = pending_draft
    task_write_result = None

    if tool_name == "prepare_task_context":
        result = execute_prepare_task_context(
            tool_args,
            npc_name=npc_name,
            npc_faction=npc_faction,
            npc_challenge=npc_challenge,
            player_progress=player_progress,
            npc_affinity=npc_affinity,
            npc_states=npc_states,
            game_data=game_data,
        )

    elif tool_name == "draft_agent_task":
        result, updated_draft = execute_draft_agent_task(
            tool_args,
            pending_draft=pending_draft,
            npc_name=npc_name,
            player_progress=player_progress,
            npc_affinity=npc_affinity,
            game_data=game_data,
        )

    elif tool_name == "update_task_draft":
        result, updated_draft = execute_update_task_draft(
            tool_args,
            pending_draft=pending_draft,
            npc_name=npc_name,
            player_progress=player_progress,
            npc_affinity=npc_affinity,
            game_data=game_data,
        )

    elif tool_name == "confirm_agent_task":
        result, updated_draft, task_write_result = execute_confirm_agent_task(
            tool_args,
            pending_draft=pending_draft,
            npc_name=npc_name,
            player_progress=player_progress,
            npc_affinity=npc_affinity,
            game_data=game_data,
        )

    elif tool_name == "cancel_agent_task":
        result, updated_draft = execute_cancel_agent_task(
            tool_args,
            pending_draft=pending_draft,
        )

    elif tool_name == "update_npc_mood":
        result = execute_update_npc_mood(tool_args)

    elif tool_name == "search_knowledge":
        result = execute_search_knowledge(
            tool_args,
            retrieve_fn=retrieve_fn,
        )

    else:
        result = json.dumps({
            "status": "error",
            "message": f"未知工具: {tool_name}",
        }, ensure_ascii=False)

    if tool_name in _task_tool_names:
        try:
            # 截断避免控制台刷屏（尤其是 prepare_task_context 返回的大列表）
            preview = (result or "")
            if isinstance(preview, str) and len(preview) > 500:
                preview = preview[:500] + "…"
        except Exception:
            preview = "<preview-unavailable>"
        # Debug
        print('——————【结果】——————')
        print(f"————/n[agent_tool_result] 工具名称： {tool_name} result={preview}")

    return result, updated_draft, task_write_result
