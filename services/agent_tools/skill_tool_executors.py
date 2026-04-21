"""
各 skill 对应的可调用实现（由 services/skills/*/handler 引用）。

与 ``tool_executor.dispatch_tool_call`` 解耦，避免与草案格式化函数循环依赖。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from services.game_data.registry import GameDataRegistry, get_game_data_registry
from services.game_progress import get_progress_stage_config
from services.agent_tools.context_builder import prepare_task_context
from services.agent_tools.draft_formatting import _detailed_draft_summary
from services.agent_tools.schemas import normalize_reward_types_for_prepare_context
from services.agent_tools.validator import validate_task_draft, DraftValidationContext
from services.agent_tools.task_tools import collect_existing_task_titles

logger = logging.getLogger(__name__)


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
    bargain_rate: float = 1.0,
) -> DraftValidationContext:
    cfg = get_progress_stage_config(player_progress)
    main_task_max_id = cfg.main_task_max_id if cfg else 0
    max_level = cfg.max_level if cfg else 50
    return DraftValidationContext(
        main_task_max_id=main_task_max_id or 0,
        max_level=max_level or 50,
        stage=player_progress,
        affinity=npc_affinity,
        npc_name=npc_name or None,
        bargain_rate=bargain_rate,
    )


def _title_duplicate_warning(title: Any, game_data: Optional[GameDataRegistry]) -> Optional[dict[str, Any]]:
    if game_data is None:
        return None
    s = str(title or "").strip()
    if not s:
        return None
    try:
        existing = collect_existing_task_titles(game_data)
    except Exception:
        return None
    if s not in existing:
        return None
    return {
        "step": "TITLE_DUPLICATE",
        "warning": (
            f"当前草案标题「{s}」与已有任务重复。"
            "建议在 confirm_agent_task 时适当调整标题内容。"
        ),
        "title": s,
    }


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
    requirement_keywords = args.get("requirement_keywords")
    reward_keywords = args.get("reward_keywords")
    if not isinstance(requirement_keywords, list):
        requirement_keywords = None
    if not isinstance(reward_keywords, list):
        reward_keywords = None
    reward_types = normalize_reward_types_for_prepare_context(
        args.get("reward_types"),
        reward_keywords,
    )
    return prepare_task_context(
        task_type=task_type,
        reward_types=reward_types,
        npc_name=npc_name,
        npc_faction=npc_faction,
        npc_challenge=npc_challenge,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
        npc_states=npc_states,
        requirement_keywords=requirement_keywords,
        reward_keywords=reward_keywords,
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
    rag_context_text: Optional[str] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    if game_data is None:
        game_data = get_game_data_registry()

    draft_id = str(uuid.uuid4())[:8]
    args_clean = dict(args)
    for _k in ("description", "get_dialogue", "finish_dialogue"):
        args_clean.pop(_k, None)

    draft: dict[str, Any] = {
        "draft_id": draft_id,
        "npc_name": npc_name,
        "bargain_count": 0,
        **args_clean,
    }
    draft.pop("_draft_commit_valid", None)

    validation_ctx = _build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )

    result = validate_task_draft(draft, context=validation_ctx, game_data=game_data)
    detailed = _detailed_draft_summary(draft, game_data, rag_context_text=rag_context_text)
    if not result.success:
        draft["_draft_commit_valid"] = False
        return json.dumps({
            "status": "validation_failed",
            "draft_id": draft_id,
            "errors": result.validation_errors,
            "draft_summary": "",
        }, ensure_ascii=False), draft
    draft["_draft_commit_valid"] = True
    payload: dict[str, Any] = {
        "status": "draft_created",
        "draft_id": draft_id,
        "message": "任务草案已创建，等待玩家确认。",
        "draft_summary": detailed,
        "bargain_remaining": 2,
    }
    warnings = list(result.validation_warnings or [])
    w_title = _title_duplicate_warning(draft.get("title"), game_data)
    if w_title:
        warnings.append(w_title)
    if warnings:
        payload["warnings"] = warnings
    return json.dumps(payload, ensure_ascii=False), draft


def execute_update_task_draft(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    game_data: Optional[GameDataRegistry] = None,
    rag_context_text: Optional[str] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
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
    _text_only_on_confirm = {"description", "get_dialogue", "finish_dialogue"}
    modify_fields = {k: v for k, v in modify_fields.items() if k not in _text_only_on_confirm}
    if not modify_fields:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    "modify_fields 不能为空。"
                    "任务说明与接取/完成对话请仅在玩家接受任务时通过 confirm_agent_task 传入，"
                    "不要写在 update_task_draft 中。"
                ),
            },
            ensure_ascii=False,
        ), pending_draft

    prev_commit_ok = bool(pending_draft.get("_draft_commit_valid"))

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
        if reward_actually_changed and prev_commit_ok:
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
        bargain_rate=1.5 if (is_bargain and reward_actually_changed and prev_commit_ok) else 1.0,
    )
    changed = set(modify_fields.keys())
    result = validate_task_draft(
        pending_draft, context=validation_ctx,
        changed_fields=changed, game_data=game_data,
    )
    detailed = _detailed_draft_summary(pending_draft, game_data, rag_context_text=rag_context_text)
    if not result.success:
        return json.dumps({
            "status": "validation_failed",
            "draft_id": pending_draft.get("draft_id", ""),
            "errors": result.validation_errors,
            "draft_summary": "",
        }, ensure_ascii=False), pending_draft

    pending_draft["_draft_commit_valid"] = True
    if is_bargain and reward_actually_changed and prev_commit_ok:
        pending_draft["bargain_count"] = int(pending_draft.get("bargain_count", 0)) + 1

    payload = {
        "status": "draft_updated",
        "draft_id": pending_draft.get("draft_id", ""),
        "message": "草案已更新，等待玩家确认。",
        "draft_summary": detailed,
        "bargain_remaining": max(0, 2 - int(pending_draft.get("bargain_count", 0))),
    }
    warnings = list(result.validation_warnings or [])
    w_title = _title_duplicate_warning(pending_draft.get("title"), game_data)
    if w_title:
        warnings.append(w_title)
    if warnings:
        payload["warnings"] = warnings
    return json.dumps(payload, ensure_ascii=False), pending_draft


def execute_confirm_agent_task(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    game_data: Optional[GameDataRegistry] = None,
    rag_context_text: Optional[str] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
    if pending_draft is None:
        return json.dumps({
            "status": "error",
            "message": "当前没有待确认的草案。",
        }, ensure_ascii=False), None, None

    if game_data is None:
        game_data = get_game_data_registry()

    arg_draft_id = str(args.get("draft_id") or "").strip()
    pending_id = str(pending_draft.get("draft_id") or "").strip()
    if not arg_draft_id or arg_draft_id != pending_id:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    "draft_id 与当前待确认草案不一致或缺失；请使用工具返回的 draft_summary 中的草案 ID。"
                ),
                "expected_draft_id": pending_id,
                "got_draft_id": arg_draft_id,
            },
            ensure_ascii=False,
        ), pending_draft, None

    desc = args.get("description", "")
    title = args.get("title", "")
    if not isinstance(desc, str):
        desc = str(desc) if desc is not None else ""
    if not isinstance(title, str):
        title = str(title) if title is not None else ""
    get_dg = args.get("get_dialogue")
    fin_dg = args.get("finish_dialogue")
    if not isinstance(get_dg, list):
        get_dg = []
    if not isinstance(fin_dg, list):
        fin_dg = []

    draft_for_commit = dict(pending_draft)
    draft_for_commit["title"] = title
    draft_for_commit["description"] = desc
    draft_for_commit["get_dialogue"] = get_dg
    draft_for_commit["finish_dialogue"] = fin_dg

    validation_ctx = _build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )
    result = validate_task_draft(draft_for_commit, context=validation_ctx, game_data=game_data)
    if not result.success:
        return json.dumps({
            "status": "validation_failed",
            "errors": result.validation_errors,
            "message": "草案校验未通过，无法确认。",
        }, ensure_ascii=False), pending_draft, None

    try:
        from services.agent_tools.task_tools import write_confirmed_agent_task_files

        write_desc, task_id = write_confirmed_agent_task_files(
            draft=draft_for_commit,
            npc_name_fallback=npc_name or str(pending_draft.get("npc_name") or ""),
            game_data=game_data,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"任务写入失败：{str(e)}",
                "draft_summary": "",
            },
            ensure_ascii=False,
        ), pending_draft, None

    detailed = _detailed_draft_summary(
        draft_for_commit, game_data, rag_context_text=rag_context_text,
    )
    confirm_payload: dict[str, Any] = {
        "status": "confirmed",
        "task_id": task_id,
        "message": write_desc,
        "instruction_for_assistant": (
            "任务已成功发布并写入。本轮不要再次发布或确认任务，也不要再调用任何任务相关工具"
        ),
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
    return json.dumps({
        "status": "ok",
        "message": "情绪与好感度变化已记录。",
    }, ensure_ascii=False)


def execute_search_knowledge(
    args: dict[str, Any],
    *,
    retrieve_fn: Any = None,
) -> str:
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
