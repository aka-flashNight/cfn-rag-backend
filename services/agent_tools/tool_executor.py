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
        **args,
    }

    validation_ctx = _build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )

    errors = validate_task_draft(draft, context=validation_ctx, game_data=game_data)
    if errors:
        return json.dumps({
            "status": "validation_failed",
            "draft_id": draft_id,
            "errors": errors,
            "draft_summary": _summarize_draft(draft),
        }, ensure_ascii=False), draft
    else:
        return json.dumps({
            "status": "draft_created",
            "draft_id": draft_id,
            "message": "任务草案已创建，等待玩家确认。",
            "draft_summary": _summarize_draft(draft),
        }, ensure_ascii=False), draft


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
    for k, v in modify_fields.items():
        pending_draft[k] = v

    validation_ctx = _build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )
    changed = set(modify_fields.keys())
    errors = validate_task_draft(
        pending_draft, context=validation_ctx,
        changed_fields=changed, game_data=game_data,
    )
    if errors:
        return json.dumps({
            "status": "validation_failed",
            "draft_id": pending_draft.get("draft_id", ""),
            "errors": errors,
            "draft_summary": _summarize_draft(pending_draft),
        }, ensure_ascii=False), pending_draft
    else:
        return json.dumps({
            "status": "draft_updated",
            "draft_id": pending_draft.get("draft_id", ""),
            "message": "草案已更新，等待玩家确认。",
            "draft_summary": _summarize_draft(pending_draft),
        }, ensure_ascii=False), pending_draft


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
    errors = validate_task_draft(pending_draft, context=validation_ctx, game_data=game_data)
    if errors:
        return json.dumps({
            "status": "validation_failed",
            "errors": errors,
            "message": "草案校验未通过，无法确认。",
        }, ensure_ascii=False), pending_draft, None

    # 分配正式任务 ID
    task_id = game_data.tasks.get_max_agent_task_id() + 1
    pending_draft["id"] = task_id

    write_desc = f"任务 {task_id}「{pending_draft.get('title', '')}」已写入。"
    return json.dumps({
        "status": "confirmed",
        "task_id": task_id,
        "message": write_desc,
        "draft_summary": _summarize_draft(pending_draft),
    }, ensure_ascii=False), None, write_desc


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

    return result, updated_draft, task_write_result
