"""任务发布流水线的纯业务函数（无 LLM，无 state mutation）。

由 ``services/tools/<category>/<name>.py`` 的 ``run()`` 方法按需调用；v3 中由
后台子 Agent（TaskRunner）与 orchestrator 的同步动作（confirm/cancel）使用。

对应 docs/v3-developer/05：
- 校验走聚合模式（ValidationReport），错误一次报全；
- draft/update 失败前先尝试 auto_repair（全 issue 可修时），成功则带 repaired_notes 落草案；
- ``draft_summary`` **永远**返回当前草案快照（修 D4：失败时模型也能看到草案现状）；
- bargain_count / _draft_commit_valid 不再混入草案 JSON（修 D5），经 ToolContext /
  ToolResult 独立传递，由调用方（orchestrator）落到 SQLite 独立列。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from services.game_data.registry import GameDataRegistry, get_game_data_registry
from services.game_progress import get_progress_stage_config
from services.agent_tools.context_builder import prepare_task_context
from services.agent_tools.draft_formatting import _detailed_draft_summary
from services.agent_tools.fallback import build_fallback_draft
from services.agent_tools.repair import auto_repair
from services.agent_tools.schemas import normalize_reward_types_for_prepare_context
from services.agent_tools.validator import (
    DraftValidationContext,
    ValidationReport,
    validate_task_draft,
)
from services.agent_tools.task_tools import collect_existing_task_titles

logger = logging.getLogger(__name__)

# 讨价还价上限（业务红线，沿用旧约定）
BARGAIN_LIMIT = 2
# 更新时禁止触碰的字段（仅在 confirm 时写入）
_TEXT_ONLY_ON_CONFIRM = {"description", "get_dialogue", "finish_dialogue"}
# 触发讨价还价计数的奖励类字段
_BARGAIN_KEYS = {"rewards", "finish_submit_items", "finish_contain_items"}


@dataclass
class DraftOpOutcome:
    """draft/update/confirm/cancel 的统一返回（tools 层映射到 ToolResult）。"""

    result_json: str
    draft: Optional[dict[str, Any]] = None          # None = 清空草案（confirm/cancel 成功）
    draft_commit_valid: bool = False
    bargain_count: int = 0
    task_write_result: Optional[str] = None
    fallback_used: bool = False
    payload: dict[str, Any] = field(default_factory=dict)  # result_json 解析后的对象（免二次解析）


def _reward_field_value_changed(cur: Any, new: Any) -> bool:
    if cur is None and (new is None or new == []):
        return False
    if (cur is None or cur == []) and new is None:
        return False
    if cur == new:
        return False
    return True


def build_validation_ctx(
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


def _title_duplicate_warning(
    title: Any,
    game_data: Optional[GameDataRegistry],
) -> Optional[dict[str, Any]]:
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


def _failed_payload(
    report: ValidationReport,
    draft: dict[str, Any],
    game_data: Optional[GameDataRegistry],
    *,
    rag_context_text: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    repaired_notes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """校验失败载荷：全量 issue + 当前草案快照（修 D4，draft_summary 不再返回空串）。

    repaired_notes：后端已自动微调的内容——随打回告知模型"这些已修好，勿改动"。
    """
    payload: dict[str, Any] = report.to_model_json()
    if repaired_notes:
        payload["auto_repaired"] = repaired_notes
    payload["draft_summary"] = _detailed_draft_summary(
        draft, game_data, rag_context_text=rag_context_text,
    )
    if extra:
        payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# prepare_task_context
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


# ---------------------------------------------------------------------------
# draft_agent_task（05 §4 新流程）
# ---------------------------------------------------------------------------

def execute_draft_agent_task(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    bargain_count: int = 0,
    game_data: Optional[GameDataRegistry] = None,
    rag_context_text: Optional[str] = None,
) -> DraftOpOutcome:
    if game_data is None:
        game_data = get_game_data_registry()

    draft_id = str(uuid.uuid4())[:8]
    args_clean = dict(args)
    for k in ("description", "get_dialogue", "finish_dialogue", "ui_hint"):
        args_clean.pop(k, None)

    draft: dict[str, Any] = {
        "draft_id": draft_id,
        "npc_name": npc_name,
        **args_clean,
    }

    validation_ctx = build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )

    report = validate_task_draft(draft, context=validation_ctx, game_data=game_data)
    notes: list[str] = []

    if not report.ok:
        # 数值类 issue 先由后端全量微调（V7 取整/V2 clamp/V10 降级/V11 去重），
        # 微调后仍存在选择类问题（V1/V3~V6/V8）才打回（05 §4）
        if report.issues:
            draft, notes, report = auto_repair(
                draft, report, context=validation_ctx, game_data=game_data,
            )

    if not report.ok:
        return DraftOpOutcome(
            result_json=json.dumps(
                _failed_payload(report, draft, game_data, rag_context_text=rag_context_text,
                                repaired_notes=notes),
                ensure_ascii=False,
            ),
            draft=draft,
            draft_commit_valid=False,
            bargain_count=bargain_count,
        )

    draft.setdefault("repaired_notes", [])
    if notes:
        draft["repaired_notes"] = notes

    detailed = _detailed_draft_summary(draft, game_data, rag_context_text=rag_context_text)
    payload: dict[str, Any] = {
        "status": "draft_created",
        "draft_id": draft_id,
        "message": "任务草案已创建，等待玩家确认。",
        "draft_summary": detailed,
        "bargain_remaining": max(0, BARGAIN_LIMIT - bargain_count),
    }
    if notes:
        payload["repaired_notes"] = notes
    warnings = list(report.warnings or [])
    w_title = _title_duplicate_warning(draft.get("title"), game_data)
    if w_title:
        warnings.append(w_title)
    if warnings:
        payload["warnings"] = warnings
    return DraftOpOutcome(
        result_json=json.dumps(payload, ensure_ascii=False),
        draft=draft,
        draft_commit_valid=True,
        bargain_count=bargain_count,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# update_task_draft（讨价还价 / 局部修改）
# ---------------------------------------------------------------------------

def execute_update_task_draft(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    bargain_count: int = 0,
    draft_commit_valid: bool = False,
    game_data: Optional[GameDataRegistry] = None,
    rag_context_text: Optional[str] = None,
) -> DraftOpOutcome:
    if pending_draft is None:
        return DraftOpOutcome(result_json=json.dumps({
            "status": "error",
            "message": "当前没有待修改的草案，请先调用 draft_agent_task 创建草案。",
        }, ensure_ascii=False))

    if game_data is None:
        game_data = get_game_data_registry()

    modify_fields = args.get("modify_fields", {})
    if not isinstance(modify_fields, dict):
        modify_fields = {}
    modify_fields = {k: v for k, v in modify_fields.items() if k not in _TEXT_ONLY_ON_CONFIRM}
    if not modify_fields:
        return DraftOpOutcome(result_json=json.dumps(
            {
                "status": "error",
                "message": (
                    "modify_fields 不能为空。"
                    "任务说明与接取/完成对话请仅在玩家接受任务时通过 confirm_agent_task 传入，"
                    "不要写在 update_task_draft 中。"
                ),
            },
            ensure_ascii=False,
        ))

    working = dict(pending_draft)
    for k, v in modify_fields.items():
        working[k] = v

    bargain_keys_touched = _BARGAIN_KEYS & set(modify_fields)
    is_bargain = bool(bargain_keys_touched)
    reward_actually_changed = False
    if is_bargain:
        for k in bargain_keys_touched:
            if _reward_field_value_changed(pending_draft.get(k), modify_fields[k]):
                reward_actually_changed = True
                break
        if reward_actually_changed and draft_commit_valid:
            if bargain_count >= BARGAIN_LIMIT:
                return DraftOpOutcome(result_json=json.dumps({
                    "status": "error",
                    "message": f"最多允许讨价还价{BARGAIN_LIMIT}次，已达上限。请让玩家接受或拒绝任务，或取消/拒绝发布。",
                    "draft_id": pending_draft.get("draft_id", ""),
                    "draft_summary": _detailed_draft_summary(
                        pending_draft, game_data, rag_context_text=rag_context_text,
                    ),
                }, ensure_ascii=False), draft=pending_draft,
                    draft_commit_valid=draft_commit_valid, bargain_count=bargain_count)

    validation_ctx = build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
        bargain_rate=1.5 if (is_bargain and reward_actually_changed and draft_commit_valid) else 1.0,
    )
    changed = set(modify_fields.keys())
    report = validate_task_draft(
        working, context=validation_ctx, changed_fields=changed, game_data=game_data,
    )
    notes: list[str] = []
    if not report.ok:
        if report.issues:
            working, notes, report = auto_repair(
                working, report, context=validation_ctx, game_data=game_data,
            )

    if not report.ok:
        return DraftOpOutcome(
            result_json=json.dumps(
                _failed_payload(report, working, game_data, rag_context_text=rag_context_text,
                                repaired_notes=notes),
                ensure_ascii=False,
            ),
            draft=working,
            draft_commit_valid=False,
            bargain_count=bargain_count,
        )

    if notes:
        prev_notes = list(working.get("repaired_notes") or [])
        working["repaired_notes"] = prev_notes + notes

    new_bargain_count = bargain_count
    if is_bargain and reward_actually_changed and draft_commit_valid:
        new_bargain_count = bargain_count + 1

    detailed = _detailed_draft_summary(working, game_data, rag_context_text=rag_context_text)
    payload: dict[str, Any] = {
        "status": "draft_updated",
        "draft_id": working.get("draft_id", ""),
        "message": "草案已更新，等待玩家确认。",
        "draft_summary": detailed,
        "bargain_remaining": max(0, BARGAIN_LIMIT - new_bargain_count),
    }
    if notes:
        payload["repaired_notes"] = notes
    warnings = list(report.warnings or [])
    w_title = _title_duplicate_warning(working.get("title"), game_data)
    if w_title:
        warnings.append(w_title)
    if warnings:
        payload["warnings"] = warnings
    return DraftOpOutcome(
        result_json=json.dumps(payload, ensure_ascii=False),
        draft=working,
        draft_commit_valid=True,
        bargain_count=new_bargain_count,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# confirm_agent_task（校验 + 原子写；orchestrator 同步动作使用）
# ---------------------------------------------------------------------------

def execute_confirm_agent_task(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
    npc_name: str = "",
    player_progress: int = 1,
    npc_affinity: int = 0,
    game_data: Optional[GameDataRegistry] = None,
    rag_context_text: Optional[str] = None,
) -> DraftOpOutcome:
    if pending_draft is None:
        return DraftOpOutcome(result_json=json.dumps({
            "status": "error",
            "message": "当前没有待确认的草案。",
        }, ensure_ascii=False))

    if game_data is None:
        game_data = get_game_data_registry()

    arg_draft_id = str(args.get("draft_id") or "").strip()
    pending_id = str(pending_draft.get("draft_id") or "").strip()
    if not arg_draft_id or arg_draft_id != pending_id:
        return DraftOpOutcome(result_json=json.dumps(
            {
                "status": "error",
                "message": (
                    "draft_id 与当前待确认草案不一致或缺失；请使用工具返回的 draft_summary 中的草案 ID。"
                ),
                "expected_draft_id": pending_id,
                "got_draft_id": arg_draft_id,
            },
            ensure_ascii=False,
        ), draft=pending_draft)

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

    validation_ctx = build_validation_ctx(
        npc_name=npc_name,
        player_progress=player_progress,
        npc_affinity=npc_affinity,
    )
    report = validate_task_draft(draft_for_commit, context=validation_ctx, game_data=game_data)
    if not report.ok and report.issues:
        # 草案存续期玩家进度变化导致回归时先修复再判（05 §6，S5）
        draft_for_commit, notes, report = auto_repair(
            draft_for_commit, report, context=validation_ctx, game_data=game_data,
        )
        if notes:
            draft_for_commit["repaired_notes"] = list(draft_for_commit.get("repaired_notes") or []) + notes
    if not report.ok:
        return DraftOpOutcome(
            result_json=json.dumps(
                {
                    "status": "validation_failed",
                    "message": "草案校验未通过，无法确认。",
                    **report.to_model_json(),
                },
                ensure_ascii=False,
            ),
            draft=pending_draft,
        )

    try:
        from services.agent_tools.task_tools import write_confirmed_agent_task_files

        write_desc, task_id = write_confirmed_agent_task_files(
            draft=draft_for_commit,
            npc_name_fallback=npc_name or str(pending_draft.get("npc_name") or ""),
            game_data=game_data,
        )
    except Exception as e:
        logger.exception("任务写入失败")
        return DraftOpOutcome(result_json=json.dumps(
            {
                "status": "error",
                "message": f"任务写入失败：{str(e)}",
            },
            ensure_ascii=False,
        ), draft=pending_draft)

    detailed = _detailed_draft_summary(
        draft_for_commit, game_data, rag_context_text=rag_context_text,
    )
    confirm_payload: dict[str, Any] = {
        "status": "confirmed",
        "task_id": task_id,
        "message": write_desc,
        "draft_summary": detailed,
    }
    if report.warnings:
        confirm_payload["warnings"] = report.warnings
    return DraftOpOutcome(
        result_json=json.dumps(confirm_payload, ensure_ascii=False),
        draft=None,
        draft_commit_valid=False,
        task_write_result=write_desc,
        payload=confirm_payload,
    )


# ---------------------------------------------------------------------------
# cancel_agent_task
# ---------------------------------------------------------------------------

def execute_cancel_agent_task(
    args: dict[str, Any],
    *,
    pending_draft: Optional[dict[str, Any]] = None,
) -> DraftOpOutcome:
    if pending_draft is None:
        return DraftOpOutcome(result_json=json.dumps({
            "status": "error",
            "message": "当前没有待取消的草案。",
        }, ensure_ascii=False))

    draft_id = pending_draft.get("draft_id", "")
    payload = {
        "status": "cancelled",
        "draft_id": draft_id,
        "message": "任务草案已取消。",
    }
    return DraftOpOutcome(
        result_json=json.dumps(payload, ensure_ascii=False),
        draft=None,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# 兜底草案入口（TaskRunner 轮限耗尽时调用）
# ---------------------------------------------------------------------------

def execute_fallback_draft(
    *,
    direction: str,
    reward_hint: str = "",
    npc_name: str,
    npc_faction: str = "",
    npc_challenge: Optional[str] = None,
    player_progress: int = 1,
    npc_affinity: int = 0,
    npc_states: Optional[dict[str, Any]] = None,
    game_data: Optional[GameDataRegistry] = None,
    rag_context_text: Optional[str] = None,
) -> DraftOpOutcome:
    """后端兜底草案（05 §5）：保证「拟定了方向就能拿出草案」。"""
    if game_data is None:
        game_data = get_game_data_registry()
    try:
        draft = build_fallback_draft(
            direction=direction,
            reward_hint=reward_hint,
            npc_name=npc_name,
            npc_faction=npc_faction,
            npc_challenge=npc_challenge,
            player_progress=player_progress,
            npc_affinity=npc_affinity,
            npc_states=npc_states,
            game_data=game_data,
        )
    except Exception as e:
        logger.warning("兜底草案构造失败: %s", e)
        return DraftOpOutcome(result_json=json.dumps({
            "status": "error",
            "message": f"兜底草案构造失败：{e}",
        }, ensure_ascii=False))

    detailed = _detailed_draft_summary(draft, game_data, rag_context_text=rag_context_text)
    payload = {
        "status": "draft_created",
        "draft_id": draft.get("draft_id", ""),
        "message": "任务草案已按保守方案生成，等待玩家确认。",
        "draft_summary": detailed,
        "fallback": True,
        "fallback_note": draft.get("fallback_note", ""),
        "bargain_remaining": BARGAIN_LIMIT,
    }
    return DraftOpOutcome(
        result_json=json.dumps(payload, ensure_ascii=False),
        draft=draft,
        draft_commit_valid=True,
        bargain_count=0,
        fallback_used=True,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# search_knowledge
# ---------------------------------------------------------------------------

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


__all__ = [
    "BARGAIN_LIMIT",
    "DraftOpOutcome",
    "build_validation_ctx",
    "execute_prepare_task_context",
    "execute_draft_agent_task",
    "execute_update_task_draft",
    "execute_confirm_agent_task",
    "execute_cancel_agent_task",
    "execute_fallback_draft",
    "execute_search_knowledge",
]
