from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.game_data.registry import GameDataRegistry


_WRITE_LOCK = threading.Lock()


def _atomic_write_text(path: Path, text: str) -> None:
    """
    原子写入：先写临时文件再 os.replace。
    """
    tmp_path = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    try:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            # 清理失败不影响主流程
            pass


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return default
    return json.loads(raw)


def _dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _reward_items_to_expr(items: Any) -> List[str]:
    """
    [{item_name, count}] -> ["物品名#数量", ...]
    """
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("item_name")
        count = it.get("count")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            n = int(count)
        except Exception:
            continue
        if n <= 0:
            continue
        out.append(f"{name}#{n}")
    return out


def _stage_reqs_to_strings(stage_reqs: Any) -> List[str]:
    """
    [{stage_area, stage_name, difficulty}] -> ["关卡名#难度", ...]
    """
    out: List[str] = []
    if not isinstance(stage_reqs, list):
        return out
    for sr in stage_reqs:
        if not isinstance(sr, dict):
            continue
        stage_name = sr.get("stage_name")
        difficulty = sr.get("difficulty")
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue
        if not isinstance(difficulty, str) or not difficulty.strip():
            continue
        out.append(f"{stage_name}#{difficulty}")
    return out


def _ensure_int_list(v: Any) -> List[int]:
    if not isinstance(v, list):
        return []
    out: List[int] = []
    for x in v:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def write_confirmed_agent_task_files(
    *,
    draft: Dict[str, Any],
    npc_name_fallback: str,
    game_data: GameDataRegistry,
) -> str:
    """
    将已确认的任务草案写入：
    - resources/data/task/agent_tasks.json
    - resources/data/task/text/agent_text.json

    返回写入描述字符串。
    """
    with _WRITE_LOCK:
        data_root = game_data.data_root
        agent_tasks_path = (data_root / "task" / "agent_tasks.json").resolve()
        agent_text_path = (data_root / "task" / "text" / "agent_text.json").resolve()

        task_id = draft.get("id")
        if task_id is None:
            raise ValueError("draft.id 缺失，无法写入 agent_tasks.json")

        try:
            task_id_int = int(task_id)
        except Exception as e:
            raise ValueError(f"draft.id 非法: {task_id}") from e

        get_npc = (
            draft.get("get_npc")
            if isinstance(draft.get("get_npc"), str) and draft.get("get_npc").strip()
            else draft.get("npc_name")
        )
        finish_npc = (
            draft.get("finish_npc")
            if isinstance(draft.get("finish_npc"), str) and draft.get("finish_npc").strip()
            else draft.get("npc_name")
        )
        if not isinstance(get_npc, str) or not get_npc.strip():
            get_npc = npc_name_fallback
        if not isinstance(finish_npc, str) or not finish_npc.strip():
            finish_npc = npc_name_fallback

        title = draft.get("title") if isinstance(draft.get("title"), str) else ""
        description = (
            draft.get("description") if isinstance(draft.get("description"), str) else ""
        )
        get_text = draft.get("get_conversation_text")
        finish_text = draft.get("finish_conversation_text")
        if not isinstance(get_text, str):
            get_text = ""
        if not isinstance(finish_text, str):
            finish_text = ""

        get_req_list = _ensure_int_list(draft.get("get_requirements"))
        finish_reqs = _stage_reqs_to_strings(draft.get("finish_requirements"))
        finish_submit = _reward_items_to_expr(draft.get("finish_submit_items"))
        finish_contain = _reward_items_to_expr(draft.get("finish_contain_items"))
        rewards = _reward_items_to_expr(draft.get("rewards"))

        # -------- agent_tasks.json --------
        tasks_doc = _read_json(agent_tasks_path, default={"tasks": []})
        if not isinstance(tasks_doc, dict):
            tasks_doc = {"tasks": []}
        tasks = tasks_doc.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = []

        # 防止重复写入同 id
        tasks = [t for t in tasks if not (isinstance(t, dict) and str(t.get("id")) == str(task_id_int))]

        tasks.append(
            {
                "id": task_id_int,
                "title": f"$AGENT_TITLE_{task_id_int}",
                "description": f"$AGENT_DESCRIPTION_{task_id_int}",
                "get_requirements": get_req_list,
                "get_conversation": f"$AGENT_GET_{task_id_int}",
                "get_npc": get_npc,
                "finish_requirements": finish_reqs,
                "finish_submit_items": finish_submit,
                "finish_contain_items": finish_contain,
                "finish_conversation": f"$AGENT_FINISH_{task_id_int}",
                "finish_npc": finish_npc,
                "rewards": rewards,
                "announcement": "",
                "chain": "委托",
            }
        )
        new_tasks_doc = {"tasks": tasks}

        # -------- agent_text.json --------
        text_doc = _read_json(agent_text_path, default={})
        if not isinstance(text_doc, dict):
            text_doc = {}

        text_doc[f"$AGENT_TITLE_{task_id_int}"] = title
        text_doc[f"$AGENT_DESCRIPTION_{task_id_int}"] = description
        text_doc[f"$AGENT_GET_{task_id_int}"] = [
            {
                "name": get_npc,
                "title": get_npc,
                "char": get_npc,
                "text": get_text,
            }
        ]
        text_doc[f"$AGENT_FINISH_{task_id_int}"] = [
            {
                "name": finish_npc,
                "title": finish_npc,
                "char": finish_npc,
                "text": finish_text,
            }
        ]

        _atomic_write_text(agent_tasks_path, _dump_json(new_tasks_doc))
        _atomic_write_text(agent_text_path, _dump_json(text_doc))

        return f"任务 {task_id_int} 已写入 agent_tasks.json 与 agent_text.json。"

