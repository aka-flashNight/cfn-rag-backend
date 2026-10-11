"""「本 NPC 发布的 Agent 委托已被玩家完成」的实时拼接（阶段三·任务发布→完成）。

只能直读磁盘：TaskRegistry 看不到按存档的 data/rag/agent_saves/<slotKey>/ 覆盖层，
而玩家台词不进检索语料，完成对话原文只能从 agent_text.json 取。
读取范围＝共享文件 ∪ 本存档文件，按 id 合并（存档优先），与语料加载口径一致。

无状态：每轮按当前存档投影里的 tasks_finished 实时算，不做「已通知」记账——
投影只在打开面板/读档/存档落盘/任务完成四处刷新，任务完成后这一轮就能看到。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from services.integration.save_paths import agent_save_task_paths, shared_task_paths

logger = logging.getLogger(__name__)

#: 一轮里最多拼接几条已完成委托（按 id 从新到旧取，控制 token）
MAX_FINISHED_TASKS_PER_TURN = 3


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        text = path.read_text("utf-8")
        return json.loads(text) if text.strip() else default
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 %s 失败，按空处理: %s", path, exc)
        return default


def _merged_docs(data_root: Path, slot_key: str | None) -> tuple[dict[int, dict], dict[str, Any]]:
    """(任务记录 by id, 文本表)；共享文件先读、本存档文件后读并覆盖同 id。"""
    tasks: dict[int, dict] = {}
    texts: dict[str, Any] = {}
    pairs = [shared_task_paths(data_root)]
    if slot_key:
        pairs.append(agent_save_task_paths(data_root, slot_key))
    for tasks_path, text_path in pairs:
        doc = _read_json(tasks_path, {"tasks": []})
        records = doc.get("tasks") if isinstance(doc, dict) else None
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            try:
                task_id = int(record.get("id"))
            except (TypeError, ValueError):
                continue
            tasks[task_id] = record
        text_doc = _read_json(text_path, {})
        if isinstance(text_doc, dict):
            texts.update(text_doc)
    return tasks, texts


def _dialogue_lines(items: Any) -> list[str]:
    """完成对话数组 → 「说话人：台词」行；玩家占位符 $PC 显示为「玩家」。"""
    lines: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        name = str(item.get("name") or "").strip()
        speaker = "玩家" if name == "$PC" else (name or "对方")
        lines.append(f"{speaker}：{text}")
    return lines


def build_finished_task_block(
    *,
    data_root: Path,
    slot_key: str | None,
    npc_name: str,
    finished_ids: Any,
) -> str:
    """该 NPC 发布、且当前存档已完成的委托块；没有匹配项＝空串（整块不注入）。"""
    npc = (npc_name or "").strip()
    if not npc:
        return ""
    try:
        finished = {int(v) for v in (finished_ids or [])}
    except (TypeError, ValueError):
        logger.warning("存档投影 tasks_finished 含非整数项，跳过完成委托拼接")
        return ""
    if not finished:
        return ""

    tasks, texts = _merged_docs(Path(data_root), slot_key)
    matched = [
        (task_id, record)
        for task_id, record in tasks.items()
        if task_id in finished and str(record.get("get_npc") or "").strip() == npc
    ]
    if not matched:
        return ""
    matched.sort(key=lambda pair: pair[0], reverse=True)

    sections: list[str] = []
    for task_id, record in matched[:MAX_FINISHED_TASKS_PER_TURN]:
        title = str(texts.get(f"$AGENT_TITLE_{task_id}") or "").strip()
        if not title:
            logger.warning("委托 %d 缺少标题文本，跳过完成拼接", task_id)
            continue
        lines = _dialogue_lines(texts.get(f"$AGENT_FINISH_{task_id}"))
        section = f"此前你发布的委托「{title}」已被玩家完成"
        if lines:
            section += "，以下对话已经发生：\n" + "\n".join(lines)
        else:
            section += "。"
        sections.append(section)

    if not sections:
        return ""
    return "【你发布的委托已被玩家完成】\n" + "\n\n".join(sections)
