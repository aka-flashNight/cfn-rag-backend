"""NPC 日常对话覆盖层 ``dialogue_overrides.json`` 的读写与合并（附录 B-5）。

- 结构：扁平 ``overrides`` 数组，每条自带 ``npc`` 与 ``action``，不按 NPC 分节；
- ``replace`` 按 (``npc``, ``group_id``) 整组替换台词，``TaskRequirement`` 沿用被替换的组；
- ``append`` 新增一组，``group_id`` 由后端按 NPC 作用域分配 ``agent_<n>``（大模型不给 id）；
- 数组顺序即应用顺序，后应用的赢；同 ``group_id`` 重复应用不产生重复组；
- 单文件条数硬上限 200，超出先淘汰最旧的 ``append``（``replace`` 不淘汰）；
- fail-closed：文件不存在＝该存档无增量（静默跳过）；解析失败＝记 error 并按空处理，
  绝不让半份数据糊掉基础对话；条目引用了基础数据里不存在的组＝记警告并忽略该条。

基础 XML 永不改写（FR-8）。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.dialogue.models import DialogueGroup, DialogueLine

logger = logging.getLogger(__name__)

#: 单文件条数硬上限（附录 B-5）
MAX_OVERRIDES_PER_FILE = 200

#: 新增组的 group_id 前缀（与基础数据的纯数字 id 永不相撞）
AGENT_GROUP_PREFIX = "agent_"

_VALID_ACTIONS = ("replace", "append")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DialogueOverride:
    """覆盖层的一条记录。"""

    npc: str
    action: str
    group_id: str
    lines: list[DialogueLine] = field(default_factory=list)
    #: replace 且目标是基础组时＝该组的 ``<Dialogue id>``；目标是新增组时为 None
    base_dialogue_id: str | None = None
    reason: str = ""
    source_task_id: int | None = None
    #: 触发来源（关系变化/态度变化/关键信息/重要任务），仅用于排查与统计
    source_trigger: str = ""
    created_at: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "base_dialogue_id": self.base_dialogue_id,
            "group_id": self.group_id,
            "action": self.action,
            "npc": self.npc,
            "subdialogues": [line.to_subdialogue() for line in self.lines],
            "reason": self.reason,
            "source_task_id": self.source_task_id,
            "source_trigger": self.source_trigger,
            "created_at": self.created_at or _now_iso(),
        }

    @classmethod
    def from_json(cls, raw: Any) -> "DialogueOverride | None":
        if not isinstance(raw, dict):
            return None
        npc = str(raw.get("npc") or "").strip()
        action = str(raw.get("action") or "").strip()
        group_id = str(raw.get("group_id") or "").strip()
        if not npc or action not in _VALID_ACTIONS or not group_id:
            logger.warning("覆盖层条目字段不全或 action 非法，忽略: %r", raw)
            return None
        lines = [
            parsed
            for parsed in (
                DialogueLine.from_subdialogue(item)
                for item in (raw.get("subdialogues") or [])
            )
            if parsed is not None
        ]
        if not lines:
            logger.warning("覆盖层条目 %s/%s 无有效台词，忽略", npc, group_id)
            return None
        base_id = raw.get("base_dialogue_id")
        try:
            task_id = int(raw["source_task_id"]) if raw.get("source_task_id") is not None else None
        except (TypeError, ValueError):
            task_id = None
        return cls(
            npc=npc,
            action=action,
            group_id=group_id,
            lines=lines,
            base_dialogue_id=str(base_id).strip() if base_id is not None else None,
            reason=str(raw.get("reason") or "").strip(),
            source_task_id=task_id,
            source_trigger=str(raw.get("source_trigger") or "").strip(),
            created_at=str(raw.get("created_at") or "").strip(),
        )


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def load_overrides(path: Path) -> list[DialogueOverride]:
    """读覆盖层；文件不存在＝无增量（正常），解析失败＝记 error 并按空处理。"""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("对话覆盖层解析失败，按空处理（保留基础数据）: %s: %s", path, exc)
        return []
    if not isinstance(doc, dict):
        logger.error("对话覆盖层顶层不是对象，按空处理: %s", path)
        return []
    raw_entries = doc.get("overrides")
    if not isinstance(raw_entries, list):
        logger.error("对话覆盖层缺 overrides 数组，按空处理: %s", path)
        return []
    entries: list[DialogueOverride] = []
    for raw in raw_entries:
        parsed = DialogueOverride.from_json(raw)
        if parsed is not None:
            entries.append(parsed)
    return entries


def save_overrides(path: Path, entries: list[DialogueOverride]) -> None:
    """原子写覆盖层（先写临时文件再 os.replace，与 task_tools 同一口径）。"""
    path = Path(path)
    doc = {"overrides": [entry.to_json() for entry in entries]}
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 合并
# ---------------------------------------------------------------------------

def merge_groups(
    base_groups: list[DialogueGroup],
    entries: list[DialogueOverride],
    npc: str,
) -> list[DialogueGroup]:
    """把某 NPC 的覆盖层按数组顺序应用到基础组上，返回合并后的组列表。

    基础组顺序保持不变；新增组追加在末尾（``TaskRequirement`` 恒 0，始终可用）。
    """
    target = (npc or "").strip()
    merged = [group.copy() for group in base_groups if group.npc == target]
    for entry in entries:
        if entry.npc != target:
            continue
        index = next(
            (i for i, g in enumerate(merged) if g.group_id == entry.group_id), None,
        )
        if index is not None:
            # 已存在（基础组被 replace，或同一 group_id 再次应用）→ 就地替换，不产生重复组
            merged[index].lines = [line.copy() for line in entry.lines]
            merged[index].from_overlay = True
            continue
        if entry.action != "append":
            logger.warning(
                "覆盖层引用了不存在的对话组，忽略该条: npc=%s group_id=%s", target, entry.group_id,
            )
            continue
        merged.append(DialogueGroup(
            npc=target,
            group_id=entry.group_id,
            lines=[line.copy() for line in entry.lines],
            task_requirement=0,
            from_overlay=True,
        ))
    return merged


def allocate_group_id(
    npc: str,
    base_groups: list[DialogueGroup],
    entries: list[DialogueOverride],
) -> str:
    """按 NPC 作用域分配 ``agent_<n>``（n 从 1 起自增）。

    扫描范围＝该 NPC 基础文件里的全部组 id ∪ 该存档覆盖层里已分配的 ``agent_*``，
    故与纯数字 id 永不相撞，且换存档/二次新增都不会重号。
    """
    used: set[str] = {
        group.group_id for group in base_groups if group.npc == (npc or "").strip()
    }
    used.update(entry.group_id for entry in entries if entry.npc == (npc or "").strip())

    highest = 0
    for group_id in used:
        if group_id.startswith(AGENT_GROUP_PREFIX):
            suffix = group_id[len(AGENT_GROUP_PREFIX):]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return f"{AGENT_GROUP_PREFIX}{highest + 1}"


def enforce_cap(entries: list[DialogueOverride]) -> list[DialogueOverride]:
    """超出单文件上限时淘汰最旧的 ``append`` 条目（``replace`` 不淘汰）。"""
    if len(entries) <= MAX_OVERRIDES_PER_FILE:
        return list(entries)

    overflow = len(entries) - MAX_OVERRIDES_PER_FILE
    dropped: set[int] = set()
    for index, entry in enumerate(entries):
        if entry.action == "append":
            dropped.add(index)
            if len(dropped) >= overflow:
                break
    if len(dropped) < overflow:
        logger.error(
            "对话覆盖层仍超出上限 %d（append 已淘汰完），保留全部条目不丢弃数据",
            MAX_OVERRIDES_PER_FILE,
        )
    else:
        logger.info("对话覆盖层超出上限，淘汰 %d 条最旧的 append 条目", len(dropped))
    return [entry for index, entry in enumerate(entries) if index not in dropped]


def upsert_override(
    entries: list[DialogueOverride],
    *,
    npc: str,
    group_id: str,
    action: str,
    lines: list[DialogueLine],
    base_dialogue_id: str | None = None,
    reason: str = "",
    source_task_id: int | None = None,
    source_trigger: str = "",
) -> list[DialogueOverride]:
    """写入一条覆盖记录并返回新的条目表（不落盘，由调用方决定何时 save）。

    同一 (npc, group_id) 已有记录时**就地更新**而不是追加——replace 天然去重，
    append 重复写也只是刷新内容，避免同一组在文件里出现两条互相打架的记录。
    """
    result = list(entries)
    for index, entry in enumerate(result):
        if entry.npc == npc and entry.group_id == group_id:
            result[index] = DialogueOverride(
                npc=npc,
                action=action,
                group_id=group_id,
                lines=[line.copy() for line in lines],
                base_dialogue_id=base_dialogue_id if base_dialogue_id is not None else entry.base_dialogue_id,
                reason=reason or entry.reason,
                source_task_id=source_task_id if source_task_id is not None else entry.source_task_id,
                source_trigger=source_trigger or entry.source_trigger,
                created_at=entry.created_at,
            )
            return enforce_cap(result)

    result.append(DialogueOverride(
        npc=npc,
        action=action,
        group_id=group_id,
        lines=[line.copy() for line in lines],
        base_dialogue_id=base_dialogue_id,
        reason=reason,
        source_task_id=source_task_id,
        source_trigger=source_trigger,
        created_at=_now_iso(),
    ))
    return enforce_cap(result)
