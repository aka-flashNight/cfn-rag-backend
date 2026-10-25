"""既有任务对话文本的 key 级覆盖层 ``task_text_overrides.json``（附录 B-5 同目录）。

内容是**平铺的 ``$KEY → 数组``**，只含被改的 key：AS2 ``ListLoader.dictMerge()``
与宿主 ``MapTaskCatalog`` 的文本表都是「后到覆盖同名 key」，覆盖层排在最后加载即
天然生效，**不需要任何新合并代码**（开发计划 §2.0.8）。

数组项与 ``data/task/text/*.json`` 逐字同形（小写 ``name/title/char/text``，可带
``imageurl``），故改写既有条目时必须原样保留附加字段——那是任务对话里的插图，
丢掉就是删信息。

fail-closed：文件不存在＝该存档无增量（静默跳过）；解析失败＝记 error 并按空处理；
非 ``$`` 开头的 key 一律拒收（会污染游戏侧的文本表）。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from services.dialogue.models import DialogueLine

logger = logging.getLogger(__name__)


def _is_text_key(key: Any) -> bool:
    return isinstance(key, str) and key.startswith("$") and len(key) > 1


def load_task_text_overrides(path: Path) -> dict[str, list[dict[str, Any]]]:
    """读任务文本覆盖层，返回 ``$KEY → 条目数组``。"""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("任务文本覆盖层解析失败，按空处理（保留基础数据）: %s: %s", path, exc)
        return {}
    if not isinstance(doc, dict):
        logger.error("任务文本覆盖层顶层不是对象，按空处理: %s", path)
        return {}

    mapping: dict[str, list[dict[str, Any]]] = {}
    for key, value in doc.items():
        if not _is_text_key(key):
            logger.warning("任务文本覆盖层含非文本 key，忽略: %r", key)
            continue
        if not isinstance(value, list):
            logger.warning("任务文本覆盖层 %s 的值不是数组，忽略", key)
            continue
        items = [item for item in value if isinstance(item, dict)]
        if items:
            mapping[key] = items
    return mapping


def save_task_text_overrides(path: Path, mapping: dict[str, list[dict[str, Any]]]) -> None:
    """原子写任务文本覆盖层（先写临时文件再 os.replace）。"""
    path = Path(path)
    text = json.dumps(mapping, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def lines_from_items(items: Any) -> list[DialogueLine]:
    """任务文本数组 → DialogueLine 列表（跳过非对象与空文本项）。"""
    out: list[DialogueLine] = []
    for raw in items if isinstance(items, list) else []:
        line = DialogueLine.from_task_text_item(raw)
        if line is not None:
            out.append(line)
    return out


def items_from_lines(lines: list[DialogueLine]) -> list[dict[str, Any]]:
    """DialogueLine 列表 → 任务文本数组（附加字段原样带回）。"""
    return [line.to_task_text_item() for line in lines]


def set_key(
    mapping: dict[str, list[dict[str, Any]]],
    key: str,
    lines: list[DialogueLine],
) -> dict[str, list[dict[str, Any]]]:
    """写入一个 ``$KEY`` 并返回新表（不落盘）；key 非法或无有效台词时原样返回。"""
    if not _is_text_key(key):
        logger.warning("拒绝写入非文本 key 的任务覆盖: %r", key)
        return dict(mapping)
    items = items_from_lines(lines)
    if not items:
        logger.warning("任务覆盖 %s 无有效台词，跳过写入", key)
        return dict(mapping)
    result = dict(mapping)
    result[key] = items
    return result


def merged_value(
    key: str,
    base_value: Any,
    mapping: dict[str, list[dict[str, Any]]],
) -> Any:
    """该 key 的最终值：覆盖层有就整份用覆盖层，否则用基础值。"""
    override = mapping.get(key)
    if override:
        return override
    return base_value
