"""查询期覆盖层替换（作者裁定 2026-10-17）。

覆盖层不进向量库、也不计入语料指纹（理由见 ``loader`` 模块 docstring：AI 终端改对话、
发任务都写这些文件，计入等于每次都全量重建 23–31 秒），本模块负责另一半：检索命中
**之后**把结果节点的正文换成当前绑定存档的覆盖层文本。分词与打分仍按基础语料算
（索引里就只有基础语料），给模型看的文本是改写后的。

两类覆盖，定位方式不同：

- ``dialogue_overrides.json``（组级）：按 (块名 ``<Dialogues><Name>``, 组 id ``<Dialogue id>``)
  整组换台词。节点 id 是加载序 ``dialogue-{seq}``，反查靠 ``loader.iter_dialogue_corpus_rows``
  ——语料装配与本模块共用同一次展开，对应关系不可能漂移。
- ``task_text_overrides.json``（``$KEY`` 级）：按文本 key 整段换台词。节点 id
  ``task-{id段}-{kind}-{行序}`` 自带 kind 与行序，id 段→key 由
  ``loader.shared_task_conversation_keys`` 反查。

行序对齐是近似：覆盖层把某组/某 key 的台词整段换掉，行数可能增减，本模块按**同序号行**
取新文本，序号超出覆盖层行数（或该行已变成玩家台词）时该节点直接退出结果——不去补新行，
也不重排。覆盖层**新增**的内容（``append`` 的新对话组、Agent 新发布的任务）不在索引里，
检索不到；这是把高频写盘挡在索引外的已知代价。

缓存只用于「节点 id → 定位键」的映射（只依赖基础数据文件，按这些文件的 mtime/size
集合失效）；覆盖层文件本身每次都重读——它们正是被替换的对象，缓存旧内容等于把改写
挡在检索外。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from services.dialogue import overrides as dialogue_overlay
from services.dialogue import task_text as task_text_overlay
from services.integration.save_paths import agent_save_override_paths
from services.integration.state import get_integration_state
from services.retrieval import loader
from services.retrieval.hybrid import RetrievalBundle, ScoredNode

logger = logging.getLogger(__name__)

#: bundle 里承载检索结果的字段（实体池没有可覆盖的台词，一并跳过靠 type 判定）
_BUNDLE_POOL_FIELDS = (
    "npc_dialogue", "world_lore", "loading", "npc_task",
    "supp_intel", "other_npc", "entity_items", "entity_stages",
)

#: 对话映射缓存：(基础对话文件状态集合) → {节点 id: (块名, 组 id)}
_dialogue_map_cache: tuple[Any, Any] | None = None
#: 任务 key 映射缓存：(任务配置文件状态集合) → {(id 段, kind): 文本 key}
_task_key_cache: tuple[Any, Any] | None = None


def apply_slot_overrides(bundle: RetrievalBundle | None) -> RetrievalBundle | None:
    """把当前绑定存档的覆盖层文本替换进检索结果；未绑定、无覆盖层或出错都原样返回。

    best-effort：任何异常只记日志并返回原 bundle，绝不因覆盖失败打断回合。
    """
    if bundle is None:
        return None
    slot_key = get_integration_state().slot_key
    if not slot_key:
        return bundle
    try:
        root = loader._resources_dir()
        dialogue_path, task_text_path = agent_save_override_paths(root / "data", slot_key)
        dialogue_entries = _dialogue_entry_map(dialogue_path)
        task_overrides = _task_text_map(task_text_path)
        if not dialogue_entries and not task_overrides:
            return bundle
        for field in _BUNDLE_POOL_FIELDS:
            picked = getattr(bundle, field, None)
            if not picked:
                continue
            kept = [
                replaced
                for replaced in (
                    _replace_one(sn, dialogue_entries, task_overrides, root) for sn in picked
                )
                if replaced is not None
            ]
            setattr(bundle, field, kept)
    except Exception as exc:  # 覆盖失败退回基础文本
        logger.warning("检索结果覆盖层替换失败，退回基础语料: %s", exc)
    return bundle


# ---------------------------------------------------------------------------
# 单节点替换
# ---------------------------------------------------------------------------

def _replace_one(
    sn: ScoredNode,
    dialogue_entries: dict[tuple[str, str], Any],
    task_overrides: dict[str, list],
    root: Path,
) -> ScoredNode | None:
    node = sn.node
    if node.type == "dialogue":
        return _replace_dialogue(sn, dialogue_entries, root)
    if node.type == "task":
        return _replace_task(sn, task_overrides, root)
    return sn


def _replace_dialogue(
    sn: ScoredNode, dialogue_entries: dict[tuple[str, str], Any], root: Path,
) -> ScoredNode | None:
    if not dialogue_entries:
        return sn
    key = _dialogue_key_map(root).get(sn.node.id)
    if key is None:
        return sn
    entry = dialogue_entries.get(key)
    if entry is None:
        return sn
    text, character_key = loader._dialogue_node_text(entry.lines, key[0])
    if not text or not character_key:
        return None  # 改写后整组只剩玩家台词，不进检索结果
    return _rewrapped(sn, text=text, character=character_key.strip().lower())


def _replace_task(
    sn: ScoredNode, task_overrides: dict[str, list], root: Path,
) -> ScoredNode | None:
    if not task_overrides:
        return sn
    parsed = _parse_task_node_id(sn.node.id)
    if parsed is None:
        return sn
    id_part, kind, idx = parsed
    key = _task_conversation_key_map(root).get((id_part, kind))
    if not key:
        return sn
    lines = task_overrides.get(key)
    if not isinstance(lines, list) or idx >= len(lines):
        return None  # 覆盖层把这段改短了，原序号行已不存在
    item = lines[idx]
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()
    character = loader._task_character_from_item(item)
    if not text or not character:
        return None
    return _rewrapped(sn, text=text, character=character)


def _rewrapped(sn: ScoredNode, *, text: str, character: str) -> ScoredNode:
    """换正文与说话人，保留节点 id 与分数（分数始终按基础语料算）。"""
    return ScoredNode(
        node=replace(sn.node, text=text, character=character),
        dense_score=sn.dense_score,
        fused_score=sn.fused_score,
    )


def _parse_task_node_id(node_id: str) -> tuple[str, str, int] | None:
    """``task-{id段}-{kind}-{行序}`` → (id段, kind, 行序)；id 段可以含 ``-``，故从右切。"""
    if not node_id.startswith("task-"):
        return None
    parts = node_id[len("task-"):].rsplit("-", 2)
    if len(parts) != 3 or parts[1] not in ("get", "finish") or not parts[2].isdigit():
        return None
    return parts[0], parts[1], int(parts[2])


# ---------------------------------------------------------------------------
# 覆盖层读取（每次都读，不缓存内容）
# ---------------------------------------------------------------------------

def _dialogue_entry_map(path: Path) -> dict[tuple[str, str], Any]:
    """(块名, 组 id) → 覆盖层条目；同键后写的赢（与 overrides.merge_groups 一致）。"""
    out: dict[tuple[str, str], Any] = {}
    for entry in dialogue_overlay.load_overrides(path):
        out[(entry.npc, entry.group_id)] = entry
    return out


def _task_text_map(path: Path) -> dict[str, list]:
    return task_text_overlay.load_task_text_overrides(path)


# ---------------------------------------------------------------------------
# 节点 id → 定位键（只依赖基础数据，按文件状态缓存）
# ---------------------------------------------------------------------------

def _dialogue_key_map(root: Path) -> dict[str, tuple[str, str]]:
    global _dialogue_map_cache
    states = _file_states(root / "data" / "dialogues", "*.xml")
    cache = _dialogue_map_cache
    if cache is not None and cache[0] == states:
        return cache[1]
    out = {node_id: (block_name, group_id) for node_id, block_name, group_id, *_rest in
           loader.iter_dialogue_corpus_rows(root)}
    _dialogue_map_cache = (states, out)
    return out


def _task_conversation_key_map(root: Path) -> dict[tuple[str, str], str]:
    global _task_key_cache
    states = _file_states(root / "data" / "task", "*tasks*.json")
    cache = _task_key_cache
    if cache is not None and cache[0] == states:
        return cache[1]
    out = loader.shared_task_conversation_keys(root)
    _task_key_cache = (states, out)
    return out


def _file_states(base: Path, pattern: str) -> tuple[tuple[str, int, int], ...]:
    """目录下匹配文件的 (名称, mtime_ns, size) 有序集合，作缓存失效键。"""
    if not base.is_dir():
        return ()
    out = []
    for f in sorted(base.rglob(pattern)):
        if not f.is_file():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        out.append((f.name, st.st_mtime_ns, st.st_size))
    return tuple(out)


def reset_cache_for_tests() -> None:
    """测试用：清空两份映射缓存（tmp 资源树之间会互相撞缓存）。"""
    global _dialogue_map_cache, _task_key_cache
    _dialogue_map_cache = None
    _task_key_cache = None
