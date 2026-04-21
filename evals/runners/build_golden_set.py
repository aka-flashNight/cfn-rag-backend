"""
从当前向量索引分层采样构造 golden jsonl（expected_doc_ids 与索引 node_id 对齐）。

用法（在仓库根目录）::

    python -m evals.runners.build_golden_set --tiny
    python -m evals.runners.build_golden_set --full
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_engine.game_data_loader import get_cached_index, iter_docstore_nodes

# Golden 抽样时排除的 NPC 阵营（非典型对话样本；与 npc_state_db.json 中 faction 字段一致）
EXCLUDED_GOLDEN_FACTIONS: frozenset[str] = frozenset({"彩蛋", "成员"})


def _load_character_faction_map() -> dict[str, str]:
    """
    角色名（小写） -> 阵营字符串。
    若无法读取 npc_state_db.json，返回空 dict（不排除任何节点，并依赖上游告警）。
    """
    try:
        from services.game_data.paths import find_resources_directory
    except Exception:
        return {}
    try:
        root = find_resources_directory()
    except Exception:
        return {}
    path = root / "data" / "rag" / "npc_state_db.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for name, item in raw.items():
        if not isinstance(item, dict):
            continue
        fac = str(item.get("faction") or "").strip()
        key = str(name).strip().lower()
        if key:
            out[key] = fac
    return out


def _faction_excluded(character_lower: str, faction_map: dict[str, str]) -> bool:
    fac = faction_map.get((character_lower or "").strip().lower(), "")
    return bool(fac and fac in EXCLUDED_GOLDEN_FACTIONS)


def _filter_nodes_by_faction(nodes: list, faction_map: dict[str, str]) -> list:
    """排除彩蛋/成员阵营 NPC 的节点（dialogue / task 等带 character 的池）。"""
    kept: list = []
    for node in nodes:
        meta = getattr(node, "metadata", None) or {}
        ch = str(meta.get("character") or "").strip().lower()
        if _faction_excluded(ch, faction_map):
            continue
        kept.append(node)
    return kept


def _text_preview(text: str, n: int = 100) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


def _sample_from_pool(
    rng: random.Random,
    pool: list,
    n: int,
) -> list:
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def build_rows(
    rng: random.Random,
    *,
    n_dialogue: int,
    n_world: int,
    n_task: int,
    n_intel: int,
) -> list[dict]:
    index = get_cached_index()
    nodes = iter_docstore_nodes(index)
    by_type: dict[str, list] = defaultdict(list)
    for node in nodes:
        t = (node.metadata or {}).get("type") or "unknown"
        by_type[str(t)].append(node)

    faction_map = _load_character_faction_map()
    if not faction_map:
        print(
            "[golden] 警告: 未读到 npc_state_db.json 或 faction 为空，"
            "无法按阵营排除彩蛋/成员；请确认 resources/data/rag/npc_state_db.json 存在且含 faction。",
            file=sys.stderr,
        )
    else:
        n_before_d = len(by_type.get("dialogue", []))
        n_before_t = len(by_type.get("task", []))
        by_type["dialogue"] = _filter_nodes_by_faction(by_type.get("dialogue", []), faction_map)
        by_type["task"] = _filter_nodes_by_faction(by_type.get("task", []), faction_map)
        print(
            f"[golden] 已排除阵营 {sorted(EXCLUDED_GOLDEN_FACTIONS)}："
            f"dialogue {n_before_d}->{len(by_type['dialogue'])}，"
            f"task {n_before_t}->{len(by_type['task'])}",
        )

    rows: list[dict] = []
    idx = 0

    d_pool = by_type.get("dialogue", [])
    if len(d_pool) < n_dialogue:
        print(
            f"[golden] 错误: 过滤后 dialogue 池仅 {len(d_pool)} 条，需要 {n_dialogue} 条。"
            "请检查索引与 npc_state_db。",
            file=sys.stderr,
        )
        sys.exit(1)
    for node in _sample_from_pool(rng, d_pool, min(n_dialogue, len(d_pool))):
        ch = (node.metadata or {}).get("character") or "unknown"
        text = getattr(node, "text", "") or ""
        q = (
            f"在「{ch}」的台词中，是否出现过与下列表述相近的内容？"
            f"{_text_preview(text, 80)}"
        )
        rows.append(
            {
                "id": f"g-dlg-{idx}",
                "type": "dialogue",
                "filter_type": "dialogue",
                "filter_character": str(ch).lower(),
                "npc_name": ch,
                "question": q,
                "retrieve_query": q,
                "expected_doc_ids": [node.node_id],
                "expected_answer_contains": [ch[:4], "台词"],
            }
        )
        idx += 1

    w_pool = by_type.get("world_lore", [])
    for node in _sample_from_pool(rng, w_pool, min(n_world, len(w_pool))):
        text = getattr(node, "text", "") or ""
        q = f"根据核心世界观设定，下列片段讨论的主题是什么？{_text_preview(text, 120)}"
        rows.append(
            {
                "id": f"g-wl-{idx}",
                "type": "world_lore",
                "filter_type": "world_lore",
                "npc_name": "Andy Law",
                "question": q,
                "retrieve_query": q,
                "expected_doc_ids": [node.node_id],
                "expected_answer_contains": ["世界", "设定"],
            }
        )
        idx += 1

    t_pool = by_type.get("task", [])
    if len(t_pool) < n_task:
        print(
            f"[golden] 错误: 过滤后 task 池仅 {len(t_pool)} 条，需要 {n_task} 条。",
            file=sys.stderr,
        )
        sys.exit(1)
    for node in _sample_from_pool(rng, t_pool, min(n_task, len(t_pool))):
        ch = (node.metadata or {}).get("character") or "unknown"
        text = getattr(node, "text", "") or ""
        q = f"与任务对话相关：{_text_preview(text, 100)}"
        rows.append(
            {
                "id": f"g-task-{idx}",
                "type": "task",
                "filter_type": "task",
                "filter_character": str(ch).lower(),
                "npc_name": ch,
                "question": q,
                "retrieve_query": q,
                "expected_doc_ids": [node.node_id],
                "expected_answer_contains": ["任务"],
            }
        )
        idx += 1

    i_pool = by_type.get("intelligence", [])
    for node in _sample_from_pool(rng, i_pool, min(n_intel, len(i_pool))):
        text = getattr(node, "text", "") or ""
        q = f"情报档案中是否包含下列信息？{_text_preview(text, 90)}"
        rows.append(
            {
                "id": f"g-int-{idx}",
                "type": "intelligence",
                "filter_type": "intelligence",
                "npc_name": "Andy Law",
                "question": q,
                "retrieve_query": q,
                "expected_doc_ids": [node.node_id],
                "expected_answer_contains": ["情报"],
            }
        )
        idx += 1

    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[golden] wrote {len(rows)} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="5 条微型集（调通流水线）")
    parser.add_argument("--full", action="store_true", help="完整分层采样（约 80 条）")
    args = parser.parse_args()

    rng = random.Random(42)
    out_dir = _ROOT / "evals" / "datasets"

    if args.tiny:
        rows = build_golden(
            rng,
            n_dialogue=2,
            n_world=1,
            n_task=1,
            n_intel=1,
        )
        _write_jsonl(out_dir / "tiny_golden.jsonl", rows)
        return

    if args.full:
        rows = build_golden(
            rng,
            n_dialogue=50,
            n_world=30,
            n_task=30,
            n_intel=20,
        )
        _write_jsonl(out_dir / "golden_v1.jsonl", rows)
        return

    parser.error("请指定 --tiny 或 --full")


def build_golden(
    rng: random.Random,
    *,
    n_dialogue: int,
    n_world: int,
    n_task: int,
    n_intel: int,
) -> list[dict]:
    return build_rows(
        rng,
        n_dialogue=n_dialogue,
        n_world=n_world,
        n_task=n_task,
        n_intel=n_intel,
    )


if __name__ == "__main__":
    main()
