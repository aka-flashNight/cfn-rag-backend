"""「本 NPC 发布的委托已被玩家完成」实时拼接（阶段三·任务发布→完成）。

只能直读磁盘：TaskRegistry 看不到按存档的 data/rag/agent_saves/<slotKey>/ 覆盖层，
玩家台词也不进检索语料，完成对话原文只在 agent_text.json 里。
"""

from __future__ import annotations

import json

import pytest

from services.agent_tools.finished_tasks import (
    MAX_FINISHED_TASKS_PER_TURN,
    build_finished_task_block,
)
from services.integration.save_paths import agent_save_task_paths, shared_task_paths

NPC = "铁匠"


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _task(task_id: int, get_npc: str = NPC, finish_npc: str = NPC) -> dict:
    return {
        "id": task_id,
        "title": f"$AGENT_TITLE_{task_id}",
        "get_npc": get_npc,
        "finish_npc": finish_npc,
        "finish_conversation": f"$AGENT_FINISH_{task_id}",
        "chain": "委托",
    }


@pytest.fixture
def data_root(tmp_path):
    """共享侧一条已完成委托（含玩家台词与 NPC 台词）。"""
    tasks_path, text_path = shared_task_paths(tmp_path / "data")
    _write(tasks_path, {"tasks": [_task(200001)]})
    _write(text_path, {
        "$AGENT_TITLE_200001": "取回祖传铁锤",
        "$AGENT_FINISH_200001": [
            {"name": "$PC", "title": "", "char": "", "text": "铁锤给你带回来了。"},
            {"name": NPC, "title": "", "char": "", "text": "太好了，这份情我记着。"},
        ],
    })
    return tmp_path / "data"


def test_block_contains_title_and_dialogue(data_root):
    block = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert block.startswith("【你发布的委托已被玩家完成】")
    assert "此前你发布的委托「取回祖传铁锤」已被玩家完成，以下对话已经发生：" in block
    assert "玩家：铁锤给你带回来了。" in block
    assert f"{NPC}：太好了，这份情我记着。" in block
    # 玩家占位符 $PC 不暴露给模型
    assert "$PC" not in block


def test_empty_when_nothing_finished_or_no_match(data_root):
    assert build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[],
    ) == ""
    # 完成了但不是这个 NPC 发布的
    assert build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name="药师", finished_ids=[200001],
    ) == ""
    # 这个 NPC 发布的但还没完成
    assert build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[299999],
    ) == ""
    assert build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name="", finished_ids=[200001],
    ) == ""


def test_malformed_finished_ids_degrades_to_empty(data_root):
    assert build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=["abc"],
    ) == ""
    assert build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=None,
    ) == ""


def test_slot_overlay_wins_over_shared(data_root):
    """本存档文件后读并覆盖同 id：标题与完成对话都以存档侧为准。"""
    tasks_path, text_path = agent_save_task_paths(data_root, "slot_1")
    _write(tasks_path, {"tasks": [_task(200001), _task(200002)]})
    _write(text_path, {
        "$AGENT_TITLE_200001": "取回祖传铁锤（本存档改题）",
        "$AGENT_TITLE_200002": "护送商队",
        "$AGENT_FINISH_200002": [{"name": "$PC", "text": "送到了。"}],
    })

    block = build_finished_task_block(
        data_root=data_root, slot_key="slot_1", npc_name=NPC,
        finished_ids=[200001, 200002],
    )
    assert "取回祖传铁锤（本存档改题）" in block
    assert "护送商队" in block
    # 200001 存档侧只覆盖了标题，完成对话仍从共享文本表取到
    assert "玩家：铁锤给你带回来了。" in block


def test_slot_tasks_invisible_to_other_slots(data_root):
    tasks_path, text_path = agent_save_task_paths(data_root, "slot_1")
    _write(tasks_path, {"tasks": [_task(200005)]})
    _write(text_path, {"$AGENT_TITLE_200005": "只有存档一有的委托"})

    assert build_finished_task_block(
        data_root=data_root, slot_key="slot_1", npc_name=NPC, finished_ids=[200005],
    ) != ""
    assert build_finished_task_block(
        data_root=data_root, slot_key="slot_2", npc_name=NPC, finished_ids=[200005],
    ) == ""


def test_missing_title_text_skips_entry(data_root):
    """缺标题文本＝没法向 NPC 指认是哪件委托，跳过而不是拼个空标题。"""
    tasks_path, _ = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(200001), _task(200003)]})

    block = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001, 200003],
    )
    assert "取回祖传铁锤" in block
    assert "200003" not in block


def test_finish_without_dialogue_uses_short_form(data_root):
    _, text_path = shared_task_paths(data_root)
    _write(text_path, {"$AGENT_TITLE_200001": "取回祖传铁锤", "$AGENT_FINISH_200001": []})

    block = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert "此前你发布的委托「取回祖传铁锤」已被玩家完成。" in block
    assert "以下对话已经发生" not in block


def test_capped_at_max_per_turn_newest_first(data_root):
    """一轮最多拼 MAX_FINISHED_TASKS_PER_TURN 条，按 id 从新到旧取。"""
    ids = list(range(200001, 200001 + MAX_FINISHED_TASKS_PER_TURN + 2))
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(i) for i in ids]})
    _write(text_path, {f"$AGENT_TITLE_{i}": f"委托{i}" for i in ids})

    block = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=ids,
    )
    kept = ids[-MAX_FINISHED_TASKS_PER_TURN:]
    for task_id in kept:
        assert f"委托{task_id}" in block
    for task_id in ids[: len(ids) - MAX_FINISHED_TASKS_PER_TURN]:
        assert f"委托{task_id}" not in block
    # 新的排在前面
    assert block.index(f"委托{kept[-1]}") < block.index(f"委托{kept[0]}")


def test_corrupt_files_degrade_to_empty(data_root):
    tasks_path, _ = shared_task_paths(data_root)
    tasks_path.write_text("{broken json", encoding="utf-8")
    assert build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    ) == ""


def test_missing_files_degrade_to_empty(tmp_path):
    assert build_finished_task_block(
        data_root=tmp_path / "data", slot_key="slot_1", npc_name=NPC, finished_ids=[200001],
    ) == ""
