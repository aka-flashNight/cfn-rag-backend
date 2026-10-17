"""未完成委托的两块：刚发布（工作项 1）与尚在途中（工作项 2）。

两块与已完成块同源（共享 ∪ 本存档，存档覆盖共享），差别只在：
- 多两段「完成 NPC 场景」与「关卡与怪物」（查不到就整行省略，不写「未知」）；
- 措辞一律是「未完成」，绝不出现「已完成/已交付」；
- **纯读**：不写数据库、不写 npc_state_db、不改 progress、不写 status。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.agent_tools.finished_tasks import (
    MAX_ACTIVE_TASKS_PER_TURN,
    build_active_tasks_block,
    build_published_task_block,
)
from services.integration.save_paths import agent_save_task_paths, shared_task_paths

NPC = "铁匠"
OTHER = "药师"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _task(task_id: int, get_npc: str = NPC, finish_npc: str = NPC) -> dict:
    return {
        "id": task_id,
        "title": f"$AGENT_TITLE_{task_id}",
        "get_npc": get_npc,
        "finish_npc": finish_npc,
        "finish_requirements": ["堕落城#普通"],
        "rewards": ["金币#900"],
        "chain": "委托",
    }


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    tasks_path, text_path = shared_task_paths(root)
    _write(tasks_path, {"tasks": [_task(200001)]})
    _write(text_path, {
        "$AGENT_TITLE_200001": "取回祖传铁锤",
        "$AGENT_DESCRIPTION_200001": "把落在堕落城的祖传铁锤找回来。",
    })
    return root


@pytest.fixture
def rendered(monkeypatch):
    """把场景与怪物查询替换成可编排的桩（本文件只验证装配口径，不验证渲染细节）。"""
    from services.agent_tools import draft_formatting
    from services.npc import scene_context

    scenes: dict[str, str] = {}
    monsters: dict[str, str] = {}
    monkeypatch.setattr(
        scene_context, "npc_scene_text", lambda map_reg, name: scenes.get(name, ""),
    )
    monkeypatch.setattr(
        draft_formatting, "_stage_monster_text", lambda game_data, stage: monsters.get(stage, ""),
    )
    return scenes, monsters


class _FakeGameData:
    """只提供渲染需要的两样东西（map_registry 与 stages/enemy_registry）。"""

    def __init__(self) -> None:
        self.map_registry = object()


# ---------------------------------------------------------------------------
# 刚发布的委托
# ---------------------------------------------------------------------------

def test_published_block_full_shape(data_root, rendered):
    scenes, monsters = rendered
    scenes[NPC] = "废城军械库"
    monsters["堕落城"] = "该关卡有以下怪物：僵尸。"

    block = build_published_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, task_id=200001,
        game_data=_FakeGameData(),
    )
    assert block == (
        "【你刚发布的委托（玩家尚未完成）】\n"
        "此前你发布的委托「取回祖传铁锤」（任务 id 200001）刚刚交给玩家，目前尚未完成。\n"
        "任务说明：把落在堕落城的祖传铁锤找回来。\n"
        "报酬：金币×900\n"
        "任务达成内容（玩家达成）：通关堕落城（普通）\n"
        "完成 NPC：铁匠（废城军械库）\n"
        "关卡与怪物：\n"
        "- 堕落城：该关卡有以下怪物：僵尸。\n"
        "这份委托由你刚发布，玩家尚未完成；玩家问起时按上述要点说明，不要重复发布同类委托。"
    )
    # 未完成块不许出现「已完成」措辞
    assert "已完成" not in block
    assert "已交付" not in block


def test_published_block_dedupes_stage_rows(data_root, rendered):
    """同一关卡只出一行怪物块（难度不同也不例外），顺序沿用任务记录顺序。"""
    scenes, monsters = rendered
    monsters["堕落城"] = "该关卡有以下怪物：僵尸。"
    monsters["医院"] = "该关卡有以下怪物：护士。"
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [{
        **_task(200001),
        "finish_requirements": ["堕落城#普通", "医院#困难", "堕落城#困难"],
    }]})
    _write(text_path, {"$AGENT_TITLE_200001": "取回祖传铁锤"})

    block = build_published_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, task_id=200001,
        game_data=_FakeGameData(),
    )
    assert block.count("- 堕落城：") == 1
    assert block.count("- 医院：") == 1
    assert block.index("- 堕落城：") < block.index("- 医院：")


def test_published_block_degrades_without_scene_or_monsters(data_root, rendered):
    """查不到场景就不写括号、查不到怪物就整行省略——都不留占位、不写「未知」。"""
    block = build_published_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, task_id=200001,
        game_data=_FakeGameData(),
    )
    assert "完成 NPC：铁匠\n" in block
    assert "完成 NPC：铁匠（" not in block  # 场景查不到＝不写括号，也不写「未知」
    assert "关卡与怪物" not in block
    assert "- " not in block


def test_published_block_without_game_data_keeps_core_lines(data_root):
    """game_data 缺失（渲染材料不可用）时仍给名称/说明/报酬/达成内容，只少两段附加信息。"""
    block = build_published_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, task_id=200001,
    )
    assert "任务达成内容（玩家达成）：通关堕落城（普通）" in block
    assert "完成 NPC：铁匠" in block
    assert "关卡与怪物" not in block


def test_published_block_empty_when_title_or_npc_or_id_missing(data_root):
    base = dict(data_root=data_root, slot_key=None, npc_name=NPC)
    # 任务不存在
    assert build_published_task_block(**base, task_id=299999) == ""
    # 不是该 NPC 发布的
    assert build_published_task_block(
        data_root=data_root, slot_key=None, npc_name=OTHER, task_id=200001,
    ) == ""
    # 任务 id 非法
    assert build_published_task_block(**base, task_id=None) == ""
    assert build_published_task_block(**base, task_id="abc") == ""
    # 缺标题文本＝无法指认是哪件委托
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(200002)]})
    _write(text_path, {"$AGENT_TITLE_200001": "取回祖传铁锤"})
    assert build_published_task_block(**base, task_id=200002) == ""


def test_published_block_reads_slot_overlay(data_root):
    """本存档文件后读覆盖共享：标题与达成内容都以存档侧为准。"""
    tasks_path, text_path = agent_save_task_paths(data_root, "slot_1")
    _write(tasks_path, {"tasks": [{**_task(200001), "finish_requirements": ["医院#困难"]}]})
    _write(text_path, {"$AGENT_TITLE_200001": "取回祖传铁锤（本存档改题）"})

    block = build_published_task_block(
        data_root=data_root, slot_key="slot_1", npc_name=NPC, task_id=200001,
    )
    assert "取回祖传铁锤（本存档改题）" in block
    assert "通关医院（困难）" in block


# ---------------------------------------------------------------------------
# 尚在途中的委托
# ---------------------------------------------------------------------------

def test_active_block_lists_in_progress(data_root, rendered):
    scenes, monsters = rendered
    scenes[NPC] = "废城军械库"
    monsters["堕落城"] = "该关卡有以下怪物：僵尸。"

    block = build_active_tasks_block(
        data_root=data_root, slot_key=None, npc_name=NPC,
        active_ids=[200001], finished_ids=[], exclude_ids=[],
        game_data=_FakeGameData(),
    )
    assert block.startswith("【你发布的委托尚在途中（玩家尚未完成）】\n")
    assert "此前你发布的委托「取回祖传铁锤」（任务 id 200001）仍在进行中，玩家尚未完成。" in block
    assert "任务达成内容（玩家达成）：通关堕落城（普通）" in block
    assert "完成 NPC：铁匠（废城军械库）" in block
    assert block.endswith(
        "以上委托由你发布，玩家均未完成；玩家问起时按当前进度说明，"
        "不要声称已经完成，也不要重复发布同类委托。"
    )
    assert "已完成" not in block


def test_active_block_filters_finished_excluded_and_other_npc(data_root, rendered):
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [
        _task(200001),
        _task(200002),
        _task(200003),
        _task(200004, get_npc=OTHER),
    ]})
    _write(text_path, {f"$AGENT_TITLE_{i}": f"委托{i}" for i in (200001, 200002, 200003, 200004)})

    block = build_active_tasks_block(
        data_root=data_root, slot_key=None, npc_name=NPC,
        active_ids=[200001, 200002, 200003, 200004],
        finished_ids=[200001],   # 投影短暂不同步时也不重复说「未完成」
        exclude_ids=[200002],    # 正在被「刚发布」块展示的那条
    )
    assert "委托200003" in block
    assert "委托200001" not in block
    assert "委托200002" not in block
    assert "委托200004" not in block


def test_active_block_caps_and_reports_remainder(data_root, rendered):
    ids = list(range(200001, 200001 + MAX_ACTIVE_TASKS_PER_TURN + 2))
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(i) for i in ids]})
    _write(text_path, {f"$AGENT_TITLE_{i}": f"委托{i}" for i in ids})

    block = build_active_tasks_block(
        data_root=data_root, slot_key=None, npc_name=NPC,
        active_ids=ids, finished_ids=[], exclude_ids=[],
    )
    kept = ids[-MAX_ACTIVE_TASKS_PER_TURN:]
    for task_id in kept:
        assert f"委托{task_id}" in block
    for task_id in ids[: len(ids) - MAX_ACTIVE_TASKS_PER_TURN]:
        assert f"委托{task_id}" not in block
    # 截断必须说出来，不能静默
    assert f"另有 {len(ids) - MAX_ACTIVE_TASKS_PER_TURN} 条进行中的委托未展开。" in block
    # 新的排在前面
    assert block.index(f"委托{kept[-1]}") < block.index(f"委托{kept[0]}")


def test_active_block_empty_cases(data_root):
    base = dict(data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[], exclude_ids=[])
    # 旧游戏不发 tasks_active
    assert build_active_tasks_block(**base, active_ids=None) == ""
    # 空数组
    assert build_active_tasks_block(**base, active_ids=[]) == ""
    # 投影里的 id 全是本 NPC 看不到的
    assert build_active_tasks_block(**base, active_ids=[299999]) == ""
    # 非法项被跳过，不抛
    assert build_active_tasks_block(**base, active_ids=["abc", None, 200001]) != ""
    # 没有 NPC 名
    assert build_active_tasks_block(
        data_root=data_root, slot_key=None, npc_name="", active_ids=[200001],
        finished_ids=[], exclude_ids=[],
    ) == ""


def test_active_block_skips_entries_without_title(data_root):
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(200001), _task(200002)]})
    _write(text_path, {"$AGENT_TITLE_200002": "有标题的那条"})

    block = build_active_tasks_block(
        data_root=data_root, slot_key=None, npc_name=NPC,
        active_ids=[200001, 200002], finished_ids=[], exclude_ids=[],
    )
    assert "有标题的那条" in block
    assert "200001" not in block


def test_unfinished_blocks_write_nothing(data_root):
    """两块都是纯读：跑完目录内容与 mtime 都不变（不落 status、不写存档）。"""
    def snapshot() -> dict[str, int]:
        return {
            str(p.relative_to(data_root)): p.stat().st_size
            for p in sorted(data_root.rglob("*")) if p.is_file()
        }

    before = snapshot()
    build_published_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, task_id=200001,
    )
    build_active_tasks_block(
        data_root=data_root, slot_key=None, npc_name=NPC,
        active_ids=[200001], finished_ids=[], exclude_ids=[],
    )
    assert snapshot() == before
