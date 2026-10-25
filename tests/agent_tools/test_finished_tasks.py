"""「本 NPC 发布的委托已被玩家完成」实时拼接（阶段三·任务发布→完成）。

只能直读磁盘：TaskRegistry 看不到按存档的 data/rag/agent_saves/<slotKey>/ 覆盖层，
玩家台词也不进检索语料，完成对话原文只在 agent_text.json 里。
"""

from __future__ import annotations

import json

import pytest

from services.agent_tools.finished_tasks import (
    MARKED_TASK_HINT,
    MAX_FINISHED_TASKS_PER_TURN,
    build_finished_task_block,
    load_task_rewrite_material,
)
from services.integration.save_paths import agent_save_task_paths, shared_task_paths

NPC = "铁匠"


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _block(**kwargs) -> str:
    """拼接结果的文本部分（``FinishedTaskBlock.text``）——下面的断言都针对这段字。"""
    return build_finished_task_block(**kwargs).text


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
    block = _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert block == (
        "【你发布的委托已被玩家完成】\n"
        "此前你发布的委托「取回祖传铁锤」（任务 id 200001）已被玩家完成。\n"
        "任务达成内容（玩家达成）：无\n"
        "任务后：\n"
        "玩家：铁锤给你带回来了。\n"
        f"{NPC}：太好了，这份情我记着。"
    )
    # 玩家占位符 $PC 不暴露给模型
    assert "$PC" not in block
    # 没有接取对话与报酬时不给空分组、不给空行
    assert "任务前" not in block
    assert "报酬" not in block
    assert "任务说明" not in block


def test_block_splits_get_and_finish_dialogue_and_carries_reward(data_root):
    """任务前＝接取对话、任务后＝完成对话，达成内容行夹在两组对话中间；报酬行排在对话之前。"""
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [{**_task(200001), "rewards": ["金币#90000", "光剑#1"]}]})
    _write(text_path, {
        "$AGENT_TITLE_200001": "取回祖传铁锤",
        "$AGENT_GET_200001": [
            {"name": NPC, "text": "帮我把祖传铁锤找回来。"},
            {"name": "$PC", "text": "说好的报酬呢？"},
        ],
        "$AGENT_FINISH_200001": [{"name": NPC, "text": "干得漂亮。"}],
    })

    block = _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert (
        "报酬：金币×90000、光剑×1\n"
        "任务前：\n"
        f"{NPC}：帮我把祖传铁锤找回来。\n"
        "玩家：说好的报酬呢？\n"
        "任务达成内容（玩家达成）：无\n"
        "任务后：\n"
        f"{NPC}：干得漂亮。"
    ) in block


def test_block_carries_description_and_requirements(data_root):
    """NPC 要能复述自己发过的是什么活：说明文本与达成内容都排在完成对话之前。
    落盘记录里没有 task_type（只存在于草案），类型由达成内容体现。"""
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [{
        **_task(200001),
        "finish_submit_items": ["祖传铁锤#1", "兽皮#3"],
        "finish_requirements": ["堕落城#普通"],
    }]})
    _write(text_path, {
        "$AGENT_TITLE_200001": "取回祖传铁锤",
        "$AGENT_DESCRIPTION_200001": "把落在堕落城的祖传铁锤找回来。",
        "$AGENT_FINISH_200001": [{"name": "$PC", "text": "铁锤给你带回来了。"}],
    })

    block = _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert "任务说明：把落在堕落城的祖传铁锤找回来。" in block
    assert (
        "任务达成内容（玩家达成）：提交祖传铁锤×1、兽皮×3给铁匠；通关堕落城（普通）" in block
    )
    assert block.index("任务说明") < block.index("任务达成内容") < block.index("任务后：")


def test_completion_phrase_carry_phrase_for_contain_only(data_root):
    """只要收集不缴纳 → 「获取X（不提交）」；关卡与物品并存时两段用分号隔开。"""
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [{
        **_task(200001), "finish_contain_items": ["兽皮#3"],
    }]})
    _write(text_path, {"$AGENT_TITLE_200001": "取回祖传铁锤"})

    block = _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert block == (
        "【你发布的委托已被玩家完成】\n"
        "此前你发布的委托「取回祖传铁锤」（任务 id 200001）已被玩家完成。\n"
        "任务达成内容（玩家达成）：获取兽皮×3（不提交）"
    )


def test_empty_when_nothing_finished_or_no_match(data_root):
    assert _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[],
    ) == ""
    # 完成了但不是这个 NPC 发布的
    assert _block(
        data_root=data_root, slot_key=None, npc_name="药师", finished_ids=[200001],
    ) == ""
    # 这个 NPC 发布的但还没完成
    assert _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[299999],
    ) == ""
    assert _block(
        data_root=data_root, slot_key=None, npc_name="", finished_ids=[200001],
    ) == ""


def test_malformed_finished_ids_degrades_to_empty(data_root):
    assert _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=["abc"],
    ) == ""
    assert _block(
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

    block = _block(
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

    assert _block(
        data_root=data_root, slot_key="slot_1", npc_name=NPC, finished_ids=[200005],
    ) != ""
    assert _block(
        data_root=data_root, slot_key="slot_2", npc_name=NPC, finished_ids=[200005],
    ) == ""


def test_missing_title_text_skips_entry(data_root):
    """缺标题文本＝没法向 NPC 指认是哪件委托，跳过而不是拼个空标题。"""
    tasks_path, _ = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(200001), _task(200003)]})

    block = _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001, 200003],
    )
    assert "取回祖传铁锤" in block
    assert "200003" not in block


def test_finish_without_dialogue_uses_short_form(data_root):
    _, text_path = shared_task_paths(data_root)
    _write(text_path, {"$AGENT_TITLE_200001": "取回祖传铁锤", "$AGENT_FINISH_200001": []})

    block = _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert "此前你发布的委托「取回祖传铁锤」（任务 id 200001）已被玩家完成。" in block
    assert "任务前" not in block
    assert "任务后" not in block


def test_capped_at_max_per_turn_newest_first(data_root):
    """一轮最多拼 MAX_FINISHED_TASKS_PER_TURN 条，按 id 从新到旧取。"""
    ids = list(range(200001, 200001 + MAX_FINISHED_TASKS_PER_TURN + 2))
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(i) for i in ids]})
    _write(text_path, {f"$AGENT_TITLE_{i}": f"委托{i}" for i in ids})

    block = _block(
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
    assert _block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    ) == ""


def test_missing_files_degrade_to_empty(tmp_path):
    assert _block(
        data_root=tmp_path / "data", slot_key="slot_1", npc_name=NPC, finished_ids=[200001],
    ) == ""


def test_marked_task_carries_no_duplicate_call_hint(data_root):
    """发布时挂了对话重写标记＝后台已自动委派，常驻提醒模型别因这条委托重复调用；
    未挂标记的委托不给这句，且只有它才让委派工具保留任务兜底表述（has_unmarked）。"""
    tasks_path, text_path = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [_task(200001), {**_task(200002), "dialogue_rewrite": True}]})
    _write(text_path, {
        "$AGENT_TITLE_200001": "取回祖传铁锤",
        "$AGENT_TITLE_200002": "护送商队",
    })

    marked = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200002],
    )
    assert MARKED_TASK_HINT in marked.text
    assert marked.has_unmarked is False

    unmarked = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert MARKED_TASK_HINT not in unmarked.text
    assert unmarked.has_unmarked is True


def test_rewrite_material_drops_chat_only_hint(data_root):
    """同一份委托：聊天侧完成块保留「勿重复调用」提醒，子 Agent 的材料里不留这句。

    被派来改对话的那个 Agent 读到「已自动触发对话修改子 Agent」只会自我怀疑，
    这句只服务于聊天主 Agent 的重复派发判断。
    """
    tasks_path, _ = shared_task_paths(data_root)
    _write(tasks_path, {"tasks": [{**_task(200001), "dialogue_rewrite": True,
                                  "dialogue_rewrite_instruction": "语气更敬重些"}]})

    material = load_task_rewrite_material(
        data_root=data_root, slot_key=None, task_id=200001,
    )
    assert material is not None
    assert material.marked is True
    assert material.instruction == "语气更敬重些"
    assert MARKED_TASK_HINT not in material.detail_block
    assert "你发布的委托「取回祖传铁锤」（任务 id 200001），玩家刚刚完成。" in material.detail_block

    chat_block = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=[200001],
    )
    assert MARKED_TASK_HINT in chat_block.text


def test_has_unmarked_only_counts_tasks_actually_in_block(data_root):
    """条件闸只看真的拼进这一轮的委托：被每轮上限砍掉的委托模型也指认不了 id。"""
    ids = list(range(200001, 200001 + MAX_FINISHED_TASKS_PER_TURN + 1))
    tasks_path, text_path = shared_task_paths(data_root)
    # 最旧那条没挂标记，但它排在截断之外（按 id 从新到旧取）
    _write(tasks_path, {"tasks": [
        _task(ids[0]),
        *[{**_task(i), "dialogue_rewrite": True} for i in ids[1:]],
    ]})
    _write(text_path, {f"$AGENT_TITLE_{i}": f"委托{i}" for i in ids})

    block = build_finished_task_block(
        data_root=data_root, slot_key=None, npc_name=NPC, finished_ids=ids,
    )
    assert f"委托{ids[0]}" not in block.text
    assert block.has_unmarked is False
