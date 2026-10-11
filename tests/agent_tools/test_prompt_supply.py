"""B8-2 / B8-3 子 Agent 提示词信息供给测试：

- 传话/问候候选轮：发布 NPC 视角场景块（publisher_scene）＋候选条目 scene 字段＋规则句；
- 草案摘要：完成NPC≠发布NPC 时补「所在场景」行与指示句；
- 选关类草案逐关卡追加怪物块（切磋不附）；RAG 已出现的关卡改锚一行。
"""

from __future__ import annotations

import json

from services.agent_tools.context_builder import _TASK_RULES, prepare_task_context
from services.agent_tools.draft_formatting import (
    _build_draft_entity_context_lines,
    _detailed_draft_summary,
)
from services.agent_tools.schemas import normalize_reward_types_for_prepare_context
from services.npc.manager import NPCState

# 与 game_data 侧测试同源的逐字块（地铁站，含无 displayname 兵种 293 的跳过）
TIETIE_MONSTER_BLOCK = (
    "该关卡有以下怪物：黄衣僵尸，金发女僵尸，红衣僵尸，小儿麻痹僵尸，胖僵尸，"
    "眼镜僵尸，光头军人僵尸，狂暴僵尸，防暴僵尸，特战僵尸，狂躁僵尸，"
    "裸体兽化僵尸，兽化僵尸，巨型僵尸。"
    "其中数量最多的是兽化僵尸，大致数量约16个。强度最高的是巨型僵尸。"
    "根据你的身份，你可能知道也可能不知道这个关卡的详情，但绝对不要编造信息，"
    "要么表现的不清楚，要么根据实际信息去说话，不能编造未提供的信息。"
)

EXCLUDE_SENTENCE = (
    "传话/问候任务通常要排除同场景npc，因为你可以和他们直接面对面说话。"
    "优先选择其他场景的npc。"
)

FINISH_NPC_HINT = "非同阵营npc你未必知道准确场景名，但知道大致情况，根据你的人设决定。"

STATES = {
    "Pig": NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-右前锋"]),
    "Boy": NPCState(sex="男", faction="A兵团元老", titles=["BPK战队-队长"]),
    "King": NPCState(sex="女", faction="A兵团元老", titles=["BPK战队-左前锋"]),
    "冷兵器商人": NPCState(sex="男", faction="商人", titles=["传统卫道士"]),
    "武器大师": NPCState(sex="男", faction="闲杂人等", titles=["武器大师"]),
    "lochan": NPCState(sex="女", faction="闲杂人等", titles=["谜之兔女郎"]),
}


def _prepare(game_data, task_type: str, npc_name: str, npc_states=None) -> dict:
    raw = prepare_task_context(
        task_type=task_type,
        reward_types=normalize_reward_types_for_prepare_context(None, ["金币"]),
        npc_name=npc_name,
        player_progress=1,
        npc_states=npc_states,
        game_data=game_data,
    )
    return json.loads(raw)


def _entry(ctx: dict, name: str) -> dict:
    for it in ctx["npc_list"]:
        if it["name"] == name:
            return it
    raise AssertionError(f"npc_list 缺少条目：{name}")


# ---------------------------------------------------------------------------
# B8-2 传话/问候候选轮
# ---------------------------------------------------------------------------

def test_prepare_supplies_publisher_scene_and_candidate_scenes(game_data):
    ctx = _prepare(game_data, "传话", "Pig", STATES)

    assert ctx["publisher_scene"] == (
        "你所在场景：基地车库（基地场景-基地车库）。\n"
        "同场景npc（均在你视线范围内）：Boy，King，冷兵器商人。"
    )
    assert _entry(ctx, "Pig")["scene"] == "基地车库（基地场景-基地车库）"
    assert _entry(ctx, "冷兵器商人")["scene"] == "基地车库（基地场景-基地车库）"
    # 多驻地用「、」连接
    assert _entry(ctx, "武器大师")["scene"] == (
        "健身房（基地场景-健身房）、第一防线（地图-第一防线防区）"
    )
    # 无驻地候选不携带 scene 字段
    assert "scene" not in _entry(ctx, "lochan")


def test_prepare_omits_publisher_scene_for_no_location_npc(game_data):
    ctx = _prepare(game_data, "问候", "lochan", STATES)
    assert "publisher_scene" not in ctx
    assert "scene" not in _entry(ctx, "lochan")


def test_prepare_degraded_without_map_registry(game_data, monkeypatch):
    monkeypatch.setattr(game_data, "map_registry", None)
    ctx = _prepare(game_data, "传话", "Pig", STATES)
    assert "publisher_scene" not in ctx
    assert all("scene" not in it for it in ctx["npc_list"])


def test_greeting_and_relay_rules_carry_exclusion_sentence():
    assert EXCLUDE_SENTENCE in _TASK_RULES["问候"]
    assert EXCLUDE_SENTENCE in _TASK_RULES["传话"]
    assert "同场景npc" not in _TASK_RULES["通关"]


# ---------------------------------------------------------------------------
# B8-3 草案摘要：完成NPC 场景行
# ---------------------------------------------------------------------------

def _draft(**over) -> dict:
    base = {
        "draft_id": "d-t1",
        "npc_name": "Pig",
        "task_type": "传话",
        "title": "给武器大师捎个话",
    }
    base.update(over)
    return base


def test_finish_npc_gets_scene_line_and_instruction(game_data):
    text = _detailed_draft_summary(_draft(finish_npc="武器大师"), game_data)
    assert (
        "完成NPC: 武器大师；所在场景："
        "健身房（基地场景-健身房）、第一防线（地图-第一防线防区）"
    ) in text
    assert FINISH_NPC_HINT in text


def test_finish_npc_same_as_publisher_or_missing_gets_plain_line(game_data):
    same = _detailed_draft_summary(_draft(finish_npc="Pig"), game_data)
    assert "完成NPC: Pig" in same
    assert "所在场景" not in same
    assert FINISH_NPC_HINT not in same

    missing = _detailed_draft_summary(_draft(), game_data)
    assert "完成NPC: Pig" in missing
    assert "所在场景" not in missing
    assert FINISH_NPC_HINT not in missing


def test_finish_npc_without_location_gets_no_scene_or_instruction(game_data):
    text = _detailed_draft_summary(_draft(finish_npc="lochan"), game_data)
    assert "完成NPC: lochan" in text
    assert "所在场景" not in text
    assert FINISH_NPC_HINT not in text


# ---------------------------------------------------------------------------
# B8-3 草案摘要：逐关卡怪物块
# ---------------------------------------------------------------------------

def test_stage_lines_append_monster_block(game_data):
    draft = _draft(task_type="通关", finish_requirements=[
        {"stage_name": "地铁站", "difficulty": "简单"},
    ])
    lines = _build_draft_entity_context_lines(draft, game_data)
    assert len(lines) == 2
    assert lines[0].startswith("[关卡] 关卡名称：地铁站；")
    assert lines[0].endswith("；草案要求难度：简单")
    assert lines[1] == TIETIE_MONSTER_BLOCK


def test_each_stage_gets_own_block_and_duplicates_collapse(game_data):
    draft = _draft(task_type="通关", finish_requirements=[
        {"stage_name": "地铁站", "difficulty": "简单"},
        {"stage_name": "堕落城深处", "difficulty": "困难"},
        {"stage_name": "地铁站", "difficulty": "困难"},
    ])
    lines = _build_draft_entity_context_lines(draft, game_data)
    assert sum(1 for ln in lines if ln.startswith("[关卡] ")) == 2
    blocks = [ln for ln in lines if ln.startswith("该关卡有以下怪物")]
    assert len(blocks) == 2
    assert any("跳蚤、铁拳" in ln for ln in blocks)


def test_challenge_draft_gets_no_monster_block(game_data):
    draft = _draft(task_type="切磋", finish_requirements=[
        {"stage_name": "地铁站", "difficulty": "简单"},
    ])
    lines = _build_draft_entity_context_lines(draft, game_data)
    assert any(ln.startswith("[关卡] 关卡名称：地铁站") for ln in lines)
    assert not any("该关卡有以下怪物" in ln for ln in lines)


def test_stage_already_in_rag_anchors_single_monster_line(game_data):
    draft = _draft(task_type="通关", finish_requirements=[
        {"stage_name": "地铁站", "difficulty": "简单"},
    ])
    rag = "【玩家可能提到的关卡】\n关卡名称：地铁站；所在区域：基地门口"
    lines = _build_draft_entity_context_lines(draft, game_data, rag_context_text=rag)
    assert lines == [f"[关卡] 地铁站：{TIETIE_MONSTER_BLOCK}"]
