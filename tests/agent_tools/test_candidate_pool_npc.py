"""NPC 候选池收窄 + 过渡语「范围说明」闸的测试。

口径（用户 2026-10-18）：
- NPC 池上限 10，命中的全进、超额留 5 + 其余随机、不足随机补足；
- 标签：同阵营/其他阵营、男/女、商人、可切磋、同区域/跨区域、阵营名；
- 只有整池都由关键词命中填满时，才把关键词＋说明交给过渡语（`pool_note`）。
"""

from __future__ import annotations

import json

from services.agent_tools import pool_tags as pt
from services.agent_tools.context_builder import NPC_POOL_CAP, build_pool_note, prepare_task_context
from services.npc.manager import NPCManager, get_npc_state_path
from services.subagents.prompts import build_task_system

RT = {"regular": ["金币"], "optional": ["材料"]}


def _states(game_data) -> dict:
    raw = json.loads(get_npc_state_path().read_text("utf-8"))
    global_rows, _saves = NPCManager._split_scopes(raw)
    return {k: NPCManager._state_from_item(v) for k, v in global_rows.items()}


def _ctx(game_data, *, task_type="传话", keywords=None, npc="Boy", stage=4) -> dict:
    return json.loads(prepare_task_context(
        task_type=task_type,
        reward_types=RT,
        npc_name=npc,
        player_progress=stage,
        npc_states=_states(game_data),
        requirement_keywords=keywords,
        game_data=game_data,
    ))


def test_npc_pool_cap(game_data):
    rows = _ctx(game_data).get("npc_list") or []
    assert len(rows) == NPC_POOL_CAP


def test_npc_rows_have_matched_by(game_data):
    rows = _ctx(game_data, keywords=["男"]).get("npc_list") or []
    assert rows
    for r in rows:
        assert r.get("matched_by")


def test_same_faction_tag(game_data):
    ctx = _ctx(game_data, keywords=["同阵营"])
    rows = ctx["npc_list"]
    states = _states(game_data)
    pub_faction = states["Boy"].faction
    for r in rows:
        if r["matched_by"] == "同阵营":
            assert r["faction"] == pub_faction


def test_gender_tag(game_data):
    rows = _ctx(game_data, keywords=["女"])["npc_list"]
    states = _states(game_data)
    hit = [r for r in rows if r["matched_by"] == "女"]
    assert hit
    for r in hit:
        assert states[r["name"]].sex == "女"


def test_merchant_tag(game_data):
    rows = _ctx(game_data, keywords=["商人"])["npc_list"]
    assert rows
    for r in rows:
        assert r["matched_by"] == "商人"
        assert game_data.shops.has_shop(r["name"])


def test_exact_name_ranks_first(game_data):
    rows = _ctx(game_data, keywords=["King"])["npc_list"]
    assert rows and rows[0]["name"] == "King"


def test_pool_note_present_when_pool_fully_covered(game_data):
    ctx = _ctx(game_data, task_type="通关", keywords=["势均力敌"])
    note = ctx.get("pool_note") or ""
    assert note.startswith("候选范围：")
    assert "势均力敌" in note


def test_pool_note_absent_when_pool_has_random(game_data):
    ctx = _ctx(game_data, task_type="通关", keywords=None)
    assert "pool_note" not in ctx


def test_pool_note_absent_when_partially_covered(game_data):
    """命中不足 10 条、靠随机补足时不给过渡语（避免说出池里没有的方向）。"""
    ctx = _ctx(game_data, task_type="资源收集", keywords=["药剂"], stage=7)
    assert "pool_note" not in ctx


def test_build_pool_note_empty_without_pools():
    assert build_pool_note({}) == ""


def test_warlord_prompt_has_no_nearby_tag():
    system = build_task_system(mode="draft", npc_faction="军阀")
    assert "周边地区" not in system
    assert "敌对区域" in system


def test_blackiron_prompt_has_nearby_tag():
    system = build_task_system(mode="draft", npc_faction="黑铁会")
    assert "周边地区" in system


def test_tag_catalog_has_no_package_tag():
    for tags in pt.ITEM_TAGS_PER_TASK.values():
        assert "礼包" not in tags
