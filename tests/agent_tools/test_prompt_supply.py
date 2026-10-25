"""B8-2 / B8-3 子 Agent 提示词信息供给测试：

- 传话/问候候选轮：发布 NPC 视角场景块（publisher_scene）＋候选条目 scene 字段＋规则句；
- 草案摘要：完成NPC≠发布NPC 时补其属性（性别/阵营/身份或称呼）与「所在场景」行及指示句；
- 选关类草案逐关卡追加怪物块（切磋不附）；RAG 已出现的关卡改锚一行；数量最多句附相对强度档位；
- 关卡候选行逐字带关卡描述（第三批），空描述整段省略不留占位。
"""

from __future__ import annotations

import json

from services.agent_tools.context_builder import (
    _TASK_RULES,
    _group_stage_rows_by_area,
    prepare_task_context,
)
from services.agent_tools.draft_formatting import (
    _build_draft_entity_context_lines,
    _detailed_draft_summary,
)
from services.agent_tools.schemas import normalize_reward_types_for_prepare_context
from services.game_entity_prompts import format_stage_detail_line, get_stage_info_for_name
from services.npc.manager import NPCState

# 与 game_data 侧测试同源的逐字块（地铁站，含无 displayname 兵种 293 的跳过）
TIETIE_MONSTER_BLOCK = (
    "该关卡有以下怪物：黄衣僵尸，金发女僵尸，红衣僵尸，小儿麻痹僵尸，胖僵尸，"
    "眼镜僵尸，光头军人僵尸，狂暴僵尸，防暴僵尸，特战僵尸，狂躁僵尸，"
    "裸体兽化僵尸，兽化僵尸，巨型僵尸。"
    "其中数量最多的是兽化僵尸，大致数量约16个，强度较强。强度最高的是巨型僵尸。"
    "根据你的身份，你可能知道也可能不知道这个关卡的详情；"
    "如果你所扮演的NPC看起来不擅长情报或战斗，或者没有这方面的资料，"
    "那你就表现得不知道具体的关卡怪物分布，只知道模糊的大概，"
    "例如那里是有僵尸还是有盗贼等；"
    "但绝对不要编造信息，要么表现的不清楚，要么根据实际信息去说话，"
    "不能编造未提供的信息。"
    "上面提到的数量只是让你心里有数，不要向玩家复述具体个数；"
    "需要说数量时用模糊的概括，例如“数十个”“很多”“不少”“大量”“较多”。"
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


def test_challenge_rule_separates_work_requests_from_sparring():
    """切磋的适用判断：想切磋/动手才优先；找活干/找委托时不许选。"""
    rule = _TASK_RULES["切磋"]
    assert "优先选本类型" in rule
    assert "玩家只是在找活干、找委托、问有没有活儿时，不要选本类型" in rule
    assert "他要的是差事，不是切磋" in rule


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


def test_finish_npc_gets_attrs_scene_and_instruction(game_data):
    text = _detailed_draft_summary(
        _draft(finish_npc="武器大师"), game_data, npc_states=STATES,
    )
    assert (
        "完成NPC: 武器大师（性别：男）（阵营：闲杂人等）（身份或称呼：武器大师）；所在场景："
        "健身房（基地场景-健身房）、第一防线（地图-第一防线防区）"
    ) in text
    assert FINISH_NPC_HINT in text


def test_finish_npc_same_as_publisher_or_missing_gets_plain_line(game_data):
    same = _detailed_draft_summary(_draft(finish_npc="Pig"), game_data, npc_states=STATES)
    assert "完成NPC: Pig" in same.splitlines()
    assert "所在场景" not in same
    assert FINISH_NPC_HINT not in same

    missing = _detailed_draft_summary(_draft(), game_data, npc_states=STATES)
    assert "完成NPC: Pig" in missing.splitlines()
    assert "所在场景" not in missing
    assert FINISH_NPC_HINT not in missing


def test_finish_npc_without_location_gets_attrs_only(game_data):
    text = _detailed_draft_summary(_draft(finish_npc="lochan"), game_data, npc_states=STATES)
    assert "完成NPC: lochan（性别：女）（阵营：闲杂人等）（身份或称呼：谜之兔女郎）" in text.splitlines()
    assert "所在场景" not in text
    assert FINISH_NPC_HINT not in text


def test_finish_npc_without_record_or_states_gets_plain_name(game_data):
    no_record = _detailed_draft_summary(
        _draft(finish_npc="无名路人"), game_data, npc_states=STATES,
    )
    assert "完成NPC: 无名路人" in no_record.splitlines()

    # 未传 npc_states（旧调用）与传空字典行为一致：属性片段整体省略
    no_states = _detailed_draft_summary(_draft(finish_npc="武器大师"), game_data)
    assert (
        "完成NPC: 武器大师；所在场景："
        "健身房（基地场景-健身房）、第一防线（地图-第一防线防区）"
    ) in no_states


def test_third_party_get_npc_gets_attrs_scene_after_finish_block(game_data):
    text = _detailed_draft_summary(
        _draft(get_npc="武器大师", finish_npc="lochan"), game_data, npc_states=STATES,
    )
    lines = text.splitlines()
    finish_idx = lines.index(
        "完成NPC: lochan（性别：女）（阵营：闲杂人等）（身份或称呼：谜之兔女郎）"
    )
    get_line = (
        "接取NPC: 武器大师（性别：男）（阵营：闲杂人等）（身份或称呼：武器大师）；所在场景："
        "健身房（基地场景-健身房）、第一防线（地图-第一防线防区）"
    )
    assert lines.index(get_line) > finish_idx
    assert text.count("接取NPC:") == 1


def test_get_npc_already_in_rag_context_is_not_repeated(game_data):
    text = _detailed_draft_summary(
        _draft(get_npc="武器大师", finish_npc="lochan"),
        game_data,
        npc_states=STATES,
        rag_context_text="【你提到的角色】「武器大师」（身份或称呼：武器大师）；所在场景：健身房。",
    )
    assert "接取NPC: 武器大师" in text.splitlines()
    assert "接取NPC: 武器大师（性别" not in text


def test_get_npc_same_as_finish_or_publisher_has_no_extra_block(game_data):
    same_finish = _detailed_draft_summary(
        _draft(get_npc="武器大师", finish_npc="武器大师"), game_data, npc_states=STATES,
    )
    assert "接取NPC: 武器大师" in same_finish.splitlines()
    assert same_finish.count("接取NPC:") == 1

    same_publisher = _detailed_draft_summary(
        _draft(get_npc="Pig", finish_npc="lochan"), game_data, npc_states=STATES,
    )
    assert "接取NPC: Pig" in same_publisher.splitlines()
    assert "接取NPC: Pig（性别" not in same_publisher


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


# ---------------------------------------------------------------------------
# 第三批：关卡描述进候选行／进草案 [关卡] 行
# ---------------------------------------------------------------------------

def test_prepare_stage_candidates_carry_description_verbatim(game_data):
    """通关／通关并收集的候选行：注册表有描述就逐字带上，没有就不出这个键。"""
    for task_type, key in (("通关", "stage_list"), ("通关并收集", "stage_loot_list")):
        ctx = _prepare(game_data, task_type, "Pig", STATES)
        rows = [(b["area"], s) for b in ctx[key] for s in b["stages"]]
        assert rows, f"{key} 应有候选行"
        for area, row in rows:
            si = game_data.stages._stage_infos[(area, row["name"])]
            expected = str(si.description or "").strip()
            assert ("description" in row) == bool(expected), row["name"]
            if expected:
                assert row["description"] == expected


def test_group_stage_rows_strips_and_omits_empty_description():
    """分组装配：描述去空白后才有键；纯空白与缺字段一律整段省略，不落占位。"""
    rows = [
        {"area": "甲区", "name": "带描述", "description": "  一段描述  ",
         "difficulties": [], "is_dungeon": False, "unlock_id": 1},
        {"area": "甲区", "name": "纯空白", "description": "   ",
         "difficulties": [], "is_dungeon": False, "unlock_id": 1},
        {"area": "甲区", "name": "无描述",
         "difficulties": [], "is_dungeon": False, "unlock_id": 1},
    ]
    stages = _group_stage_rows_by_area(rows, stage=1)[0]["stages"]
    assert stages[0]["description"] == "一段描述"
    assert "description" not in stages[1]
    assert "description" not in stages[2]


def test_draft_stage_line_keeps_description_when_present(game_data):
    lines = _build_draft_entity_context_lines(
        _draft(task_type="通关", finish_requirements=[
            {"stage_name": "地铁站", "difficulty": "简单"},
        ]),
        game_data,
    )
    assert (
        "[关卡] 关卡名称：地铁站；关卡描述：僵尸喜爱黑暗的环境，"
        "于是地铁站成为了整个废城最为凶险的地区，"
        "没人知道这地狱的洞窟里隐藏了多少僵尸 这里可以获得一些中级材料；"
        "所在区域：基地门口"
    ) in lines[0]


def test_draft_stage_line_omits_empty_description_without_placeholder(game_data):
    """无描述关卡的 [关卡] 行：整段省略，不再写「关卡描述：（无）」。"""
    stage_name = "外交-第一防线防区"
    assert not get_stage_info_for_name(game_data.stages, stage_name).description
    assert "关卡描述" not in format_stage_detail_line(
        get_stage_info_for_name(game_data.stages, stage_name),
    )
    lines = _build_draft_entity_context_lines(
        _draft(task_type="通关", finish_requirements=[
            {"stage_name": stage_name, "difficulty": "简单"},
        ]),
        game_data,
    )
    assert lines[0].startswith(f"[关卡] 关卡名称：{stage_name}；所在区域：")
    assert "（无）" not in lines[0]
