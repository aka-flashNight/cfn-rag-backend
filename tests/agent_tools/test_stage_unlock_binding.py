"""绑定存档的关卡候选与前置：真实解锁口径（V3.1 第 4 项修复）。

覆盖：
- 主线链值 20/21 的地铁站边界；
- AltUnlockCondition 单条/多条 OR、零门槛、缺失/坏值、同名后定义覆盖；
- 势均力敌的阶段早期回退阈值 0/2/3；
- 前置任务取已完成链锚点，不再抄阈值/截断 77；
- 绑定时候选只含真实已解锁关卡，未绑定保持旧语义。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.agent_tools.context_builder import (
    _get_all_stages_for_progress,
    prepare_task_context,
)
from services.agent_tools.fallback import FallbackDraftError, _pick_stage_requirements
from services.agent_tools.task_tools import _stage_reqs_unlock_ids
from services.agent_tools.validator import (
    DraftValidationContext,
    _validate_v3_dungeon_recommended_level,
    _validate_v4_stage_unlock_condition,
    _validate_v5_replica_stage_difficulty,
)
from services.game_data.models import AltUnlockCondition, StageInfo, Task
from services.game_data.stage_registry import StageRegistry
from services.stage_access import (
    BindingFacts,
    matched_stage_for_tags,
    stage_is_unlocked,
    stage_unlock_task_ids,
    stage_unlock_task_ids_unbound,
)

RT = {"regular": ["金币", "经验值"], "optional": ["材料"]}


def _facts(
    *,
    mainline: int | None = None,
    completed: list[int] | None = None,
    chains: dict[str, int] | None = None,
    level: int | None = None,
    slot: str = "slot_a",
) -> BindingFacts:
    progress: dict = {
        "tasks_finished": list(completed or []),
        "task_chains_progress": dict(chains or {}),
    }
    if mainline is not None:
        progress["task_chains_progress"]["主线"] = mainline
    facts = BindingFacts.from_save_info({
        "slot_key": slot,
        "progress": progress,
        "player": {"level": level} if level is not None else {},
    })
    assert facts is not None
    return facts


def _rows(context: dict, key: str = "stage_list") -> list[tuple[str, dict]]:
    return [(b["area"], s) for b in context.get(key, []) for s in b["stages"]]


def _stage_names(context: dict, key: str = "stage_list") -> set[str]:
    return {row["name"] for _area, row in _rows(context, key)}


# ---------------------------------------------------------------------------
# 主线链值 20 / 21：地铁站
# ---------------------------------------------------------------------------

def test_metro_station_enters_pool_only_after_chain_21(game_data):
    ctx20 = prepare_task_context(
        task_type="通关", reward_types=RT, npc_name="测试NPC", player_progress=1,
        npc_states={}, requirement_keywords=["地铁站"], game_data=game_data,
        save_info={"slot_key": "slot_a", "progress": {
            "task_chains_progress": {"主线": 20},
            "tasks_finished": list(range(20)),
        }, "player": {"level": 12}},
    )
    ctx21 = prepare_task_context(
        task_type="通关", reward_types=RT, npc_name="测试NPC", player_progress=1,
        npc_states={}, requirement_keywords=["地铁站"], game_data=game_data,
        save_info={"slot_key": "slot_a", "progress": {
            "task_chains_progress": {"主线": 21},
            "tasks_finished": list(range(21)),
        }, "player": {"level": 12}},
    )
    names20 = _stage_names(json.loads(ctx20))
    names21 = _stage_names(json.loads(ctx21))
    assert "地铁站" not in names20
    assert "地铁站" in names21


def test_metro_prerequisite_is_completed_task_20(game_data):
    sit = game_data.stages._stage_infos[("基地门口", "地铁站")]
    req = {"stage_name": "地铁站", "difficulty": "简单"}

    facts20 = _facts(mainline=20, completed=list(range(20)))
    facts21 = _facts(mainline=21, completed=list(range(21)))

    assert _stage_reqs_unlock_ids(
        finish_requirements=[req], game_data=game_data, facts=facts20,
    ) == []
    assert _stage_reqs_unlock_ids(
        finish_requirements=[req], game_data=game_data, facts=facts21,
    ) == [20]
    assert stage_unlock_task_ids(sit, facts21, game_data.tasks.list_all_tasks()) == [20]


def test_chain_value_is_not_written_as_task_id_unbound(game_data):
    req = {"stage_name": "地铁站", "difficulty": "简单"}
    assert _stage_reqs_unlock_ids(
        finish_requirements=[req], game_data=game_data,
    ) == [20]


# ---------------------------------------------------------------------------
# AltUnlockCondition / 零门槛 / 缺失 / 坏值
# ---------------------------------------------------------------------------

def _si(
    name: str = "测试关",
    unlock: int | None = None,
    alts: list[tuple[str, int]] | None = None,
) -> StageInfo:
    return StageInfo(
        area="基地门口",
        name=name,
        unlock_condition=unlock,
        alt_unlock_conditions=[
            AltUnlockCondition(chain=c, min=m) for c, m in (alts or [])
        ],
    )


def test_alt_unlock_is_or_with_mainline():
    si = _si(unlock=999, alts=[("大学", 7)])
    assert stage_is_unlocked(si, _facts(mainline=1000)) is True
    assert stage_is_unlocked(si, _facts(mainline=5, chains={"大学": 7})) is True
    assert stage_is_unlocked(si, _facts(mainline=5, chains={"大学": 6})) is False
    multi = _si(unlock=999, alts=[("大学", 7), ("公会", 3)])
    assert stage_is_unlocked(multi, _facts(mainline=5, chains={"公会": 3})) is True


def test_zero_threshold_requires_known_zero_progress():
    always = _si(unlock=0)
    assert stage_is_unlocked(always, _facts(mainline=0)) is True
    assert stage_is_unlocked(always, _facts()) is False
    missing = _si(unlock=None)
    assert stage_is_unlocked(missing, _facts(mainline=100)) is False


def test_alt_prerequisite_uses_completed_chain_anchor():
    si = _si(unlock=999, alts=[("大学", 7)])
    tasks = [
        Task(id=120, title="旧", chain="大学#5"),
        Task(id=121, title="中", chain="大学#7"),
        Task(id=122, title="新", chain="大学#8"),
    ]
    facts = _facts(mainline=5, chains={"大学": 8}, completed=[121])
    assert stage_unlock_task_ids(si, facts, tasks) == [121]
    # 只完成活动任务不算完成，也不拿链值冒充
    facts_active_only = _facts(mainline=5, chains={"大学": 8})
    assert stage_unlock_task_ids(si, facts_active_only, tasks) == []


def test_anchor_id_above_77_is_not_clamped():
    si = _si(unlock=21)
    tasks = [
        Task(id=120, title="锚", chain="主线#21"),
        Task(id=121, title="后", chain="主线#22"),
    ]
    facts = _facts(mainline=21, completed=[120])
    assert stage_unlock_task_ids(si, facts, tasks) == [120]


def test_unbound_mapping_does_not_claim_completion():
    si = _si(unlock=21)
    tasks = [Task(id=20, title="锚", chain="主线#21")]
    assert stage_unlock_task_ids_unbound(si, tasks) == [20]


def test_parse_alt_zero_missing_and_bad(tmp_path):
    area_dir = Path(tmp_path) / "stages" / "基地门口"
    area_dir.mkdir(parents=True)
    (area_dir / "__list__.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<Stages>
  <StageInfo><Name>零门槛</Name><UnlockCondition>0</UnlockCondition></StageInfo>
  <StageInfo><Name>坏值</Name><UnlockCondition>abc</UnlockCondition></StageInfo>
  <StageInfo><Name>缺失主路径</Name>
    <AltUnlockCondition chain="大学" min="7"/>
    <AltUnlockCondition chain="公会" min="3"/>
  </StageInfo>
  <StageInfo><Name>重复关</Name><UnlockCondition>21</UnlockCondition></StageInfo>
  <StageInfo><Name>重复关</Name><UnlockCondition>30</UnlockCondition></StageInfo>
</Stages>
""",
        encoding="utf-8",
    )
    (Path(tmp_path) / "stages" / "list.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?><root><stages>基地门口</stages></root>',
        encoding="utf-8",
    )
    reg = StageRegistry(data_root=tmp_path)
    reg.load()

    zero = reg._stage_infos[("基地门口", "零门槛")]
    bad = reg._stage_infos[("基地门口", "坏值")]
    alt_only = reg._stage_infos[("基地门口", "缺失主路径")]
    assert zero.unlock_condition == 0
    assert bad.unlock_condition is None
    assert alt_only.unlock_condition is None
    assert [(a.chain, a.min) for a in alt_only.alt_unlock_conditions] == [
        ("大学", 7), ("公会", 3),
    ]
    # 同名定义按加载顺序后者覆盖
    assert reg._stage_infos[("基地门口", "重复关")].unlock_condition == 30


# ---------------------------------------------------------------------------
# 势均力敌：阶段早期回退
# ---------------------------------------------------------------------------

def _tag_rows(count: int, *, area: str, dungeon: bool = False) -> list[dict]:
    return [
        {"area": area, "name": f"{area}{i}", "is_dungeon": dungeon}
        for i in range(count)
    ]


@pytest.mark.parametrize(("count", "expected"), [(0, 3), (2, 3), (3, 4)])
def test_stage_early_fallback_boundary(count, expected):
    rows = _tag_rows(count, area="黑铁会总部")
    assert matched_stage_for_tags(rows, stage=4) == expected


def test_stage_early_fallback_stops_at_stage_one():
    assert matched_stage_for_tags([], stage=1) == 1


def test_previous_stage_targets_become_even_match(game_data):
    # 主线 29 处阶段 3；基地房顶此时 0 个可攻打关卡 → 势均力敌退回阶段 2（下限 22）。
    facts = _facts(mainline=29, completed=list(range(29)))
    raw = prepare_task_context(
        task_type="通关", reward_types=RT, npc_name="测试NPC", player_progress=3,
        npc_states={}, requirement_keywords=["堕落城区"], game_data=game_data,
        save_info={
            "slot_key": "slot_a",
            "progress": facts.progress,
            "player": {"level": 17},
        },
    )
    rows = [row for _area, row in _rows(json.loads(raw)) if row["name"] == "堕落城区"]
    assert rows, "堕落城区应作为关键词命中进池"
    assert not rows[0].get("below_progress")


def test_current_stage_full_pool_keeps_older_stage_below(game_data):
    raw = prepare_task_context(
        task_type="通关", reward_types=RT, npc_name="测试NPC", player_progress=2,
        npc_states={}, requirement_keywords=["新手练习场"], game_data=game_data,
        save_info={
            "slot_key": "slot_a",
            "progress": {"task_chains_progress": {"主线": 780}, "tasks_finished": []},
            "player": {"level": 60},
        },
    )
    rows = [row for _area, row in _rows(json.loads(raw)) if row["name"] == "新手练习场"]
    assert rows and rows[0].get("below_progress")


# ---------------------------------------------------------------------------
# 绑定候选：缺字段为空、锁定目标不补回
# ---------------------------------------------------------------------------

def test_missing_binding_fields_yield_empty_pool(game_data):
    raw = prepare_task_context(
        task_type="通关", reward_types=RT, npc_name="测试NPC", player_progress=1,
        npc_states={}, game_data=game_data,
        save_info={"slot_key": "slot_a", "progress": {}, "player": {}},
    )
    assert _rows(json.loads(raw)) == []


def test_locked_stage_never_returned_even_by_keyword(game_data):
    raw = prepare_task_context(
        task_type="通关", reward_types=RT, npc_name="测试NPC", player_progress=1,
        npc_states={}, requirement_keywords=["突围"], game_data=game_data,
        save_info={
            "slot_key": "slot_a",
            "progress": {"task_chains_progress": {"主线": 21}},
            "player": {"level": 12},
        },
    )
    assert "突围" not in _stage_names(json.loads(raw))


# ---------------------------------------------------------------------------
# 校验：绑定后不做生成后资格拒绝；未绑定保持旧语义
# ---------------------------------------------------------------------------

def test_v4_skips_bound_but_keeps_unbound(game_data):
    draft = {"finish_requirements": [{"stage_name": "突围", "difficulty": "简单"}]}
    unbound = _validate_v4_stage_unlock_condition(
        draft=draft, stage_registry=game_data.stages, main_task_max_id=21,
    )
    bound = _validate_v4_stage_unlock_condition(
        draft=draft, stage_registry=game_data.stages, main_task_max_id=21,
        binding=_facts(mainline=5),
    )
    assert unbound and unbound[0].rule == "V4"
    assert bound == []


def _mercenary_with_level(game_data, max_level: int = 1):
    for mt in game_data.mercenary_tasks.list_all():
        if mt.stage_name and mt.recommended_min_level is not None:
            if mt.recommended_min_level > max_level:
                return mt
    return None


def test_v3r_skips_bound_but_keeps_unbound(game_data):
    mt = _mercenary_with_level(game_data)
    if mt is None:
        pytest.skip("没有推荐等级高于 1 的副本任务")
    draft = {"finish_requirements": [{"stage_name": mt.stage_name, "difficulty": "简单"}]}
    context_unbound = DraftValidationContext(main_task_max_id=21, max_level=1)
    context_bound = DraftValidationContext(
        main_task_max_id=21, max_level=1, binding=_facts(mainline=5),
    )
    assert _validate_v3_dungeon_recommended_level(
        draft=draft, context=context_unbound, game_data=game_data,
    )
    assert _validate_v3_dungeon_recommended_level(
        draft=draft, context=context_bound, game_data=game_data,
    ) == []


def test_v5_bound_allows_configured_difficulty_without_level_gate(game_data):
    mt = None
    for cand in game_data.mercenary_tasks.list_all():
        if (
            cand.stage_name
            and cand.challenge_difficulty
            and cand.challenge_difficulty != "简单"
            and (cand.challenge_recommended_min_level or 0) > 1
        ):
            mt = cand
            break
    if mt is None:
        pytest.skip("没有带挑战难度的副本任务")
    draft = {"finish_requirements": [
        {"stage_name": mt.stage_name, "difficulty": mt.challenge_difficulty},
    ]}
    context_unbound = DraftValidationContext(main_task_max_id=21, max_level=1)
    context_bound = DraftValidationContext(
        main_task_max_id=21, max_level=1, binding=_facts(mainline=5),
    )
    assert _validate_v5_replica_stage_difficulty(
        draft=draft, context=context_unbound, game_data=game_data,
    )
    assert _validate_v5_replica_stage_difficulty(
        draft=draft, context=context_bound, game_data=game_data,
    ) == []


# ---------------------------------------------------------------------------
# 兜底草案：绑定时不再用旧阶段上限二次筛
# ---------------------------------------------------------------------------

class _FakeStages:
    def __init__(self) -> None:
        self._stage_infos = {
            ("测试区", "测试关卡"): StageInfo(
                area="测试区", name="测试关卡", unlock_condition=999,
            ),
        }


class _FakeGameData:
    stages = _FakeStages()


def test_fallback_picks_filtered_candidate_when_bound():
    candidates = {"stage_list": [{"area": "测试区", "stages": [
        {"name": "测试关卡", "unlock_id": 999},
    ]}]}
    context = DraftValidationContext(main_task_max_id=21, max_level=12)
    with pytest.raises(FallbackDraftError):
        _pick_stage_requirements(
            task_type="通关", candidates=candidates, context=context,
            game_data=_FakeGameData(),
        )
    assert _pick_stage_requirements(
        task_type="通关", candidates=candidates, context=context,
        game_data=_FakeGameData(), facts=_facts(mainline=1000),
    ) == [{"stage_name": "测试关卡", "difficulty": "简单"}]
