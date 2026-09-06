"""自动修复层测试（对应 docs/v3-developer/05 §3/§8.2）。

原则：后端只动数值，不替模型做选择（V1/V3/V4/V5/V6/V8 不可自动修）。
"""

from __future__ import annotations

import pytest

from services.agent_tools.repair import auto_repair
from services.agent_tools.validator import validate_task_draft


def test_repair_v7_scales_reward(game_data, vctx):
    """V7 偏差 ≤10%：等比调整一个奖励项到区间内最近值，note 正确。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "rewards": [{"item_name": "金币", "count": 28000}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    assert not report.ok

    new_draft, notes, fresh = auto_repair(draft, report, context=vctx, game_data=game_data)
    assert fresh.ok
    assert new_draft["rewards"][0]["count"] == 27000
    assert any("28000" in n and "27000" in n for n in notes)


def test_repair_v2_and_v11_together(game_data, vctx):
    """V2 数量为 0 + V11 重名：一次修复全部并过检（05 §8.2）。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "finish_submit_items": [{"item_name": "金币", "count": 0}],
        "rewards": [{"item_name": "金币", "count": 25000}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    rules = {i.rule for i in report.issues}
    assert rules == {"V2", "V11"}

    new_draft, notes, fresh = auto_repair(draft, report, context=vctx, game_data=game_data)
    assert fresh.ok
    assert any("V2" in n or "数量" in n for n in notes)
    assert any("重名" in n or "移除" in n for n in notes)


def test_repair_v11_keeps_rewards_side(game_data, vctx):
    """V11 去重保留 rewards 侧（报酬误填进提交的场景）。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "finish_submit_items": [{"item_name": "金币", "count": 1}],
        "rewards": [{"item_name": "金币", "count": 25000}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    new_draft, notes, fresh = auto_repair(draft, report, context=vctx, game_data=game_data)
    assert fresh.ok
    assert new_draft["rewards"] == [{"item_name": "金币", "count": 25000}]
    assert new_draft["finish_submit_items"] == []


def test_repair_v10_demotes_equipment(game_data, vctx):
    """V10 装备超等级：降级到阶段上限内的同类型装备（若有）。"""
    items = game_data.items
    # 找一件类型为「武器」、等级超上限、且类型在历史奖励中出现（V8 合规）的超等级装备
    valid_reward_names = set(game_data.tasks.list_reward_types())
    over = None
    for it in items.items:
        if it.type == "武器" and (it.level or 0) > vctx.max_level and it.name in valid_reward_names:
            over = it
            break
    if over is None:
        pytest.skip("数据中无符合条件的超等级武器奖励")

    # 补金币把总值抬进 V7 区间
    base = [{"item_name": over.name, "count": 1}, {"item_name": "金币", "count": 9000}]
    total = over.price + 9000
    if total > 27000:
        base = [{"item_name": over.name, "count": 1}, {"item_name": "金币", "count": max(0, 27000 - over.price)}]
        total = over.price + base[1]["count"]
    draft = {"task_type": "装备缴纳", "title": "测试", "rewards": base}
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    assert any(i.rule == "V10" for i in report.issues)

    new_draft, notes, fresh = auto_repair(draft, report, context=vctx, game_data=game_data)
    names = [r["item_name"] for r in new_draft["rewards"]]
    assert over.name not in names
    assert any("降级" in n or "移除" in n for n in notes)


def test_no_repair_for_existence_rules(game_data, vctx):
    """V1（存在性）不可自动修：报告原样返回。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "rewards": [{"item_name": "不存在的灵药", "count": 1}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    new_draft, notes, fresh = auto_repair(draft, report, context=vctx, game_data=game_data)
    assert notes == []
    assert not fresh.ok
    assert any(i.rule == "V1" for i in fresh.issues)
