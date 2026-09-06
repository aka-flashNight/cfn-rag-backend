"""聚合校验测试（对应 docs/v3-developer/05 §2/§8.1/§8.3）。

- 一次跑完所有规则全量收集（修 D1：不再串行短路一次只报一个错）；
- 反馈增强：V1 模糊匹配候选、V2 直给建议值、V7 直给差额数字。
"""

from __future__ import annotations

from services.agent_tools.validator import validate_task_draft


def test_aggregate_reports_all_issues_at_once(game_data, vctx):
    """同时违反 V1+V7+V11 的草案：一次返回 3 条 issue（05 §8.1）。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试委托",
        "finish_submit_items": [{"item_name": "不存在的灵药", "count": 1}],
        "rewards": [{"item_name": "不存在的灵药", "count": 1}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)

    assert not report.ok
    rules = sorted(i.rule for i in report.issues)
    assert rules == ["V1", "V11", "V7"]
    # 模型可见 JSON：全量 issue + 每条带 fix_hint
    payload = report.to_model_json()
    assert payload["issue_count"] == 3
    for issue in payload["issues"]:
        assert issue["fix_hint"]
        assert issue["message"]


def test_v1_fuzzy_candidates_contain_real_item(game_data, vctx):
    """V1 增强：candidates 含模糊匹配到的真实物品名（取最长物品名的截断作查询，子串必命中）。"""
    longest = max((it.name for it in game_data.items.items), key=len)
    query = longest[:-2] if len(longest) > 2 else longest[:-1]
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "rewards": [{"item_name": query, "count": 1}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    v1 = next(i for i in report.issues if i.rule == "V1")
    assert longest in v1.candidates
    assert len(v1.candidates) <= 5


def test_v2_gives_suggested_value(game_data, vctx):
    """V2 增强：直给「当前 X，需调整到 [a,b]，建议值 m」。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "rewards": [{"item_name": "金币", "count": 999999}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    v2 = next(i for i in report.issues if i.rule == "V2")
    assert v2.detail["allowed_range"][1] == 400000  # 历史统计上限 200000 × 2
    assert "999999" in v2.fix_hint
    assert str(v2.detail["allowed_range"][1]) in v2.fix_hint
    assert v2.auto_repairable


def test_v7_gives_exact_delta(game_data, vctx):
    """V7 增强：fix_hint 含具体差额数字（05 §8.3）。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "rewards": [{"item_name": "金币", "count": 28000}],  # 超上限 27000，偏差 ≤10%
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    v7 = next(i for i in report.issues if i.rule == "V7")
    assert v7.detail["allowed_range"] == [9000, 27000]
    assert "28000" in v7.fix_hint and "27000" in v7.fix_hint
    assert v7.auto_repairable  # 偏差 1000/27000 ≈ 3.7% ≤ 10%


def test_v7_far_overflow_not_auto_repairable(game_data, vctx):
    """V7 偏差 >10% 不可自动修复（必须模型重新选择）。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "rewards": [{"item_name": "金币", "count": 90000}],  # 超上限 3 倍多
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    v7 = next(i for i in report.issues if i.rule == "V7")
    assert not v7.auto_repairable


def test_v9_stays_warning(game_data, vctx):
    """V9 雷同维持 warning 不阻塞（业务红线保留）。"""
    draft = {
        "task_type": "资源收集",
        "title": "测试",
        "rewards": [{"item_name": "金币", "count": 25000}],
    }
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    # 无论是否有雷同警告，都不能出现在 issues 里
    assert all(i.rule != "V9" for i in report.issues)
