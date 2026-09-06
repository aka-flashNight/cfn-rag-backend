"""后端兜底草案测试（对应 docs/v3-developer/05 §5/§8.4）。

保证「拟定了方向就能拿出草案」：构造结果 100% 通过全量校验。
"""

from __future__ import annotations

from services.agent_tools.fallback import build_fallback_draft, detect_task_type
from services.agent_tools.handlers import execute_fallback_draft
from services.agent_tools.validator import validate_task_draft


def test_detect_task_type():
    assert detect_task_type("收集 3 个食材") == "资源收集"
    assert detect_task_type("通关废弃矿坑") == "通关"
    assert detect_task_type("拿一把新武器来") == "装备缴纳"
    assert detect_task_type("跟我切磋一下") == "切磋"
    assert detect_task_type("随便") == "资源收集"  # 缺省


def test_fallback_draft_passes_full_validation(game_data, vctx):
    draft = build_fallback_draft(
        direction="收集类，目标是食材，报酬金币",
        npc_name="铁匠",
        player_progress=1,
        game_data=game_data,
    )
    assert draft["fallback"] is True
    assert draft["draft_id"]
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    assert report.ok, [i.message for i in report.issues]
    assert draft["rewards"], "兜底草案必须有奖励"


def test_fallback_draft_for_combat_type(game_data, vctx):
    draft = build_fallback_draft(
        direction="通关一个副本",
        npc_name="铁匠",
        player_progress=1,
        game_data=game_data,
    )
    report = validate_task_draft(draft, context=vctx, game_data=game_data)
    assert report.ok, [i.message for i in report.issues]
    assert draft["finish_requirements"], "通关类兜底草案必须带关卡要求"


def test_execute_fallback_draft_outcome(game_data):
    outcome = execute_fallback_draft(
        direction="收集食材",
        npc_name="铁匠",
        player_progress=1,
        game_data=game_data,
    )
    assert outcome.fallback_used
    assert outcome.draft_commit_valid
    payload = outcome.payload
    assert payload["status"] == "draft_created"
    assert payload["fallback"] is True
