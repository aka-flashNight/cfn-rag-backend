"""plan 单测：F-05/F-05b/F-06/F-07/F-08 逐字段，以及越界/end/冲突的宽容处理。"""

from __future__ import annotations

from services.dialogue_interject.parser import JSONLStreamParser, ParseOptions
from services.dialogue_interject.plan import normalize_task_plan
from services.dialogue_interject.schemas import RemainingLine
from tests.dialogue_interject import fixtures as fx


def _remaining(count: int = 4):
    return {
        line["index"]: RemainingLine.model_validate(line)
        for line in fx.TASK_REMAINING[:count]
    }


def _normalize(jsonl: list[str], remaining_count: int = 4):
    remaining = _remaining(remaining_count)
    parser = JSONLStreamParser(
        ParseOptions(mode="task", npc=fx.NPC, remaining=remaining)
    )
    entries = parser.feed("".join(jsonl))
    entries.extend(parser.finish())
    return normalize_task_plan(entries, remaining_count), parser


def test_f05_end_resume_with_edit_and_implied_nochange():
    plan, _ = _normalize(fx.F05_JSONL)
    assert plan.mute is False
    assert plan.end == 3
    assert [edit.model_dump() for edit in plan.edits] == [
        {
            "n": 2,
            "kind": "edit",
            "lines": [{"text": "北边仓库昨晚丢了一批货。", "emotion": "普通", "split": None}],
        }
    ]
    assert plan.implied_nochange == [1]  # end 前未提及＝nochange，但不写进 edits


def test_f05b_edit_lines_carry_split_marks():
    plan, _ = _normalize(fx.F05B_JSONL)
    assert plan.end == 4
    assert len(plan.edits) == 1
    edit = plan.edits[0]
    assert edit.n == 3 and edit.kind == "edit"
    assert len(edit.lines) == 2
    assert [line.split.count for line in edit.lines] == [2, 2]
    assert [line.split.index for line in edit.lines] == [0, 1]
    assert "".join(line.text for line in edit.lines) == fx.F05B_LONG_EDIT
    assert all(len(line.text) <= 165 for line in edit.lines)
    assert plan.implied_nochange == [1, 2]


def test_f06_remove_before_end():
    plan, _ = _normalize(fx.F06_JSONL)
    assert plan.end == 3
    assert [edit.model_dump() for edit in plan.edits] == [
        {"n": 2, "kind": "remove", "lines": []}
    ]
    assert plan.implied_nochange == [1]


def test_f07_missing_nochange_is_tolerated():
    plan, _ = _normalize(fx.F07_JSONL)
    assert plan.end == 4
    assert [edit.n for edit in plan.edits] == [3]
    assert plan.implied_nochange == [1, 2]  # 未提及的 1、2 视为 nochange
    assert all(edit.kind != "remove" for edit in plan.edits)


def test_f08_no_end_means_whole_group_replacement():
    plan, _ = _normalize(fx.F08_JSONL)
    assert plan.end is None
    assert [edit.model_dump() for edit in plan.edits] == [
        {
            "n": 1,
            "kind": "edit",
            "lines": [{"text": "这事我只跟你说一遍。", "emotion": "普通", "split": None}],
        },
        {"n": 2, "kind": "remove", "lines": []},
    ]
    assert plan.implied_nochange == []


def test_out_of_range_n_is_rejected_without_dropping_other_edits():
    plan, parser = _normalize([
        '{"n":9,"line":"越界","emo":"普通"}\n',
        '{"n":9,"remove":true}\n',
        '{"n":2,"line":"合法","emo":"普通"}\n',
        '{"end":3}\n',
    ])
    assert plan.end == 3
    assert [edit.n for edit in plan.edits] == [2]
    # line 越界在 parser 层就丢弃；remove 越界在 plan 层记录
    assert any("越界" in issue.reason for issue in parser.issues)
    assert any(item.n == 9 and "越界" in item.reason for item in plan.rejected)


def test_out_of_range_end_is_ignored_and_edits_stay():
    plan, _ = _normalize([
        '{"n":2,"line":"合法","emo":"普通"}\n',
        '{"end":9}\n',
    ])
    assert plan.end is None
    assert [edit.n for edit in plan.edits] == [2]
    assert any(item.kind == "end" for item in plan.rejected)


def test_end_pointing_to_removed_item_jumps_to_next_original():
    plan, _ = _normalize([
        '{"n":3,"remove":true}\n',
        '{"end":3}\n',
    ])
    assert plan.end == 4
    assert [edit.n for edit in plan.edits] == [3]


def test_end_pointing_to_last_removed_item_ends_group():
    plan, _ = _normalize([
        '{"n":4,"remove":true}\n',
        '{"end":4}\n',
    ])
    assert plan.end is None  # 后面没有可接回的原文＝结束
    assert [edit.n for edit in plan.edits] == [4]


def test_edit_at_or_after_end_is_ignored():
    plan, _ = _normalize([
        '{"n":2,"line":"之前","emo":"普通"}\n',
        '{"n":3,"line":"end 本身","emo":"普通"}\n',
        '{"end":3}\n',
    ])
    assert plan.end == 3
    assert [edit.n for edit in plan.edits] == [2]
    assert any(item.n == 3 and "end" in item.reason for item in plan.rejected)


def test_multiple_lines_same_n_merge_into_one_edit():
    plan, _ = _normalize([
        '{"n":2,"line":"第一段","emo":"普通"}\n',
        '{"n":2,"line":"第二段","emo":"普通"}\n',
        '{"end":3}\n',
    ])
    assert len(plan.edits) == 1
    assert [line.text for line in plan.edits[0].lines] == ["第一段", "第二段"]


def test_nochange_and_remove_are_explicit_edits():
    plan, _ = _normalize([
        '{"n":1,"nochange":true}\n',
        '{"n":2,"remove":true}\n',
        '{"end":3}\n',
    ])
    assert [edit.kind for edit in plan.edits] == ["nochange", "remove"]
    assert plan.implied_nochange == []


def test_line_wins_over_remove_on_same_n():
    plan, _ = _normalize([
        '{"n":2,"line":"改写","emo":"普通"}\n',
        '{"n":2,"remove":true}\n',
        '{"end":3}\n',
    ])
    assert [edit.kind for edit in plan.edits] == ["edit"]
    assert any("按 edit 处理" in item.reason for item in plan.rejected)


def test_mute_is_carried_into_plan():
    plan, _ = _normalize(fx.F09_JSONL)
    assert plan.mute is True
    assert plan.end == 1
    assert plan.edits == []
