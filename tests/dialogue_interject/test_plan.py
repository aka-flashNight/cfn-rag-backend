"""plan 单测：有序 steps、continue/end、跳号与缺终止符的兜底。"""

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


def test_f05_continue_with_keep_and_edit():
    plan, _ = _normalize(fx.F05_JSONL)
    assert plan.mute is False
    assert plan.resume == 3
    assert plan.fallback_continue is False
    assert [step.model_dump() for step in plan.steps] == [
        {"n": 1, "kind": "keep", "lines": []},
        {
            "n": 2,
            "kind": "edit",
            "lines": [{
                "text": "北边仓库昨晚丢了一批货。",
                "emotion": "普通",
                "speaker": fx.NPC,
                "split": None,
            }],
        },
    ]


def test_f05b_edit_lines_carry_split_marks():
    plan, _ = _normalize(fx.F05B_JSONL)
    assert plan.resume == 4
    assert len(plan.steps) == 3
    edit = plan.steps[2]
    assert edit.n == 3 and edit.kind == "edit"
    assert len(edit.lines) == 2
    assert [line.split.count for line in edit.lines] == [2, 2]
    assert [line.split.index for line in edit.lines] == [0, 1]
    assert "".join(line.text for line in edit.lines) == fx.F05B_LONG_EDIT
    assert all(len(line.text) <= 165 for line in edit.lines)


def test_f06_remove_before_continue():
    plan, _ = _normalize(fx.F06_JSONL)
    assert plan.resume == 3
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "remove")]


def test_f07_explicit_keep_is_preserved_in_steps():
    plan, _ = _normalize(fx.F07_JSONL)
    assert plan.resume == 4
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "keep"), (2, "keep"), (3, "edit"),
    ]


def test_f08_end_processes_all_originals_and_does_not_resume():
    plan, _ = _normalize(fx.F08_JSONL)
    assert plan.resume == 5
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "edit"), (2, "remove"), (3, "remove"), (4, "remove"),
    ]
    assert plan.fallback_continue is False


def test_out_of_range_no_is_rejected_without_dropping_other_steps():
    plan, parser = _normalize([
        '{"type":"edit","no":9,"text":"越界","emo":"普通"}\n',
        '{"type":"remove","no":9}\n',
        '{"type":"keep","no":1}\n',
        '{"type":"edit","no":2,"text":"合法","emo":"普通"}\n',
        '{"type":"continue","no":3}\n',
    ])
    assert plan.resume == 3
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "edit")]
    assert len([issue for issue in parser.issues if "越界" in issue.reason]) == 2


def test_skipped_number_is_backfilled_as_remove():
    plan, _ = _normalize([
        '{"type":"keep","no":1}\n',
        '{"type":"edit","no":3,"text":"只改第 3 条","emo":"普通"}\n',
        '{"type":"continue","no":4}\n',
    ])
    assert plan.resume == 4
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "keep"), (2, "remove"), (3, "edit"),
    ]
    assert any(item.n == 2 and "按 remove 兜底" in item.reason for item in plan.rejected)


def test_continue_with_gap_backfills_removed_gap():
    plan, _ = _normalize([
        '{"type":"keep","no":1}\n',
        '{"type":"continue","no":3}\n',
    ])
    assert plan.resume == 3
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "remove")]


def test_missing_terminator_resumes_after_last_processed_original():
    plan, _ = _normalize([
        '{"type":"keep","no":1}\n',
        '{"type":"edit","no":2,"text":"合法","emo":"普通"}\n',
    ])
    assert plan.resume == 3
    assert plan.fallback_continue is True
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "edit")]
    assert any("缺少 continue/end" in item.reason for item in plan.rejected)


def test_missing_terminator_without_original_ops_keeps_all_originals():
    plan, _ = _normalize([
        '{"type":"add","no":0,"text":"先回应一句。","emo":"普通"}\n',
    ])
    assert plan.resume == 1
    assert plan.steps == []
    assert plan.fallback_continue is True


def test_missing_terminator_after_last_edit_keeps_edit_and_ends():
    plan, _ = _normalize([
        '{"type":"keep","no":1}\n',
        '{"type":"keep","no":2}\n',
        '{"type":"keep","no":3}\n',
        '{"type":"edit","no":4,"text":"改写末条","emo":"普通"}\n',
    ])
    assert plan.resume == 5
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "keep"), (2, "keep"), (3, "keep"), (4, "edit"),
    ]


def test_invalid_continue_falls_back_to_after_last_processed():
    plan, _ = _normalize([
        '{"type":"keep","no":1}\n',
        '{"type":"edit","no":2,"text":"合法","emo":"普通"}\n',
        '{"type":"continue","no":2}\n',
    ])
    assert plan.resume == 3
    assert plan.fallback_continue is True
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "edit")]


def test_continue_one_keeps_all_originals():
    plan, _ = _normalize([
        '{"type":"add","no":0,"text":"回应。","emo":"普通"}\n',
        '{"type":"continue","no":1}\n',
    ])
    assert plan.resume == 1
    assert plan.steps == []
    assert plan.fallback_continue is False


def test_end_without_original_ops_removes_all_originals():
    plan, _ = _normalize([
        '{"type":"add","no":0,"text":"回应。","emo":"普通"}\n',
        '{"type":"end"}\n',
    ])
    assert plan.resume == 5
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "remove"), (2, "remove"), (3, "remove"), (4, "remove"),
    ]


def test_no_remaining_allows_only_end():
    plan, _ = _normalize([
        '{"type":"add","no":0,"text":"回应。","emo":"普通"}\n',
        '{"type":"end"}\n',
    ], remaining_count=0)
    assert plan.resume == 1
    assert plan.steps == []


def test_entries_after_terminator_are_ignored():
    plan, _ = _normalize([
        '{"type":"keep","no":1}\n',
        '{"type":"continue","no":2}\n',
        '{"type":"edit","no":2,"text":"不该出现","emo":"普通"}\n',
        '{"type":"add","no":0,"text":"也不该出现","emo":"普通"}\n',
        '{"type":"end"}\n',
    ])
    assert plan.resume == 2
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep")]
    assert len([item for item in plan.rejected if "之后" in item.reason]) == 3


def test_add_after_original_op_enters_steps_in_output_order():
    plan, _ = _normalize([
        '{"type":"add","no":0,"text":"前缀即时回应","emo":"普通"}\n',
        '{"type":"edit","no":1,"text":"改写 1","emo":"普通"}\n',
        '{"type":"add","no":0,"text":"插在 1 和 2 之间","emo":"普通"}\n',
        '{"type":"keep","no":2}\n',
        '{"type":"add","no":0,"text":"插在 2 和 continue 之间","emo":"普通"}\n',
        '{"type":"continue","no":3}\n',
    ])
    assert plan.resume == 3
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "edit"), (0, "add"), (2, "keep"), (0, "add"),
    ]
    assert [step.lines[0].text for step in plan.steps if step.kind == "add"] == [
        "插在 1 和 2 之间", "插在 2 和 continue 之间",
    ]


def test_plan_lines_carry_model_name_as_speaker():
    plan, _ = _normalize([
        '{"type":"keep","no":1,"name":"Andy Law"}\n',
        '{"type":"add","name":"老赵","text":"插一句。","emo":"普通"}\n'
        '{"type":"edit","no":2,"name":"$PC","text":"玩家改口。","emo":"普通"}\n'
        '{"type":"continue","no":3,"name":"Andy Law"}\n',
    ])
    assert [(step.n, step.kind) for step in plan.steps] == [
        (1, "keep"), (0, "add"), (2, "edit"),
    ]
    assert plan.steps[1].lines[0].speaker == "老赵"
    assert plan.steps[2].lines[0].speaker == "$PC"
    assert plan.to_wire()["steps"][1]["lines"][0]["speaker"] == "老赵"


def test_consecutive_edits_same_no_merge_into_split_replacement():
    plan, _ = _normalize([
        '{"type":"edit","no":1,"text":"第一段","emo":"普通"}\n',
        '{"type":"edit","no":1,"text":"第二段","emo":"普通"}\n',
        '{"type":"continue","no":2}\n',
    ])
    assert len(plan.steps) == 1
    assert [line.text for line in plan.steps[0].lines] == ["第一段", "第二段"]


def test_nonconsecutive_or_conflicting_duplicate_is_rejected():
    plan, _ = _normalize([
        '{"type":"edit","no":1,"text":"第一段","emo":"普通"}\n',
        '{"type":"add","no":0,"text":"插入","emo":"普通"}\n',
        '{"type":"edit","no":1,"text":"不应接收","emo":"普通"}\n',
        '{"type":"remove","no":1}\n',
        '{"type":"continue","no":2}\n',
    ])
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "edit"), (0, "add")]
    assert len([item for item in plan.rejected if "重复" in item.reason]) == 2


def test_out_of_order_original_number_is_rejected():
    plan, _ = _normalize([
        '{"type":"keep","no":1}\n',
        '{"type":"keep","no":2}\n',
        '{"type":"edit","no":1,"text":"倒退","emo":"普通"}\n',
        '{"type":"continue","no":3}\n',
    ])
    assert [(step.n, step.kind) for step in plan.steps] == [(1, "keep"), (2, "keep")]
    assert any("倒退" in item.reason for item in plan.rejected)


def test_mute_is_carried_into_plan():
    plan, _ = _normalize(fx.F09_JSONL)
    assert plan.mute is True
    assert plan.resume == 1
    assert plan.steps == []
