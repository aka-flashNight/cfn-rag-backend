"""parser 单测：JSONL/围栏/非 JSON、括号剥离、165 拆分与 split 标记、情绪回落。"""

from __future__ import annotations

from services.dialogue_interject.parser import (
    JSONLStreamParser,
    ParseOptions,
    clean_text,
    split_text,
)
from services.dialogue_interject.schemas import RemainingLine
from tests.dialogue_interject import fixtures as fx


def _daily_parser() -> JSONLStreamParser:
    return JSONLStreamParser(
        ParseOptions(mode="daily", npc=fx.NPC, npc_emotions=("普通", "微笑"))
    )


def test_daily_skips_fence_blank_and_non_json():
    parser = _daily_parser()
    entries = parser.feed(
        "```jsonl\n"
        '{"add":"甲","emo":"普通"}\n'
        "\n"
        "这不是 JSON\n"
        '{"closing":"乙","emo":"普通"}\n'
        "```\n"
    )
    assert [entry.kind for entry in entries] == ["add", "closing"]
    assert [entry.segments[0].text for entry in entries] == ["甲", "乙"]
    assert len(parser.issues) == 1  # 只有“这不是 JSON”；围栏行静默跳过
    assert "非 JSON" in parser.issues[0].reason


def test_line_split_across_chunks_waits_for_newline():
    parser = _daily_parser()
    assert parser.feed('{"add":"你') == []
    entries = parser.feed('好","emo":"普通"}\n')
    assert len(entries) == 1
    assert entries[0].segments[0].text == "你好"


def test_finish_flushes_trailing_line_without_newline():
    parser = _daily_parser()
    assert parser.feed('{"add":"最后一句"}') == []
    entries = parser.finish()
    assert [entry.segments[0].text for entry in entries] == ["最后一句"]


def test_bracket_content_is_stripped_and_empty_dropped():
    assert clean_text("（冷笑）你好（转身就走）") == "你好"
    assert clean_text("(sigh) 走吧 (aside)") == "走吧"

    parser = _daily_parser()
    entries = parser.feed('{"add":"（沉默不语）","emo":"普通"}\n{"add":"（点头）嗯。","emo":"普通"}\n')
    assert [entry.segments[0].text for entry in entries] == ["嗯。"]
    assert any("为空" in issue.reason for issue in parser.issues)


def test_split_prefers_period_then_other_punctuation_then_hard_cut():
    text = "甲" * 100 + "。" + "乙" * 100 + "。"
    segments = split_text(text, 165)
    assert len(segments) == 2
    assert segments[0].endswith("。")
    assert all(len(segment) <= 165 for segment in segments)
    assert "".join(segments) == text

    hard = "啊" * 400
    hard_segments = split_text(hard, 165)
    assert [len(segment) for segment in hard_segments] == [165, 165, 70]
    assert "".join(hard_segments) == hard


def test_fixture_f02_split_shape():
    segments = split_text(fx.F02_LONG_ADD, 165)
    assert len(segments) == 2
    assert segments[0].endswith("。")
    assert "".join(segments) == fx.F02_LONG_ADD

    parser = _daily_parser()
    entries = parser.feed(fx.F02_JSONL[0])
    assert len(entries) == 1
    marks = [segment.split for segment in entries[0].segments]
    assert [(mark.index, mark.count) for mark in marks] == [(0, 2), (1, 2)]


def test_emotion_falls_back_to_default_for_current_npc():
    parser = _daily_parser()
    entries = parser.feed(
        '{"add":"甲","emo":"狰狞"}\n{"add":"乙","emo":"微笑"}\n'
    )
    assert [entry.segments[0].emotion for entry in entries] == ["普通", "微笑"]


def test_emotion_falls_back_to_original_for_other_npc_edit():
    remaining = {
        1: RemainingLine(index=1, name="老赵", text="原文", emotion="阴冷"),
    }
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            npc_emotions=("普通",),
            speaker_emotions={"老赵": ["平静"]},
            remaining=remaining,
        )
    )
    entries = parser.feed('{"n":1,"line":"改写","emo":"疯狂"}\n')
    assert len(entries) == 1
    assert entries[0].segments[0].speaker == "老赵"
    assert entries[0].segments[0].emotion == "阴冷"


def test_task_out_of_range_line_is_ignored_and_recorded():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    assert parser.feed('{"n":9,"line":"越界"}\n') == []
    assert any("越界" in issue.reason for issue in parser.issues)


def test_nochange_with_body_is_marker_and_body_ignored():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    entries = parser.feed('{"n":1,"nochange":true,"line":"不该出现"}\n')
    assert [entry.kind for entry in entries] == ["nochange"]
    assert entries[0].n == 1
    assert any("nochange 带正文" in issue.reason for issue in parser.issues)


def test_task_mute_add_and_end():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    entries = parser.feed('{"mute":true}\n{"add":"好"}\n{"end":1}\n')
    assert [entry.kind for entry in entries] == ["mute", "add", "end"]
    assert entries[2].end == 1
