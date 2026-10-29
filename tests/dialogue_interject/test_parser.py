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


def test_daily_skips_fence_blank_and_broken_json_fragment():
    parser = _daily_parser()
    entries = parser.feed(
        "```jsonl\n"
        '{"add":"甲","emo":"普通"}\n'
        "\n"
        '{"add":"未闭合"\n'
        '{"closing":"乙","emo":"普通"}\n'
        "```\n"
    )
    assert [entry.kind for entry in entries] == ["add", "closing"]
    assert [entry.segments[0].text for entry in entries] == ["甲", "乙"]
    assert len(parser.issues) == 1  # 只有截断的 JSON；围栏行静默跳过
    assert "非 JSON" in parser.issues[0].reason


def test_daily_salvages_objects_glued_on_one_line():
    parser = _daily_parser()
    entries = parser.feed('{"add":"甲","emo":"微笑"}{"add":"乙","emo":"普通"}\n')

    assert [entry.kind for entry in entries] == ["add", "add"]
    assert [entry.segments[0].text for entry in entries] == ["甲", "乙"]
    assert parser.issues == []


def test_daily_salvages_complete_objects_before_truncated_tail():
    parser = _daily_parser()
    entries = parser.feed(
        '{"add":"一瓶hp药剂，拿好了。","emo":"微笑"}'
        '{"add":"要补货的话，The Girl那边储备不少。","emo":"普通"}'
        '{"add":"第三句被截断","emo":"普\n'
    )

    assert [entry.segments[0].text for entry in entries] == [
        "一瓶hp药剂，拿好了。",
        "要补货的话，The Girl那边储备不少。",
    ]
    assert len(parser.issues) == 1
    assert "截断" in parser.issues[0].reason


def test_task_structural_marker_in_glued_line_drops_whole_line():
    parser = JSONLStreamParser(
        ParseOptions(mode="task", npc=fx.NPC, npc_emotions=("普通",))
    )
    entries = parser.feed('{"type":"add","no":0,"text":"甲"}{"type":"continue","no":1}\n')

    assert entries == []
    assert len(parser.issues) == 1
    assert "JSON 解析失败" in parser.issues[0].reason


def test_pure_text_line_falls_back_to_add_with_empty_emotion():
    parser = _daily_parser()
    entries = parser.feed("我是A兵团的线人，在这基地里打探消息、传个话。\n")

    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "add"
    assert entry.n is None
    assert entry.segments[0].text == "我是A兵团的线人，在这基地里打探消息、传个话。"
    assert entry.segments[0].emotion == ""
    assert entry.segments[0].speaker == fx.NPC
    assert parser.issues == []
    assert parser.plain_text_fallbacks == ["我是A兵团的线人，在这基地里打探消息、传个话。"]


def test_pure_text_fallback_still_strips_brackets_and_splits():
    parser = _daily_parser()
    long_line = "（压低声音）" + "甲" * 200
    entries = parser.feed(long_line + "\n")

    assert len(entries) == 1
    segments = entries[0].segments
    assert len(segments) == 2
    assert "".join(segment.text for segment in segments) == "甲" * 200
    assert [segment.split.index for segment in segments] == [0, 1]
    assert all(segment.emotion == "" for segment in segments)


def test_json_key_fragment_without_braces_is_not_treated_as_text():
    parser = _daily_parser()
    entries = parser.feed('"add":"你好"\n[1,2]\n')

    assert entries == []
    assert [issue.reason for issue in parser.issues] == ["非 JSON 行", "非 JSON 行"]
    assert parser.plain_text_fallbacks == []


def test_task_pure_text_line_falls_back_to_add():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    entries = parser.feed("这句是任务对话里的裸台词。\n")

    assert len(entries) == 1
    assert entries[0].kind == "add"
    assert entries[0].segments[0].speaker == fx.NPC
    assert entries[0].segments[0].emotion == ""


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
    entries = parser.feed('{"type":"edit","no":1,"text":"改写","emo":"疯狂"}\n')
    assert len(entries) == 1
    assert entries[0].segments[0].speaker == "老赵"
    assert entries[0].segments[0].emotion == "阴冷"


def test_task_out_of_range_edit_is_ignored_and_recorded():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    assert parser.feed('{"type":"edit","no":9,"text":"越界"}\n') == []
    assert any("越界" in issue.reason for issue in parser.issues)


def test_task_keep_remove_and_continue_use_original_number():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    entries = parser.feed(
        '{"type":"keep","no":1}\n'
        '{"type":"remove","no":1}\n'
        '{"type":"continue","no":1}\n'
    )
    assert [(entry.kind, entry.n) for entry in entries] == [
        ("keep", 1), ("remove", 1), ("continue", 1),
    ]


def test_task_add_accepts_missing_no_and_ignores_present_no():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    entries = parser.feed(
        '{"type":"add","text":"无编号","emo":"普通","name":"老赵"}\n'
        '{"type":"add","no":1,"text":"带编号也忽略","emo":"普通"}\n'
    )
    assert [(entry.kind, entry.n) for entry in entries] == [("add", 0), ("add", 0)]
    assert entries[0].segments[0].speaker == "老赵"
    assert entries[1].segments[0].speaker == fx.NPC
    assert parser.issues == []


def test_task_edit_uses_name_then_falls_back_to_original_speaker():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name="老赵", text="原文", emotion="阴冷")},
        )
    )
    entries = parser.feed(
        '{"type":"edit","no":1,"name":"新人","text":"改写一","emo":"普通"}\n'
        '{"type":"edit","no":1,"name":"","text":"改写二","emo":"普通"}\n'
    )
    assert [entry.segments[0].speaker for entry in entries] == ["新人", "老赵"]
    assert parser.issues == []


def test_task_keep_remove_continue_ignore_name():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    entries = parser.feed(
        '{"type":"keep","no":1,"name":"别的名字"}\n'
        '{"type":"remove","no":1,"name":""}\n'
        '{"type":"continue","no":1,"name":"$PC"}\n'
    )
    assert [(entry.kind, entry.n) for entry in entries] == [
        ("keep", 1), ("remove", 1), ("continue", 1),
    ]
    assert parser.issues == []


def test_task_mute_add_and_continue():
    parser = JSONLStreamParser(
        ParseOptions(
            mode="task",
            npc=fx.NPC,
            remaining={1: RemainingLine(index=1, name=fx.NPC, text="原文")},
        )
    )
    entries = parser.feed(
        '{"type":"mute"}\n'
        '{"type":"add","no":0,"text":"好","emo":"普通"}\n'
        '{"type":"continue","no":1}\n'
    )
    assert [entry.kind for entry in entries] == ["mute", "add", "continue"]
    assert entries[2].n == 1
