"""meta 行协议解析器测试（02 §6.1：正常/无 meta/半截 JSON/非法 emo/fav 越界/多行 JSON 冒充）。"""

from __future__ import annotations

import json

from services.llm.client import StreamEvent
from services.llm.meta import Meta, MetaAct, split_meta, split_meta_events

EMOTIONS = ["普通", "微笑", "生气", "惊讶"]


async def _stream_from_text(text: str, chunk_size: int = 7):
    """把整段文本切成小 chunk 的 StreamEvent 流（模拟真实 token 流）。"""
    for i in range(0, len(text), chunk_size):
        yield StreamEvent(kind="content", text=text[i : i + chunk_size])
    yield StreamEvent(kind="finish", usage={"prompt_tokens": 10, "completion_tokens": 5})


async def _collect_text(aiter) -> str:
    return "".join([chunk async for chunk in aiter])


async def test_normal_meta():
    text = '{"emo":"微笑","fav":0,"act":null}\n诶，是你啊。今天有什么想聊的？\n'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "微笑"
    assert meta.favorability_change == 0
    assert meta.act is None
    assert body == "诶，是你啊。今天有什么想聊的？\n"


async def test_no_meta_plain_text():
    text = "你来了啊，正好我有事找你。"
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通"
    assert meta.favorability_change == 0
    assert meta.act is None
    # 无 meta 时缓冲文本必须原样还给正文
    assert body == text


async def test_partial_json_line_treated_as_no_meta():
    text = '{"emo":"微笑","fav":1,"ac\n正文开始了。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通" and meta.favorability_change == 0
    assert body == text  # 半截 JSON 不吞正文


async def test_invalid_emotion_falls_back():
    text = '{"emo":"暴怒","fav":0,"act":null}\n哼。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通"  # 非法 → 回退默认
    assert body == "哼。"


async def test_fav_out_of_range_clamped():
    text = '{"emo":"生气","fav":99,"act":null}\n别怪我发火。'
    meta, _ = await split_meta(_stream_from_text(text), EMOTIONS)
    assert meta.favorability_change == 5
    text2 = '{"emo":"生气","fav":-50,"act":null}\n哼。'
    meta2, _ = await split_meta(_stream_from_text(text2), EMOTIONS)
    assert meta2.favorability_change == -5


async def test_multiline_json_impersonation_is_content():
    """多行 JSON 冒充 meta：首行只有 { → 解析失败，全文按正文处理。"""
    text = '{\n"emo": "微笑",\n"fav": 1\n}\n这是台词的一部分。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通"
    assert body == text


async def test_act_variants():
    # v3 修订：task_draft 只表达意图（prepare 参数由工具调用承载），多余字段忽略
    draft = {"emo": "微笑", "fav": 1, "act": {"kind": "task_draft", "direction": "收集3个猫爪交给铁匠"}}
    meta, rest = await split_meta(_stream_from_text(json.dumps(draft, ensure_ascii=False) + "\n好。"), EMOTIONS)
    await _collect_text(rest)
    assert meta.act == MetaAct(kind="task_draft")

    search = {"emo": "普通", "fav": 0, "act": {"kind": "search", "query": "安迪·洛的过去"}}
    meta2, rest2 = await split_meta(_stream_from_text(json.dumps(search, ensure_ascii=False) + "\n嗯。"), EMOTIONS)
    await _collect_text(rest2)
    assert meta2.act == MetaAct(kind="search", query="安迪·洛的过去")

    for kind in ("task_confirm", "task_cancel"):
        obj = {"emo": "普通", "fav": 0, "act": {"kind": kind}}
        meta3, rest3 = await split_meta(_stream_from_text(json.dumps(obj) + "\n好。"), EMOTIONS)
        await _collect_text(rest3)
        assert meta3.act == MetaAct(kind=kind)


async def test_act_task_draft_without_params_still_valid():
    """task_draft 无需任何参数（prepare 由工具调用承载）→ 意图保留。"""
    text = '{"emo":"微笑","fav":1,"act":{"kind":"task_draft"}}\n好。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    await _collect_text(rest)
    assert meta.emotion == "微笑"
    assert meta.act == MetaAct(kind="task_draft")


async def test_unknown_act_kind_becomes_null():
    text = '{"emo":"普通","fav":0,"act":{"kind":"nuke_the_world"}}\n好。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    await _collect_text(rest)
    assert meta.act is None


async def test_act_as_string_coerced():
    """act 写成字符串（模型常见手误）→ 无参 kind 宽容纠正，confirm 不被静默丢弃。"""
    for kind in ("task_confirm", "task_cancel", "task_draft"):
        text = f'{{"emo":"微笑","fav":1,"act":"{kind}"}}\n好，就这么定。'
        meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
        await _collect_text(rest)
        assert meta.act == MetaAct(kind=kind), kind


async def test_act_string_with_required_payload_becomes_null():
    """task_update/search 需要载荷，字符串形态无法恢复 → 按 null 处理。"""
    for kind in ("task_update", "search"):
        text = f'{{"emo":"普通","fav":0,"act":"{kind}"}}\n嗯。'
        meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
        await _collect_text(rest)
        assert meta.act is None, kind


async def test_meta_without_content():
    """流只有 meta 行（无换行无正文）→ meta 正常解析，正文为空。"""
    text = '{"emo":"惊讶","fav":-1,"act":null}'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "惊讶" and meta.favorability_change == -1
    assert body == ""


async def test_over_512_chars_without_newline_is_content():
    """超过 512 字符仍无换行 → 视为无 meta，缓冲文本完整回到正文。"""
    long_text = "好" * 600
    meta, rest = await split_meta(_stream_from_text(long_text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通"
    assert body == long_text


async def test_reasoning_events_never_enter_meta_or_content():
    """reasoning 增量一律丢弃（修 A7），不影响 meta 判定与正文。"""
    async def stream():
        yield StreamEvent(kind="reasoning", text="用户在跟我打招呼")
        yield StreamEvent(kind="content", text='{"emo":"微笑","fav":0,"act":null}\n')
        yield StreamEvent(kind="reasoning", text="继续思考")
        yield StreamEvent(kind="content", text="你好呀。")
        yield StreamEvent(kind="finish")

    meta, rest = await split_meta_events(stream(), EMOTIONS)
    kinds = []
    texts = []
    async for ev in rest:
        kinds.append(ev.kind)
        if ev.kind == "content":
            texts.append(ev.text)
    assert meta.emotion == "微笑"
    assert "reasoning" not in kinds
    assert "".join(texts) == "你好呀。"
