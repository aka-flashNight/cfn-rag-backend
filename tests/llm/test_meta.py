"""meta 行协议解析器测试（02 §6.1：正常/无 meta/半截 JSON/非法 emo/fav 越界/多行 JSON 冒充/同行 meta）。

v3 再次重构后 meta 行只剩 emo/fav 两个字段，委派意图改由路由工具承载
（见 tests/orchestrator/test_turn_e2e.py），故本文件不再有 act 相关断言。
"""

from __future__ import annotations

from services.llm.client import StreamEvent
from services.llm.meta import Meta, meta_prompt_block, split_meta, split_meta_events

EMOTIONS = ["普通", "微笑", "生气", "惊讶"]


async def _stream_from_text(text: str, chunk_size: int = 7):
    """把整段文本切成小 chunk 的 StreamEvent 流（模拟真实 token 流）。"""
    for i in range(0, len(text), chunk_size):
        yield StreamEvent(kind="content", text=text[i : i + chunk_size])
    yield StreamEvent(kind="finish", usage={"prompt_tokens": 10, "completion_tokens": 5})


async def _collect_text(aiter) -> str:
    return "".join([chunk async for chunk in aiter])


async def test_normal_meta():
    text = '{"emo":"微笑","fav":0}\n诶，是你啊。今天有什么想聊的？\n'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "微笑"
    assert meta.favorability_change == 0
    assert body == "诶，是你啊。今天有什么想聊的？\n"


async def test_no_meta_plain_text():
    text = "你来了啊，正好我有事找你。"
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通"
    assert meta.favorability_change == 0
    # 无 meta 时缓冲文本必须原样还给正文
    assert body == text


async def test_partial_json_line_treated_as_no_meta():
    text = '{"emo":"微笑","fav":1\n正文开始了。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通" and meta.favorability_change == 0
    assert body == text  # 半截 JSON 不吞正文


async def test_invalid_emotion_falls_back():
    text = '{"emo":"暴怒","fav":0}\n哼。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通"  # 非法 → 回退默认
    assert body == "哼。"


async def test_fav_out_of_range_clamped():
    text = '{"emo":"生气","fav":99}\n别怪我发火。'
    meta, _ = await split_meta(_stream_from_text(text), EMOTIONS)
    assert meta.favorability_change == 5
    text2 = '{"emo":"生气","fav":-50}\n哼。'
    meta2, _ = await split_meta(_stream_from_text(text2), EMOTIONS)
    assert meta2.favorability_change == -5


async def test_unknown_extra_keys_ignored():
    """协议外的多余键不影响 emo/fav 解析（模型偶发夹带字段不至于整行作废）。"""
    text = '{"emo":"微笑","fav":1,"act":{"kind":"task_draft"},"foo":"bar"}\n好。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "微笑"
    assert meta.favorability_change == 1
    assert body == "好。"


async def test_json_without_emotion_or_fav_is_content():
    """首行是合法 JSON 但既无 emo 也无 fav → 不认作 meta，整段按正文。"""
    text = '{"result":"ok"}\n这是台词。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta == Meta()
    assert body == text


async def test_multiline_json_impersonation_is_content():
    """多行 JSON 冒充 meta：首行只有 { → 解析失败，全文按正文处理。"""
    text = '{\n"emo": "微笑",\n"fav": 1\n}\n这是台词的一部分。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "普通"
    assert body == text


async def test_meta_without_content():
    """流只有 meta 行（无换行无正文）→ meta 正常解析，正文为空。

    路由轮就是这种形态：第一行 JSON 之后不写台词，只发工具调用。
    """
    text = '{"emo":"惊讶","fav":-1}'
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


async def test_meta_and_dialogue_on_same_line_are_split():
    """模型把 meta 和台词写在同一行（2026-10-16 发布轮实测）→ 按 JSON 前缀切，花括号不得漏给玩家。"""
    text = '{"emo":"微笑","fav":0}好，痛快。出了基地往东，环线公路上见真章。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "微笑" and meta.favorability_change == 0
    assert body == "好，痛快。出了基地往东，环线公路上见真章。"
    assert "{" not in body


async def test_meta_and_dialogue_on_same_line_keeps_following_paragraphs():
    """同行 meta 之后的换行分段仍归正文，不能多吃一个换行或丢掉后文。"""
    text = '{"emo":"惊讶","fav":1}这么快？\n环线都清干净了？'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "惊讶" and meta.favorability_change == 1
    assert body == "这么快？\n环线都清干净了？"


async def test_whole_meta_only_line_still_waits_for_newline():
    """只有 meta、后面还没有增量时不提前判定：等换行或流结束再定，正文不得凭空多个空行。"""
    text = '{"emo":"微笑","fav":0}\n你好呀。'
    meta, rest = await split_meta(_stream_from_text(text), EMOTIONS)
    body = await _collect_text(rest)
    assert meta.emotion == "微笑"
    assert body == "你好呀。"


async def test_reasoning_events_never_enter_meta_or_content():
    """reasoning 增量一律丢弃（修 A7），不影响 meta 判定与正文。"""
    async def stream():
        yield StreamEvent(kind="reasoning", text="用户在跟我打招呼")
        yield StreamEvent(kind="content", text='{"emo":"微笑","fav":0}\n')
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


# ---------------------------------------------------------------------------
# meta 协议说明文本：只讲情绪与好感，且对同一 NPC 跨轮恒定
# ---------------------------------------------------------------------------

def test_meta_prompt_block_covers_only_emotion_and_favorability():
    block = meta_prompt_block(EMOTIONS)
    assert "1. emo" in block and "2. fav" in block
    assert "、".join(EMOTIONS) in block
    # 委派意图不在 meta 协议里：任何路由/工具词汇都不得出现
    for forbidden in ("act", "task_draft", "task_update", "task_confirm",
                      "task_cancel", "search", "prepare_task_context", "route_"):
        assert forbidden not in block


def test_meta_prompt_block_mentions_routing_tool_as_the_other_branch():
    """第一行之后「要么台词、要么路由工具调用」必须写清，否则模型会在路由轮硬凑台词。"""
    block = meta_prompt_block(EMOTIONS)
    assert "路由工具调用" in block
    assert "两者只取其一" in block


def test_meta_prompt_block_drops_routing_branch_when_agent_disabled():
    """关闭 Agent 能力时不提路由：既无工具也无路由说明块，提了就是悬空指代。"""
    block = meta_prompt_block(EMOTIONS, agent_enabled=False)
    assert "路由" not in block
    assert "第一行之后是台词正文" in block
    # 情绪/好感规则本身不受开关影响
    assert "1. emo" in block and "2. fav" in block
    assert "、".join(EMOTIONS) in block


def test_meta_prompt_block_is_stable_across_calls():
    """前缀缓存要求：同 NPC、同开关状态跨轮必须逐字一致（有无草案不影响它）。"""
    assert meta_prompt_block(EMOTIONS) == meta_prompt_block(EMOTIONS)
    assert (meta_prompt_block(EMOTIONS, agent_enabled=False)
            == meta_prompt_block(EMOTIONS, agent_enabled=False))


def test_meta_prompt_block_empty_emotions_falls_back():
    block = meta_prompt_block([])
    assert "普通" in block
