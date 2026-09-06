"""流式事件测试（02 §6.4/6.5）：reasoning 不混入 content、usage 回收、tool_calls 聚合。"""

from __future__ import annotations

from services.llm.client import ChatRequest, LLMClient, LLMConfig
from services.llm.meta import split_meta_events

from tests.conftest import FakeDelta, FakeOpenAI, FakeStreamChunk, FakeToolCall


def _make_client(handler) -> LLMClient:
    from services.llm.profiles import ModelProfile

    return LLMClient(
        LLMConfig(api_key="k", api_base="http://unit.test/v1", model_name="test-model"),
        profile=ModelProfile(match="test-model", vision=False, thinking_body={}),
        openai_client=FakeOpenAI(handler),
    )


async def test_reasoning_isolated_from_content():
    """reasoning_content delta 一律归入 kind=reasoning，绝不混入 content（修 A7）。"""

    def handler(kwargs):
        async def gen():
            yield FakeStreamChunk(delta=FakeDelta(reasoning_content="让我想想"))
            yield FakeStreamChunk(delta=FakeDelta(content="你"))
            yield FakeStreamChunk(delta=FakeDelta(reasoning_content="他好像在打招呼"))
            yield FakeStreamChunk(delta=FakeDelta(content="好呀。"))
        return gen()

    llm = _make_client(handler)
    events = [ev async for ev in llm.chat_stream(ChatRequest(messages=[{"role": "user", "content": "hi"}]))]
    reasoning = "".join(ev.text for ev in events if ev.kind == "reasoning")
    content = "".join(ev.text for ev in events if ev.kind == "content")
    assert reasoning == "让我想想他好像在打招呼"
    assert content == "你好呀。"
    for ev in events:
        if ev.kind == "content":
            assert ev.text not in ("让我想想", "他好像在打招呼")


async def test_tool_calls_aggregated_at_finish():
    def handler(kwargs):
        async def gen():
            yield FakeStreamChunk(delta=FakeDelta(tool_calls=[FakeToolCall(0, id="call_1", name="draft_agent_task")]))
            yield FakeStreamChunk(delta=FakeDelta(tool_calls=[FakeToolCall(0, arguments='{"task_type"')]))
            yield FakeStreamChunk(delta=FakeDelta(tool_calls=[FakeToolCall(0, arguments=":1}")]))
            yield FakeStreamChunk(finish_reason="tool_calls")
        return gen()

    llm = _make_client(handler)
    events = [ev async for ev in llm.chat_stream(ChatRequest(messages=[{"role": "user", "content": "发任务"}]))]
    finish = events[-1]
    assert finish.kind == "finish"
    assert finish.tool_calls == [
        {"type": "function", "id": "call_1", "function": {"name": "draft_agent_task", "arguments": '{"task_type":1}'}}
    ]


async def test_usage_via_final_chunk_reaches_finish():
    def handler(kwargs):
        assert kwargs.get("stream_options") == {"include_usage": True}

        async def gen():
            yield FakeStreamChunk(delta=FakeDelta(content="嗯"))
            yield FakeStreamChunk(usage=type("U", (), {"prompt_tokens": 900, "completion_tokens": 7, "total_tokens": 907, "prompt_tokens_details": type("D", (), {"cached_tokens": 640})()})())
        return gen()

    llm = _make_client(handler)
    events = [ev async for ev in llm.chat_stream(ChatRequest(messages=[{"role": "user", "content": "hi"}]))]
    usage_events = [ev for ev in events if ev.kind == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage == {
        "prompt_tokens": 900, "completion_tokens": 7, "total_tokens": 907, "cached_tokens": 640,
    }
    assert events[-1].kind == "finish"
    assert events[-1].usage["cached_tokens"] == 640


async def test_split_meta_events_forward_usage_and_finish():
    """split_meta_events 转发 usage/finish（编排器 done 事件依赖）；reasoning 丢弃。"""

    def handler(kwargs):
        async def gen():
            yield FakeStreamChunk(delta=FakeDelta(content='{"emo":"微笑","fav":1,"act":null}\n'))
            yield FakeStreamChunk(delta=FakeDelta(content="见到你真高兴。"))
            yield FakeStreamChunk(usage=type("U", (), {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14, "prompt_tokens_details": None})())
        return gen()

    llm = _make_client(handler)
    meta, events = await split_meta_events(
        llm.chat_stream(ChatRequest(messages=[{"role": "user", "content": "hi"}])),
        ["普通", "微笑"],
    )
    assert meta.emotion == "微笑"
    kinds = []
    body = ""
    async for ev in events:
        kinds.append(ev.kind)
        if ev.kind == "content":
            body += ev.text
    assert body == "见到你真高兴。"
    assert "usage" in kinds and "finish" in kinds
    assert "reasoning" not in kinds
