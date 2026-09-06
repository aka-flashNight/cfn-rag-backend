"""LLMClient 降级链测试（02 §6.3，mock 客户端）：
思考参数 400→剥离重试成功；剥离后仍 400→裸请求；图像 400→去图重试且后续不再带图；
其他 400→上抛真实错误；401→不重试；usage 回收。
"""

from __future__ import annotations

import asyncio

import pytest

from services.llm import client as client_module
from services.llm.client import ChatRequest, LLMClient, LLMConfig, is_image_unsupported_model
from services.llm.profiles import ModelProfile

from tests.conftest import FakeOpenAI, FakeResponse, make_status_error


@pytest.fixture(autouse=True)
def _clear_image_marks():
    client_module._IMAGE_UNSUPPORTED_MODELS.clear()
    yield
    client_module._IMAGE_UNSUPPORTED_MODELS.clear()


def _make_client(handler, profile: ModelProfile | None = None) -> tuple[LLMClient, FakeOpenAI]:
    fake = FakeOpenAI(handler)
    config = LLMConfig(api_key="sk-test", api_base="http://unit.test/v1", model_name="test-model")
    return LLMClient(config, profile=profile, openai_client=fake), fake


_THINK_PROFILE = ModelProfile(match="test-model", vision=True, thinking_body={"thinking": {"type": "disabled"}})
_IMG_MESSAGES = [
    {"role": "user", "content": [
        {"type": "text", "text": "看看这个"},
        {"type": "image_url", "image_url": {"url": "data:image/webp;base64,xxxx"}},
    ]},
]


def _has_image_part(messages) -> bool:
    return any(
        isinstance(p, dict) and p.get("type") == "image_url"
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
    )


async def test_thinking_param_error_strips_and_retries():
    """思考参数被拒 → 剥离 extra_body 重试成功（仅 2 次调用）。"""
    calls = []

    def handler(kwargs):
        calls.append(kwargs)
        if "extra_body" in kwargs:
            raise make_status_error(400, "unknown parameter: thinking", body={
                "error": {"message": "Unknown parameter: 'thinking'", "param": "thinking"}
            })
        return FakeResponse(content="好的。")

    llm, fake = _make_client(handler, profile=_THINK_PROFILE)
    result = await llm.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert result.content == "好的。"
    assert len(fake.chat.completions.calls) == 2
    assert "extra_body" not in fake.chat.completions.calls[1]


async def test_strip_fails_then_bare_request_succeeds():
    """剥离后仍 400 → 只留 model/messages/stream[/tools] 的裸请求最后重试。"""
    def handler(kwargs):
        if "extra_body" in kwargs:
            raise make_status_error(400, "unknown parameter: thinking")
        if "max_tokens" in kwargs:
            raise make_status_error(400, "unknown parameter: max_tokens")
        return FakeResponse(content="ok")

    llm, fake = _make_client(handler, profile=_THINK_PROFILE)
    result = await llm.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=100))
    assert result.content == "ok"
    bare = fake.chat.completions.calls[-1]
    assert set(bare.keys()) <= {"model", "messages", "stream", "tools"}


async def test_image_error_strips_image_and_marks_model():
    """图像不支持 → 去图重试成功；该模型后续回合不再发图。"""
    def handler(kwargs):
        if _has_image_part(kwargs["messages"]):
            raise make_status_error(400, "image_url is not supported by this endpoint")
        return FakeResponse(content="看到了。")

    llm, fake = _make_client(handler, profile=_THINK_PROFILE)
    result = await llm.chat(ChatRequest(messages=_IMG_MESSAGES))
    assert result.content == "看到了。"
    assert len(fake.chat.completions.calls) == 2
    second = fake.chat.completions.calls[1]
    assert not _has_image_part(second["messages"])
    # session 级标记生效：后续请求直接不发图
    assert is_image_unsupported_model("test-model")
    await llm.chat(ChatRequest(messages=_IMG_MESSAGES))
    third = fake.chat.completions.calls[2]
    assert not _has_image_part(third["messages"])


async def test_other_400_raises_real_error_no_silent_degrade():
    """其他 400 → 直接上抛真实错误（修 A1：不再静默降级）。"""
    def handler(kwargs):
        raise make_status_error(400, "model `no-such-model` does not exist")

    llm, fake = _make_client(handler, profile=_THINK_PROFILE)
    with pytest.raises(Exception) as excinfo:
        await llm.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert "no-such-model" in str(excinfo.value)
    assert len(fake.chat.completions.calls) == 1  # 不重试


async def test_401_no_retry_raises_auth():
    from services.llm.errors import LLMAuthError

    def handler(kwargs):
        raise make_status_error(401, "invalid api key")

    llm, fake = _make_client(handler)
    with pytest.raises(LLMAuthError):
        await llm.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert len(fake.chat.completions.calls) == 1


async def test_429_raises_quota():
    from services.llm.errors import LLMQuotaError

    llm, _ = _make_client(lambda kwargs: (_ for _ in ()).throw(make_status_error(429, "rate limit")))
    with pytest.raises(LLMQuotaError):
        await llm.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))


async def test_network_error_retries_once(monkeypatch):
    from services.llm.errors import LLMNetworkError

    sleeps: list[float] = []

    async def fake_sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    state = {"n": 0}

    def handler(kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise make_status_error(503, "service unavailable")
        return FakeResponse(content="恢复。")

    llm, fake = _make_client(handler)
    result = await llm.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert result.content == "恢复。"
    assert state["n"] == 2
    assert sleeps == [1.0]


async def test_non_stream_usage_recaptured():
    usage_obj = type("U", (), {"prompt_tokens": 128, "completion_tokens": 32, "total_tokens": 160, "prompt_tokens_details": None})()

    def handler(kwargs):
        return FakeResponse(content="好", usage=usage_obj)

    llm, _ = _make_client(handler)
    result = await llm.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert result.usage == {"prompt_tokens": 128, "completion_tokens": 32, "total_tokens": 160}


async def test_subagent_purpose_forces_no_image():
    """purpose != chat 时强制不带图（即使 send_image=True 且 vision=True）。"""
    def handler(kwargs):
        assert not _has_image_part(kwargs["messages"])
        return FakeResponse(content="工具结果")

    llm, _ = _make_client(handler, profile=_THINK_PROFILE)
    result = await llm.chat(ChatRequest(messages=_IMG_MESSAGES, purpose="subagent"))
    assert result.content == "工具结果"


async def test_non_vision_profile_strips_images_without_call_failure():
    """vision=False 的 Profile（如 deepseek-v4-flash 纯文本档）直接剥图，不浪费调用。"""
    calls = []

    def handler(kwargs):
        calls.append(kwargs)
        return FakeResponse(content="好的")

    llm, _ = _make_client(handler, profile=ModelProfile(match="test-model", vision=False, thinking_body={}))
    await llm.chat(ChatRequest(messages=_IMG_MESSAGES))
    assert len(calls) == 1
    assert not _has_image_part(calls[0]["messages"])


async def test_stream_chain_with_stream_options():
    """流式：include_usage 正常；思考参数报错同样走剥离重试。"""
    async def stream_gen():
        yield type("C", (), {"choices": [], "usage": None})()

    def handler(kwargs):
        if "extra_body" in kwargs:
            raise make_status_error(400, "unknown parameter: thinking")
        if kwargs.get("stream_options") is None:
            raise make_status_error(400, "missing stream_options")
        async def gen():
            yield type("C", (), {"choices": [type("CH", (), {"delta": type("D", (), {"content": "你好", "reasoning_content": None, "tool_calls": None})(), "finish_reason": None})()], "usage": None})()
            yield type("C", (), {"choices": [], "usage": type("U", (), {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "prompt_tokens_details": None})()})()
        return gen()

    llm, fake = _make_client(handler, profile=_THINK_PROFILE)
    events = [ev async for ev in llm.chat_stream(ChatRequest(messages=[{"role": "user", "content": "hi"}]))]
    kinds = [ev.kind for ev in events]
    assert kinds == ["content", "usage", "finish"]
    usage_event = [ev for ev in events if ev.kind == "usage"][0]
    assert usage_event.usage["prompt_tokens"] == 10
    finish = events[-1]
    assert finish.usage["completion_tokens"] == 2
    assert len(fake.chat.completions.calls) == 2
