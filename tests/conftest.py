"""pytest 共享夹具：项目根入 sys.path、假 LLM 客户端、假嵌入器。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 假 LLM（openai 客户端形状）
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, content="", tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class FakeChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, prompt=100, completion=20, cached=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        if cached is not None:
            self.prompt_tokens_details = type("D", (), {"cached_tokens": cached})()


class FakeResponse:
    def __init__(self, content="", tool_calls=None, usage=None, reasoning_content=None):
        self.choices = [FakeChoice(FakeMessage(content, tool_calls, reasoning_content))]
        self.usage = usage


class FakeDelta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class FakeStreamChunk:
    def __init__(self, delta=None, usage=None, finish_reason=None):
        self.choices = [type("C", (), {"delta": delta or FakeDelta(), "finish_reason": finish_reason})()]
        self.usage = usage


class FakeToolCall:
    """流式 tool_call delta（聚合用）。"""

    def __init__(self, index, id="", name="", arguments=""):
        self.index = index
        self.id = id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeCompletions:
    """handler(kwargs) -> FakeResponse / 异步生成器（流式）；异常原样抛出。"""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._handler(kwargs)
        if hasattr(result, "__anext__"):
            return result
        return result


class FakeOpenAI:
    def __init__(self, handler):
        self.chat = type("Chat", (), {"completions": FakeCompletions(handler)})()


def make_status_error(status_code: int, message: str, *, body: dict | None = None):
    """构造 openai APIStatusError（带结构化错误体）。"""
    import httpx
    from openai import APIStatusError

    payload = body if body is not None else {"error": {"message": message}}
    request = httpx.Request("POST", "http://unit.test/v1/chat/completions")
    response = httpx.Response(status_code, json=payload, request=request)
    return APIStatusError(message, response=response, body=payload)


# ---------------------------------------------------------------------------
# 假嵌入器（确定性向量，可按文本注册精确向量）
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """确定性嵌入器：registry 命中返回注册向量，否则按文本哈希生成单位向量。"""

    dim = 16

    def __init__(self):
        self.registry: dict[str, np.ndarray] = {}
        self.encode_calls: list[list[str]] = []

    def register(self, text: str, vec: np.ndarray) -> None:
        self.registry[text] = np.asarray(vec, dtype=np.float32)

    def _default_vec(self, text: str) -> np.ndarray:
        # 用内容稳定哈希做种子：进程内 hash() 受 PYTHONHASHSEED 随机化影响，
        # 会让「无关 query 与语料正交」的假设跨进程随机失效（阈值测试 flaky 根源）
        import hashlib

        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        self.encode_calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = self.registry.get(t)
            out[i] = v if v is not None else self._default_vec(t)
        return out

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def warmup(self) -> None:
        pass


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
