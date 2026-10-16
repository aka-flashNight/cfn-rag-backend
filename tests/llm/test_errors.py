"""错误分类测试（02 §3.2 / 修 A1：不再把一切 400/422 当不支持工具）。"""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from services.llm.errors import (
    LLMAuthError,
    LLMNetworkError,
    LLMQuotaError,
    is_image_unsupported_error,
    is_param_error,
    is_tools_unsupported_error,
    wrap_http_error,
)


def _status_err(code: int, message: str, body: dict | None = None) -> APIStatusError:
    request = httpx.Request("POST", "http://unit.test/v1/chat/completions")
    response = httpx.Response(code, json=body or {"error": {"message": message}}, request=request)
    return APIStatusError(message, response=response, body=body or {"error": {"message": message}})


def test_param_error_by_structured_field():
    exc = _status_err(
        400,
        "Unknown parameter",
        body={"error": {"message": "Unknown parameter: thinking", "param": "thinking"}},
    )
    assert is_param_error(exc)


def test_param_error_by_message_keyword():
    exc = _status_err(422, "unsupported parameter: reasoning_effort is not supported by this model")
    assert is_param_error(exc)


def test_generic_400_is_not_param_error():
    """模型名写错 / 上下文超长等普通 400 不算参数错（修 A1）。"""
    exc = _status_err(400, "model not found: glm-99")
    assert not is_param_error(exc)
    exc2 = _status_err(400, "This model's maximum context length is exceeded")
    assert not is_param_error(exc2)


def test_image_unsupported_keywords():
    assert is_image_unsupported_error(_status_err(400, "image_url is not supported"))
    assert is_image_unsupported_error(_status_err(400, "expected text input, got image"))
    assert is_image_unsupported_error(
        _status_err(400, "invalid request", body={"error": {"message": "bad", "param": "image_url"}})
    )


def test_image_error_not_misclassified_as_param():
    exc = _status_err(400, "image_url is not supported by this endpoint")
    assert is_image_unsupported_error(exc)
    assert not is_param_error(exc)


def test_tools_unsupported_requires_keyword_not_bare_400():
    """裸 400（无工具关键词）不再判为不支持工具（修 A1 的核心回归点）。"""
    assert not is_tools_unsupported_error(_status_err(400, "invalid model name"))
    assert not is_tools_unsupported_error(_status_err(400, "max tokens exceeded"))
    msg = "tools / function calling is not supported on this deployment"
    assert is_tools_unsupported_error(_status_err(400, msg))


# --- 中文错误体（2026-10-16 智谱 glm-5.3-flash 实测口径）---------------------
# 国产端点回中文且不点名参数名，特征表只写英文就会漏判、整轮硬失败。

def test_zh_param_error_real_zhipu_wording():
    """实测智谱 400：不点名参数，只说「始终思考，不支持关闭思考」。"""
    exc = _status_err(400, "该模型始终思考，不支持关闭思考；请使用 low、high 或 max")
    assert is_param_error(exc)
    assert not is_tools_unsupported_error(exc)


def test_zh_image_error_content_type_restricted_to_text():
    """纯文本档把 content 类型收窄到 'text'，整条消息不出现 image 字样。"""
    exc = _status_err(400, "messages.content.type 参数非法，取值范围 ['text']")
    assert is_image_unsupported_error(exc)
    assert not is_param_error(exc)


def test_zh_image_and_tools_error():
    assert is_image_unsupported_error(_status_err(400, "该模型不支持图片输入"))
    assert is_tools_unsupported_error(_status_err(400, "该部署不支持工具调用"))
    assert is_tools_unsupported_error(_status_err(400, "无法识别函数调用参数"))


def test_zh_generic_400_is_still_not_param_error():
    """中文但与被剥参数无关的 400 不能算参数错（避免把模型名写错也当降级）。"""
    assert not is_param_error(_status_err(400, "模型不存在，请检查模型名称"))
    assert not is_image_unsupported_error(_status_err(400, "模型不存在，请检查模型名称"))


def test_wrap_auth_and_quota():
    assert isinstance(wrap_http_error(_status_err(401, "invalid api key")), LLMAuthError)
    assert isinstance(wrap_http_error(_status_err(403, "forbidden")), LLMAuthError)
    assert isinstance(wrap_http_error(_status_err(429, "rate limited")), LLMQuotaError)


def test_wrap_network_5xx_and_timeout():
    assert isinstance(wrap_http_error(_status_err(500, "boom")), LLMNetworkError)
    assert wrap_http_error(_status_err(503, "unavailable")).retryable

    request = httpx.Request("POST", "http://unit.test")
    timeout = APITimeoutError(request)
    assert isinstance(wrap_http_error(timeout), LLMNetworkError)
    assert wrap_http_error(timeout).retryable

    conn = APIConnectionError(request=request)
    assert isinstance(wrap_http_error(conn), LLMNetworkError)


def test_wrap_preserves_message():
    err = wrap_http_error(_status_err(400, "超长上下文无法处理"))
    assert "超长上下文" in str(err)
