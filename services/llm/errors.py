"""LLM 错误分类：可降级参数错误 / 图像不支持 / 工具不支持 / 配额 / 网络。

对应 docs/v3-developer/02-LLM接入层.md §3.2 降级链。这是全项目唯一的
LLM 错误分类位置（旧 5 处复制回退矩阵已删除，修 A1/A2/S8）。
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """LLM 调用层错误的基类。message 保留给前端 error 事件的原始信息。"""

    retryable: bool = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMAuthError(LLMError):
    """401/403：鉴权失败，直接上抛，给前端明确提示。"""

    retryable = False


class LLMQuotaError(LLMError):
    """429 / 配额耗尽，直接上抛。"""

    retryable = False


class LLMNetworkError(LLMError):
    """连接失败 / 超时 / 5xx：可重试一次（指数退避 1s）。"""

    retryable = True


class LLMParamError(LLMError):
    """400/422 且错误体命中「未知/非法参数」特征：剥离思考参数后重试。"""

    retryable = False


class LLMUnsupportedImageError(LLMError):
    """错误体命中「图像不支持」特征：去图重试并给该模型打 session 级标记。"""

    retryable = False


class LLMUnsupportedToolsError(LLMError):
    """仅子 Agent 路径：模型不支持 tools/function calling，子 Agent 直接失败。"""

    retryable = False


def exc_status_code(exc: BaseException) -> int | None:
    """从 openai/httpx 异常中取状态码。"""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else None


def exc_message(exc: BaseException) -> str:
    """从异常中提取可读错误信息（含 OpenAI 兼容平台的 error.message）。"""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if body.get("message"):
            return str(body["message"])
        return str(body)
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(exc)


def _lower(exc: BaseException) -> str:
    return exc_message(exc).lower()


# 思考控制参数名（与 profiles.py 的 thinking_body 键保持一致，供参数错误识别）
THINKING_PARAM_NAMES = (
    "thinking",
    "reasoning_effort",
    "enable_thinking",
    "thinking_level",
)

# 「未知/非法参数」特征：结构化字段点名参数名，或消息含 unknown/unsupported/invalid + 参数名
_PARAM_FEATURE_WORDS = ("unknown", "unsupported", "unrecognized", "invalid", "unexpected")


def _error_param_names(exc: BaseException) -> set[str]:
    """从错误体结构化字段中提取被点名的参数名（如 extra_fields / param / path）。"""
    names: set[str] = set()
    body = getattr(exc, "body", None)
    candidates: list[Any] = []
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            candidates.append(err)
        candidates.append(body)
    for cand in candidates:
        for key in ("param", "field", "path"):
            v = cand.get(key)
            if isinstance(v, str) and v:
                names.add(v.split(".")[-1].strip().lower())
        # OpenAI 400 结构化字段：extra_fields / details
        ef = cand.get("extra_fields") or cand.get("details")
        if isinstance(ef, dict):
            names.update(str(k).lower() for k in ef.keys())
        elif isinstance(ef, list):
            for item in ef:
                if isinstance(item, dict) and isinstance(item.get("loc"), list) and item["loc"]:
                    names.add(str(item["loc"][-1]).lower())
    return names


def is_param_error(exc: BaseException) -> bool:
    """400/422 且命中「未知/非法参数」特征（我们发送的思考参数被点名）。"""
    code = exc_status_code(exc)
    if code not in (400, 422):
        return False
    named = _error_param_names(exc)
    if named and any(name in THINKING_PARAM_NAMES for name in named):
        return True
    msg = _lower(exc)
    if not any(w in msg for w in _PARAM_FEATURE_WORDS):
        return False
    return any(p in msg for p in THINKING_PARAM_NAMES)


# 「图像不支持」特征：先查结构化字段，最后才关键词嗅探（S8 已知风险，记录在案）
_IMAGE_KEYWORDS = ("image_url", "image", "vision", "multimodal", "multimodal")


def is_image_unsupported_error(exc: BaseException) -> bool:
    """判断错误是否为「API 不支持 image_url / 多模态图片」类。"""
    code = exc_status_code(exc)
    if code not in (400, 415, 422):
        return False
    msg = _lower(exc)
    if "image_url" in msg:
        return True
    if ("expected" in msg or "expecting" in msg) and "text" in msg and "image" in msg:
        return True
    named = _error_param_names(exc)
    if any("image" in n for n in named):
        return True
    for kw in ("vision", "multimodal", "image"):
        if kw in msg and (
            "not support" in msg
            or "unsupported" in msg
            or "does not support" in msg
            or "do not support" in msg
        ):
            return True
    return False


# 「工具不支持」特征：仅关键词探测（修 A1 的「一刀切判 400/422」）
_TOOLS_KEYWORDS = ("tool", "function_call", "function call", "function calling")


def is_tools_unsupported_error(exc: BaseException) -> bool:
    """判断错误是否为「API 不支持 tools/function calling」类。仅限子 Agent 路径使用。"""
    if is_image_unsupported_error(exc):
        return False
    code = exc_status_code(exc)
    msg = _lower(exc)
    for kw in _TOOLS_KEYWORDS:
        if kw in msg and (
            "not support" in msg
            or "unsupported" in msg
            or "does not support" in msg
            or "do not support" in msg
            or "invalid" in msg
        ):
            return True
    # 部分平台对未知参数 tools 直接报未知参数错（405/501 少见，400 常见）
    if code in (400, 404, 422) and ("tools" in msg) and any(w in msg for w in _PARAM_FEATURE_WORDS):
        return True
    return False


def wrap_http_error(exc: BaseException) -> LLMError:
    """把 openai/httpx 异常映射为 LLMError 子类（网络/5xx → 可重试，其余按状态码）。"""
    import httpx
    from openai import APIConnectionError, APITimeoutError, APIStatusError

    if isinstance(exc, (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.TransportError)):
        return LLMNetworkError(exc_message(exc))
    if isinstance(exc, APIStatusError):
        code = exc_status_code(exc) or 0
        if code in (401, 403):
            return LLMAuthError(exc_message(exc), status_code=code)
        if code == 429:
            return LLMQuotaError(exc_message(exc), status_code=code)
        if code >= 500:
            return LLMNetworkError(exc_message(exc), status_code=code)
        return LLMError(exc_message(exc), status_code=code)
    return LLMError(exc_message(exc))
