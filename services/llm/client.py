"""LLMClient：统一调用入口（流式/非流式），连接复用、超时重试、参数降级。

对应 docs/v3-developer/02-LLM接入层.md §2/§3。要点：
- AsyncOpenAI 实例按 (api_base, api_key, proxy_url, model) 进程内缓存复用（修 A5）；
- 代理经 httpx 客户端级 proxy 参数传入，禁止改 os.environ（修 E3）；
- 采样参数默认不传（思考模型下失效甚至报错）；
- reasoning_content 一律归入 reasoning 事件，绝不混入 content（修 A7）；
- usage 回收（流式 stream_options.include_usage / 非流式 resp.usage）；
- §3.2 降级链是全项目唯一的降级位置。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

import httpx
from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel, model_validator

from core.config import get_settings
from services.llm.errors import (
    exc_status_code,
    is_image_unsupported_error,
    is_param_error,
    wrap_http_error,
)
from services.llm.profiles import ModelProfile, get_profile

logger = logging.getLogger(__name__)

LLMPurpose = Literal["chat", "subagent", "summary"]

# 连接 10s，读 120s（02 §2.2）
_LLM_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class LLMConfig(BaseModel):
    """一次 LLM 调用的连接配置（前端可按请求覆盖，本地单用户合理）。"""

    api_key: str = ""
    api_base: str = ""
    model_name: str = ""
    proxy_url: str = ""

    def merged_with_settings(self) -> "LLMConfig":
        """空缺字段回落到 core/config 的全局默认。"""
        s = get_settings()
        return LLMConfig(
            api_key=self.api_key or s.llm_api_key,
            api_base=self.api_base or s.llm_api_base,
            model_name=self.model_name or s.llm_model_name,
            proxy_url=self.proxy_url or s.llm_proxy_url,
        )


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None  # 仅后台子 Agent 传
    purpose: LLMPurpose = "chat"
    send_image: bool = True  # purpose != "chat" 时强制 False（图片只进聊天轮）
    max_tokens: int | None = None

    @model_validator(mode="after")
    def _force_no_image_for_non_chat(self) -> "ChatRequest":
        if self.purpose != "chat":
            self.send_image = False
        return self


@dataclass
class ChatResult:
    """非流式调用结果。"""

    content: str = ""
    reasoning: str = ""  # 思考内容（隔离存放，不进正文）
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None


@dataclass
class StreamEvent:
    """流式事件（02 §2.3）。content/reasoning 增量为 text；finish 带聚合 tool_calls。"""

    kind: Literal["content", "reasoning", "tool_calls", "usage", "finish"]
    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None


def normalize_usage(raw: Any) -> dict[str, Any] | None:
    """把 SDK usage 对象规整为 dict；无值返回 None。"""
    if raw is None:
        return None
    get = getattr(raw, "get", None)
    if callable(get):
        data = raw
    else:
        data = {
            "prompt_tokens": getattr(raw, "prompt_tokens", None),
            "completion_tokens": getattr(raw, "completion_tokens", None),
            "total_tokens": getattr(raw, "total_tokens", None),
        }
        details = getattr(raw, "prompt_tokens_details", None)
        if details is not None:
            data["cached_tokens"] = getattr(details, "cached_tokens", None)
        return {k: v for k, v in data.items() if v is not None}
    out: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if data.get(key) is not None:
            out[key] = data[key]
    details = data.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        out["cached_tokens"] = details["cached_tokens"]
    return out or None


def _is_image_part(part: dict[str, Any]) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def messages_have_images(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(part, list) and any(_is_image_part(p) for p in part)
        for m in messages
        if isinstance(m.get("content"), (list, tuple))
        for part in [m["content"]]
    )


def strip_image_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉消息中的 image_url part（文本 part 在前的约定不变）；空 content 兜底为空文本。"""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, (list, tuple)):
            kept = [p for p in content if not _is_image_part(p)]
            if not kept:
                kept = [{"type": "text", "text": ""}]
            out.append({**m, "content": kept})
        else:
            out.append(m)
    return out


# session 级标记：该模型曾返回「图像不支持」，后续回合直接不发图（02 §3.2.c）
_IMAGE_UNSUPPORTED_MODELS: set[str] = set()
_IMAGE_UNSUPPORTED_LOCK = threading.Lock()


def mark_image_unsupported(model_name: str) -> None:
    with _IMAGE_UNSUPPORTED_LOCK:
        _IMAGE_UNSUPPORTED_MODELS.add((model_name or "").strip().lower())


def is_image_unsupported_model(model_name: str) -> bool:
    with _IMAGE_UNSUPPORTED_LOCK:
        return (model_name or "").strip().lower() in _IMAGE_UNSUPPORTED_MODELS


class LLMClient:
    """单个 (config) 维度的 LLM 客户端。用 for_config 获取缓存实例。"""

    def __init__(
        self,
        config: LLMConfig,
        profile: ModelProfile | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config
        self.profile = profile or get_profile(config.model_name)
        self._client = openai_client  # 测试注入用；None 时懒建

    # ------------------------------------------------------------------
    # 构造与缓存
    # ------------------------------------------------------------------

    @classmethod
    def for_config(cls, cfg: LLMConfig) -> "LLMClient":
        cfg = cfg.merged_with_settings()
        key = (cfg.api_base, cfg.api_key, cfg.proxy_url, cfg.model_name)
        cached = _CLIENT_CACHE.get(key)
        if cached is not None:
            return cached
        with _CLIENT_CACHE_LOCK:
            cached = _CLIENT_CACHE.get(key)
            if cached is not None:
                return cached
            client = cls(config=cfg)
            _CLIENT_CACHE[key] = client
            return client

    @property
    def _openai(self) -> AsyncOpenAI:
        if self._client is None:
            cfg = self.config
            kwargs: dict[str, Any] = {
                "api_key": cfg.api_key or "EMPTY",
                "base_url": cfg.api_base or None,
                "timeout": _LLM_TIMEOUT,
                "max_retries": 0,  # 重试由本类自管（网络 1 次 + 降级链）
            }
            if cfg.proxy_url.strip():
                # 代理只作用于本客户端（httpx 客户端级），不改进程环境变量（修 E3）
                kwargs["http_client"] = httpx.AsyncClient(
                    proxy=cfg.proxy_url.strip(), timeout=_LLM_TIMEOUT
                )
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # 请求组装
    # ------------------------------------------------------------------

    def _prepare_messages(self, req: ChatRequest, *, allow_images: bool) -> list[dict[str, Any]]:
        messages = req.messages
        if allow_images and (self.profile.vision is False or is_image_unsupported_model(self.config.model_name)):
            allow_images = False
        if not allow_images:
            messages = strip_image_parts(messages)
        return messages

    def _build_kwargs(
        self,
        req: ChatRequest,
        *,
        stream: bool,
        with_thinking: bool,
        allow_images: bool,
        with_stream_options: bool,
    ) -> dict[str, Any]:
        messages = self._prepare_messages(req, allow_images=allow_images)
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
        }
        if with_thinking and self.profile.thinking_body:
            # 非标准思考控制参数全部经 extra_body 进入请求体
            kwargs["extra_body"] = dict(self.profile.thinking_body)
        if req.tools:
            kwargs["tools"] = req.tools
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if stream and with_stream_options:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _allow_images(self, req: ChatRequest) -> bool:
        return req.purpose == "chat" and req.send_image and self.profile.vision is not False

    # ------------------------------------------------------------------
    # 调用与降级链
    # ------------------------------------------------------------------

    async def chat(self, req: ChatRequest) -> ChatResult:
        """非流式调用（后台子 Agent / 摘要）。"""
        allow_images = self._allow_images(req)
        kwargs = self._build_kwargs(
            req, stream=False, with_thinking=True, allow_images=allow_images,
            with_stream_options=False,
        )
        resp = await self._execute_with_chain(kwargs, has_images=allow_images and messages_have_images(kwargs["messages"]))

        message = resp.choices[0].message
        content = message.content or ""
        if not isinstance(content, str):
            try:
                content = "".join(part.get("text", "") for part in content)  # type: ignore[arg-type]
            except Exception:
                content = str(content)
        reasoning = getattr(message, "reasoning_content", None) or ""
        tool_calls = [
            tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        return ChatResult(
            content=content,
            reasoning=str(reasoning),
            tool_calls=tool_calls,
            usage=normalize_usage(getattr(resp, "usage", None)),
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        """流式调用（聊天主 Agent）。yield StreamEvent；reasoning 单独成事件。"""
        allow_images = self._allow_images(req)
        kwargs = self._build_kwargs(
            req, stream=True, with_thinking=True, allow_images=allow_images,
            with_stream_options=True,
        )
        stream = await self._execute_with_chain(
            kwargs, has_images=allow_images and messages_have_images(kwargs["messages"])
        )
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        last_usage: dict[str, Any] | None = None

        async for chunk in stream:
            usage = normalize_usage(getattr(chunk, "usage", None))
            if usage:
                last_usage = usage
                yield StreamEvent(kind="usage", usage=usage)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.delta
            # reasoning_content / reasoning 一律单独成事件，绝不混入 content（修 A7）
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning:
                yield StreamEvent(kind="reasoning", text=str(reasoning))
            if getattr(delta, "content", None):
                yield StreamEvent(kind="content", text=delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", None)
                if idx is None:
                    continue
                acc = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    acc["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        acc["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        acc["arguments"] += fn.arguments

        aggregated = [
            {
                "type": "function",
                "id": tool_calls_acc[i]["id"],
                "function": {
                    "name": tool_calls_acc[i]["name"],
                    "arguments": tool_calls_acc[i]["arguments"],
                },
            }
            for i in sorted(tool_calls_acc)
        ]
        yield StreamEvent(kind="finish", tool_calls=aggregated or None, usage=last_usage)

    # ------------------------------------------------------------------
    # 底层执行：网络重试 + §3.2 降级链（全项目唯一降级位置）
    # ------------------------------------------------------------------

    async def _execute_with_chain(self, kwargs: dict[str, Any], *, has_images: bool) -> Any:
        """执行一次调用（含网络重试与降级链）；内部异常保持原始 openai 异常，
        向调用方抛出时统一包装为 LLMError 子类。"""
        try:
            return await self._call_once(kwargs, retry_network=True)
        except Exception as first_exc:  # noqa: BLE001 —— 分类后逐一处理
            return await self._degrade(kwargs, first_exc, has_images=has_images)

    async def _call_once(self, kwargs: dict[str, Any], *, retry_network: bool) -> Any:
        """单次真实 API 调用；异常原样抛出（网络错误可选重试 1 次）。"""
        try:
            return await self._openai.chat.completions.create(**kwargs)
        except Exception as exc:
            if retry_network and wrap_http_error(exc).retryable:
                # 网络错误/5xx 重试 1 次（指数退避 1s）；仅首个请求享有，降级重试不再重试
                logger.warning("LLM 网络错误，1s 后重试一次: %s", exc)
                await asyncio.sleep(1.0)
                return await self._openai.chat.completions.create(**kwargs)
            raise

    async def _degrade(self, kwargs: dict[str, Any], first_exc: Exception, *, has_images: bool) -> Any:
        code = exc_status_code(first_exc)

        # 非 400/422：包装后原样上抛（401/403/429/网络重试后仍失败等）
        if code not in (400, 422):
            raise wrap_http_error(first_exc) from first_exc

        # a. 思考参数不被识别 → 剥离思考参数重试（仅 1 次）
        if is_param_error(first_exc) and kwargs.get("extra_body"):
            no_think = {k: v for k, v in kwargs.items() if k != "extra_body"}
            logger.warning("模型 %s 拒绝思考参数，剥离后重试: %s", self.config.model_name, first_exc)
            try:
                return await self._call_once(no_think, retry_network=False)
            except Exception as second_exc:
                # b. 剥离后仍 400/422 → 只留 model/messages/stream[/tools]，最后重试 1 次
                if exc_status_code(second_exc) in (400, 422):
                    bare = self._bare_kwargs(no_think)
                    logger.warning("模型 %s 剥参后仍报错，裸请求重试: %s", self.config.model_name, second_exc)
                    try:
                        return await self._call_once(bare, retry_network=False)
                    except Exception as third_exc:
                        raise wrap_http_error(third_exc) from third_exc
                raise wrap_http_error(second_exc) from second_exc

        # c. 图像不支持 → 去图重试，并给该模型打标记（后续回合直接不发图）
        if is_image_unsupported_error(first_exc) and has_images:
            mark_image_unsupported(self.config.model_name)
            no_img = {**kwargs, "messages": strip_image_parts(kwargs["messages"])}
            logger.warning("模型 %s 不支持图片，去图重试（已标记后续不发图）: %s", self.config.model_name, first_exc)
            try:
                return await self._call_once(no_img, retry_network=False)
            except Exception as second_exc:
                raise wrap_http_error(second_exc) from second_exc

        # d. 其他 400/422 → 直接上抛真实错误（禁止旧式「一律当不支持工具」，修 A1）
        raise wrap_http_error(first_exc) from first_exc

    @staticmethod
    def _bare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """剥离全部非标准参数，只留 model/messages/stream[/tools]。"""
        allowed = {"model", "messages", "stream", "tools"}
        return {k: v for k, v in kwargs.items() if k in allowed}


_CLIENT_CACHE: dict[tuple[str, str, str, str], LLMClient] = {}
_CLIENT_CACHE_LOCK = threading.Lock()
