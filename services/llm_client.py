"""
OpenAI 兼容的 LLM 调用客户端：非流式与流式对话，支持多模态（图像）与 Function Calling。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, AsyncIterator, List, Tuple

from openai import AsyncOpenAI

from services.portrait_utils import prepare_portrait_for_ai


async def call_llm(
    *,
    api_key: str,
    api_base: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    image_path: Path | None = None,
    image_description: str | None = None,
    emotion_hint: str | None = None,
    tools: List[dict] | None = None,
) -> Tuple[str, List[dict]]:
    """
    调用任意 OpenAI 兼容的大模型生成回复。
    支持 Function Calling：传入 tools 时，返回 (回复正文, tool_calls 列表)；未传时 tool_calls 为空列表。

    Args:
        api_key: API Key
        api_base: API Base URL
        model_name: 模型名
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        image_path: 可选的图像文件路径，如果提供则会将图像作为多模态输入传给模型
        image_description: 图像描述，用于在提示中说明传入了什么图像
        emotion_hint: 可选的上一轮情绪说明（如「你之前的情绪是「开心」。」），用于连贯与情绪过渡
        tools: 可选的工具定义列表（OpenAI tools 格式），用于 Function Calling

    Returns:
        (content, tool_calls_list)
    """
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    prefix_parts: List[str] = []
    if image_description and image_description.strip():
        prefix_parts.append(image_description.strip() + "。")
    if emotion_hint and emotion_hint.strip():
        prefix_parts.append(emotion_hint.strip())
    prompt_prefix = ("".join(prefix_parts) + "\n\n") if prefix_parts else ""

    if image_path and image_path.is_file():
        portrait_bytes, media_type = prepare_portrait_for_ai(image_path)
        image_data = base64.b64encode(portrait_bytes).decode("utf-8")
        prompt_with_desc = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
        messages = [
            {"role": "system", "content": prompt_with_desc},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    else:
        system_content = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

    kwargs: dict = {"model": model_name, "messages": messages}
    if tools:
        kwargs["tools"] = tools

    completion = await client.chat.completions.create(**kwargs)

    message = completion.choices[0].message
    content = message.content or ""
    if not isinstance(content, str):
        try:
            content = "".join(part.get("text", "") for part in content)  # type: ignore[arg-type]
        except Exception:
            content = str(content)

    tool_calls_raw = getattr(message, "tool_calls", None) or []
    tool_calls_list: List[dict] = []
    for tc in tool_calls_raw:
        if hasattr(tc, "model_dump"):
            tool_calls_list.append(tc.model_dump())
        elif hasattr(tc, "dict"):
            tool_calls_list.append(tc.dict())
        elif isinstance(tc, dict):
            tool_calls_list.append(tc)
        else:
            fn = getattr(tc, "function", None)
            tool_calls_list.append({
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(fn, "name", "") if fn else "",
                    "arguments": getattr(fn, "arguments", "") if fn else "",
                },
            })

    return content, tool_calls_list


async def call_llm_stream(
    *,
    api_key: str,
    api_base: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    image_path: Path | None = None,
    image_description: str | None = None,
    emotion_hint: str | None = None,
    tools: List[dict] | None = None,
) -> AsyncIterator[Tuple[str, Any]]:
    """
    流式调用 LLM：yield ("content", delta_str) 逐片输出正文；
    结束时 yield ("finished", (full_content, tool_calls_list)) 供调用方做后处理。
    """
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    prefix_parts: List[str] = []
    if image_description and image_description.strip():
        prefix_parts.append(image_description.strip() + "。")
    if emotion_hint and emotion_hint.strip():
        prefix_parts.append(emotion_hint.strip())
    prompt_prefix = ("".join(prefix_parts) + "\n\n") if prefix_parts else ""

    if image_path and image_path.is_file():
        portrait_bytes, media_type = prepare_portrait_for_ai(image_path)
        image_data = base64.b64encode(portrait_bytes).decode("utf-8")
        prompt_with_desc = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
        messages = [
            {"role": "system", "content": prompt_with_desc},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        # print(prompt_with_desc);
        # print("————————");
        # print(user_prompt);
    else:
        system_content = f"{prompt_prefix}{system_prompt}" if prompt_prefix else system_prompt
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

    kwargs: dict = {"model": model_name, "messages": messages, "stream": True}
    if tools:
        kwargs["tools"] = tools

    stream = await client.chat.completions.create(**kwargs)
    content_parts: List[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None) and delta.content:
            content_parts.append(delta.content)
            yield ("content", delta.content)
        tool_calls_delta = getattr(delta, "tool_calls", None) or []
        for tc in tool_calls_delta:
            idx = getattr(tc, "index", None)
            if idx is None:
                continue
            if idx not in tool_calls_acc:
                tool_calls_acc[idx] = {"id": getattr(tc, "id", "") or "", "name": "", "arguments": ""}
            fn = getattr(tc, "function", None)
            if fn:
                if getattr(fn, "name", None):
                    tool_calls_acc[idx]["name"] = (tool_calls_acc[idx].get("name") or "") + (fn.name or "")
                if getattr(fn, "arguments", None):
                    tool_calls_acc[idx]["arguments"] = (tool_calls_acc[idx].get("arguments") or "") + (fn.arguments or "")

    full_content = "".join(content_parts)
    tool_calls_list = []
    for i in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[i]
        tool_calls_list.append({
            "type": "function",
            "id": acc.get("id", ""),
            "function": {
                "name": acc.get("name", ""),
                "arguments": acc.get("arguments", ""),
            },
        })
    yield ("finished", (full_content, tool_calls_list))
