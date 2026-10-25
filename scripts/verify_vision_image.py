#!/usr/bin/env python3
"""vision 带图链路一次性验证（P7 验收用，dev 工具，不进 exe）。

流程：
1. provider 取某 NPC 当前情绪立绘的 data URL（manifest 查表 + bounds 裁剪 + WebP）；
2. 用真实 LLMClient 发一次「带图」与一次「无图」的同 prompt 流式调用；
3. 对比两次 usage.prompt_tokens：带图请求的增量即图片进入请求体并被模型计费的证据。

用法：python scripts/verify_vision_image.py [NPC名] [情绪]
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from services.llm import ChatRequest, LLMClient, LLMConfig  # noqa: E402
from services.portraits import get_portrait_data_url  # noqa: E402


async def _run(client: LLMClient, prompt: str, image_url: str | None) -> dict | None:
    content: str | list[dict] = prompt
    if image_url:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    req = ChatRequest(
        messages=[{"role": "user", "content": content}],
        purpose="chat",
        send_image=bool(image_url),
        max_tokens=100,
    )
    usage = None
    text_parts: list[str] = []
    async for ev in client.chat_stream(req):
        if ev.kind == "content":
            text_parts.append(ev.text)
        elif ev.kind == "usage":
            usage = ev.usage
        elif ev.kind == "finish":
            break  # 消费到 finish 再退出，避免流未关闭的噪音
    print(f"  回复: {''.join(text_parts)[:60]!r}")
    return usage


async def main() -> None:
    npc = sys.argv[1] if len(sys.argv) > 1 else "Andy Law"
    emotion = sys.argv[2] if len(sys.argv) > 2 else "微笑"

    image_url = get_portrait_data_url(npc, emotion)
    if image_url is None:
        print(f"[错误] 未取到立绘（{npc}/{emotion}），检查 CFN_GAME_PROJECT_DIR 与 manifest")
        sys.exit(1)
    print(f"立绘 data URL 长度: {len(image_url)} 字符（WebP base64）")

    client = LLMClient.for_config(LLMConfig())  # 空配置回落 .env 默认（需为 vision 模型）
    print(f"模型: {client.config.model_name}（profile.vision={client.profile.vision}）")

    prompt = "用一句话描述你看到的这张人物立绘的穿着与气质。"
    print("\n[1] 带图请求:")
    usage_img = await _run(client, prompt, image_url)
    print(f"  usage: {usage_img}")

    print("\n[2] 无图请求（同 prompt）:")
    usage_txt = await _run(client, prompt, None)
    print(f"  usage: {usage_txt}")

    if usage_img and usage_txt:
        delta = (usage_img.get("prompt_tokens") or 0) - (usage_txt.get("prompt_tokens") or 0)
        print(f"\nprompt_tokens 增量（即图片 token 计入）: +{delta}")
        if delta > 50:
            print("✅ 图片已进入请求体并被模型处理")
        else:
            print("❌ 增量过小，图片可能未生效")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
