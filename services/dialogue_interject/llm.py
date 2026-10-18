"""日常插话的最小真 LLM 流（B1）。

- 复用 `services/llm/` 的客户端、配置与降级链；不新增 provider/依赖。
- 只走日常：system 用稳定块（`assemble_daily_blocks`），user 用请求现场
  （`build_user_context`），不读 session 历史。
- 不向模型暴露工具（tools=None），避免半成品任务/委派链。
- 任务模式不进这里；API 在收到 task 且未注入假流时直接回 `task_mode_not_ready`。
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from services.dialogue_interject.context import DailyBlocks, assemble_daily_blocks
from services.dialogue_interject.prompts import build_messages
from services.dialogue_interject.schemas import InterjectRequest
from services.llm import ChatRequest, LLMClient, LLMConfig


logger = logging.getLogger(__name__)


class InterjectLLMUnavailable(RuntimeError):
    """真实 LLM 未配置（无 api_key）。"""

    code = "llm_not_configured"


def resolve_llm_config(llm_config: LLMConfig | None = None) -> LLMConfig:
    """请求不带模型配置：沿用全局 settings（与终端默认一致）。"""
    return (llm_config or LLMConfig()).merged_with_settings()


def llm_is_configured(llm_config: LLMConfig | None = None) -> bool:
    return bool(resolve_llm_config(llm_config).api_key.strip())


async def stream_daily_llm(
    request: InterjectRequest,
    *,
    llm_config: LLMConfig | None = None,
    client: Any = None,
    blocks: DailyBlocks | None = None,
    npc_manager: Any = None,
    game_data: Any = None,
) -> AsyncIterator[str]:
    """把真 LLM 的正文增量转成插话解析器可消费的纯文本流。

    `client` / `blocks` 仅用于离线测试注入；生产路径用全局配置懒建 LLMClient。
    """
    cfg = resolve_llm_config(llm_config)
    if not cfg.api_key.strip():
        raise InterjectLLMUnavailable("未配置 llm_api_key，插话真模型路径不可用")

    if blocks is None:
        blocks = await assemble_daily_blocks(
            request.npc,
            npc_manager=npc_manager,
            game_data=game_data,
            prompt_max=request.limits.line_prompt_max,
        )
    messages = build_messages(request, daily_system=blocks.system_prompt)

    llm = client or LLMClient.for_config(cfg)
    started = time.perf_counter()
    first_content_ms: float | None = None
    content_chars = 0

    logger.info(
        "[interject] LLM 请求 dialogue=%s turn=%s model=%s system_chars=%d user_chars=%d",
        request.dialogue_instance_id, request.turn_id, cfg.model_name,
        len(messages[0]["content"]), len(messages[1]["content"]),
    )
    # 不传 tools：B1 不向模型暴露任何工具
    chat_request = ChatRequest(
        messages=messages,
        tools=None,
        purpose="chat",
        send_image=False,
    )
    async for event in llm.chat_stream(chat_request):
        if event.kind != "content" or not event.text:
            continue
        if first_content_ms is None:
            first_content_ms = (time.perf_counter() - started) * 1000
        content_chars += len(event.text)
        yield event.text

    logger.info(
        "[interject] LLM 完成 dialogue=%s turn=%s first_content_ms=%s chars=%d elapsed_ms=%.0f",
        request.dialogue_instance_id, request.turn_id,
        f"{first_content_ms:.0f}" if first_content_ms is not None else "n/a",
        content_chars, (time.perf_counter() - started) * 1000,
    )
