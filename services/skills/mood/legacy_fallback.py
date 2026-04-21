"""
流式/工具不可用时的正文末尾 mood JSON 兜底（已弃用主路径）。

仅当环境变量 ``CFN_ENABLE_MOOD_TEXT_FALLBACK`` 为 ``1`` / ``true`` / ``yes`` 时启用，
由 ``parse_mood_node`` 与 ``_ask_stream_legacy`` 调用 ``strip_trailing_mood_json``。
"""

from __future__ import annotations

import os

from services.npc_mood_agent import strip_trailing_mood_json

__all__ = ["is_mood_text_fallback_enabled", "strip_trailing_mood_json"]


def is_mood_text_fallback_enabled() -> bool:
    v = (os.environ.get("CFN_ENABLE_MOOD_TEXT_FALLBACK") or "").strip().lower()
    return v in ("1", "true", "yes")
