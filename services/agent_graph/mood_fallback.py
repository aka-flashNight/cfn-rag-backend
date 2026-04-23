"""
Mood parsing legacy fallback（默认关闭）。

仅当环境变量 ``CFN_ENABLE_MOOD_TEXT_FALLBACK`` 为 ``1``/``true``/``yes`` 时启用。
主路径请使用原生流式 tool_calls 协议（见 ``services/agent_graph/nodes.py`` 的 generate_response_stream）。
"""

from __future__ import annotations

import os

from services.npc_mood_agent import strip_trailing_mood_json

__all__ = ["is_mood_text_fallback_enabled", "strip_trailing_mood_json"]


def is_mood_text_fallback_enabled() -> bool:
    v = (os.environ.get("CFN_ENABLE_MOOD_TEXT_FALLBACK") or "").strip().lower()
    return v in ("1", "true", "yes")
