from __future__ import annotations

import html
import re


_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_WS_RE = re.compile(r"\s+")


def strip_game_markup(text: str | None) -> str:
    """
    去除游戏数据里常见的 HTML/XML 片段与实体（如 &lt;BR&gt;），压缩空白。
    用于物品/关卡描述入库与向量化前的清洗。
    """

    if not text:
        return ""
    t = html.unescape(str(text))
    t = _TAG_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t
