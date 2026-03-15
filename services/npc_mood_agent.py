"""
NPC 情绪与好感度 Agent 工具：工具定义、Function Calling 解析与正文回退解析。

供 GameRAGService 在 RAG 对话中注入 update_npc_mood 工具，并从 LLM 返回的
tool_calls 或正文中解析出好感度变化与情绪标签。后续可在此模块扩展更多 agent 能力。
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Function Calling：update_npc_mood 工具定义
# ---------------------------------------------------------------------------
UPDATE_NPC_MOOD_TOOL = {
    "type": "function",
    "function": {
        "name": "update_npc_mood",
        "description": (
            "在每次以 NPC 身份回复玩家后调用，用于上报本次对话的好感度变化与当前情绪。"
            "好感度变化取值范围为 -5 到 5，常规对话可传 0；情绪必须从当前 NPC 的可用情绪标签中选择。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "favorability_change": {
                    "type": "integer",
                    "description": "本次对话对玩家的好感度变化，范围 -5 到 5。0 表示不变，正数增加好感，负数减少。",
                },
                "emotion": {
                    "type": "string",
                    "description": "当前回复对应的情绪标签，用于立绘展示，必须从系统提供的可用情绪列表中选择。",
                },
            },
            "required": ["favorability_change", "emotion"],
        },
    },
}


def is_tools_unsupported_error(exc: BaseException) -> bool:
    """
    判断是否为「API 不支持 tools/function calling」类错误。
    用于降级：带 tools 请求失败时重试不带 tools，保证远古模型也能正常返回对话。
    """
    raw = getattr(exc, "message", None) or getattr(exc, "body", None) or str(exc)
    if isinstance(raw, dict):
        msg = str(raw.get("error", raw)).lower()
    else:
        msg = str(raw).lower()
    if getattr(exc, "status_code", None) in (400, 422):
        return True
    for kw in ("tool", "function_call", "function call", "not support"):
        if kw in msg:
            return True
    return False


def has_update_npc_mood_tool_call(tool_calls: List[dict]) -> bool:
    """判断 tool_calls 中是否包含 update_npc_mood 调用。"""
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        f = tc.get("function") if tc.get("type") == "function" else tc.get("function")
        if isinstance(f, dict) and f.get("name") == "update_npc_mood":
            return True
    return False


def strip_trailing_tool_call_text(reply_text: str) -> str:
    """
    按双换行分段；若某段内出现任一截断条件（工具调用、tool_calls_list、含关键词的 HTML 注释、
    含关键词的 {}、或直接出现 update_npc_mood/emotion/favorability_change），则从该段起删掉该段及之后全部内容，
    只保留该段之前的段落，避免各种变体漏网。移除前调用方应先对全文做 parse_mood_from_text / strip_trailing_mood_json 以解析工具参数。
    """
    if not reply_text or not reply_text.strip():
        return reply_text
    _TOOL_KEYWORDS = ("update_npc_mood", "emotion", "favorability_change", "tool_calls_list")

    def _segment_has_trigger(seg: str) -> bool:
        if "工具调用" in seg:
            return True
        seg_lower = seg.lower()
        if any(kw in seg_lower for kw in _TOOL_KEYWORDS):
            return True
        for pat in [r"<!---.*?--->", r"<!--.*?-->"]:
            for m in re.finditer(pat, seg, re.IGNORECASE | re.DOTALL):
                if any(kw in m.group(0).lower() for kw in _TOOL_KEYWORDS):
                    return True
        pos = 0
        while True:
            i = seg.find("{", pos)
            if i == -1:
                break
            depth = 0
            j = i
            while j < len(seg):
                if seg[j] == "{":
                    depth += 1
                elif seg[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                break
            if any(kw in seg[i : j + 1].lower() for kw in _TOOL_KEYWORDS):
                return True
            pos = j + 1
        return False

    paragraphs = re.split(r"\n\s*\n", reply_text)
    cut = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if _segment_has_trigger(p):
            cut = i
            break
    return "\n\n".join(paragraphs[:cut]).strip()


def parse_mood_from_text(text: str) -> Tuple[int | None, str | None]:
    """
    在删除前从正文中用正则解析 favorability_change 与 emotion，兼容 =/:、有引号/无引号等。
    返回 (delta, emotion_raw)，未找到则对应为 None。调用方需用 allowed_emotions 校验 emotion。
    """
    if not text or not text.strip():
        return None, None
    delta: int | None = None
    emotion_raw: str | None = None
    for m in re.finditer(
        r"favorability_change\s*[=:]\s*[\"']*(-?\d+)[\"']*",
        text,
        re.IGNORECASE,
    ):
        try:
            n = int(m.group(1))
            delta = max(-5, min(5, n))
        except (TypeError, ValueError):
            pass
    for m in re.finditer(
        r"emotion\s*[=:]\s*[\"']([^\"']*)[\"']|emotion\s*[=:]\s*([^,}\s\)]+)",
        text,
        re.IGNORECASE,
    ):
        em = (m.group(1) or m.group(2) or "").strip()
        if em:
            emotion_raw = em
    return (delta, emotion_raw)


def _delta_emotion_from_obj(obj: dict, allowed_emotions: List[str]) -> Tuple[int, str] | None:
    if not isinstance(obj, dict) or "emotion" not in obj:
        return None
    emo_raw = obj.get("emotion")
    if not isinstance(emo_raw, str) or not emo_raw.strip():
        return None
    try:
        delta = max(-5, min(5, int(obj.get("favorability_change", 0))))
    except (TypeError, ValueError):
        delta = 0
    default_emotion = (
        "普通" if "普通" in allowed_emotions else (allowed_emotions[0] if allowed_emotions else "普通")
    )
    emotion = (
        emo_raw.strip()
        if emo_raw.strip() in allowed_emotions
        else default_emotion
    )
    return delta, emotion


def strip_trailing_mood_json(
    reply_text: str, allowed_emotions: List[str]
) -> Tuple[str, int | None, str | None]:
    """
    若回复末尾是类似 update_npc_mood 的 JSON（模型未走工具而把参数写在内容里），
    则剥离并解析为 (delta, emotion)，返回 (剥离后的回复, delta, emotion)；
    否则返回 (原回复, None, None)。
    """
    if not reply_text or not reply_text.strip():
        return reply_text, None, None
    text = reply_text

    brace_indices = [i for i, c in enumerate(text) if c == "{"]
    for idx in reversed(brace_indices):
        suffix = text[idx:]
        try:
            obj = json.loads(suffix)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        mood = _delta_emotion_from_obj(obj, allowed_emotions)
        if mood is not None:
            cleaned = text[:idx].rstrip()
            return cleaned, mood[0], mood[1]
        args_str = obj.get("arguments")
        if isinstance(args_str, str) and args_str.strip():
            try:
                inner = json.loads(args_str)
                mood = _delta_emotion_from_obj(inner, allowed_emotions) if isinstance(inner, dict) else None
                if mood is not None:
                    cleaned = text[:idx].rstrip()
                    return cleaned, mood[0], mood[1]
            except Exception:
                pass
    return reply_text, None, None


def parse_update_npc_mood_tool_calls(
    tool_calls: List[dict],
    allowed_emotions: List[str],
) -> Tuple[int, str]:
    """
    从 Function Calling 的 tool_calls 中解析并校验 update_npc_mood 的 (好感度变化, 情绪)。
    未找到有效调用时返回 (0, "普通")。
    """
    if not allowed_emotions:
        allowed_emotions = ["普通"]
    default_emotion = "普通" if "普通" in allowed_emotions else allowed_emotions[0]

    delta = 0
    emotion = default_emotion

    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        f = tc.get("function") if tc.get("type") == "function" else None
        if not f or f.get("name") != "update_npc_mood":
            continue
        args_str = f.get("arguments")
        if not args_str:
            break
        try:
            obj = json.loads(args_str)
        except Exception:
            break
        raw_delta = obj.get("favorability_change", 0)
        try:
            delta = int(raw_delta)
        except (TypeError, ValueError):
            delta = 0
        delta = max(-5, min(5, delta))
        emo_raw = obj.get("emotion")
        if isinstance(emo_raw, str) and emo_raw.strip():
            emo_candidate = emo_raw.strip()
            if emo_candidate in allowed_emotions:
                emotion = emo_candidate
        break

    if emotion not in allowed_emotions:
        emotion = default_emotion
    return delta, emotion
