"""逐行 JSONL 解析、括号剥离、165 拆分与情绪回落。

对应《协议》§5.3 / §5.4 / §5.5 / §5.6 与实施指引 §7。
原则：宽容解析，不整段打回——非 JSON 行、未知字段、越界编号一律记录后跳过；
正文只做结构处理（括号、字数、情绪），不做内容层面的漏信息校验。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from services.dialogue_interject.schemas import (
    LINE_HARD_MAX,
    RemainingLine,
    SplitMark,
)
from services.llm.meta import DEFAULT_EMOTION, default_emotion_for

#: 括号内容剥离：全角/半角成对括号，重复应用直到稳定（不做嵌套深度解析）
_BRACKET_RE = re.compile(r"[（(][^（）()]*[）)]")
#: 拆分优先级（协议 5.6）：句号 → 感叹/问号/省略/分号 → 逗号/顿号/冒号 → 硬切
_SPLIT_PRIORITIES: tuple[str, ...] = ("。", "！？…；", "，、：")

ParsedKind = Literal["add", "closing", "mute", "nochange", "line", "remove", "end"]


@dataclass(frozen=True)
class ParseIssue:
    """一条被跳过的原始行与原因（用于日志/走查，不进入 SSE）。"""

    raw: str
    reason: str


@dataclass(frozen=True)
class TextSegment:
    """一条正文按 165 硬上限切出的段。"""

    text: str
    emotion: str
    speaker: str
    split: SplitMark | None = None
    interim: bool = False


@dataclass(frozen=True)
class ParsedEntry:
    """一行 JSONL 的规范化结果。

    add/closing/line 带 `segments`（已拆分）；mute/nochange/remove/end 只有标记。
    """

    kind: ParsedKind
    n: int | None = None
    end: int | None = None
    segments: tuple[TextSegment, ...] = ()


@dataclass(frozen=True)
class ParseOptions:
    """解析所需的现场信息：当前 NPC、情绪白名单、原始剩余台词。"""

    mode: Literal["daily", "task"]
    npc: str
    npc_emotions: Sequence[str] = (DEFAULT_EMOTION,)
    #: 其他 NPC 的情绪白名单（键为 NPC 名）；缺失时回落原台词情绪
    speaker_emotions: Mapping[str, Sequence[str]] = field(default_factory=dict)
    #: 任务模式：编号 → 原始剩余台词（说话人/情绪/原文）
    remaining: Mapping[int, RemainingLine] = field(default_factory=dict)
    hard_max: int = LINE_HARD_MAX


# ---------------------------------------------------------------------------
# 文本处理
# ---------------------------------------------------------------------------


def strip_brackets(text: str) -> str:
    """剥离 `（…）` / `(…)` 内容（协议 D-13 的机制闸）；重复到稳定。"""
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _BRACKET_RE.sub("", current)
    return current


def clean_text(text: str) -> str:
    """正文清洗：去 CR、剥括号、去首尾空白。"""
    return strip_brackets(text.replace("\r", "")).strip()


def _longest_prefix(text: str, hard_max: int) -> int:
    """取 ≤hard_max 的最长前缀切点（优先句末标点），返回切点长度。"""
    window = text[:hard_max]
    for puncts in _SPLIT_PRIORITIES:
        cut = max((window.rfind(ch) for ch in puncts), default=-1)
        if cut >= 0:
            return cut + 1
    return hard_max


def split_text(text: str, hard_max: int = LINE_HARD_MAX) -> list[str]:
    """按协议 5.6 拆分：≤硬上限原样；否则按句末优先级切，最后硬切。

    不变量：`"".join(结果) == text`，且每段长度 ≤hard_max。
    """
    if hard_max <= 0:
        raise ValueError("hard_max 必须为正整数")
    if len(text) <= hard_max:
        return [text]
    segments: list[str] = []
    rest = text
    while len(rest) > hard_max:
        cut = _longest_prefix(rest, hard_max)
        if cut <= 0:  # 理论不可达；防御死循环
            cut = hard_max
        segments.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        segments.append(rest)
    return segments


def split_marks(count: int) -> tuple[SplitMark | None, ...]:
    """拆分段数 → 每段的 split 标记；自然单条不标记（协议 5.2）。"""
    if count <= 1:
        return (None,) * count
    return tuple(SplitMark(index=i, count=count) for i in range(count))


def resolve_emotion(
    raw: object,
    *,
    speaker: str,
    options: ParseOptions,
    original_emotion: str | None = None,
) -> str:
    """情绪回落：当前 NPC 非法→默认情绪；其他 NPC 非法→原台词情绪或“普通”。"""
    candidate = raw.strip() if isinstance(raw, str) else ""
    if speaker == options.npc:
        if candidate and candidate in options.npc_emotions:
            return candidate
        return default_emotion_for(options.npc_emotions)

    whitelist = options.speaker_emotions.get(speaker) or ()
    if candidate and candidate in whitelist:
        return candidate
    if original_emotion:
        return original_emotion
    return default_emotion_for(whitelist)


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------


class JSONLStreamParser:
    """按行缓冲的 JSONL 解析器：feed 任意文本增量，返回已完成的解析结果。"""

    def __init__(self, options: ParseOptions) -> None:
        self.options = options
        self.issues: list[ParseIssue] = []
        self._buffer = ""

    def feed(self, chunk: str) -> list[ParsedEntry]:
        if not chunk:
            return []
        self._buffer += chunk
        entries: list[ParsedEntry] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            entries.extend(self._parse_line(line))
        return entries

    def finish(self) -> list[ParsedEntry]:
        """流结束：处理末尾无换行的残行。"""
        if not self._buffer.strip():
            self._buffer = ""
            return []
        line, self._buffer = self._buffer, ""
        return self._parse_line(line)

    # ------------------------------------------------------------------

    def _issue(self, raw: str, reason: str) -> None:
        self.issues.append(ParseIssue(raw=raw[:500], reason=reason))

    def _parse_line(self, line: str) -> list[ParsedEntry]:
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            return []
        # 代码围栏（```json / ```jsonl / ```）整行跳过
        if stripped.startswith("```"):
            return []
        if not (stripped.startswith("{") and stripped.endswith("}")):
            self._issue(stripped, "非 JSON 行")
            return []
        try:
            obj = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            self._issue(stripped, "JSON 解析失败")
            return []
        if not isinstance(obj, dict):
            self._issue(stripped, "JSON 不是对象")
            return []

        if self.options.mode == "daily":
            return self._parse_daily(obj, stripped)
        return self._parse_task(obj, stripped)

    # ------------------------------------------------------------------
    # 日常：只接受 add / closing
    # ------------------------------------------------------------------

    def _parse_daily(self, obj: dict, raw: str) -> list[ParsedEntry]:
        if isinstance(obj.get("add"), str):
            return self._text_entry("add", obj, obj["add"], raw, speaker=self.options.npc)
        if isinstance(obj.get("closing"), str):
            return self._text_entry(
                "closing", obj, obj["closing"], raw, speaker=self.options.npc
            )
        self._issue(raw, "日常模式下无 add/closing 字段")
        return []

    # ------------------------------------------------------------------
    # 任务：mute / add / nochange / line / remove / end
    # ------------------------------------------------------------------

    def _parse_task(self, obj: dict, raw: str) -> list[ParsedEntry]:
        if obj.get("mute") is True:
            return [ParsedEntry(kind="mute")]

        if isinstance(obj.get("add"), str):
            return self._text_entry("add", obj, obj["add"], raw, speaker=self.options.npc)

        if "end" in obj:
            end_raw = obj.get("end")
            if isinstance(end_raw, bool) or not isinstance(end_raw, int):
                self._issue(raw, "end 不是整数")
                return []
            return [ParsedEntry(kind="end", end=end_raw)]

        # 标记类必须先于 line 判定：nochange/remove 可能同时带正文（协议 5.4-6）
        if obj.get("nochange") is True:
            n = self._parse_n(obj, raw)
            if n is None:
                return []
            if isinstance(obj.get("line"), str) and obj["line"].strip():
                self._issue(raw, "nochange 带正文，按 nochange 处理并忽略正文")
            return [ParsedEntry(kind="nochange", n=n)]

        if obj.get("remove") is True:
            n = self._parse_n(obj, raw)
            if n is None:
                return []
            if isinstance(obj.get("line"), str) and obj["line"].strip():
                self._issue(raw, "remove 带正文，按 remove 处理并忽略正文")
            return [ParsedEntry(kind="remove", n=n)]

        if isinstance(obj.get("line"), str):
            n = self._parse_n(obj, raw)
            if n is None:
                return []
            original = self.options.remaining.get(n)
            if original is None:
                self._issue(raw, f"line 的 n={n} 越界（当前剩余 1..{len(self.options.remaining)}）")
                return []
            return self._text_entry(
                "line",
                obj,
                obj["line"],
                raw,
                speaker=original.name or self.options.npc,
                n=n,
                original_emotion=original.emotion,
            )

        self._issue(raw, "任务模式下无可识别字段")
        return []

    def _parse_n(self, obj: dict, raw: str) -> int | None:
        n_raw = obj.get("n")
        if isinstance(n_raw, bool) or not isinstance(n_raw, int):
            self._issue(raw, "n 不是整数")
            return None
        return n_raw

    # ------------------------------------------------------------------

    def _text_entry(
        self,
        kind: Literal["add", "closing", "line"],
        obj: dict,
        text: object,
        raw: str,
        *,
        speaker: str,
        n: int | None = None,
        original_emotion: str | None = None,
    ) -> list[ParsedEntry]:
        if not isinstance(text, str):
            self._issue(raw, "正文不是字符串")
            return []
        cleaned = clean_text(text)
        if not cleaned:
            self._issue(raw, "剥离括号后正文为空")
            return []
        emotion = resolve_emotion(
            obj.get("emo"),
            speaker=speaker,
            options=self.options,
            original_emotion=original_emotion,
        )
        pieces = split_text(cleaned, self.options.hard_max)
        marks = split_marks(len(pieces))
        interim = obj.get("interim") is True
        segments = tuple(
            TextSegment(
                text=piece, emotion=emotion, speaker=speaker, split=mark, interim=interim,
            )
            for piece, mark in zip(pieces, marks)
        )
        return [ParsedEntry(kind=kind, n=n, segments=segments)]
