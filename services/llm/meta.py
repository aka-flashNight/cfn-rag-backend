"""meta 行协议：解析器 + 校验 + prompt 说明文本生成。

对应 docs/v3-developer/02-LLM接入层.md §4。聊天主 Agent 的响应第一行是
单行紧凑 JSON（情绪/好感/委派指令），第二行起为 NPC 台词正文。
解析器全容错：解析失败、缺字段、类型错误一律按「无 meta」处理，绝不丢正文。
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Sequence

from services.llm.client import StreamEvent

# act 合法 kind（01 §3 / 02 §4.1）
ACT_KINDS = ("task_draft", "task_update", "task_confirm", "task_cancel", "search")

# 首行缓冲上限：超过 512 字符仍无换行即视为无 meta
_MAX_META_LINE_CHARS = 512

DEFAULT_EMOTION = "普通"
_FAV_MIN, _FAV_MAX = -5, 5
_DIRECTION_MAX_CHARS = 60
_HINT_MAX_CHARS = 30
_QUERY_MAX_CHARS = 40


@dataclass(frozen=True)
class MetaAct:
    """委派指令。kind ∈ ACT_KINDS；task_draft 需 direction（任务大方向）。"""

    kind: str
    direction: str = ""
    reward_hint: str = ""
    note: str = ""
    query: str = ""


@dataclass(frozen=True)
class Meta:
    """本轮 meta：情绪、好感变化、委派指令。"""

    emotion: str = DEFAULT_EMOTION
    favorability_change: int = 0
    act: MetaAct | None = None


def default_emotion_for(npc_emotions: Sequence[str]) -> str:
    """NPC 情绪回退值：优先「普通」，否则列表第一个，列表为空回「普通」。"""
    if npc_emotions and DEFAULT_EMOTION in npc_emotions:
        return DEFAULT_EMOTION
    if npc_emotions:
        return npc_emotions[0]
    return DEFAULT_EMOTION


def _clip(text: str, limit: int) -> str:
    return text.strip()[:limit]


def parse_meta_obj(obj: object, npc_emotions: Sequence[str]) -> Meta | None:
    """校验并规范化 meta JSON 对象；整体不是 dict 时返回 None（按无 meta 处理）。"""
    if not isinstance(obj, dict):
        return None

    # emo：非法 → 回退默认情绪
    emo_raw = obj.get("emo")
    emotion = (
        emo_raw.strip()
        if isinstance(emo_raw, str) and emo_raw.strip() in npc_emotions
        else default_emotion_for(npc_emotions)
    )

    # fav：整数，clamp 到 [-5, 5]；缺失/非法 → 0
    fav_raw = obj.get("fav", 0)
    try:
        fav = max(_FAV_MIN, min(_FAV_MAX, int(fav_raw)))
    except (TypeError, ValueError):
        fav = 0

    # act：非法形态一律置 None（保住 emo/fav，不丢整个 meta）
    act_raw = obj.get("act")
    act: MetaAct | None = None
    if isinstance(act_raw, dict):
        kind = str(act_raw.get("kind") or "").strip()
        if kind == "task_draft":
            direction = act_raw.get("direction")
            if isinstance(direction, str) and direction.strip():
                act = MetaAct(
                    kind=kind,
                    direction=_clip(direction, _DIRECTION_MAX_CHARS),
                    reward_hint=_clip(str(act_raw.get("reward_hint") or ""), _HINT_MAX_CHARS)
                    if isinstance(act_raw.get("reward_hint"), str)
                    else "",
                    note=_clip(str(act_raw.get("note") or ""), _HINT_MAX_CHARS)
                    if isinstance(act_raw.get("note"), str)
                    else "",
                )
        elif kind == "task_update":
            note = act_raw.get("note")
            act = MetaAct(
                kind=kind,
                note=_clip(note, _HINT_MAX_CHARS) if isinstance(note, str) else "",
            )
        elif kind in ("task_confirm", "task_cancel"):
            act = MetaAct(kind=kind)
        elif kind == "search":
            query = act_raw.get("query")
            if isinstance(query, str) and query.strip():
                act = MetaAct(kind=kind, query=_clip(query, _QUERY_MAX_CHARS))
    return Meta(emotion=emotion, favorability_change=fav, act=act)


def _try_parse_first_line(line: str) -> dict | None:
    """首行是单行紧凑 JSON 对象则返回 dict，否则 None（多行 JSON 冒充必然失败）。"""
    stripped = line.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _passes_meta_schema(obj: dict, npc_emotions: Sequence[str]) -> bool:
    """首行是否被认定为 meta 行：emo/fav/act 三键至少出现一个（防台词恰似 JSON）。"""
    return any(key in obj for key in ("emo", "fav", "act"))


def _forwardable(kind: str) -> bool:
    """聊天主 Agent 只转发 content / usage / finish；reasoning 一律丢弃（修 A7）。"""
    return kind in ("content", "usage", "finish")


async def split_meta_events(
    stream: AsyncIterator[StreamEvent],
    npc_emotions: Sequence[str],
) -> tuple[Meta, AsyncIterator[StreamEvent]]:
    """缓冲流的开头直到凑齐第一行（\\n）或超过 512 字符，切出 meta 与事件流。

    - 首非空字符是 '{' 且首行 json.loads 成功且通过 schema 校验 → (Meta, 剩余事件流)
    - 否则 → (默认 Meta, 已缓冲文本作为 content 事件原样还给事件流)
    - 剩余事件流只含 content / usage / finish（reasoning 丢弃）。
    """
    buffer = ""
    held_events: list[StreamEvent] = []  # 在 meta 判定前到达的 usage/finish 事件

    async def _rest(leftover: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
        for ev in leftover:
            yield ev
        async for ev in stream:
            if _forwardable(ev.kind):
                yield ev

    def _finish(leftover: list[StreamEvent], meta: Meta) -> tuple[Meta, AsyncIterator[StreamEvent]]:
        return meta, _rest(leftover)

    async for ev in stream:
        if ev.kind == "content":
            buffer += ev.text
            nl = buffer.find("\n")
            if nl < 0:
                if len(buffer) > _MAX_META_LINE_CHARS:
                    # 超过 512 字符仍无换行 → 无 meta，缓冲原样还给正文
                    return _finish([StreamEvent(kind="content", text=buffer)], Meta())
                continue
            first_line, rest_text = buffer[:nl], buffer[nl + 1 :]
            obj = _try_parse_first_line(first_line)
            if obj is not None and _passes_meta_schema(obj, npc_emotions):
                meta = parse_meta_obj(obj, npc_emotions) or Meta()
                leftover = [StreamEvent(kind="content", text=rest_text)] if rest_text else []
                return _finish(leftover, meta)
            return _finish([StreamEvent(kind="content", text=buffer)], Meta())
        if ev.kind == "reasoning":
            continue  # 思考内容绝不进正文、不参与 meta 判定
        # usage / finish：先扣住，等 meta 判定后随事件流交还
        held_events.append(ev)

    # 流结束仍未凑齐换行：整段 buffer 尝试按首行解析（流可能只有 meta 无正文）
    if buffer.strip():
        obj = _try_parse_first_line(buffer)
        if obj is not None and _passes_meta_schema(obj, npc_emotions):
            return _finish(held_events, parse_meta_obj(obj, npc_emotions) or Meta())
        return _finish([*held_events, StreamEvent(kind="content", text=buffer)], Meta())
    return _finish(held_events, Meta())


async def split_meta(
    stream: AsyncIterator[StreamEvent],
    npc_emotions: Sequence[str],
) -> tuple[Meta, AsyncIterator[str]]:
    """split_meta_events 的纯文本版：返回 (Meta, 正文增量迭代器)。

    注意：usage/finish 事件在纯文本迭代器中被丢弃，需要 usage 的调用方
    （聊天主 Agent 的 done 事件）请改用 split_meta_events。
    """
    meta, events = await split_meta_events(stream, npc_emotions)

    async def _texts() -> AsyncIterator[str]:
        async for ev in events:
            if ev.kind == "content":
                yield ev.text

    return meta, _texts()


def meta_prompt_block(npc_emotions: Sequence[str]) -> str:
    """生成写进聊天尾部指令的 meta 行协议说明（02 §4.3）。"""
    emotions = list(npc_emotions) or [DEFAULT_EMOTION]
    emo_list = "、".join(emotions)
    lines = [
        "【输出格式（最高优先级）】",
        "你的回复第一行必须是一个单行 JSON（不要解释、不要用代码块包裹），第二行起为台词正文。格式示例：",
        '{"emo":"微笑","fav":1,"act":null}',
        "诶，是你啊。今天有什么想聊的？",
        "",
        "字段规则：",
        f"1. emo：本轮情绪，必须从列表中选择：{emo_list}。",
        f"2. fav：本条消息对玩家好感度的变化，整数，范围 -5~5，无变化写 0。",
        "3. act：本轮委派指令，取以下之一：",
        '   - null：纯聊天，不委派任何事。',
        '   - {"kind":"task_draft","direction":"<任务大方向>","reward_hint":"<奖励方向>","note":"<备注>"}：玩家明确要求委托/工作/任务时使用。direction 必须具体到可执行（任务类型/目标物/目标关卡至少其一明确），不超过 60 字。**拟定了方向就必须委派，不要自己编任务细节**。',
        '   - {"kind":"task_update","note":"<玩家的新条件>"}：已存在待确认草案且玩家在谈条件/要求修改时使用。',
        '   - {"kind":"task_confirm"}：存在待确认草案且玩家明确接受时使用。',
        '   - {"kind":"task_cancel"}：存在待确认草案且玩家明确拒绝/放弃时使用。',
        '   - {"kind":"search","query":"<查询要点>"}：需要查证设定/物品/关卡信息才能回答时使用，query 不超过 40 字。',
        "",
        "重要提醒：",
        "- 存在待确认草案时：玩家接受 → task_confirm；玩家拒绝 → task_cancel；玩家谈条件 → task_update；玩家岔开话题 → 保持 null（草案会保留若干回合）。",
        "- **草案只是拟定，不是已发布**。确认前禁止说「任务已发布」。",
        "- 委派 task_draft / task_update / search 时，第一段正文只写 1~2 句过渡话（如「我想想给你安排点什么……」），不要写任何具体任务内容或数字。",
        "- 第一行 JSON 之后的所有内容都是 NPC 台词本身，不要再输出任何 JSON。",
    ]
    return "\n".join(lines)
