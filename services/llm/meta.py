"""meta 行协议：解析器 + 校验 + prompt 说明文本生成。

对应 docs/v3-developer/02-LLM接入层.md §4。聊天主 Agent 的响应第一行是
单行紧凑 JSON（情绪/好感），第二行起为 NPC 台词正文。模型漏掉那个换行时按
JSON 前缀切一刀（实测 glm-5.3-flash 的发布轮会写成同一行）。
解析器全容错：解析失败、缺字段、类型错误一律按「无 meta」处理，绝不丢正文。

v3 再次重构（路由工具化）：meta 行只剩情绪与好感两个字段。委派意图不再走 meta——
聊天主 Agent 通过 services/tools/route/ 下的路由声明工具把本轮交给后台系统，
路由判据写在工具描述里，本轮不输出台词。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from services.llm.client import StreamEvent

logger = logging.getLogger(__name__)

# 首行缓冲上限：超过 512 字符仍无换行即视为无 meta
_MAX_META_LINE_CHARS = 512

DEFAULT_EMOTION = "普通"
_FAV_MIN, _FAV_MAX = -5, 5


@dataclass(frozen=True)
class Meta:
    """本轮 meta：情绪与好感变化。"""

    emotion: str = DEFAULT_EMOTION
    favorability_change: int = 0


def default_emotion_for(npc_emotions: Sequence[str]) -> str:
    """NPC 情绪回退值：优先「普通」，否则列表第一个，列表为空回「普通」。"""
    if npc_emotions and DEFAULT_EMOTION in npc_emotions:
        return DEFAULT_EMOTION
    if npc_emotions:
        return npc_emotions[0]
    return DEFAULT_EMOTION


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

    return Meta(emotion=emotion, favorability_change=fav)


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
    """首行是否被认定为 meta 行：emo/fav 两键至少出现一个（防台词恰似 JSON）。"""
    return any(key in obj for key in ("emo", "fav"))


def _decode_meta_prefix(text: str) -> tuple[dict, int] | None:
    """text 以合法 meta JSON 开头时返回 (对象, 该 JSON 之后的下标)，否则 None。

    实测有模型把 meta 和台词挤在同一行（`{"emo":"坏笑","fav":0}好，痛快。……`），
    此时没有换行可依据；按 JSON 前缀切一刀才能既取出情绪又不把花括号漏给玩家。
    """
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None
    try:
        obj, idx = json.JSONDecoder().raw_decode(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not _passes_meta_schema(obj, ()):
        return None
    return obj, len(text) - len(stripped) + idx


def _forwardable(kind: str) -> bool:
    """聊天主 Agent 只转发 content / usage / finish；reasoning 一律丢弃（修 A7）。"""
    return kind in ("content", "usage", "finish")


async def split_meta_events(
    stream: AsyncIterator[StreamEvent],
    npc_emotions: Sequence[str],
) -> tuple[Meta, AsyncIterator[StreamEvent]]:
    """缓冲流的开头直到凑齐一个 meta JSON（或换行、或超过 512 字符），切出 meta 与事件流。

    - 首非空字符是 '{' 且 json.loads 成功且通过 schema 校验 → (Meta, 剩余事件流)
    - meta JSON 之后紧跟台词（模型漏了换行）→ 按 JSON 前缀切，剩下的仍归正文
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
                hit = _decode_meta_prefix(buffer)
                if hit is not None and buffer[hit[1]:]:
                    # JSON 已完整且后面就是台词（模型漏了换行）；此时继续等换行会把
                    # 花括号连同台词一起当成正文。
                    obj, consumed = hit
                    rest_text = buffer[consumed:]
                    meta = parse_meta_obj(obj, npc_emotions) or Meta()
                    # 正文必须排在 held usage/finish 之前，否则「finish 即终止」的转发会丢正文
                    leftover = [StreamEvent(kind="content", text=rest_text), *held_events]
                    logger.warning("meta 与台词同行（缺换行），已按 JSON 前缀切出: %.80r", buffer)
                    return _finish(leftover, meta)
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
            hit = _decode_meta_prefix(first_line)
            if hit is not None:
                obj, consumed = hit
                tail = first_line[consumed:]
                body = (tail + "\n" + rest_text) if rest_text else tail
                logger.warning("meta 与台词同行（缺换行），已按 JSON 前缀切出: %.80r", first_line)
                meta = parse_meta_obj(obj, npc_emotions) or Meta()
                return _finish([StreamEvent(kind="content", text=body)] if body else [], meta)
            # 观测点：模型尝试发 meta 但格式不合法（warning），或完全没发 meta（debug）
            if first_line.lstrip().startswith("{"):
                logger.warning("meta 首行 JSON 解析失败，整段按正文处理: %.160r", first_line)
            else:
                logger.debug("模型未输出 meta 行，按纯正文处理: %.80r", first_line)
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
        if buffer.lstrip().startswith("{"):
            logger.warning("meta 首行 JSON 解析失败（流结束无换行），按正文处理: %.160r", buffer)
        # 缓冲正文必须排在 held usage/finish 之前，否则下游「finish 即终止」的转发会丢正文
        return _finish([StreamEvent(kind="content", text=buffer), *held_events], Meta())
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


def meta_prompt_block(npc_emotions: Sequence[str], *, agent_enabled: bool = True) -> str:
    """生成写进聊天 system 的 meta 行协议说明（02 §4.3）。

    v3 再次重构后本块只讲情绪与好感两个字段，因此对同一 NPC 跨轮完全稳定
    （有无待确认草案不再影响它，前缀缓存命中率随之提高）。路由判据不在这里：
    它们写在 services/tools/route/ 各工具的描述与
    orchestrator.prompts.build_routing_guide 中。

    唯一的分支是 agent_enabled：关闭 Agent 能力时既不下发路由工具、也不注入路由
    说明块，此处再提「或者发一个路由工具调用」就成了悬空指代，会诱导模型凭空
    编一个调用（编排层会忽略它，但这一轮就白跑了）。
    """
    emotions = list(npc_emotions) or [DEFAULT_EMOTION]
    emo_list = "、".join(emotions)
    after_first_line = (
        "第一行之后要么是台词正文，要么是一个路由工具调用——两者只取其一（路由见下方说明）。格式示例："
        if agent_enabled else
        "第一行之后是台词正文。格式示例："
    )
    return "\n".join([
        "【输出格式（最高优先级）】",
        "你的回复第一行必须是一个单行 JSON（不要解释、不要用代码块包裹），写完就换行、"
        "台词从第二行开始。" + after_first_line,
        '{"emo":"微笑","fav":0}',
        "诶，是你啊。今天有什么想聊的？",
        "",
        "字段规则：",
        f"1. emo：本轮情绪，必须从列表中选择：{emo_list}。",
        "2. fav：本条消息对玩家好感度的变化，整数，范围 -5~5，无变化写 0。"
        "日常对话通常无需变化，有情绪起伏/情感变化时可调整好感度。",
        "- 第一行 JSON 之后的所有内容都是 NPC 台词本身，不要再输出任何 JSON。",
    ])
