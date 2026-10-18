"""测试辅助：假流工厂、编排器构造、SSE 解析（不放 fixture，避免 conftest 循环导入）。"""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterable

from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.schemas import InterjectRequest


def make_request(payload: dict) -> InterjectRequest:
    return InterjectRequest.model_validate(payload)


def chunk_stream(chunks: Iterable[str], error: Exception | None = None):
    """构造假流工厂：逐块 yield，可在末尾抛异常模拟断流。"""

    async def _stream(_request: InterjectRequest) -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk
        if error is not None:
            raise error

    return _stream


def make_orchestrator(
    payload: dict,
    *,
    chunks: Iterable[str],
    repair_chunks: Iterable[str] | None = None,
    error: Exception | None = None,
    repair_error: Exception | None = None,
    session_id: str = "session-test",
    npc_emotions=("普通", "微笑", "冷漠"),
    speaker_emotions=None,
) -> InterjectOrchestrator:
    request = make_request(payload)
    return InterjectOrchestrator(
        request,
        session_id=session_id,
        stream_factory=chunk_stream(chunks, error),
        repair_factory=chunk_stream(repair_chunks, repair_error) if repair_chunks is not None else None,
        npc_emotions=npc_emotions,
        speaker_emotions=speaker_emotions,
    )


async def collect_events(orchestrator: InterjectOrchestrator) -> list:
    return [event async for event in orchestrator.run()]


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """解析 SSE 文本为 (event, data) 列表；忽略注释/空帧。"""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if name:
            events.append((name, data if data is not None else {}))
    return events
