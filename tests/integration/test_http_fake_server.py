"""真实 HTTP 假流：SSE 形状、finalize、任务 error、断连取消。

全部离线：Uvicorn 跑在本机临时端口，假流回放 fixture；MemoryStore 用 tmp_path。
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import parse_sse
from tests.integration.fake_interject_server import FakeFixture, build_app
from tests.integration.helpers import run_server


@pytest.mark.asyncio
async def test_http_f01_sse_and_idempotent_finalize(fake_memory):
    app = build_app(forced="F-01")
    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base}/api/game/dialogue-interject", json=fx.daily_request(),
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = parse_sse(response.text)
            assert [name for name, _ in events] == ["status", "line", "line", "line", "done"]
            assert [data["seq"] for name, data in events if name == "line"] == [0, 1, 2]
            session_id = events[-1][1]["session_id"]

            first = await client.post(
                f"{base}/api/game/dialogue-interject/finalize", json=fx.F01_FINALIZE,
            )
            assert first.status_code == 200
            body = first.json()
            assert body == {
                "ok": True,
                "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
                "session_id": session_id,
                "appended": 4,
                "total": 5,
            }

            second = await client.post(
                f"{base}/api/game/dialogue-interject/finalize", json=fx.F01_FINALIZE,
            )
            assert second.json()["appended"] == 0
            assert second.json()["total"] == 5

    ledger = fake_memory.dialogue_ledger_read_sync(session_id)
    assert [row.seq for row in ledger] == [0, 1, 2, 3, 4]
    assert [row.role for row in ledger] == [
        "assistant", "user", "assistant", "assistant", "assistant"
    ]


@pytest.mark.asyncio
async def test_http_f09_task_fake_path_emits_mute_before_line(fake_memory):
    app = build_app(forced="F-09")
    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base}/api/game/dialogue-interject", json=fx.task_request(),
            )
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "mute", "line", "plan", "done"]
    assert events[1][1] == {"mute": True}
    assert events[3][1]["mute"] is True


@pytest.mark.asyncio
async def test_http_f10_zero_valid_and_f11_partial(fake_memory):
    app = build_app(forced="F-10")
    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            zero = await client.post(
                f"{base}/api/game/dialogue-interject",
                json=fx.daily_request(player_input="来个零有效"),
            )
    zero_events = parse_sse(zero.text)
    assert [name for name, _ in zero_events] == ["status", "error"]
    assert zero_events[-1][1]["code"] == "no_valid_lines"

    app = build_app(forced="F-11")
    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            partial = await client.post(
                f"{base}/api/game/dialogue-interject",
                json=fx.daily_request(player_input="来个部分有效"),
            )
    partial_events = parse_sse(partial.text)
    assert [name for name, _ in partial_events] == ["status", "line", "error"]
    assert partial_events[-1][1]["code"] == "stream_failed"
    assert partial_events[-1][1]["valid_lines"] == 1


@pytest.mark.asyncio
async def test_http_bind_stub_is_memory_only(fake_memory):
    app = build_app(forced="F-01")
    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            bound = await client.post(
                f"{base}/api/integration/bind",
                json={"slot_key": "fake_slot", "character": {"name": "甲"}},
            )
            assert bound.status_code == 200
            assert bound.json()["mode"] == "embedded"
            status = (await client.get(f"{base}/api/integration/status")).json()
            assert status["bound_slot"] == "fake_slot"
            assert status["character"]["name"] == "甲"


@pytest.mark.asyncio
async def test_http_client_disconnect_cancels_upstream_and_writes_nothing(fake_memory):
    """真实 HTTP 断连（不是只取消 token）：服务端假流收到 CancelledError。"""
    cancel_event = threading.Event()
    slow = FakeFixture(
        name="SLOW",
        chunks=['{"add":"第一句","emo":"普通"}\n'],
        hold_after_first_s=30.0,
        cancel_event=cancel_event,
    )
    app = build_app({"SLOW": slow}, forced="SLOW")

    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream(
                "POST",
                f"{base}/api/game/dialogue-interject",
                json=fx.daily_request(),
                # 明确要求服务端不复用连接：断连后 uvicorn 立即发 http.disconnect
                headers={"Connection": "close"},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data:") and "第一句" in line:
                        break
            # 离开 stream 上下文即关闭连接 → Starlette/uvicorn 取消响应任务
        assert await asyncio.to_thread(cancel_event.wait, 10.0), "服务端未收到取消"

    info = fake_memory.dialogue_session_get_sync(fx.DAILY_DIALOGUE_ID)
    assert info is not None
    ledger = fake_memory.dialogue_ledger_read_sync(info.session_id)
    assert [row.seq for row in ledger] == [0]  # 只种入请求 played；AI 台词未落库
    assert slow.calls == ["primary"]


@pytest.mark.asyncio
async def test_http_f04_reference_continuation_request_streams(fake_memory):
    """F-04：请求带 reference_continuation，假流正常出首句。"""
    app = build_app(forced="F-04")
    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base}/api/game/dialogue-interject", json=fx.daily_turn_b_request(),
            )
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "line", "line", "done"]
    assert events[1][1]["text"] == "任务是吧，我这儿正好有个跑腿的活儿。"


@pytest.mark.asyncio
async def test_http_f03_shape_streams_final_dialogue_after_draft(fake_memory):
    """F-03 形状：只回放草案结果后的最终对话，不实际建草案（工具链见单测）。"""
    app = build_app(forced="F-03")
    payload = fx.daily_request(player_input="有没有什么活儿给我？")
    with run_server(app) as base:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base}/api/game/dialogue-interject", json=payload,
            )
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "line", "line", "line", "done"]
    assert events[1][1]["text"] == "行，那我直说了。"
    assert events[3][1]["kind"] == "closing"
