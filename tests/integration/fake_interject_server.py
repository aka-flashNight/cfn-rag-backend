"""仅测试用的离线插话 HTTP 服务（绝不进入生产默认路径）。

用途：给游戏窗口做真实 SSE/取消/终态联调，不需要真模型、不消耗 API 配额。
实现：真实 Uvicorn + 真实 HTTP，用 `set_stream_factories` 回放 fixture JSONL；
MemoryStore 指向临时目录，不写生产 `tools/memory.db`。

启动（仓库根目录）：
    .\\venv\\Scripts\\Activate.ps1
    python -m tests.integration.fake_interject_server --port 7088 --fixture auto

fixture 选择：
- `--fixture auto`（默认）：task→F-09；player_input 含「零有效」→F-10；
  含「部分」→F-11；含「北边」→F-02；其余→F-01。
- `--fixture F-01|F-02|F-09|F-10|F-11`：所有请求强制回放该 fixture。
- `--chunk-delay-ms N`：每个 chunk 之间延迟 N 毫秒（便于观察逐句/取消）。

端点：
    POST http://127.0.0.1:7088/api/game/dialogue-interject
    POST http://127.0.0.1:7088/api/game/dialogue-interject/finalize
    GET  http://127.0.0.1:7088/api/health

注意：本服务不实现 `/api/integration/bind`，也不读写游戏存档；插话 session 的
slot_key 为空。游戏窗口联调时如 Host 强依赖 bind，请改用真实后端。
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterable

import uvicorn
from fastapi import FastAPI

from services.dialogue_interject.api import router as interject_router
from services.dialogue_interject.api import set_stream_factories
from services.dialogue_interject.schemas import InterjectRequest
from services.integration.state import get_integration_state
from services.memory.store import MemoryStore, set_memory_store
from schemas.integration_schema import BindRequest, BindResponse
from tests.dialogue_interject import fixtures as fx

DEFAULT_PORT = 7088


@dataclass
class FakeFixture:
    """一个可回放的假流场景。"""

    name: str
    chunks: list[str]
    repair_chunks: list[str] | None = None
    error: Exception | None = None
    repair_error: Exception | None = None
    delay_ms: int = 0
    #: 首块之后挂起多久（秒）；用于验证真实客户端断开→服务端取消
    hold_after_first_s: float = 0.0
    cancel_event: threading.Event | None = None
    calls: list[str] = field(default_factory=list)


def default_fixtures() -> dict[str, FakeFixture]:
    """B1 联调固定场景（与游戏侧 G0-3 使用同一组 fixture）。"""
    return {
        "F-01": FakeFixture("F-01", list(fx.F01_JSONL)),
        "F-02": FakeFixture("F-02", list(fx.F02_JSONL)),
        "F-09": FakeFixture("F-09", list(fx.F09_JSONL)),
        "F-10": FakeFixture(
            "F-10", list(fx.F10_TEXT), repair_chunks=list(fx.F10_TEXT),
        ),
        "F-11": FakeFixture(
            "F-11", list(fx.F11_JSONL), error=RuntimeError("simulated stream break"),
        ),
    }


def select_fixture(
    request: InterjectRequest,
    fixtures: dict[str, FakeFixture],
    forced: str | None = None,
) -> FakeFixture:
    if forced:
        return fixtures[forced]
    if request.mode == "task":
        return fixtures["F-09"]
    text = request.player_input
    if "零有效" in text:
        return fixtures["F-10"]
    if "部分" in text:
        return fixtures["F-11"]
    if "北边" in text:
        return fixtures["F-02"]
    return fixtures["F-01"]


async def _replay(
    fixture: FakeFixture, *, repair: bool, delay_ms: int,
) -> AsyncIterator[str]:
    fixture.calls.append("repair" if repair else "primary")
    chunks: Iterable[str] = (
        fixture.repair_chunks if repair and fixture.repair_chunks is not None else fixture.chunks
    )
    try:
        for index, chunk in enumerate(chunks):
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)
            yield chunk
            if fixture.hold_after_first_s and index == 0:
                await asyncio.sleep(fixture.hold_after_first_s)
        error = fixture.repair_error if repair else fixture.error
        if error is not None:
            raise error
    except asyncio.CancelledError:
        # 真实客户端断开 → Starlette 取消响应任务 → 这里收到取消
        if fixture.cancel_event is not None:
            fixture.cancel_event.set()
        raise


def build_app(
    fixtures: dict[str, FakeFixture] | None = None,
    *,
    forced: str | None = None,
    chunk_delay_ms: int = 0,
) -> FastAPI:
    """构造离线插话 app；同时把假流注入全局端点（仅测试进程）。"""
    fixtures = fixtures or default_fixtures()
    if forced is not None and forced not in fixtures:
        raise ValueError(f"未知 fixture: {forced}")

    def _fixture_for(request: InterjectRequest) -> FakeFixture:
        return select_fixture(request, fixtures, forced)

    async def _primary(request: InterjectRequest) -> AsyncIterator[str]:
        async for chunk in _replay(
            _fixture_for(request), repair=False, delay_ms=chunk_delay_ms,
        ):
            yield chunk

    async def _repair(request: InterjectRequest) -> AsyncIterator[str]:
        async for chunk in _replay(
            _fixture_for(request), repair=True, delay_ms=chunk_delay_ms,
        ):
            yield chunk

    set_stream_factories(_primary, _repair)

    app = FastAPI(title="CFN-RAG Interject Fake Server (test only)")
    app.include_router(interject_router, prefix="/api/game")

    @app.get("/api/health")
    async def health() -> dict:
        snapshot = get_integration_state().snapshot()
        return {
            "status": "ok",
            "app": "CFN-RAG Interject Fake Server (test only)",
            "version": "test",
            "mode": snapshot["mode"],
            "bound_slot": snapshot["bound_slot"],
            "frontend_url": "",
        }

    # 联调便利：只写内存绑定，不建槽目录、不刷新索引、不碰游戏存档
    @app.post("/api/integration/bind", response_model=BindResponse)
    async def bind(payload: BindRequest) -> BindResponse:
        state = get_integration_state()
        state.bind(
            slot_key=payload.slot_key,
            character=payload.character.model_dump() if payload.character else None,
            progress=payload.progress.model_dump() if payload.progress else None,
            scene=payload.scene.model_dump() if payload.scene else None,
            player=payload.player,
        )
        return BindResponse(mode=state.mode, frontend_url="")

    @app.post("/api/integration/unbind")
    async def unbind() -> dict:
        state = get_integration_state()
        state.unbind()
        return {"ok": True, "mode": state.mode}

    @app.get("/api/integration/status")
    async def integration_status() -> dict:
        snapshot = get_integration_state().snapshot()
        return {
            "ok": True,
            **snapshot,
            "frontend_url": "",
            "stage": None,
            "stage_name": None,
        }

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线插话假流服务（仅供联调）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--fixture", default="auto",
        help="auto 或 F-01/F-02/F-09/F-10/F-11（强制所有请求回放同一个）",
    )
    parser.add_argument("--chunk-delay-ms", type=int, default=0)
    parser.add_argument(
        "--data-dir", default="",
        help="插话 session 的临时目录；缺省自动建系统临时目录",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    forced = None if args.fixture == "auto" else args.fixture
    fixtures = default_fixtures()
    if forced is not None and forced not in fixtures:
        raise SystemExit(f"未知 fixture: {forced}；可选 auto/{'/'.join(fixtures)}")

    data_dir = Path(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="cfn-interject-fake-"))
    data_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(data_dir / "memory.db")
    set_memory_store(store)

    app = build_app(fixtures, forced=forced, chunk_delay_ms=args.chunk_delay_ms)
    print(f"[fake-interject] 监听 http://127.0.0.1:{args.port}")
    print(f"[fake-interject] fixture={args.fixture} delay={args.chunk_delay_ms}ms")
    print(f"[fake-interject] 临时数据目录：{data_dir}")
    print(f"[fake-interject] POST /api/game/dialogue-interject")
    print(f"[fake-interject] POST /api/game/dialogue-interject/finalize")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
