"""游戏内嵌集成 API（`/api/integration/*`，见开发计划 §1.2-B1-4、§附 B）。

- `bind`：游戏 Host 推送存档投影（幂等；同槽位重复 bind＝投影刷新）；
- `unbind`：解除绑定（回标题/无存档状态）；
- `status`：当前模式与绑定快照（前端/联调查询用）；
- `shutdown`：游戏退出时请求终端进程优雅退出（仅限本进程由 `--embedded` 启动）。

阶段一只落内存态；任务/记忆/NPC 状态的作用域切换在阶段二、三接通。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from schemas.integration_schema import (
    BindRequest,
    BindResponse,
    IntegrationStatusResponse,
    ShutdownResponse,
    UnbindResponse,
)
from services.integration.state import (
    frontend_url_for,
    get_integration_state,
    request_process_shutdown,
)

logger = logging.getLogger(__name__)
router: APIRouter = APIRouter()


@router.post("/bind", response_model=BindResponse, summary="绑定/刷新存档投影")
async def bind(payload: BindRequest) -> BindResponse:
    state = get_integration_state()
    state.bind(
        slot_key=payload.slot_key,
        character=payload.character.model_dump() if payload.character else None,
        progress=payload.progress.model_dump() if payload.progress else None,
    )
    logger.info("[integration] bind slot=%s mode=%s", payload.slot_key, state.mode)
    return BindResponse(mode=state.mode, frontend_url=frontend_url_for(state.mode))


@router.post("/unbind", response_model=UnbindResponse, summary="解除存档绑定")
async def unbind() -> UnbindResponse:
    state = get_integration_state()
    state.unbind()
    logger.info("[integration] unbind mode=%s", state.mode)
    return UnbindResponse(mode=state.mode)


@router.get("/status", response_model=IntegrationStatusResponse, summary="集成状态快照")
async def status() -> IntegrationStatusResponse:
    state = get_integration_state()
    snapshot = state.snapshot()
    return IntegrationStatusResponse(
        mode=snapshot["mode"],
        launch_mode=snapshot["launch_mode"],
        bound_slot=snapshot["bound_slot"],
        frontend_url=frontend_url_for(snapshot["mode"]),
        character=snapshot["character"],
        progress=snapshot["progress"],
    )


@router.post("/shutdown", response_model=ShutdownResponse, summary="请求终端进程退出")
async def shutdown() -> ShutdownResponse:
    # 只允许关闭本进程由 --embedded 启动的实例；用户手动启动的 standalone
    # 即使已被游戏 bind（mode=embedded 口径）也不由游戏退出链处理。
    if get_integration_state().launch_mode != "embedded":
        raise HTTPException(status_code=409, detail="standalone 启动的进程不接受游戏侧退出请求")
    if not request_process_shutdown():
        raise HTTPException(status_code=503, detail="进程未接入退出回调")
    logger.info("[integration] shutdown requested")
    return ShutdownResponse()
