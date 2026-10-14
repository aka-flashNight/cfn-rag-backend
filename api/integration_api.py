"""游戏内嵌集成 API（`/api/integration/*`，见开发计划 §1.2-B1-4、§附 B）。

- `bind`：游戏 Host 推送存档投影（幂等；同槽位重复 bind＝投影刷新）；
- `unbind`：解除绑定（回标题/无存档状态）；
- `status`：当前模式与绑定快照（前端/联调查询用）；
- `task-completed`：任务完成事件（阶段四触发一），挂了对话重写标记的重要任务据此
  自动委派后台对话修改子 Agent；同存档同任务只受理一次（幂等在数据层）；
- `shutdown`：游戏退出时请求终端进程优雅退出（仅限本进程由 `--embedded` 启动）。

阶段一只落内存态；任务/记忆/NPC 状态的作用域切换在阶段二、三接通。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.startup import ensure_agent_save_files
from schemas.integration_schema import (
    BindRequest,
    BindResponse,
    IntegrationStatusResponse,
    ShutdownResponse,
    TaskCompletedRequest,
    TaskCompletedResponse,
    UnbindResponse,
)
from services.dialogue.delegation import handle_task_completed
from services.integration.state import (
    frontend_url_for,
    get_integration_state,
    request_process_shutdown,
)
from services.game_progress import derive_progress_stage, get_progress_stage_name
from services.retrieval.refresh import refresh_index_if_stale

logger = logging.getLogger(__name__)
router: APIRouter = APIRouter()


@router.post("/bind", response_model=BindResponse, summary="绑定/刷新存档投影")
async def bind(payload: BindRequest) -> BindResponse:
    state = get_integration_state()
    state.bind(
        slot_key=payload.slot_key,
        character=payload.character.model_dump() if payload.character else None,
        progress=payload.progress.model_dump() if payload.progress else None,
        scene=payload.scene.model_dump() if payload.scene else None,
        player=payload.player,
    )
    # B2-4：按需建槽目录与空文件（best-effort，失败不影响 bind）
    ensure_agent_save_files(payload.slot_key)
    # B2-7：绑定后按新槽作用域检查语料（指纹失配则后台重建，best-effort）
    refresh_index_if_stale()
    logger.info("[integration] bind slot=%s mode=%s", payload.slot_key, state.mode)
    return BindResponse(mode=state.mode, frontend_url=frontend_url_for(state.mode))


@router.post("/unbind", response_model=UnbindResponse, summary="解除存档绑定")
async def unbind() -> UnbindResponse:
    state = get_integration_state()
    state.unbind()
    # B2-7：解绑后语料作用域回共享（指纹失配则后台重建，best-effort）
    refresh_index_if_stale()
    logger.info("[integration] unbind mode=%s", state.mode)
    return UnbindResponse(mode=state.mode)


@router.get("/status", response_model=IntegrationStatusResponse, summary="集成状态快照")
async def status() -> IntegrationStatusResponse:
    state = get_integration_state()
    snapshot = state.snapshot()
    stage = derive_progress_stage(snapshot["progress"], snapshot["player"])
    return IntegrationStatusResponse(
        mode=snapshot["mode"],
        launch_mode=snapshot["launch_mode"],
        bound_slot=snapshot["bound_slot"],
        frontend_url=frontend_url_for(snapshot["mode"]),
        character=snapshot["character"],
        progress=snapshot["progress"],
        scene=snapshot["scene"],
        player=snapshot["player"],
        stage=stage,
        stage_name=get_progress_stage_name(stage),
    )


@router.post(
    "/task-completed",
    response_model=TaskCompletedResponse,
    summary="任务完成事件（阶段四触发一）",
)
async def task_completed(payload: TaskCompletedRequest) -> TaskCompletedResponse:
    """受理任务完成事件：发布时挂了对话重写标记的重要任务→自动委派对话修改。

    委派是后台链路，本接口不等结果；幂等由数据层保证（同存档同任务只受理一次，
    重复推送静默跳过）。槽位与当前绑定不符／未挂标记／频率限制命中都是 accepted=False。
    """
    outcome = await handle_task_completed(
        slot_key=payload.slot_key, task_id=payload.task_id, npc=payload.npc,
    )
    logger.info(
        "[integration] task-completed slot=%s task_id=%s accepted=%s detail=%s",
        payload.slot_key, payload.task_id, outcome.accepted, outcome.detail,
    )
    return TaskCompletedResponse(
        accepted=outcome.accepted, detail=outcome.detail, job_id=outcome.job_id,
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
