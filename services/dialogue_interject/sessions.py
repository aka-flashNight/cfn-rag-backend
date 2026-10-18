"""插话 session：按 dialogue_instance_id 建/复用、角色映射、幂等 finalize。

对应《协议》§6 与实施指引 §9。
- 一组对话一个 session；首次插入建，后续复用；日常重播换新 id＝新 session。
- session 只记“真正播过”的行：请求 `played` 只种入历史，本轮 `$PC` 由 finalize 追加。
- 角色映射（协议 6.2）：当前 NPC/ai → assistant；玩家 → user；其他 NPC → user
  且正文写成 `【NPC名：台词】`。
- finalize.transcript 是截至当时的全量已播行；按 seq 幂等对账，重复调用不重复写。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from services.dialogue_interject.schemas import (
    FinalizeRequest,
    FinalizeResponse,
    InterjectRequest,
    PlayedLine,
)
from services.integration.state import get_integration_state
from services.memory.store import MemoryStore, get_memory_store

logger = logging.getLogger(__name__)

_TASK_KIND_LABEL = {"task_pre": "任务前", "task_post": "任务后"}


@dataclass(frozen=True)
class SessionBinding:
    """一次插话请求对应的 session 绑定结果。"""

    dialogue_instance_id: str
    session_id: str
    title: str
    created: bool


def _now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def default_session_title(request: InterjectRequest) -> str:
    """标题建议（协议 6.3）：日常 `日常插话 · NPC · 时间`；任务带任务名与前后。"""
    stamp = _now_label()
    if request.mode == "daily":
        return f"日常插话 · {request.npc} · {stamp}"
    label = _TASK_KIND_LABEL.get(request.dialogue.kind, "任务")
    title = ""
    context = request.task_context
    if context is not None:
        for summary in (context.upcoming, context.completed):
            if summary is not None and summary.title.strip():
                title = summary.title.strip()
                break
    subject = title or request.npc
    return f"任务插话 · {subject} · {label} · {stamp}"


def map_played_line(line: PlayedLine) -> dict | None:
    """游戏已播行 → chat_history 行（协议 6.2 的角色映射）。

    返回 None＝该行无正文，不落库。
    """
    text = line.text.replace("\r", "").strip()
    if not text:
        return None
    if line.role in ("npc", "ai"):
        role = "assistant"
        content = text
    elif line.role == "player":
        role = "user"
        content = text
    else:  # other_npc
        role = "user"
        name = line.name.strip() or "其他NPC"
        content = f"【{name}：{text}】"
    return {"seq": line.seq, "role": role, "content": content}


def _map_transcript(lines: list[PlayedLine]) -> list[dict]:
    rows: list[dict] = []
    for line in lines:
        row = map_played_line(line)
        if row is not None:
            rows.append(row)
    return rows


async def get_session(
    dialogue_instance_id: str, store: MemoryStore | None = None,
) -> tuple[str, str] | None:
    """取 (session_id, title)；未建立返回 None。"""
    memory = store or get_memory_store()
    info = await memory.dialogue_session_get(dialogue_instance_id)
    return (info.session_id, info.title) if info is not None else None


async def ensure_session(
    request: InterjectRequest, store: MemoryStore | None = None,
) -> SessionBinding:
    """建/取本组 session，并把请求 `played` 按 seq 幂等种入历史。

    注意：本轮 `$PC` 由 `player_input` 表示，**不**在这里写入（协议 5.1 / 6.2）。
    """
    memory = store or get_memory_store()
    title = request.session_title or default_session_title(request)
    slot_key = get_integration_state().slot_key
    info, created = await memory.dialogue_session_ensure(
        request.dialogue_instance_id, request.npc, title, slot_key,
    )
    rows = _map_transcript(request.dialogue.played)
    if rows:
        appended = await memory.dialogue_played_append(info.session_id, rows)
        if appended:
            logger.info(
                "[interject] session=%s 种入已播 %d 行 seq=%s",
                info.session_id, len(appended), appended,
            )
    logger.info(
        "[interject] session=%s dialogue=%s created=%s mode=%s kind=%s",
        info.session_id, request.dialogue_instance_id, created,
        request.mode, request.dialogue.kind,
    )
    return SessionBinding(
        dialogue_instance_id=request.dialogue_instance_id,
        session_id=info.session_id,
        title=info.title,
        created=created,
    )


def _npc_from_transcript(lines: list[PlayedLine]) -> str:
    for line in lines:
        if line.role in ("npc", "ai", "other_npc") and line.name.strip():
            return line.name.strip()
    return ""


async def finalize(
    request: FinalizeRequest, store: MemoryStore | None = None,
) -> FinalizeResponse:
    """以 Host 的已播 transcript 幂等收尾。

    - 未播内容不会出现在 transcript 中，后端也没有其他写入路径，因此天然不落库。
    - 重复调用：同 seq 同内容跳过；内容被修正时改原消息，不重复插入。
    - session 尚不存在（请求轮失败/进程重启后没落过库）时按 transcript 补建。
    """
    memory = store or get_memory_store()
    info = await memory.dialogue_session_get(request.dialogue_instance_id)
    if info is None:
        npc = _npc_from_transcript(request.transcript)
        if not npc:
            logger.warning(
                "[interject] finalize 未知 session 且 transcript 为空：dialogue=%s",
                request.dialogue_instance_id,
            )
            return FinalizeResponse(
                ok=False,
                dialogue_instance_id=request.dialogue_instance_id,
                session_id="",
            )
        info, created = await memory.dialogue_session_ensure(
            request.dialogue_instance_id,
            npc,
            f"对话插话 · {npc} · {_now_label()}",
            get_integration_state().slot_key,
        )
        logger.warning(
            "[interject] finalize 补建 session=%s dialogue=%s created=%s",
            info.session_id, request.dialogue_instance_id, created,
        )

    rows = _map_transcript(request.transcript)
    appended = await memory.dialogue_played_append(info.session_id, rows) if rows else []
    ledger = await memory.dialogue_ledger_read(info.session_id)
    logger.info(
        "[interject] finalize session=%s dialogue=%s transcript=%d appended=%d total=%d",
        info.session_id, request.dialogue_instance_id, len(request.transcript),
        len(appended), len(ledger),
    )
    return FinalizeResponse(
        ok=True,
        dialogue_instance_id=request.dialogue_instance_id,
        session_id=info.session_id,
        appended=len(appended),
        total=len(ledger),
    )
