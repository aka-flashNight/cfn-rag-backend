"""对话修改委派的汇聚入口（阶段四 B4-1）：三个触发共用同一条链路。

三个触发（开发计划 §4.0.2）：

1. **重要任务**：发布时挂标记 → 游戏推来完成事件（``POST /api/integration/task-completed``）
   → ``handle_task_completed`` 自动委派；
2. **好感度跨档**：只拼提示（``build_relationship_hint``），调不调仍由模型自己决定；
3. **聊天中主动调用**：模型在聊天轮里调用委派工具（``build_delegate_tool``），
   主用途是关键信息与态度/关系重大变化，任务只兜「发布时漏标记」的遗漏。

成本控制全放在初判与频率限制上（§4.0.1「委派上限放宽一点」＋§4.0.4）：

- **硬幂等**：同存档同任务只委派一次，由 ``dialogue_jobs`` 的部分唯一索引在库层挡住
  （「只有首次传有效，后续静默跳过」）；
- **冷却窗**：同 ``(slot_key, npc)`` 在窗内已委派过即 dismissed；
- **每会话上限**：同一聊天会话累计委派到顶即 dismissed；
- **覆盖层无处落盘即不受理**：未绑定存档 / 非 embedded 一律 dismissed（写侧 fail-closed）。

委派是 fire-and-forget：本模块只把材料交给后台 worker，不等结果、不回写聊天链路
（§4.0.1 独立链路）。所有 dismiss 都落 ``dialogue_jobs`` 一行，便于事后核查为什么没改。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from services.llm import LLMConfig
from services.memory.long_term import (
    CHAT_TOOL_SPEECH_REQUIREMENT,
    parse_arguments,
    tool_call_name,
)
from services.memory.store import (
    DIALOGUE_JOB_DELEGATED,
    DIALOGUE_JOB_DISMISSED,
    DIALOGUE_JOB_FAILED,
    DIALOGUE_JOB_JUDGING,
    DIALOGUE_JOB_RECEIVED,
    get_memory_store,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 更改类型（§4.0.4：调用时必须传）
# ---------------------------------------------------------------------------

TRIGGER_RELATIONSHIP = "关系变化"
TRIGGER_ATTITUDE = "态度变化"
TRIGGER_KEY_INFO = "关键信息"
TRIGGER_IMPORTANT_TASK = "重要任务"

#: 关系变化 / 态度变化 / 关键信息由触发二、三产出；重要任务主要由触发一产出
TRIGGER_TYPES = (TRIGGER_RELATIONSHIP, TRIGGER_ATTITUDE, TRIGGER_KEY_INFO, TRIGGER_IMPORTANT_TASK)

# ---------------------------------------------------------------------------
# 频率限制（施工取值：用户口径只说「加频率限制控制成本」，未给具体数值）
# ---------------------------------------------------------------------------

#: 同 (slot_key, npc) 的委派冷却窗（秒）
DIALOGUE_DELEGATE_COOLDOWN_S = 600.0
#: 同一聊天会话累计委派上限（HTTP 事件没有会话，不受此限）
DIALOGUE_DELEGATE_PER_SESSION = 5
#: 好感度跨档提示持续轮数（§4.0.6 用户裁定）
RELATIONSHIP_HINT_ROUNDS = 3
#: 发布那场对话的精确记录最多取多少条（token 预算，施工取值）
PUBLISH_DIALOGUE_LIMIT = 40

#: 委派工具名（聊天侧手拼下发，不进全局注册表——与长期记忆工具同一做法）
DELEGATE_TOOL_NAME = "delegate_dialogue_rewrite"


@dataclass
class DelegationOutcome:
    """一次委派受理的结果。accepted=False 时 detail 是 dismissed/failed 的原因。"""

    accepted: bool
    detail: str = ""
    job_id: str = ""


# ---------------------------------------------------------------------------
# 提示词（触发二）
# ---------------------------------------------------------------------------

def build_relationship_hint(old_level: str, new_level: str) -> str:
    """好感度跨档提示（§4.0.2-2 逐字口径，上升下降都用这一句）。"""
    return (
        f"你的关系从 {old_level} 变为了 {new_level}，"
        "可以考虑调用对话修改的子 Agent，调整你的对话措辞。"
    )


# ---------------------------------------------------------------------------
# 工具（触发三，手拼 OpenAI tool dict）
# ---------------------------------------------------------------------------

_DELEGATE_BASE_DESCRIPTION = (
    f"{CHAT_TOOL_SPEECH_REQUIREMENT}"
    "派一个后台子 Agent 去修改你自己的对话（日常对话与你的任务对话），"
    "把这轮发生的长期影响落到你以后说的话上。"
    "调用时必须给出 change_type（更改类型），reason 写清发生了什么（相关事实），"
    "instruction 写你希望自己的对话怎么变。"
    "只在真的发生了很重要的对话、导致产生很长期的影响时调用；如无必要，不要调用。"
)

#: 任务兜底表述（§4.0.2-3：仅在「保留任务兜底」那一版里出现，条件闸见 build_delegate_tool）
_DELEGATE_TASK_FALLBACK = (
    "当你首次获知某任务已经完成，且该任务非常重要时，"
    "你可以在确保说完聊天对话的同时，主动调用该子 Agent，并给出指示——"
    "此时 change_type 选重要任务，task_id 填已完成委托块里给出的那个任务 id。"
)

_CHANGE_TYPE_DESCRIPTION = "更改类型（必传）"
_CHANGE_TYPE_DESCRIPTION_TASK = "更改类型（必传）。选重要任务时必须同时给出 task_id"


def build_delegate_tool(*, include_task_fallback: bool) -> dict[str, Any]:
    """委派工具（同名不同形）：任务兜底表述按当轮上下文二选一（§4.0.2-3 条件闸）。

    ``include_task_fallback=False`` 时任务相关表述与 ``task_id`` 参数**整段不出现**
    ——与注册闸同一精神：给了才提，不该给就一个字都不提。
    """
    change_types = [*TRIGGER_TYPES] if include_task_fallback else [
        t for t in TRIGGER_TYPES if t != TRIGGER_IMPORTANT_TASK
    ]
    properties: dict[str, Any] = {
        "change_type": {
            "type": "string",
            "enum": change_types,
            "description": (
                _CHANGE_TYPE_DESCRIPTION_TASK if include_task_fallback else _CHANGE_TYPE_DESCRIPTION
            ),
        },
        "reason": {
            "type": "string",
            "description": "发生了什么（相关事实）",
        },
        "instruction": {
            "type": "string",
            "description": "你希望自己的对话怎么变（改写指示，可留空）",
        },
    }
    description = _DELEGATE_BASE_DESCRIPTION
    if include_task_fallback:
        properties["task_id"] = {
            "type": "integer",
            "description": "任务 id（取自已完成委托块，仅更改类型为重要任务时给）",
        }
        description = f"{description}{_DELEGATE_TASK_FALLBACK}"
    return {
        "type": "function",
        "function": {
            "name": DELEGATE_TOOL_NAME,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["change_type", "reason"],
            },
        },
    }


def delegate_tool_or_none(*, include_task_fallback: bool) -> dict[str, Any] | None:
    """注册闸（§4.0.4）：不该注册时返回 None，提示词里也一个字都不提这个工具。

    两条约束：① 后台有对话修改子 Agent 正在执行（避免并发写同一份覆盖层）；
    ② 未绑定存档（覆盖层是按存档数据，没有落点，写侧 fail-closed）。
    """
    from services.dialogue.worker import is_dialogue_agent_running
    from services.integration.state import get_integration_state

    if is_dialogue_agent_running():
        return None
    if get_integration_state().slot_key is None:
        return None
    return build_delegate_tool(include_task_fallback=include_task_fallback)


# ---------------------------------------------------------------------------
# 委派受理（三个触发共用）
# ---------------------------------------------------------------------------

async def submit_delegation(
    *,
    npc_name: str,
    trigger: str,
    reason: str = "",
    instruction: str = "",
    task_id: int | None = None,
    session_id: str = "",
    require_marker: bool = False,
    llm_config: LLMConfig | None = None,
) -> DelegationOutcome:
    """受理一次委派：初判 → 频率限制 → 落 dialogue_jobs → 交给后台 worker。

    ``require_marker=True`` 只用于完成事件那条路径（触发一「看标记自动发起」）：
    任务记录没挂标记就 dismissed。聊天侧兜底调用（触发三）恰恰是为漏标记的任务，
    故不要求标记。
    """
    from services.integration.state import get_integration_state

    npc = (npc_name or "").strip()
    kind = (trigger or "").strip()
    reason = (reason or "").strip()
    instruction = (instruction or "").strip()
    session = (session_id or "").strip()
    state = get_integration_state()
    slot_key = state.slot_key or ""
    store = get_memory_store()

    if kind not in TRIGGER_TYPES:
        return DelegationOutcome(accepted=False, detail=f"未知的更改类型：{kind!r}")

    async def _reject(detail: str, *, job_id: str = "") -> DelegationOutcome:
        """不受理：已有记录就把它推到 dismissed，还没有就补一条 dismissed 记录留痕。"""
        if job_id:
            await store.dialogue_job_set_status(job_id, DIALOGUE_JOB_DISMISSED, detail=detail)
        elif npc and slot_key:
            await store.dialogue_job_create(
                slot_key=slot_key,
                npc=npc,
                trigger=kind,
                reason=reason,
                instruction=instruction,
                task_id=task_id,
                session_id=session or None,
                status=DIALOGUE_JOB_DISMISSED,
                detail=detail,
            )
        logger.info("对话修改委派未受理（%s）: npc=%s trigger=%s", detail, npc, kind)
        return DelegationOutcome(accepted=False, detail=detail, job_id=job_id)

    # ---- 初判（任务元数据规则 + 轻量评估，§B4-1）----
    # 任务材料先读：记录里的 get_npc 才是权威发布 NPC，事件/调用给的名字只作兜底。
    task_block = ""
    publish_block = ""
    if kind == TRIGGER_IMPORTANT_TASK:
        if task_id is None:
            return DelegationOutcome(accepted=False, detail="更改类型为重要任务但没给任务 id")
        material = await _load_task_material(task_id)
        if material is None:
            return await _reject(f"任务 {task_id} 的记录读不到")
        if require_marker and not material.marked:
            return await _reject(f"任务 {task_id} 发布时没挂对话重写标记")
        npc = material.npc or npc
        task_block = material.detail_block
        instruction = instruction or material.instruction
        reason = reason or f"玩家完成了你发布的委托（任务 id {task_id}）。"
        if material.publish_session_id:
            publish_block = await _load_publish_dialogue(material.publish_session_id, npc)
    elif not reason:
        return await _reject("没写发生了什么，判断不了关键性")

    if not npc:
        return await _reject("缺少 NPC 名")
    if state.mode != "embedded" or not slot_key:
        return await _reject("未绑定存档或非内嵌模式，对话覆盖层无处落盘")

    row = await store.dialogue_job_create(
        slot_key=slot_key,
        npc=npc,
        trigger=kind,
        reason=reason,
        instruction=instruction,
        task_id=task_id,
        session_id=session or None,
        status=DIALOGUE_JOB_RECEIVED,
    )
    if row is None:
        # 部分唯一索引命中：该存档该任务已委派过（首次有效，后续静默跳过）
        logger.info("任务 %s 已委派过，静默跳过本次触发", task_id)
        return DelegationOutcome(accepted=False, detail=f"任务 {task_id} 已委派过，静默跳过")
    await store.dialogue_job_set_status(row.job_id, DIALOGUE_JOB_JUDGING)

    # ---- 频率限制 ----
    recent = await store.dialogue_job_delegated_since(
        slot_key, npc, time.time() - DIALOGUE_DELEGATE_COOLDOWN_S,
    )
    if recent > 0:
        return await _reject(
            f"冷却窗内已为 {npc} 委派过 {recent} 次", job_id=row.job_id,
        )
    if session:
        used = await store.dialogue_job_session_delegated(session)
        if used >= DIALOGUE_DELEGATE_PER_SESSION:
            return await _reject(
                f"本会话委派已达上限 {DIALOGUE_DELEGATE_PER_SESSION} 次", job_id=row.job_id,
            )

    # ---- 交给后台 worker（槽位在受理当刻钉死）----
    from services.dialogue.worker import DialogueJob, get_dialogue_worker

    submitted = get_dialogue_worker().submit(DialogueJob(
        job_id=row.job_id,
        npc_name=npc,
        slot_key=slot_key,
        llm_config=llm_config or LLMConfig(),
        source_trigger=kind,
        reason=reason,
        instruction=instruction,
        source_task_id=task_id if kind == TRIGGER_IMPORTANT_TASK else None,
        task_block=task_block,
        publish_block=publish_block,
    ))
    if not submitted:
        await store.dialogue_job_set_status(
            row.job_id, DIALOGUE_JOB_FAILED, detail="后台队列已满，本次委派被丢弃",
        )
        return DelegationOutcome(accepted=False, detail="后台队列已满", job_id=row.job_id)
    await store.dialogue_job_set_status(row.job_id, DIALOGUE_JOB_DELEGATED)
    logger.info(
        "对话修改委派已受理: job=%s npc=%s trigger=%s task_id=%s",
        row.job_id, npc, kind, task_id,
    )
    return DelegationOutcome(accepted=True, detail="已受理", job_id=row.job_id)


async def handle_task_completed(
    *, slot_key: str, task_id: int, npc: str = "",
) -> DelegationOutcome:
    """触发一：游戏推来的任务完成事件（``POST /api/integration/task-completed``）。

    只受理当前绑定槽位的事件（换档后旧存档的事件一律 dismissed），
    且只认发布时挂了对话重写标记的任务。
    """
    from services.integration.state import get_integration_state

    bound = get_integration_state().slot_key
    if bound is None or bound != (slot_key or "").strip():
        logger.info(
            "任务完成事件与当前绑定槽位不符，已忽略: 事件=%s 当前=%s task_id=%s",
            slot_key, bound, task_id,
        )
        return DelegationOutcome(accepted=False, detail="事件槽位与当前绑定槽位不符")
    return await submit_delegation(
        npc_name=npc,
        trigger=TRIGGER_IMPORTANT_TASK,
        reason=f"玩家完成了你发布的重要委托（任务 id {task_id}）。",
        task_id=task_id,
        require_marker=True,
    )


# ---------------------------------------------------------------------------
# 材料装配
# ---------------------------------------------------------------------------

async def _load_task_material(task_id: int) -> Any:
    """任务详情与对话＋发布时挂的标记（读不到＝None）。"""
    from services.agent_tools.finished_tasks import load_task_rewrite_material
    from services.integration.state import get_integration_state

    data_root = _data_root()
    if data_root is None:
        return None
    slot_key = get_integration_state().slot_key
    return await asyncio.to_thread(
        lambda: load_task_rewrite_material(
            data_root=data_root, slot_key=slot_key, task_id=task_id,
        )
    )


async def _load_publish_dialogue(session_id: str, npc_name: str) -> str:
    """发布那场对话的摘要＋摘要后的精确对话（§4.0.2-1 的材料口径）。"""
    store = get_memory_store()
    try:
        summary = await store.get_summary(session_id)
        after = (await store.count_messages(session_id)) - await store.get_summary_message_count(session_id)
        messages = await store.get_history(
            session_id, limit=min(max(after, 0), PUBLISH_DIALOGUE_LIMIT), order="asc",
        ) if after > 0 else []
    except Exception as exc:
        logger.warning("读取发布对话记录失败（按无记录继续）: %s", exc)
        return ""
    parts: list[str] = []
    if summary:
        parts.append(f"【当时的对话摘要】\n{summary}")
    if messages:
        speaker = npc_name or "对方"
        lines = [
            f"玩家：{m.content}" if m.role == "user" else f"{speaker}：{m.content}"
            for m in messages
        ]
        parts.append("【摘要之后的对话原文】\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _data_root() -> Any:
    try:
        from services.game_data.registry import get_game_data_registry

        return get_game_data_registry().data_root
    except Exception as exc:
        logger.warning("读取游戏数据目录失败，无法装配任务材料: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 聊天侧调用落地（触发三）
# ---------------------------------------------------------------------------

async def apply_delegate_calls(
    tool_calls: list[dict[str, Any]],
    *,
    npc_name: str,
    session_id: str,
    llm_config: LLMConfig | None = None,
) -> list[str]:
    """执行本轮聊天 Agent 的委派调用（工具不在全局注册表里，这里直接落地）。

    返回逐条结果文本，供空正文兜底轮告诉 NPC 已经派了什么；
    任何失败只记日志，绝不阻断本轮对话（§4.0.1 独立链路）。
    """
    results: list[str] = []
    for tool_call in tool_calls:
        if tool_call_name(tool_call) != DELEGATE_TOOL_NAME:
            continue
        args = parse_arguments(tool_call)
        trigger = str(args.get("change_type") or "").strip()
        reason = str(args.get("reason") or "").strip()
        task_id = _as_task_id(args.get("task_id"))
        try:
            outcome = await submit_delegation(
                npc_name=npc_name,
                trigger=trigger,
                reason=reason,
                instruction=str(args.get("instruction") or "").strip(),
                task_id=task_id,
                session_id=session_id,
                llm_config=llm_config,
            )
        except Exception as exc:
            logger.warning("对话修改委派调用执行失败（忽略）: %s", exc)
            continue
        results.append(f"{trigger}：{outcome.detail}")
    return results


def _as_task_id(raw: Any) -> int | None:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
