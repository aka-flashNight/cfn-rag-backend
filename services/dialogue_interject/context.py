"""插话上下文的纯文本装配（日常/任务共用稳定块）。

原则（实施指引 §4.1）：请求现场是权威——`played` / `remaining` /
`reference_continuation` / `player_input` / `task_context` 全部来自游戏；
session 历史不参与本轮 prompt 拼装。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from services.dialogue_interject.item_claims import ITEM_CLAIMS
from services.dialogue_interject.item_ledger import load_item_ledger_block
from services.dialogue_interject.schemas import (
    InterjectRequest,
    PlayedLine,
    ReferenceLine,
    RemainingLine,
    TaskSummary,
)
from services.integration.state import get_integration_state
from services.stage_access import BindingFacts
from services.llm.meta import DEFAULT_EMOTION

logger = logging.getLogger(__name__)

#: 上下文预算省略标记（协议 4.4：编号仍按原索引，不能打乱）
OMISSION_MARK = "（中间省略）"

_TASK_KIND_LABEL = {"task_pre": "任务前", "task_post": "任务后"}


@dataclass(frozen=True)
class DailyBlocks:
    """插话（日常/任务）的 system/现场块与工具运行所需的只读快照。

    `system_prompt` 只放稳定块；RAG/长期记忆/草案摘要/任务上下文属于现场块，
    由 `build_user_context` 拼进 user；工具运行数据只读，不进入 prompt 之外的地方。
    """

    system_prompt: str
    #: 工具结果后的最终台词轮 system：不含 tool guide（tools=None 时不描述工具）
    tool_free_system_prompt: str = ""
    npc_emotions: tuple[str, ...] = (DEFAULT_EMOTION,)
    npc_state: Any = None
    npc_states: dict[str, Any] = field(default_factory=dict)
    game_data: Any = None
    engine: Any = None
    registry: Any = None
    memory: Any = None
    session_id: str = ""
    slot_key: str | None = None
    progress_stage: int | None = None
    player_level: int | None = None
    progress_desc: str = ""
    has_shop: bool = False
    shop_reward_types: tuple[str, ...] = ()
    player_can_challenge: bool = False
    rag_context_text: str = ""
    retrieval_bundle: Any = None
    retrieve_fn: Any = None
    long_term: Any = None
    long_term_block: str = ""
    pending_draft_row: Any = None
    pending_draft_summary: str = ""
    recent_dialogue: str = ""
    task_publish_disabled: bool = False
    delegate_enabled: bool = False
    save_info: Any = None
    #: 物品往来现场块（跨组账本进 daily/task；其余仅 daily）
    item_memory_block: str = ""
    gift_detail_block: str = ""
    item_recent_block: str = ""
    pending_claim_block: str = ""
    pending_item_claim: Any = None
    item_tools_enabled: bool = False
    turn_no: int = 0


def render_played(lines: Sequence[PlayedLine], *, npc: str, max_lines: int | None = None) -> list[str]:
    """已播历史逐行渲染：当前 NPC/ai → `NPC：…`；玩家 → `玩家：…`；其他 NPC → `【…】`。"""
    rendered: list[str] = []
    for line in _trim(lines, max_lines):
        if line is None:
            rendered.append(OMISSION_MARK)
            continue
        text = line.text.strip()
        if not text:
            continue
        if line.role in ("npc", "ai"):
            rendered.append(f"{npc}：{text}")
        elif line.role == "player":
            rendered.append(f"玩家：{text}")
        else:
            rendered.append(f"【{line.name or '其他NPC'}：{text}】")
    return rendered


def render_remaining(
    lines: Sequence[RemainingLine], *, max_lines: int | None = None, numbered: bool = True,
) -> list[str]:
    """原始剩余台词：任务模式带 `[n]`；日常只作参考、不编号。"""
    rendered: list[str] = []
    for line in _trim(lines, max_lines):
        if line is None:
            rendered.append(OMISSION_MARK)
            continue
        prefix = f"[{line.index}] " if numbered else ""
        rendered.append(f"{prefix}{line.name}：{line.text.strip()}")
    return rendered


def render_reference(lines: Sequence[ReferenceLine]) -> list[str]:
    """参考续写：明确标“上一轮未播”，不参与编号、不进已播历史。"""
    rendered: list[str] = []
    for line in lines:
        text = line.text.strip()
        if line.kind == "add":
            rendered.append(f"上一轮未播新增：{text}")
        elif line.kind == "closing":
            if text:
                rendered.append(f"上一轮未播结束语：{text}")
        elif line.kind == "edit":
            rendered.append(f"上一轮原第 {line.index} 条改写：{text}")
        elif line.kind == "remove":
            rendered.append(f"上一轮原第 {line.index} 条移除")
    return rendered


def render_task_summary(summary: TaskSummary, *, label: str) -> list[str]:
    """任务摘要一块：标题/NPC/目标/奖励/（已完成）玩家做了什么。"""
    lines = [f"{label}：{summary.title.strip() or '（未命名任务）'}"]
    if summary.task_id is not None:
        lines.append(f"任务 id：{summary.task_id}")
    if summary.npc.strip():
        lines.append(f"发布 NPC：{summary.npc.strip()}")
    if summary.objectives:
        lines.append("目标：" + "；".join(item.strip() for item in summary.objectives if item.strip()))
    if summary.rewards:
        lines.append("奖励：" + "；".join(item.strip() for item in summary.rewards if item.strip()))
    if summary.outcome.strip():
        lines.append(f"玩家完成了什么：{summary.outcome.strip()}")
    return lines


def build_user_context(
    request: InterjectRequest,
    *,
    rag_context_text: str = "",
    long_term_block: str = "",
    pending_draft_summary: str = "",
    item_memory_block: str = "",
    gift_detail_block: str = "",
    item_recent_block: str = "",
    pending_claim_block: str = "",
) -> str:
    """按实施指引 §4.2 的建议顺序拼装 user 上下文（请求现场为准）。

    物品块紧跟长期记忆：跨组账本（daily/task）、本组赠送明细/最近往来/待处理索取
    （仅 daily）。它们是易变现场，不进 system 稳定前缀。
    """
    blocks: list[str] = []
    dialogue = request.dialogue

    if long_term_block.strip():
        blocks.append(long_term_block.strip())
    for item_block in (
        item_memory_block, gift_detail_block, item_recent_block, pending_claim_block,
    ):
        if item_block.strip():
            blocks.append(item_block.strip())

    if rag_context_text.strip():
        blocks.append(
            "【检索到的设定资料（仅用于保持设定与说话风格，不要逐字复读原文）】\n"
            + rag_context_text.strip()
        )

    if dialogue.played:
        blocks.append("【本组已播历史】\n" + "\n".join(render_played(dialogue.played, npc=request.npc)))

    if dialogue.remaining:
        if request.mode == "daily":
            header = "【原对话剩余台词（仅作参考，不要求逐字续写）】"
            body = render_remaining(dialogue.remaining, numbered=False)
        else:
            header = "【原始剩余台词（按编号）】"
            body = render_remaining(dialogue.remaining, numbered=True)
        blocks.append(header + "\n" + "\n".join(body))

    if dialogue.reference_continuation:
        blocks.append(
            "【参考续写（上一轮未播，仅供参考，可不采用）】\n"
            + "\n".join(render_reference(dialogue.reference_continuation))
        )

    if request.mode == "task":
        task_lines: list[str] = []
        context = request.task_context
        if context is not None:
            if context.completed is not None:
                task_lines.extend(render_task_summary(context.completed, label="已完成任务"))
            if context.upcoming is not None:
                task_lines.extend(render_task_summary(context.upcoming, label="将完成任务"))
        kind_label = _TASK_KIND_LABEL.get(dialogue.kind, "任务")
        if task_lines:
            blocks.append(f"【任务上下文（{kind_label}）】\n" + "\n".join(task_lines))
        else:
            blocks.append(f"【任务上下文（{kind_label}）】\n（本次没有可用的已完成/将完成任务摘要）")

    if pending_draft_summary.strip():
        blocks.append(
            "【待确认的委托草案（已拟定、尚未发布）】\n"
            f"{pending_draft_summary.strip()}\n"
            "玩家可能会接受、拒绝、讨价还价或岔开话题；点头之前不要说已发布。"
        )

    blocks.append("【玩家刚刚说的话】\n" + request.player_input)
    return "\n\n".join(blocks)


def _trim(items: Sequence, max_lines: int | None) -> Iterable:
    """预算截断：保留开头少量 + 最近 M 条，中间插省略标记。"""
    if max_lines is None or max_lines <= 0 or len(items) <= max_lines:
        return list(items)
    head = min(2, max_lines // 3)
    tail = max_lines - head
    return [*items[:head], None, *items[-tail:]]


# ---------------------------------------------------------------------------
# 日常上下文装配（B2：稳定块 + RAG/长期记忆/草案 + 工具运行快照）
# ---------------------------------------------------------------------------


async def assemble_blocks(
    request: InterjectRequest,
    *,
    npc_manager: Any = None,
    game_data: Any = None,
    engine: Any = None,
    memory: Any = None,
    registry: Any = None,
    prompt_max: int = 135,
    rag_enabled: bool = True,
    session_id: str | None = None,
) -> DailyBlocks:
    """按 `request.mode` 装配插话块（日常/任务共用稳定块与检索/记忆）。

    - system：世界观/扮演/NPC 状态/场景/同阵营/输出纪律/工具说明；
    - user 现场：长期记忆、RAG、played、remaining、reference、草案摘要（仅日常）、
      任务上下文（仅任务）、玩家输入；
    - 不读取 session 聊天历史（协议 4.1：session 历史不参与本轮 prompt）；
    - 任务模式不读草案、不开任务工具、不注册委派（协议 4.1）；
    - 草案与工具写盘一律用 MemoryStore 的 session_id（与终端继续聊同一把钥匙），
      传空时按 dialogue_instance_id 反查，查不到才退回原值。
    """
    is_task = request.mode == "task"
    npc_name = request.npc
    if npc_manager is None:
        from services.npc.manager import get_npc_manager

        npc_manager = await get_npc_manager()

    state = await npc_manager.get(npc_name)
    states = await npc_manager.all_states()

    if memory is None:
        from services.memory.store import get_memory_store

        memory = get_memory_store()
    if registry is None:
        from services.tools.base import get_tool_registry

        registry = get_tool_registry()

    if game_data is None:
        try:
            from services.game_data.registry import get_game_data_registry

            game_data = get_game_data_registry()
        except Exception as exc:  # 资源目录缺失/未初始化：降级为无场景块
            logger.warning("插话场景数据不可用，按无场景处理: %s", exc)
            game_data = None
    map_reg = getattr(game_data, "map_registry", None)

    if engine is None and rag_enabled:
        try:
            from services.retrieval import get_retrieval_engine

            engine = get_retrieval_engine()
        except Exception as exc:  # 检索不可用：本轮按无 RAG 处理
            logger.warning("插话检索引擎不可用，按无 RAG 处理: %s", exc)
            engine = None

    snap = get_integration_state().snapshot()
    slot_key = snap["bound_slot"]
    save_info: dict[str, Any] | None = None
    if slot_key is not None:
        scene = dict(snap["scene"]) if isinstance(snap["scene"], dict) else {}
        # 场景段补显示标签：stageFlag → map_definition label；查不到保持空串
        if map_reg is not None and str(scene.get("stageFlag") or ""):
            try:
                scene["label"] = map_reg.label_for_scene_name(str(scene["stageFlag"]))
            except Exception:
                scene["label"] = ""
        save_info = {
            "slot_key": slot_key,
            "character": snap["character"],
            "progress": snap["progress"],
            "scene": scene,
            "player": snap["player"],
        }

    progress_stage = None
    player_level = None
    progress_desc = ""
    if save_info is not None:
        try:
            from services.game_progress import (
                derive_progress_stage,
                get_player_level,
                get_progress_stage_config,
            )
            from services.orchestrator.prompts import progress_stage_desc

            progress_stage = derive_progress_stage(snap["progress"], snap["player"])
            player_level = get_player_level(snap["player"]) if snap["player"] else None
            stage_cfg = get_progress_stage_config(progress_stage) if progress_stage else None
            progress_desc = progress_stage_desc(progress_stage, stage_cfg)
        except Exception as exc:
            logger.warning("插话进度块装配失败，降级为空: %s", exc)

    # ---- RAG：与终端同源的 Tier-1 检索（同步实现放线程池）----
    from services.orchestrator.context import (
        build_retrieve_fn,
        entity_hints_block,
        forbidden_other_chars,
        retrieve_tier1,
    )

    forbidden = forbidden_other_chars(states)
    titles = list(state.titles or [])
    last_npc_message = next(
        (line.text for line in reversed(request.dialogue.played) if line.role in ("npc", "ai")),
        None,
    )
    rag_context_text = ""
    retrieval_bundle: Any = None
    retrieve_fn = None
    if engine is not None and rag_enabled:
        retrieve_fn = build_retrieve_fn(
            engine=engine,
            npc_name=npc_name,
            npc_titles=titles,
            npc_faction=state.faction,
            forbidden=forbidden,
        )

        def _retrieve() -> tuple[str, Any]:
            bundle = retrieve_tier1(
                engine=engine,
                user_query=request.player_input,
                npc_name=npc_name,
                npc_titles=titles,
                npc_faction=state.faction,
                npc_last_message=last_npc_message,
                forbidden=forbidden,
            )
            text = ""
            if bundle is not None and not bundle.is_empty:
                from services.retrieval import format_retrieval_context

                text = format_retrieval_context(bundle, forbidden)
            hints = entity_hints_block(bundle, game_data)
            if hints:
                text = f"{text}\n\n{hints}" if text else hints
            return text, bundle

        try:
            rag_context_text, retrieval_bundle = await asyncio.to_thread(_retrieve)
        except Exception as exc:
            logger.warning("插话检索装配失败，降级为空: %s", exc)

    # ---- 长期记忆（跨对话窗口；不是 session 聊天历史）----
    long_term = None
    long_term_block = ""
    try:
        from services.memory.long_term import format_block, load_snapshot

        long_term = await load_snapshot(memory, slot_key=slot_key, npc=npc_name)
        long_term_block = format_block(long_term)
    except Exception as exc:
        logger.warning("插话长期记忆装配失败，降级为空: %s", exc)

    # ---- 物品往来（跨组账本进 daily/task；claim 与组内明细仅 daily）----
    item_memory_block = await load_item_ledger_block(
        memory, slot_key_value=slot_key, npc=npc_name,
    )
    pending_item_claim = None
    gift_detail_block = ""
    item_recent_block = ""
    pending_claim_block = ""
    turn_no = 0
    if not is_task:
        turn_no = ITEM_CLAIMS.turn_number(request.dialogue_instance_id, request.turn_id)
        gift_detail_block = ITEM_CLAIMS.gift_detail_block(request.dialogue_instance_id)
        item_recent_block = ITEM_CLAIMS.recent_block(request.dialogue_instance_id)
        # 选项直连请求正在解决 claim：不再把它当待处理项注入本轮现场
        if request.action is None:
            pending_item_claim = ITEM_CLAIMS.pending(request.dialogue_instance_id)
        if pending_item_claim is not None:
            claim_lines = [
                "【待处理的索取】",
                f"你正在向玩家索取 {pending_item_claim.quantity} 个 "
                f"{pending_item_claim.item_name}，玩家还没有交付。",
            ]
            counts = getattr(request, "player_item_counts", None) or {}
            try:
                have = int(counts.get(pending_item_claim.item_name, 0) or 0)
            except (TypeError, ValueError):
                have = 0
            if have <= 0:
                claim_lines.append("玩家看起来没有这个物品。")
            elif have < int(pending_item_claim.quantity or 0):
                claim_lines.append("玩家看起来拥有该物品的数量不足。")
            pending_claim_block = "\n".join(claim_lines)

    # ---- 待确认草案（B2 起支持 confirm/cancel/update；任务模式不读）----
    # 草案与终端共用一把钥匙：优先 MemoryStore 的 session_id；未建会话时退回
    # dialogue_instance_id（离线单测/异常降级），并兼容 B2 早期按 instance id 落库的草案。
    session_id = (session_id or "").strip()
    if not session_id:
        session_id = request.dialogue_instance_id
        try:
            info = await memory.dialogue_session_get(request.dialogue_instance_id)
            if info is not None:
                session_id = info.session_id
        except Exception as exc:
            logger.warning("插话 session 反查失败，按 dialogue_instance_id 处理: %s", exc)

    pending_draft_row = None
    pending_draft_summary = ""
    draft_keys: list[str] = []
    if not is_task:
        draft_keys = [session_id]
        if request.dialogue_instance_id and request.dialogue_instance_id != session_id:
            draft_keys.append(request.dialogue_instance_id)
    for draft_key in draft_keys:
        if not draft_key:
            continue
        try:
            pending_draft_row = await memory.get_draft(draft_key)
        except Exception as exc:
            logger.warning("插话草案读取失败，按无草案处理: %s", exc)
            pending_draft_row = None
        if pending_draft_row is not None:
            break
    if pending_draft_row is not None and isinstance(pending_draft_row.draft, dict):
        try:
            from services.agent_tools.draft_formatting import _detailed_draft_summary

            pending_draft_summary = _detailed_draft_summary(
                pending_draft_row.draft,
                game_data,
                rag_context_text=rag_context_text,
                npc_states=states,
            )
        except Exception as exc:
            logger.warning("插话草案摘要装配失败，降级为空: %s", exc)

    # ---- 商店/切磋能力（任务 Agent 选型硬边界）----
    has_shop = False
    shop_reward_types: list[str] = []
    try:
        shops = getattr(game_data, "shops", None)
        if shops is not None and shops.has_shop(npc_name):
            has_shop = True
            seen: set[str] = set()
            for item_name in shops.get_npc_shop(npc_name):
                item = game_data.items.get_by_name(item_name)
                if item and item.type and item.type not in seen:
                    seen.add(item.type)
                    shop_reward_types.append(item.type)
    except Exception:
        pass

    player_can_challenge = False
    try:
        if (state.challenge or "").strip():
            from services.agent_tools.context_builder import _build_challenge_targets

            challenge_result = _build_challenge_targets(
                game_data, npc_name, state.challenge, max(1, min(7, progress_stage or 1)),
                facts=BindingFacts.from_save_info(save_info),
            )
            player_can_challenge = isinstance(challenge_result, list)
    except Exception:
        player_can_challenge = False

    scene_block = ""
    if map_reg is not None:
        try:
            from services.npc.scene_context import build_scene_block

            scene_block = build_scene_block(
                map_reg=map_reg, npc_states=states, npc_name=npc_name,
            )
        except Exception as exc:
            logger.warning("插话场景块装配失败，降级为空: %s", exc)

    same_faction_block = ""
    try:
        from services.orchestrator.context import same_faction_npcs_block

        # 稳定块不随玩家这句话变化：不排除任何被提到的角色（前缀缓存友好）
        same_faction_block = same_faction_npcs_block(
            npc_name, state.faction, states, set(), map_reg,
        )
    except Exception as exc:
        logger.warning("插话同阵营块装配失败，降级为空: %s", exc)

    # NPC 固定层装备：与终端静态 system 同源，避免插话时 NPC 不知道自己的装备。
    equipment_block = ""
    try:
        from services.npc.npc_equipment import build_npc_equipment_block

        equipment_block = build_npc_equipment_block(
            npc_name=npc_name, game_data=game_data,
        )
    except Exception as exc:
        logger.warning("插话装备块装配失败，降级为空: %s", exc)

    from services.npc.task_publish_policy import is_task_publish_disabled

    # 任务插话：禁掉全部任务类工具与委派（协议 4.1），查询/记忆保留
    task_publish_disabled = True if is_task else is_task_publish_disabled(npc_name)
    # 委派工具按终端注册闸（后台有 worker / 未绑定存档时不注册，提示词也不提）
    delegate_enabled = False
    if not is_task:
        try:
            from services.dialogue.delegation import delegate_tool_or_none

            delegate_enabled = delegate_tool_or_none(include_task_fallback=False) is not None
        except Exception as exc:
            logger.warning("插话委派工具判定失败，按不注册处理: %s", exc)

    # 局部导入：prompts 顶层依赖本模块的现场渲染，避免循环导入
    from services.dialogue_interject.prompts import (
        build_daily_system_prompt,
        build_task_system_prompt,
    )

    # 物品工具与任务草案互斥；pending claim 时只留取消索取；选项直连轮不再给物品工具
    item_tools_enabled = (
        not is_task
        and request.action is None
        and pending_item_claim is None
        and pending_draft_row is None
    )
    common_system = dict(
        npc_name=npc_name,
        state=state,
        scene_block=scene_block,
        same_faction_block=same_faction_block,
        save_info=save_info,
        progress_desc=progress_desc,
        prompt_max=prompt_max,
        equipment_block=equipment_block,
    )
    if is_task:
        system_prompt = build_task_system_prompt(**common_system)
        tool_free_system_prompt = build_task_system_prompt(
            **common_system, include_tool_guide=False,
        )
    else:
        system_prompt = build_daily_system_prompt(
            **common_system,
            has_pending_draft=pending_draft_row is not None,
            task_publish_disabled=task_publish_disabled,
            delegate_enabled=delegate_enabled,
            has_shop=has_shop,
            shop_reward_types=list(shop_reward_types),
            player_can_challenge=player_can_challenge,
            item_tools=item_tools_enabled,
            has_item_claim=pending_item_claim is not None,
        )
        tool_free_system_prompt = build_daily_system_prompt(
            **common_system,
            has_pending_draft=pending_draft_row is not None,
            task_publish_disabled=task_publish_disabled,
            delegate_enabled=delegate_enabled,
            has_shop=has_shop,
            shop_reward_types=list(shop_reward_types),
            player_can_challenge=player_can_challenge,
            item_tools=False,
            has_item_claim=False,
            include_tool_guide=False,
        )
    recent_dialogue = "\n".join(
        render_played(request.dialogue.played, npc=npc_name, max_lines=8)
    )
    emotions = tuple(state.emotions or [DEFAULT_EMOTION])
    return DailyBlocks(
        system_prompt=system_prompt,
        tool_free_system_prompt=tool_free_system_prompt,
        npc_emotions=emotions,
        npc_state=state,
        npc_states=states,
        game_data=game_data,
        engine=engine,
        registry=registry,
        memory=memory,
        session_id=session_id,
        slot_key=slot_key,
        progress_stage=progress_stage,
        player_level=player_level,
        progress_desc=progress_desc,
        has_shop=has_shop,
        shop_reward_types=tuple(shop_reward_types),
        player_can_challenge=player_can_challenge,
        rag_context_text=rag_context_text,
        retrieval_bundle=retrieval_bundle,
        retrieve_fn=retrieve_fn,
        long_term=long_term,
        long_term_block=long_term_block,
        pending_draft_row=pending_draft_row,
        pending_draft_summary=pending_draft_summary,
        recent_dialogue=recent_dialogue,
        task_publish_disabled=task_publish_disabled,
        delegate_enabled=delegate_enabled,
        save_info=save_info,
        item_memory_block=item_memory_block,
        gift_detail_block=gift_detail_block,
        item_recent_block=item_recent_block,
        pending_claim_block=pending_claim_block,
        pending_item_claim=pending_item_claim,
        item_tools_enabled=item_tools_enabled,
        turn_no=turn_no,
    )


async def assemble_daily_blocks(
    request: InterjectRequest,
    *,
    npc_manager: Any = None,
    game_data: Any = None,
    engine: Any = None,
    memory: Any = None,
    registry: Any = None,
    prompt_max: int = 135,
    rag_enabled: bool = True,
    session_id: str | None = None,
) -> DailyBlocks:
    """日常模式装配（薄包装，保留既有调用点）。"""
    return await assemble_blocks(
        request,
        npc_manager=npc_manager,
        game_data=game_data,
        engine=engine,
        memory=memory,
        registry=registry,
        prompt_max=prompt_max,
        rag_enabled=rag_enabled,
        session_id=session_id,
    )


async def assemble_task_blocks(
    request: InterjectRequest,
    *,
    npc_manager: Any = None,
    game_data: Any = None,
    engine: Any = None,
    memory: Any = None,
    registry: Any = None,
    prompt_max: int = 135,
    rag_enabled: bool = True,
    session_id: str | None = None,
) -> DailyBlocks:
    """任务模式装配：与日常共用稳定块/检索/记忆，但不读草案、不开任务工具与委派。"""
    return await assemble_blocks(
        request,
        npc_manager=npc_manager,
        game_data=game_data,
        engine=engine,
        memory=memory,
        registry=registry,
        prompt_max=prompt_max,
        rag_enabled=rag_enabled,
        session_id=session_id,
    )
