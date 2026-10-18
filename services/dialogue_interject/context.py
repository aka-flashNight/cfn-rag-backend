"""插话上下文的纯文本装配（P0 骨架，P1/P2 接真模型时复用）。

原则（实施指引 §4.1）：请求现场是权威——`played` / `remaining` /
`reference_continuation` / `player_input` / `task_context` 全部来自游戏；
session 历史不参与本轮 prompt 拼装。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from services.dialogue_interject.schemas import (
    InterjectRequest,
    PlayedLine,
    ReferenceLine,
    RemainingLine,
    TaskSummary,
)
from services.integration.state import get_integration_state
from services.llm.meta import DEFAULT_EMOTION

logger = logging.getLogger(__name__)

#: 上下文预算省略标记（协议 4.4：编号仍按原索引，不能打乱）
OMISSION_MARK = "（中间省略）"

_TASK_KIND_LABEL = {"task_pre": "任务前", "task_post": "任务后"}


@dataclass(frozen=True)
class DailyBlocks:
    """日常插话的稳定 system 块与情绪白名单（请求现场不进这里）。"""

    system_prompt: str
    npc_emotions: tuple[str, ...] = (DEFAULT_EMOTION,)


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


def build_user_context(request: InterjectRequest) -> str:
    """按实施指引 §4.2/§4.3 的建议顺序拼装 user 上下文（P0 骨架）。"""
    blocks: list[str] = []
    dialogue = request.dialogue

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
# 日常稳定块装配（B1 真 LLM 最小路径）
# ---------------------------------------------------------------------------


async def assemble_daily_blocks(
    npc_name: str,
    *,
    npc_manager: Any = None,
    game_data: Any = None,
    prompt_max: int = 135,
) -> DailyBlocks:
    """复用终端的稳定块（世界观/扮演/NPC 状态/场景），拼成日常插话 system。

    不读取 session 历史、草案、路由或说话爆发块；这些属于终端回合逻辑。
    场景/同阵营块在数据缺失时降级为空串，不影响本轮对话。
    """
    if npc_manager is None:
        from services.npc.manager import get_npc_manager

        npc_manager = await get_npc_manager()

    state = await npc_manager.get(npc_name)
    states = await npc_manager.all_states()

    if game_data is None:
        try:
            from services.game_data.registry import get_game_data_registry

            game_data = get_game_data_registry()
        except Exception as exc:  # 资源目录缺失/未初始化：降级为无场景块
            logger.warning("插话场景数据不可用，按无场景处理: %s", exc)
            game_data = None
    map_reg = getattr(game_data, "map_registry", None)

    snap = get_integration_state().snapshot()
    save_info: dict[str, Any] | None = None
    if snap["bound_slot"] is not None:
        scene = dict(snap["scene"]) if isinstance(snap["scene"], dict) else {}
        # 场景段补显示标签：stageFlag → map_definition label；查不到保持空串
        if map_reg is not None and str(scene.get("stageFlag") or ""):
            try:
                scene["label"] = map_reg.label_for_scene_name(str(scene["stageFlag"]))
            except Exception:
                scene["label"] = ""
        save_info = {
            "slot_key": snap["bound_slot"],
            "character": snap["character"],
            "progress": snap["progress"],
            "scene": scene,
            "player": snap["player"],
        }

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

    progress_desc = ""
    if save_info is not None:
        try:
            from services.game_progress import derive_progress_stage, get_progress_stage_config
            from services.orchestrator.prompts import progress_stage_desc

            stage = derive_progress_stage(snap["progress"], snap["player"])
            stage_cfg = get_progress_stage_config(stage) if stage else None
            progress_desc = progress_stage_desc(stage, stage_cfg)
        except Exception as exc:
            logger.warning("插话进度块装配失败，降级为空: %s", exc)

    # 局部导入：prompts 顶层依赖本模块的现场渲染，避免循环导入
    from services.dialogue_interject.prompts import build_daily_system_prompt

    system_prompt = build_daily_system_prompt(
        npc_name=npc_name,
        state=state,
        scene_block=scene_block,
        same_faction_block=same_faction_block,
        save_info=save_info,
        progress_desc=progress_desc,
        prompt_max=prompt_max,
    )
    emotions = tuple(state.emotions or [DEFAULT_EMOTION])
    return DailyBlocks(system_prompt=system_prompt, npc_emotions=emotions)
