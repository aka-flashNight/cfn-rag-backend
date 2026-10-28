"""关卡解锁与任务前置的只读规则（与游戏 AS2 同源，不新增权威）。

游戏口径：
- ``isStageUnlocked``：主线链值 >= UnlockCondition，或任一 AltUnlockCondition 的
  ``task_chains_progress[chain] >= min``；
- ``taskAvailable``：get_requirements 里的任务都已在 tasks_finished 中完成。

本模块只做纯计算，不读写磁盘、不引入第二个存档来源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from services.game_data.models import StageInfo, Task

MAINLINE_CHAIN = "主线"

#: 当前阶段已解锁、可参与当前任务类型的关卡少于此数时，势均力敌退回上一阶段匹配。
MATCH_EARLY_STAGE_MIN_TARGETS = 3


def _chain_values(progress: Any) -> dict[str, int]:
    if not isinstance(progress, dict):
        return {}
    chains = progress.get("task_chains_progress")
    if not isinstance(chains, dict):
        return {}
    out: dict[str, int] = {}
    for name, raw in chains.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = int(raw)
        if value >= 0:
            out[str(name)] = value
    return out


def _number(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


@dataclass(frozen=True)
class BindingFacts:
    """回合开始时捕获的绑定存档事实（只读快照）。"""

    slot_key: str
    progress: dict[str, Any]
    player: dict[str, Any]

    @classmethod
    def from_save_info(cls, save_info: Any) -> Optional["BindingFacts"]:
        """从入口已有的 save_info 取快照；无有效槽位＝未绑定。"""
        if not isinstance(save_info, dict):
            return None
        slot_key = str(save_info.get("slot_key") or "").strip()
        if not slot_key:
            return None
        progress = save_info.get("progress")
        player = save_info.get("player")
        return cls(
            slot_key=slot_key,
            progress=progress if isinstance(progress, dict) else {},
            player=player if isinstance(player, dict) else {},
        )

    def chain_value(self, name: str) -> Optional[int]:
        return _chain_values(self.progress).get(str(name or ""))

    @property
    def mainline_value(self) -> Optional[int]:
        return self.chain_value(MAINLINE_CHAIN)

    @property
    def completed_task_ids(self) -> frozenset[int]:
        raw = self.progress.get("tasks_finished")
        if not isinstance(raw, list):
            return frozenset()
        out: set[int] = set()
        for value in raw:
            if isinstance(value, bool):
                continue
            try:
                tid = int(value)
            except (TypeError, ValueError):
                continue
            if tid >= 0:
                out.add(tid)
        return frozenset(out)

    @property
    def player_level(self) -> Optional[int]:
        level = _number(self.player.get("level"))
        return level if level is not None and level > 0 else None


def stage_is_unlocked(si: StageInfo, facts: BindingFacts) -> bool:
    """与游戏 isStageUnlocked 相同的 OR 规则；无法证明即视为未解锁。"""
    mainline = facts.mainline_value
    if si.unlock_condition is not None and mainline is not None:
        if mainline >= int(si.unlock_condition):
            return True
    for alt in si.alt_unlock_conditions or ():
        value = facts.chain_value(alt.chain)
        if value is not None and value >= int(alt.min):
            return True
    return False


def _chain_node(task: Task) -> Optional[tuple[str, int]]:
    chain = getattr(task, "chain", None)
    if not isinstance(chain, str) or "#" not in chain:
        return None
    name, raw_node = chain.rsplit("#", 1)
    name = name.strip()
    if not name:
        return None
    try:
        node = int(raw_node.strip())
    except (TypeError, ValueError):
        return None
    if node < 0:
        return None
    return name, node


def _earliest_anchor(
    tasks: list[Task],
    *,
    chain_name: str,
    threshold: int,
    completed: Optional[frozenset[int]],
) -> list[int]:
    """链值达到 threshold 时最早可证明的已完成任务；未绑定不要求完成集合。"""
    if threshold <= 0:
        return []
    best: Optional[tuple[int, int]] = None
    for task in tasks:
        parsed = _chain_node(task)
        if parsed is None or parsed[0] != chain_name:
            continue
        node = parsed[1]
        if node < threshold:
            continue
        tid = int(getattr(task, "id", -1))
        if tid < 0:
            continue
        if completed is not None and tid not in completed:
            continue
        key = (node, tid)
        if best is None or key < best:
            best = key
    return [best[1]] if best is not None else []


def stage_unlock_task_ids(
    si: StageInfo,
    facts: BindingFacts,
    tasks: list[Task],
) -> list[int]:
    """目标关卡当前满足的解锁路径对应的已完成任务锚点（OR，不拼成多个 AND）。"""
    completed = facts.completed_task_ids
    mainline = facts.mainline_value
    if si.unlock_condition is not None and mainline is not None:
        if mainline >= int(si.unlock_condition):
            return _earliest_anchor(
                tasks, chain_name=MAINLINE_CHAIN,
                threshold=int(si.unlock_condition), completed=completed,
            )
    for alt in si.alt_unlock_conditions or ():
        value = facts.chain_value(alt.chain)
        if value is None or value < int(alt.min):
            continue
        return _earliest_anchor(
            tasks, chain_name=alt.chain, threshold=int(alt.min), completed=completed,
        )
    return []


def stage_unlock_task_ids_unbound(
    si: StageInfo,
    tasks: list[Task],
) -> list[int]:
    """未绑定时的推荐前置：按真实链映射取最早锚点，不声称已完成。"""
    if si.unlock_condition is not None and int(si.unlock_condition) > 0:
        return _earliest_anchor(
            tasks, chain_name=MAINLINE_CHAIN,
            threshold=int(si.unlock_condition), completed=None,
        )
    for alt in si.alt_unlock_conditions or ():
        return _earliest_anchor(
            tasks, chain_name=alt.chain, threshold=int(alt.min), completed=None,
        )
    return []


def matched_stage_for_tags(rows: list[dict[str, Any]], *, stage: int) -> int:
    """势均力敌匹配阶段：本阶段可攻打关卡不足阈值时退回前一阶段，不递归。"""
    if stage <= 1:
        return stage
    from services.game_progress import get_progress_stage_config

    cfg = get_progress_stage_config(stage)
    current_area = cfg.stage_name if cfg is not None else None
    if not current_area:
        return stage
    count = sum(
        1
        for row in rows
        if str(row.get("area") or "") == current_area and not row.get("is_dungeon")
    )
    if count >= MATCH_EARLY_STAGE_MIN_TARGETS:
        return stage
    return stage - 1
