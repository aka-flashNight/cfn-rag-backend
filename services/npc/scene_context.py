"""场景块与场景补充描述（提示词信息供给 B8-1，游戏仓开发计划 §8.1-①）。

- 场景块：聊天 NPC 层「你所在场景／同场景npc」块；单驻地两行，多驻地逐场景一行；
  无同场景 NPC 时用固定替代句；无驻地 NPC 返回空串（不拼块）。
- 场景补充：同阵营块未覆盖、且 npc_state_db 有记录的同场景 NPC，
  按同阵营块同构格式（「名字」（性别：X）（阵营：Y）（身份或称呼：Z））逐行补出。
- 名单来源＝ map_definition.json（含 npc_state_db 范围外 NPC，用地图 label 显示）；
  db 名字优先：地图名归一击中 db key 时用 db 名（大小写/别名差异对由此对齐）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from services.game_data.map_registry import _norm
from services.game_data.models import MapNpc, MapScene
from services.npc.manager import NPCState

logger = logging.getLogger(__name__)

MULTI_SCENE_HEADER = "你所在场景（以下多个场景为你常去的场景，你可能在任意其一）："


def scene_display(scene: MapScene) -> str:
    """场景显示文本：`<label>（<linkage>）`；linkage 缺失时只给 label。"""
    if scene.linkage:
        return f"{scene.label}（{scene.linkage}）"
    return scene.label


def npc_scene_text(map_reg: Any, npc_name: str) -> str:
    """NPC 驻场景显示文本（多驻地以「、」连接）；无驻地/查不到返回空串。"""
    if map_reg is None:
        return ""
    try:
        scenes = map_reg.scenes_for_npc(npc_name)
    except Exception:
        return ""
    return "、".join(scene_display(s) for s in scenes)


def resolve_db_name(map_npc: MapNpc, npc_states: dict[str, NPCState]) -> Optional[str]:
    """把地图 NPC 归一到 npc_state_db 的名字；无记录返回 None。"""
    for db_name in npc_states:
        if _norm(db_name) in map_npc.norm_names:
            return db_name
    return None


def _is_self(map_npc: MapNpc, npc_name: str) -> bool:
    return _norm(npc_name) in map_npc.norm_names


def _co_located_names(
    map_reg: Any, scene: MapScene, npc_name: str, npc_states: dict[str, NPCState],
) -> list[str]:
    """场景内其他 NPC 的显示名（db 名优先，查无记录用地图 label）。"""
    names: list[str] = []
    for map_npc in map_reg.npcs_in_scene(scene.location_id):
        if _is_self(map_npc, npc_name):
            continue
        names.append(resolve_db_name(map_npc, npc_states) or map_npc.label)
    return names


def build_scene_block(
    *, map_reg: Any, npc_states: dict[str, NPCState], npc_name: str,
) -> str:
    """聊天层场景块（无驻地/查不到 → 空串）。"""
    if map_reg is None:
        return ""
    try:
        scenes = map_reg.scenes_for_npc(npc_name)
    except Exception as exc:
        logger.warning("场景块生成失败，降级为空: %s", exc)
        return ""
    if not scenes:
        return ""

    lines: list[str] = []
    if len(scenes) == 1:
        scene = scenes[0]
        lines.append(f"你所在场景：{scene_display(scene)}。")
        names = _co_located_names(map_reg, scene, npc_name, npc_states)
        if names:
            lines.append("同场景npc（均在你视线范围内）：" + "，".join(names) + "。")
        else:
            lines.append("同场景无其他npc。")
    else:
        lines.append(MULTI_SCENE_HEADER)
        for scene in scenes:
            names = _co_located_names(map_reg, scene, npc_name, npc_states)
            head = f"{scene_display(scene)}："
            if names:
                lines.append(head + "同场景npc（均在你视线范围内）：" + "，".join(names) + "。")
            else:
                lines.append(head + "同场景无其他npc。")
    return "\n".join(lines)


def _fmt_npc(db_name: str, state: NPCState) -> str:
    parts = [f"「{db_name}」"]
    if state.sex:
        parts.append(f"（性别：{state.sex}）")
    if state.faction:
        parts.append(f"（阵营：{state.faction}）")
    if state.titles:
        parts.append(f"（身份或称呼：{'、'.join(state.titles)}）")
    return "".join(parts)


def build_scene_supplement(
    *,
    map_reg: Any,
    npc_states: dict[str, NPCState],
    npc_name: str,
    exclude_names: set[str],
) -> str:
    """同阵营块正下方的场景补充描述（仅 db 有记录者；排除已列过者）。"""
    if map_reg is None:
        return ""
    try:
        scenes = map_reg.scenes_for_npc(npc_name)
    except Exception:
        return ""
    if not scenes:
        return ""

    seen: set[str] = set(exclude_names)
    lines: list[str] = []
    for scene in scenes:
        for map_npc in map_reg.npcs_in_scene(scene.location_id):
            if _is_self(map_npc, npc_name):
                continue
            db_name = resolve_db_name(map_npc, npc_states)
            if db_name is None or db_name in seen:
                continue
            seen.add(db_name)
            lines.append(_fmt_npc(db_name, npc_states[db_name]))
    return "\n".join(lines)
