"""NPC→场景注册表（提示词信息供给前置 P1）。

唯一来源＝游戏仓 data/map/map_definition.json：
- npcs（label/runtimeNames/aliases）与 locations（label＋sceneBinding.linkage）；
- placements（npcId→locationId，按文件顺序保存；enabled=false 的落点与场景跳过）。
名字匹配一律「小写＋去空白」归一（大小写差异对 Guitar/guitar 等 4 组与
别名对 杀马特/∞天ㄙ★使的剪∞ 由此对齐）；查不到＝无驻地（不拼场景块）。
纯静态数据，进程内缓存，无落盘。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import MapNpc, MapScene
from .parsers import parse_json


def _norm(name: str) -> str:
    """小写 + 去空白（跨库名字模糊匹配；保留资产，与 orchestrator 侧口径一致）。"""
    return "".join(str(name).lower().split())


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v or "").strip()]


class MapRegistry:
    """map_definition.json 的只读查询层（加载后进程内缓存）。"""

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self._file = self.data_root / "map" / "map_definition.json"

        self._npcs: dict[str, MapNpc] = {}
        self._scenes: dict[str, MapScene] = {}
        self._npcs_by_location: dict[str, list[str]] = {}
        self._scenes_by_npc: dict[str, list[str]] = {}
        self._norm_index: dict[str, str] = {}

    def load(self) -> None:
        raw = parse_json(self._file)
        if not isinstance(raw, dict):
            raise ValueError(f"map_definition.json 结构异常（非对象）: {self._file}")

        scenes: dict[str, MapScene] = {}
        for loc_id, item in (raw.get("locations") or {}).items():
            if not isinstance(item, dict) or item.get("enabled") is False:
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            binding = item.get("sceneBinding")
            linkage = ""
            if isinstance(binding, dict):
                linkage = str(binding.get("linkage") or "").strip()
            scenes[str(loc_id)] = MapScene(
                location_id=str(loc_id), label=label, linkage=linkage,
            )

        npcs: dict[str, MapNpc] = {}
        norm_index: dict[str, str] = {}
        for npc_id, item in (raw.get("npcs") or {}).items():
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            key = str(npc_id)
            norms: set[str] = set()
            for name in [
                label,
                *_str_list(item.get("runtimeNames")),
                *_str_list(item.get("aliases")),
            ]:
                norms.add(_norm(name))
            npcs[key] = MapNpc(key=key, label=label, norm_names=norms)
            for n in norms:
                # 地图内归一名字理论唯一；万一冲突以先到者为准
                norm_index.setdefault(n, key)

        npcs_by_location: dict[str, list[str]] = {}
        scenes_by_npc: dict[str, list[str]] = {}
        for pl in (raw.get("placements") or {}).values():
            if not isinstance(pl, dict) or pl.get("enabled") is False:
                continue
            npc_key = str(pl.get("npcId") or "")
            loc_id = str(pl.get("locationId") or "")
            if npc_key not in npcs or loc_id not in scenes:
                continue
            bucket = npcs_by_location.setdefault(loc_id, [])
            if npc_key not in bucket:
                bucket.append(npc_key)
            lst = scenes_by_npc.setdefault(npc_key, [])
            if loc_id not in lst:
                lst.append(loc_id)

        self._npcs = npcs
        self._scenes = scenes
        self._npcs_by_location = npcs_by_location
        self._scenes_by_npc = scenes_by_npc
        self._norm_index = norm_index

    def scenes_for_npc(self, npc_name: str) -> list[MapScene]:
        """NPC 常去场景列表（按 placements 文件顺序）；查不到/无驻地返回空列表。"""
        key = self._norm_index.get(_norm(npc_name or ""))
        if not key:
            return []
        return [
            self._scenes[loc]
            for loc in self._scenes_by_npc.get(key, [])
            if loc in self._scenes
        ]

    def npcs_in_scene(self, location_id: str) -> list[MapNpc]:
        """场景内 NPC 列表（按 placements 文件顺序，含 npc_state_db 范围外的 NPC）。"""
        return [
            self._npcs[k]
            for k in self._npcs_by_location.get(str(location_id), [])
            if k in self._npcs
        ]
