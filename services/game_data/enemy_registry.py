"""关卡→怪物注册表（提示词信息供给前置 P2）。

链路：关卡 XML `<Enemy><Type>兵种N</Type>` → data/units/units.json（id=N → spritename）
→ data/enemy_properties/*.xml（模板 tag=spritename → displayname＋<标识>/<阶段>/<档次系数>）。
产出：`monsters_for_spawns()` 按 displayname 聚合（Quantity 求和；强度＝阶段×档次系数，
两者缺一为 None、不参与强度比较；无 displayname 的模板跳过不列）。
纯静态数据，进程内缓存，无落盘。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .models import EnemySpawn, StageMonster
from .parsers import _safe_int, discover_list_entries, parse_json, parse_xml

_UNIT_TYPE_RE = re.compile(r"兵种(\d+)")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if s == "":
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class EnemyTemplate:
    """enemy_properties 模板（强度＝阶段×档次系数，缺一为 None）。"""

    spritename: str
    displayname: str
    strength: Optional[float] = None


class EnemyRegistry:
    """units.json 与 enemy_properties 的只读查询层（加载后进程内缓存）。"""

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self._units_file = self.data_root / "units" / "units.json"
        self._props_dir = self.data_root / "enemy_properties"
        self._spritename_by_id: dict[int, str] = {}
        self._templates: dict[str, EnemyTemplate] = {}

    def load(self) -> None:
        units_raw = parse_json(self._units_file)
        by_id: dict[int, str] = {}
        if isinstance(units_raw, list):
            for entry in units_raw:
                if not isinstance(entry, dict):
                    continue
                eid = _safe_int(entry.get("id"), default=-1)
                spritename = str(entry.get("spritename") or "").strip()
                if eid >= 0 and spritename:
                    by_id[eid] = spritename
        self._spritename_by_id = by_id

        templates: dict[str, EnemyTemplate] = {}
        list_fp = self._props_dir / "list.xml"
        if list_fp.exists():
            for name in discover_list_entries(list_fp, tags={"items"}):
                fp = (self._props_dir / name).resolve()
                if not fp.exists():
                    continue
                for el in parse_xml(fp):
                    tag = (el.tag or "").strip()
                    if not tag:
                        continue
                    displayname = (el.findtext("displayname") or "").strip()
                    stage = 0
                    coef: Optional[float] = None
                    mark = el.find("标识")
                    if mark is not None:
                        stage = _safe_int(mark.findtext("阶段"), default=0)
                        coef = _safe_float(mark.findtext("档次系数"))
                    strength = float(stage) * coef if stage > 0 and coef is not None else None
                    templates.setdefault(
                        tag,
                        EnemyTemplate(spritename=tag, displayname=displayname, strength=strength),
                    )
        self._templates = templates

    def spritename_for_unit_type(self, unit_type: str) -> Optional[str]:
        """兵种 Type（兵种N）→ units.json 的 spritename；无法映射返回 None。"""
        m = _UNIT_TYPE_RE.fullmatch((unit_type or "").strip())
        if m is None:
            return None
        return self._spritename_by_id.get(int(m.group(1)))

    def monsters_for_spawns(self, spawns: list[EnemySpawn]) -> list[StageMonster]:
        """把关卡 Enemy 条目聚合为怪物列表（按首次出现顺序；无 displayname 跳过）。"""
        order: list[str] = []
        agg: dict[str, list] = {}
        for spawn in spawns or []:
            spritename = self.spritename_for_unit_type(spawn.unit_type)
            tpl = self._templates.get(spritename) if spritename else None
            if tpl is None or not tpl.displayname:
                continue
            if tpl.displayname not in agg:
                agg[tpl.displayname] = [0, []]
                order.append(tpl.displayname)
            agg[tpl.displayname][0] += int(spawn.quantity or 0)
            if tpl.strength is not None:
                agg[tpl.displayname][1].append(tpl.strength)

        monsters: list[StageMonster] = []
        for displayname in order:
            quantity, strengths = agg[displayname]
            monsters.append(StageMonster(
                displayname=displayname,
                quantity=quantity,
                strength=max(strengths) if strengths else None,
            ))
        return monsters
