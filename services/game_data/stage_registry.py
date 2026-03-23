from __future__ import annotations

from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from services.game_progress import get_progress_stage_config, is_valid_stage_root

from .models import LootCrate, LootDrop, StageInfo
from .parsers import _safe_int, discover_list_entries, parse_xml
from .text_utils import strip_game_markup


class StageRegistry:
    """
    关卡数据注册表：
    - data/stages/list.xml：列出子目录（大区）
    - data/stages/<area>/__list__.xml：列出 StageInfo（含 UnlockCondition）
    - data/stages/<area>/<stage>.xml：扫描箱子掉落（纸箱/资源箱/装备箱）
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.stages_root = (self.data_root / "stages").resolve()

        self._stage_infos: dict[tuple[str, str], StageInfo] = {}
        self._loot_cache: dict[tuple[str, str], list[LootCrate]] = {}

    def load(self) -> None:
        list_xml = self.stages_root / "list.xml"
        if not list_xml.exists():
            raise FileNotFoundError(f"未找到 stages/list.xml: {list_xml}")

        areas = discover_list_entries(list_xml, tags={"stages"})
        infos: dict[tuple[str, str], StageInfo] = {}

        for area in areas:
            area = str(area).strip()
            if not area:
                continue
            if not is_valid_stage_root(area):
                continue
            list_fp = (self.stages_root / area / "__list__.xml").resolve()
            if not list_fp.exists():
                continue
            for si in self._parse_stage_list(area, list_fp):
                infos[(si.area, si.name)] = si

        self._stage_infos = infos
        self._loot_cache = {}

    def _parse_stage_list(self, area: str, list_path: Path) -> list[StageInfo]:
        root = parse_xml(list_path)
        out: list[StageInfo] = []

        # 示例根节点为 <Stages>，子节点 <StageInfo>
        for el in root.findall(".//StageInfo"):
            name = (el.findtext("Name") or "").strip()
            if not name:
                continue
            unlock_raw = el.findtext("UnlockCondition")
            unlock = _safe_int(unlock_raw, default=0) if unlock_raw is not None else None
            if unlock is not None and unlock <= 0:
                # 文档规则：没有 UnlockCondition 不作为 agent 候选关卡
                unlock = None
            desc_raw = (el.findtext("Description") or "").strip()
            desc_clean = strip_game_markup(desc_raw) if desc_raw else None
            if desc_clean == "":
                desc_clean = None
            si = StageInfo(
                area=area,
                name=name,
                type=(el.findtext("Type") or "").strip() or None,
                unlock_condition=unlock,
                description=desc_clean,
                raw={"list_path": str(list_path)},
            )
            out.append(si)
        return out

    def get_unlock_condition(self, area: str, stage_name: str) -> int:
        si = self._stage_infos.get((area, stage_name))
        if si is None or si.unlock_condition is None:
            return 0
        return int(si.unlock_condition)

    def list_stages_for_progress(self, stage: int) -> list[StageInfo]:
        """
        按玩家 progress_stage（1-7）筛选候选关卡：
        - area 必须属于该阶段对应的大区（由 services/game_progress.py 提供）
        - 且必须有 unlock_condition
        """

        cfg = get_progress_stage_config(stage)
        area = cfg.stage_name if cfg is not None else None
        if not area:
            return []
        out: list[StageInfo] = []
        for (a, _), si in self._stage_infos.items():
            if a != area:
                continue
            if si.unlock_condition is None:
                continue
            out.append(si)
        return out

    def get_stage_loot(self, area: str, stage_name: str) -> list[LootCrate]:
        key = (area, stage_name)
        if key in self._loot_cache:
            return list(self._loot_cache[key])

        stage_fp = (self.stages_root / area / f"{stage_name}.xml").resolve()
        if not stage_fp.exists():
            self._loot_cache[key] = []
            return []

        crates = self._scan_loot_from_stage_xml(stage_fp)
        self._loot_cache[key] = crates
        return list(crates)

    @staticmethod
    def _scan_loot_from_stage_xml(stage_path: Path) -> list[LootCrate]:
        root = parse_xml(stage_path)
        allowed = {"纸箱", "资源箱", "装备箱"}

        crates: list[LootCrate] = []

        # 在任何 SubStage/Instances/Instance 下找 Identifier 与 Parameters/掉落物
        for inst in root.findall(".//Instance"):
            identifier = (inst.findtext("Identifier") or "").strip()
            if identifier not in allowed:
                continue
            params = inst.find("Parameters")
            if params is None:
                continue

            # 若箱子带有主线进度限制，则整个箱子不参与 Agent 任务候选
            has_min_prog = params.findtext("最小主线进度") is not None
            has_max_prog = params.findtext("最大主线进度") is not None
            if has_min_prog or has_max_prog:
                continue

            drops: list[LootDrop] = []
            for drop_el in params.findall(".//掉落物"):
                name = (drop_el.findtext("名字") or "").strip()
                if not name:
                    continue

                # 兼容 XML 中“缺失最小/最大数量”的默认规则：
                # - 两者都缺失或空：min=1, max=1
                # - 仅最大存在：min=1, max=最大值
                # - 仅最小存在：min=最小值, max=最小值
                min_raw = drop_el.findtext("最小数量")
                max_raw = drop_el.findtext("最大数量")
                min_present = min_raw is not None and str(min_raw).strip() != ""
                max_present = max_raw is not None and str(max_raw).strip() != ""

                if not min_present and not max_present:
                    mn = 1
                    mx = 1
                elif not min_present and max_present:
                    mn = 1
                    mx = _safe_int(max_raw, default=1)
                elif min_present and not max_present:
                    mn = _safe_int(min_raw, default=1)
                    mx = mn
                else:
                    mn = _safe_int(min_raw, default=0)
                    mx = _safe_int(max_raw, default=0)

                drops.append(LootDrop(name=name, min_count=mn, max_count=mx))

            if drops:
                crates.append(LootCrate(identifier=identifier, drops=drops, raw={"stage_path": str(stage_path)}))

        return crates

