from __future__ import annotations

from pathlib import Path
from typing import Optional

from xml.etree import ElementTree as ET

from .parsers import discover_from_list_xml, parse_xml


class EquipmentModsRegistry:
    """
    装备插件材料 registry：解析 items/equipment_mods 下的插件配置。

    能回答：
    - 给定物品名，是否为插件？
    - 若是插件，其“等级/档位”前缀（如 低级材料/中等材料/高等材料/特殊材料）是什么？
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.mods_root = (self.data_root / "items" / "equipment_mods").resolve()
        # 物品名 -> 档位前缀（如 "低级材料"），缺省为 None
        self._name_to_tier: dict[str, Optional[str]] = {}

    def load(self) -> None:
        if not self.mods_root.exists():
            self._name_to_tier = {}
            return

        list_xml = self.mods_root / "list.xml"
        if not list_xml.exists():
            self._name_to_tier = {}
            return

        # discover_from_list_xml 返回的是“绝对 Path”，其中 base_dir 为 list.xml 所在目录
        files = discover_from_list_xml(list_xml)
        mapping: dict[str, Optional[str]] = {}

        for fp in files:
            if not fp.suffix.lower().endswith(".xml"):
                continue

            if not fp.exists() or not fp.is_file():
                continue

            tier = self._extract_tier_from_filename(fp.name)
            root = parse_xml(fp)
            # 每个 <mod> 下的 <name> 视为插件物品名
            for mod_el in root.findall(".//mod"):
                name_el = mod_el.find("name")
                if name_el is None or not (name_el.text and name_el.text.strip()):
                    continue
                name = name_el.text.strip()
                # 同名插件如出现在多个文件，以“更高档位”覆盖的需求暂不考虑，后写入者覆盖先写入者即可
                mapping[name] = tier

        self._name_to_tier = mapping

    @staticmethod
    def _extract_tier_from_filename(filename: str) -> Optional[str]:
        """
        从文件名中提取“档位前缀”（可为空），例如：
        - 低级材料_刀专用.xml -> 低级材料
        - 高等材料_通用.xml -> 高等材料
        - 特殊材料_刀专用.xml -> 特殊材料
        如果没有下划线，则返回去掉后缀的整体文件名。
        """

        stem = filename
        if stem.lower().endswith(".xml"):
            stem = stem[:-4]
        if "_" in stem:
            return stem.split("_", 1)[0]
        return stem or None

    def is_plugin(self, item_name: str) -> bool:
        return item_name in self._name_to_tier

    def get_plugin_tier(self, item_name: str) -> Optional[str]:
        return self._name_to_tier.get(item_name)

