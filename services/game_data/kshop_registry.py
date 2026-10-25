from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .models import KShopItem
from .parsers import discover_from_list_xml, parse_json

logger = logging.getLogger(__name__)


class KShopRegistry:
    """
    K 点商城。

    优先读新版逐店格式：`data/kshop/list.xml` 索引各店 json，每个文件是
    [{id, item, type, price}, ...] 数组，汇总为全量条目。

    新版缺失（无 list.xml 或未解析出任何条目）时，回退旧版合并文件
    `data/kshop/kshop.json`（同为数组结构），以兼容旧版游戏数据。

    对上层暴露的接口与旧版一致：list_items / get_by_name。
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self._items: list[KShopItem] = []
        self._by_name: dict[str, KShopItem] = {}

    def load(self) -> None:
        ks_dir = self.data_root / "kshop"
        list_xml = ks_dir / "list.xml"

        rows: list = []
        source = ""
        if list_xml.exists():
            rows = self._load_per_store(list_xml)
            if rows:
                source = "新版逐店文件"

        if not rows:
            fp = ks_dir / "kshop.json"
            if not fp.exists():
                raise FileNotFoundError(
                    f"未找到 K 点商城数据：既无新版 {list_xml}（逐店），也无旧版 {fp}"
                )
            obj = parse_json(fp)
            if not isinstance(obj, list):
                raise ValueError("kshop.json 结构不正确（期望 list）")
            rows = obj
            source = "旧版 kshop.json"

        self._items, self._by_name = self._build_items(rows)
        logger.info("K 点商城数据来源: %s，%d 条", source, len(self._items))

    @staticmethod
    def _load_per_store(list_xml: Path) -> list:
        """汇总 list.xml 索引的每个商店 json（每个文件一个条目数组）。"""
        rows: list = []
        for path in discover_from_list_xml(list_xml):
            if not path.exists():
                logger.warning("K 点商店文件不存在，跳过: %s", path)
                continue
            try:
                obj = parse_json(path)
            except Exception as exc:
                logger.warning("解析 K 点商店文件 %s 失败，跳过: %s", path.name, exc)
                continue
            if isinstance(obj, list):
                rows.extend(row for row in obj if isinstance(row, dict))
            else:
                logger.warning("K 点商店文件 %s 结构非数组，跳过", path.name)
        return rows

    @staticmethod
    def _build_items(rows: list) -> tuple[list[KShopItem], dict[str, KShopItem]]:
        items: list[KShopItem] = []
        by_name: dict[str, KShopItem] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                price = int(str(row.get("price", "0")).strip() or "0")
            except Exception:
                price = 0
            it = KShopItem(
                id=str(row.get("id", "")),
                item=str(row.get("item", "")),
                type=str(row.get("type")) if row.get("type") is not None else None,
                price=price,
                raw=row,
            )
            if it.item:
                by_name[it.item] = it
            items.append(it)
        return items, by_name

    def list_items(self) -> list[KShopItem]:
        return list(self._items)

    def get_by_name(self, name: str) -> Optional[KShopItem]:
        return self._by_name.get(name)
