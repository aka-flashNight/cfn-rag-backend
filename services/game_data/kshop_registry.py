from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import KShopItem
from .parsers import parse_json


class KShopRegistry:
    """
    K 点商城：data/kshop/kshop.json
    结构：[{id, item, type, price}, ...]
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self._items: list[KShopItem] = []
        self._by_name: dict[str, KShopItem] = {}

    def load(self) -> None:
        fp = (self.data_root / "kshop" / "kshop.json").resolve()
        if not fp.exists():
            raise FileNotFoundError(f"未找到 kshop.json: {fp}")
        obj = parse_json(fp)
        if not isinstance(obj, list):
            raise ValueError("kshop.json 结构不正确（期望 list）")

        items: list[KShopItem] = []
        by_name: dict[str, KShopItem] = {}
        for row in obj:
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

        self._items = items
        self._by_name = by_name

    def list_items(self) -> list[KShopItem]:
        return list(self._items)

    def get_by_name(self, name: str) -> Optional[KShopItem]:
        return self._by_name.get(name)

