from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from xml.etree import ElementTree as ET

from .models import Item
from .parsers import (
    extract_item_attributes,
    iter_files,
    normalize_item_use,
    parse_xml,
)
from .paths import get_game_data_root
from .text_utils import strip_game_markup


@dataclass(frozen=True)
class ItemQuery:
    name: Optional[str] = None
    type: Optional[str] = None
    use: Optional[str] = None
    min_level: Optional[int] = None
    max_level: Optional[int] = None


class ItemRegistry:
    """
    加载 items/ 下所有物品 XML，建立索引，支持查询。
    """

    def __init__(self, *, data_root: Optional[Path] = None, items_dir: str = "items"):
        self.data_root = Path(data_root).resolve() if data_root is not None else get_game_data_root()
        self.items_root = (self.data_root / items_dir).resolve()

        self._items: list[Item] = []
        self._by_name: dict[str, Item] = {}
        self._by_type: dict[str, list[Item]] = {}
        self._by_use: dict[str, list[Item]] = {}

    @property
    def items(self) -> list[Item]:
        return list(self._items)

    def load(self, *, only_files: Optional[Iterable[Path]] = None) -> None:
        if not self.items_root.exists():
            raise FileNotFoundError(f"items 目录不存在: {self.items_root}")

        if only_files is not None:
            files = [Path(p).resolve() for p in only_files]
        else:
            files = iter_files(self.items_root, patterns=("*.xml",))
        items: list[Item] = []

        for fp in files:
            if not fp.exists() or not fp.is_file():
                continue
            items.extend(self._parse_items_file(fp))

        self._items = items
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._by_name = {}
        self._by_type = {}
        self._by_use = {}

        for it in self._items:
            # name 唯一：后加载覆盖先加载（便于热更新/覆盖包）
            self._by_name[it.name] = it

            if it.type:
                self._by_type.setdefault(it.type, []).append(it)
            if it.use:
                self._by_use.setdefault(it.use, []).append(it)

    def _parse_items_file(self, path: Path) -> list[Item]:
        root = parse_xml(path)

        # 兼容不同根结构：可能 root 就是 <items> 或者单个 <item>
        item_els: Iterable[ET.Element]
        if (root.tag or "").lower() == "item":
            item_els = [root]
        else:
            item_els = root.findall(".//item")

        out: list[Item] = []
        for el in item_els:
            attrs = extract_item_attributes(el)
            name = attrs.get("name")
            if not name:
                # 没有 name 的节点直接跳过
                continue

            displayname = attrs.get("displayname")
            item_type = attrs.get("type")
            use = attrs.get("use")

            normalized_use = normalize_item_use(
                item_type=item_type,
                use=use,
                source_path=str(path),
            )

            # displayname 缺失时：默认用 name（这样有效 displayname 会自然变成 None）
            if displayname is None:
                displayname = name

            # price：extract 中已尽量转 int；但这里仍保持 Optional[int] 语义
            price = attrs.get("price")
            if price == 0 and "price" not in el.attrib:
                price = None

            desc_raw = attrs.get("description")
            desc_clean = strip_game_markup(desc_raw) if desc_raw else None
            if desc_clean == "":
                desc_clean = None

            item = Item(
                name=str(name),
                displayname=str(displayname) if displayname is not None else None,
                type=str(item_type) if item_type is not None else None,
                use=str(normalized_use) if normalized_use is not None else None,
                actiontype=str(attrs.get("actiontype")) if attrs.get("actiontype") is not None else None,
                weapontype=str(attrs.get("weapontype")) if attrs.get("weapontype") is not None else None,
                price=price,
                description=desc_clean,
                weight=(str(attrs["weight"]).strip() if attrs.get("weight") is not None else None),
                clipname=(str(attrs["clipname"]).strip() if attrs.get("clipname") is not None else None),
                level=int(attrs.get("level") or 0),
                source_path=str(path),
                raw={"xml_attrib": dict(el.attrib)},
            )
            out.append(item)

        return out

    def get_by_name(self, name: str) -> Optional[Item]:
        return self._by_name.get(name)

    def get_price(self, name: str) -> int:
        it = self.get_by_name(name)
        if it is None or it.price is None:
            return 0
        return int(it.price)

    def list_by_type(self, type: str) -> list[Item]:
        return list(self._by_type.get(type, []))

    def list_by_level_range(self, min_level: int, max_level: int) -> list[Item]:
        return [it for it in self._items if min_level <= (it.level or 0) <= max_level]

    def search(
        self,
        keyword: str,
        *,
        type: Optional[str] = None,
        use: Optional[str] = None,
        limit: int = 50,
    ) -> list[Item]:
        kw = (keyword or "").strip()
        if not kw:
            return []
        kw_lower = kw.lower()

        results: list[Item] = []
        for it in self._items:
            if type and it.type != type:
                continue
            if use and it.use != use:
                continue
            hay = f"{it.name} {it.displayname or ''}".lower()
            if kw_lower in hay:
                results.append(it)
                if len(results) >= limit:
                    break
        return results

    def find(
        self,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
        use: Optional[str] = None,
        min_level: Optional[int] = None,
        max_level: Optional[int] = None,
    ) -> list[Item]:
        q = ItemQuery(name=name, type=type, use=use, min_level=min_level, max_level=max_level)
        return list(self._iter_query(q))

    def _iter_query(self, q: ItemQuery) -> Iterable[Item]:
        if q.name:
            it = self._by_name.get(q.name)
            if not it:
                return []
            if not self._match_filters(it, q):
                return []
            return [it]

        # 初始候选集：尽量用索引缩小
        if q.use and q.use in self._by_use:
            candidates = self._by_use[q.use]
        elif q.type and q.type in self._by_type:
            candidates = self._by_type[q.type]
        else:
            candidates = self._items

        return [it for it in candidates if self._match_filters(it, q)]

    @staticmethod
    def _match_filters(it: Item, q: ItemQuery) -> bool:
        if q.type and it.type != q.type:
            return False
        if q.use and it.use != q.use:
            return False
        if q.min_level is not None and it.level < q.min_level:
            return False
        if q.max_level is not None and it.level > q.max_level:
            return False
        return True

