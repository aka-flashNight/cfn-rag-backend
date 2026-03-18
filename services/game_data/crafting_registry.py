from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import Recipe
from .parsers import discover_list_entries, parse_json


class CraftingRegistry:
    """
    合成配方：data/crafting/list.xml + 多个同名 .json

    list.xml 记录的是“无后缀表名”，需要拼接 `.json`。
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.crafting_root = (self.data_root / "crafting").resolve()
        self._recipes: list[Recipe] = []
        self._by_product: dict[str, Recipe] = {}

    def load(self) -> None:
        list_xml = self.crafting_root / "list.xml"
        if not list_xml.exists():
            raise FileNotFoundError(f"未找到 crafting/list.xml: {list_xml}")

        table_names = discover_list_entries(list_xml, tags={"list"})
        recipes: list[Recipe] = []
        by_product: dict[str, Recipe] = {}

        for table in table_names:
            fp = (self.crafting_root / f"{table}.json").resolve()
            if not fp.exists():
                continue
            obj = parse_json(fp)
            if not isinstance(obj, list):
                continue
            for row in obj:
                if not isinstance(row, dict):
                    continue
                try:
                    rec = Recipe(
                        title=str(row.get("title", "")),
                        name=str(row.get("name", "")),
                        price=int(row.get("price", 0) or 0),
                        kprice=int(row.get("kprice", 0) or 0),
                        value=int(row["value"]) if "value" in row and row["value"] is not None else None,
                        materials=list(row.get("materials") or []),
                        source=table,
                        raw=row,
                    )
                except Exception:
                    continue
                if not rec.name:
                    continue
                recipes.append(rec)
                by_product[rec.name] = rec

        self._recipes = recipes
        self._by_product = by_product

    def search(self, keyword: str, *, limit: int = 50) -> list[Recipe]:
        kw = (keyword or "").strip()
        if not kw:
            return []
        kw_lower = kw.lower()

        out: list[Recipe] = []
        for r in self._recipes:
            hay = " ".join(
                [
                    r.title or "",
                    r.name or "",
                    r.source or "",
                    " ".join(r.materials or []),
                ]
            ).lower()
            if kw_lower in hay:
                out.append(r)
                if len(out) >= limit:
                    break
        return out

    def get_by_product(self, name: str) -> Optional[Recipe]:
        return self._by_product.get(name)

