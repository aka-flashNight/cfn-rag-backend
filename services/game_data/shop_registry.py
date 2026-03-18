from __future__ import annotations

from pathlib import Path
from typing import Optional

from .parsers import parse_json


class ShopRegistry:
    """
    NPC 金币商店：data/shops/shops.json
    结构：NPC名 -> {"0": "物品名", "1": "物品名", ...}
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self._shops: dict[str, dict[str, str]] = {}

    def load(self) -> None:
        fp = (self.data_root / "shops" / "shops.json").resolve()
        if not fp.exists():
            raise FileNotFoundError(f"未找到 shops.json: {fp}")
        obj = parse_json(fp)
        if not isinstance(obj, dict):
            raise ValueError("shops.json 结构不正确（期望 dict）")
        # 强制转为 str->str
        shops: dict[str, dict[str, str]] = {}
        for npc, mapping in obj.items():
            if not isinstance(mapping, dict):
                continue
            shops[str(npc)] = {str(k): str(v) for k, v in mapping.items()}
        self._shops = shops

    def has_shop(self, npc_name: str) -> bool:
        return npc_name in self._shops

    def get_npc_shop(self, npc_name: str) -> list[str]:
        mapping = self._shops.get(npc_name)
        if not mapping:
            return []
        # 索引键是字符串数字：按数值排序输出稳定列表
        def _key(k: str) -> int:
            try:
                return int(k)
            except Exception:
                return 10**9

        return [mapping[k] for k in sorted(mapping.keys(), key=_key)]

