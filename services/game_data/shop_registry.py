from __future__ import annotations

import logging
from pathlib import Path

from .parsers import discover_from_list_xml, parse_json

logger = logging.getLogger(__name__)


class ShopRegistry:
    """
    NPC 金币商店。

    优先读新版 npc-shop.v2 格式：`data/shops/list.xml` 索引 `npcs/*.json`，
    每个文件形如 {schema, shopId, title, catalog: {索引: 物品名 | {name, ...}}}，
    以 `shopId` 作为 NPC 键。

    新版缺失（无 list.xml 或未解析出任何商店）时，回退旧版合并文件
    `data/shops/shops.json`（结构：NPC名 -> {索引: 物品名}），以兼容旧版游戏数据。

    对上层暴露的接口与旧版一致：has_shop / get_npc_shop。
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self._shops: dict[str, dict[str, str]] = {}

    def load(self) -> None:
        shops_dir = self.data_root / "shops"
        list_xml = shops_dir / "list.xml"

        if list_xml.exists():
            shops = self._load_npc_shop_v2(list_xml)
            if shops:
                self._shops = shops
                logger.info("商店数据来源: npc-shop.v2，%d 个 NPC 商店", len(shops))
                return

        fp = shops_dir / "shops.json"
        if not fp.exists():
            raise FileNotFoundError(
                f"未找到商店数据：既无新版 {list_xml}（npc-shop.v2），也无旧版 {fp}"
            )
        self._shops = self._parse_legacy(parse_json(fp))
        logger.info("商店数据来源: 旧版 shops.json，%d 个 NPC 商店", len(self._shops))

    @staticmethod
    def _load_npc_shop_v2(list_xml: Path) -> dict[str, dict[str, str]]:
        """解析 npc-shop.v2：list.xml 索引的每个 npcs/*.json 一个商店。"""
        shops: dict[str, dict[str, str]] = {}
        for path in discover_from_list_xml(list_xml):
            if not path.exists():
                logger.warning("商店文件不存在，跳过: %s", path)
                continue
            try:
                doc = parse_json(path)
            except Exception as exc:
                logger.warning("解析商店文件 %s 失败，跳过: %s", path.name, exc)
                continue
            if not isinstance(doc, dict):
                continue
            catalog = doc.get("catalog")
            if not isinstance(catalog, dict):
                continue

            # shopId 是稳定身份；缺失时退回 title / 文件名，保证仍可按 NPC 查询
            key = str(doc.get("shopId") or doc.get("title") or path.stem).strip()
            mapping: dict[str, str] = {}
            for idx, value in catalog.items():
                # catalog 值支持字符串（物品名）与对象（{name, requiredInfo, purchaseLimit}）
                name = value.get("name") if isinstance(value, dict) else value
                if name is None:
                    continue
                name = str(name).strip()
                if name:
                    mapping[str(idx)] = name
            if key and mapping:
                shops[key] = mapping
        return shops

    @staticmethod
    def _parse_legacy(obj: object) -> dict[str, dict[str, str]]:
        """解析旧版合并文件：NPC名 -> {索引: 物品名}。"""
        if not isinstance(obj, dict):
            raise ValueError("shops.json 结构不正确（期望 dict）")
        shops: dict[str, dict[str, str]] = {}
        for npc, mapping in obj.items():
            if not isinstance(mapping, dict):
                continue
            shops[str(npc)] = {str(k): str(v) for k, v in mapping.items()}
        return shops

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
