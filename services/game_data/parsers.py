from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        return int(value)
    except Exception:
        return default


def parse_json(path: Path, encoding: str = "utf-8-sig") -> Any:
    # 兼容部分数据文件带 UTF-8 BOM 的情况：
    # - 正常 UTF-8 也能被 utf-8-sig 正确读取
    # - 带 BOM 的文件则不会触发 “Unexpected UTF-8 BOM” 异常
    with path.open("r", encoding=encoding) as f:
        return json.load(f)


def parse_xml(path: Path, encoding: str = "utf-8") -> ET.Element:
    # ElementTree 能处理 xml 声明里的 encoding；这里保留 encoding 以便未来扩展
    _ = encoding
    tree = ET.parse(path)
    return tree.getroot()


def discover_list_entries(
    list_xml_path: Path,
    *,
    tags: Optional[set[str]] = None,
) -> list[str]:
    """
    从 list.xml / __list__.xml 中提取“条目文本”（不强制要求后缀）。

    文档中多处 list.xml 的用法会记录：
    - 具体文件名：如 task/list.xml 里登记 tasks1.json
    - 无后缀文件名：如 crafting/list.xml 里登记 铁枪会（需外部拼接 .json）
    - 子目录名：如 stages/list.xml 里登记 基地门口（需进入对应目录二次解析）

    参数 tags:
    - None：提取所有有文本的元素
    - set：仅提取 tag 在集合内的元素（大小写不敏感）
    """

    root = parse_xml(list_xml_path)
    allowed = {t.lower() for t in tags} if tags else None

    out: list[str] = []
    seen: set[str] = set()
    for el in root.iter():
        tag = (el.tag or "").lower()
        if allowed is not None and tag not in allowed:
            continue
        if el.text is None:
            continue
        text = el.text.strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def discover_from_list_xml(list_xml_path: Path) -> list[Path]:
    """
    从 list.xml / __list__.xml 发现数据文件列表。

    兼容以下常见形态：
    - <file>relative/path.xml</file>
    - <item file="relative/path.xml" />
    - <entry path="relative/path.xml" />

    返回：相对于 list.xml 所在目录解析后的绝对 Path（存在性不在此处强校验）。
    """

    root = parse_xml(list_xml_path)
    base_dir = list_xml_path.parent

    candidates: list[str] = []
    for el in root.iter():
        tag = (el.tag or "").lower()
        # 兼容：task/list.xml 的 <task>xx.json</task>、
        # items/list.xml 的 <items>xx.xml</items>、
        # stages/list.xml 的 <stages>子目录</stages>（此类不带后缀则由 discover_list_entries 处理）
        if el.text and el.text.strip():
            text = el.text.strip()
            # 只要看起来像文件名，就加入（tag 不做限制）
            if text.lower().endswith((".xml", ".json")):
                candidates.append(text)
                continue

        for attr_key in ("file", "path", "src", "name"):
            v = el.attrib.get(attr_key)
            if v and str(v).strip().lower().endswith((".xml", ".json")):
                candidates.append(str(v).strip())

    # 去重并保持顺序
    seen = set()
    out: list[Path] = []
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        out.append((base_dir / rel).resolve())
    return out


def iter_files(
    root_dir: Path,
    *,
    list_filenames: Iterable[str] = ("list.xml", "__list__.xml"),
    patterns: Iterable[str] = ("*.xml", "*.json"),
) -> list[Path]:
    """
    通用文件发现：
    - 若 root_dir 下存在 list.xml / __list__.xml：优先使用其列出的文件
    - 否则：按 patterns 递归扫描
    """

    root_dir = root_dir.resolve()
    for name in list_filenames:
        list_xml = root_dir / name
        if list_xml.exists() and list_xml.is_file():
            return discover_from_list_xml(list_xml)

    files: list[Path] = []
    for pat in patterns:
        files.extend(root_dir.rglob(pat))

    # 排序稳定化（便于调试与一致性）
    files = sorted({p.resolve() for p in files})
    return files


def extract_data_level_from_item_element(item_el: ET.Element) -> int:
    """
    提取 data.level（支持多种常见结构）：
    1) <data level="3" />
    2) <data><level>3</level></data>
    3) <item level="3" />（兜底）
    """

    # 1) <data level="...">
    data_el = item_el.find("data")
    if data_el is not None:
        if "level" in data_el.attrib:
            return _safe_int(data_el.attrib.get("level"), 0)
        level_el = data_el.find("level")
        if level_el is not None and level_el.text is not None:
            return _safe_int(level_el.text, 0)

    # 3) 兜底：<item level="...">
    if "level" in item_el.attrib:
        return _safe_int(item_el.attrib.get("level"), 0)

    return 0


def extract_item_attributes(item_el: ET.Element) -> dict[str, Any]:
    """
    提取 items XML 里你明确要求记录的字段：
    name, displayname, type, use, actiontype, weapontype, price, data.level

    注意：weapontype 在 <item> 标签属性上（仅枪械有）。
    """

    attrs: dict[str, Any] = {}

    # 1) 优先读取 <item ...> 属性（例如 weapontype）
    for k in ("name", "displayname", "type", "use", "actiontype", "weapontype", "price"):
        if k in item_el.attrib:
            v = item_el.attrib.get(k)
            if v is not None:
                attrs[k] = str(v).strip()

    # 2) 兼容常见结构：字段放在子节点里（例如 <name>xxx</name>）
    for k in ("name", "displayname", "type", "use", "actiontype", "price"):
        if k in attrs:
            continue
        v = item_el.findtext(k)
        if v is not None and v.strip() != "":
            attrs[k] = v.strip()

    attrs["level"] = extract_data_level_from_item_element(item_el)

    # price 转 int（保留原始值在 raw 里由上层决定）
    if "price" in attrs:
        attrs["price"] = _safe_int(attrs.get("price"), default=0)

    return attrs


def normalize_item_use(
    *,
    item_type: Optional[str],
    use: Optional[str],
    source_path: Optional[str],
) -> Optional[str]:
    """
    特殊覆盖规则（基于你给的文件示例）：
    - 消耗品_材料_食材*: use=材料 → 覆盖为 食材
    - 消耗品_药剂_食品.xml: use=药剂 → 覆盖为 食品
    """

    if not use or not source_path:
        return use

    p = Path(source_path).name
    if "消耗品_材料_食材" in p and use == "材料":
        return "食材"
    if p == "消耗品_药剂_食品.xml" and use == "药剂":
        return "食品"

    # 兜底：不改
    _ = item_type
    return use

