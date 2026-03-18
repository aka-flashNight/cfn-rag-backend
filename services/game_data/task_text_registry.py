from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .parsers import discover_list_entries, parse_json


class TaskTextRegistry:
    """
    加载 data/task/text 下所有文本 JSON（按 task/text/list.xml），聚合为一个 dict。
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.text_root = (self.data_root / "task" / "text").resolve()
        self._texts: dict[str, Any] = {}

    def load(self) -> None:
        list_xml = self.text_root / "list.xml"
        if not list_xml.exists():
            raise FileNotFoundError(f"未找到 task/text/list.xml: {list_xml}")

        entries = discover_list_entries(list_xml, tags={"text"})
        merged: dict[str, Any] = {}
        for filename in entries:
            if not filename.lower().endswith(".json"):
                continue
            fp = (self.text_root / filename).resolve()
            if not fp.exists():
                continue
            obj = parse_json(fp)
            if isinstance(obj, dict):
                merged.update(obj)
        self._texts = merged

    def get(self, key: str, default: Any = None) -> Any:
        return self._texts.get(key, default)

    def all(self) -> dict[str, Any]:
        return dict(self._texts)

    def resolve_str(self, key: Optional[str]) -> Optional[str]:
        """
        若 key 是文本 key（如 $MAIN_TITLE_0）且存在映射，则返回映射值；否则原样返回。
        """

        if not key:
            return None
        v = self._texts.get(key)
        if v is None:
            return key
        if isinstance(v, str):
            return v
        return key

