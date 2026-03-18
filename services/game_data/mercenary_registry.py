from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .parsers import parse_json


@dataclass
class MercenaryTaskInfo:
    id: int
    title: str
    recommended_min_level: Optional[int]
    recommended_max_level: Optional[int]
    stage_name: Optional[str]
    raw: dict


class MercenaryTaskRegistry:
    """
    mercenary_tasks.json registry：
    - 解析 recommended_level 字符串为 (min, max)
    - 暴露与关卡名的关联（通常在 finish_requirements 里使用 "关卡名#难度"）
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self._tasks: list[MercenaryTaskInfo] = []

    def load(self) -> None:
        fp = (self.data_root / "task" / "mercenary_tasks.json").resolve()
        if not fp.exists():
            self._tasks = []
            return

        obj = parse_json(fp)
        tasks_raw = obj.get("tasks", []) if isinstance(obj, dict) else []

        out: list[MercenaryTaskInfo] = []
        for t in tasks_raw:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("id"))
            except Exception:
                continue

            rec_min, rec_max = self._parse_recommended_level(t.get("recommended_level"))
            # 尝试从 finish_requirements 中解析关卡名（"关卡名#难度"）
            stage_name = None
            for req in t.get("finish_requirements") or []:
                if not isinstance(req, str):
                    continue
                if "#" in req:
                    stage_name = req.split("#", 1)[0].strip()
                    break

            info = MercenaryTaskInfo(
                id=tid,
                title=str(t.get("title", "")),
                recommended_min_level=rec_min,
                recommended_max_level=rec_max,
                stage_name=stage_name,
                raw=t,
            )
            out.append(info)

        self._tasks = out

    @staticmethod
    def _parse_recommended_level(value) -> tuple[Optional[int], Optional[int]]:
        """
        字符串：
        - "20"      -> (20, 20)
        - "20-25"   -> (20, 25)
        非法/缺失 -> (None, None)
        """

        if value is None:
            return None, None
        s = str(value).strip()
        if not s:
            return None, None
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                mn = int(a.strip())
                mx = int(b.strip())
                return mn, mx
            except Exception:
                return None, None
        try:
            v = int(s)
            return v, v
        except Exception:
            return None, None

    def list_all(self) -> list[MercenaryTaskInfo]:
        return list(self._tasks)

