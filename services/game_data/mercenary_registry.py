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
    # challenge 模式（challenge.difficulty != "简单"）
    challenge_difficulty: Optional[str]
    challenge_recommended_min_level: Optional[int]
    challenge_recommended_max_level: Optional[int]
    challenge_description: Optional[str]
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

            # challenge 解析：仅当 challenge.difficulty 存在且非 "简单" 时才作为“额外难度”使用
            challenge_difficulty: Optional[str] = None
            challenge_description: Optional[str] = None
            cmin: Optional[int]
            cmax: Optional[int]
            cmin, cmax = None, None

            challenge_obj = t.get("challenge")
            if isinstance(challenge_obj, dict):
                cd = challenge_obj.get("difficulty")
                if isinstance(cd, str) and cd.strip():
                    challenge_difficulty = cd.strip()
                desc = challenge_obj.get("description")
                if isinstance(desc, str) and desc.strip():
                    challenge_description = desc.strip()

                # challenge.recommended_level 解析；若缺失，按你的“同逻辑”回退到根 recommended_level
                cmin, cmax = self._parse_recommended_level(challenge_obj.get("recommended_level"))
                if challenge_difficulty and challenge_difficulty != "简单":
                    if cmin is None and cmax is None:
                        cmin, cmax = rec_min, rec_max
                else:
                    # challenge == "简单" 或缺失：视为没有额外难度
                    challenge_difficulty = None
                    cmin, cmax = None, None

            info = MercenaryTaskInfo(
                id=tid,
                title=str(t.get("title", "")),
                recommended_min_level=rec_min,
                recommended_max_level=rec_max,
                challenge_difficulty=challenge_difficulty,
                challenge_recommended_min_level=cmin,
                challenge_recommended_max_level=cmax,
                challenge_description=challenge_description,
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

