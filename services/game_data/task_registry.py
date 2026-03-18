from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import Task
from .parsers import discover_list_entries, parse_json


class TaskRegistry:
    """
    加载 data/task 下所有任务 JSON（按 task/list.xml），聚合到内存并提供查询。
    """

    def __init__(self, *, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.task_root = (self.data_root / "task").resolve()
        self._by_id: dict[int, Task] = {}
        self._by_npc: dict[str, list[Task]] = {}
        self._reward_types: set[str] = set()

    def load(self) -> None:
        list_xml = self.task_root / "list.xml"
        if not list_xml.exists():
            raise FileNotFoundError(f"未找到 task/list.xml: {list_xml}")

        entries = discover_list_entries(list_xml, tags={"task"})
        tasks: list[Task] = []
        for filename in entries:
            if not filename.lower().endswith(".json"):
                continue
            fp = (self.task_root / filename).resolve()
            if not fp.exists():
                continue
            obj = parse_json(fp)
            task_list = obj.get("tasks", []) if isinstance(obj, dict) else []
            for t in task_list:
                try:
                    task = Task(**t, raw=t)
                except Exception:
                    # 允许部分文件包含额外字段或类型不严格：尽量保留 raw
                    if isinstance(t, dict) and "id" in t:
                        task = Task(
                            id=int(t["id"]),
                            title=str(t.get("title", "")),
                            description=t.get("description"),
                            get_requirements=[int(x) for x in (t.get("get_requirements") or []) if isinstance(x, (int, str))],
                            get_conversation=t.get("get_conversation"),
                            get_npc=t.get("get_npc"),
                            finish_requirements=list(t.get("finish_requirements") or []),
                            finish_submit_items=list(t.get("finish_submit_items") or []),
                            finish_contain_items=list(t.get("finish_contain_items") or []),
                            finish_conversation=t.get("finish_conversation"),
                            finish_npc=t.get("finish_npc"),
                            rewards=list(t.get("rewards") or []),
                            announcement=t.get("announcement"),
                            chain=t.get("chain"),
                            raw=t if isinstance(t, dict) else None,
                        )
                    else:
                        continue
                tasks.append(task)

        self._rebuild_indexes(tasks)

    def _rebuild_indexes(self, tasks: list[Task]) -> None:
        self._by_id = {}
        self._by_npc = {}
        self._reward_types = set()

        for t in tasks:
            self._by_id[t.id] = t

            if t.get_npc:
                self._by_npc.setdefault(t.get_npc, []).append(t)

            for reward in t.rewards or []:
                # rewards: "物品名#数量"
                name = str(reward).split("#", 1)[0].strip()
                if name:
                    self._reward_types.add(name)

    def get_by_id(self, id: int) -> Optional[Task]:
        return self._by_id.get(int(id))

    def list_by_npc(self, npc_name: str) -> list[Task]:
        return list(self._by_npc.get(npc_name, []))

    def get_max_agent_task_id(self) -> int:
        """
        仅按文档的 agent 任务 id 规划区间统计最大值：
        200001–300000（如不存在则返回 200000，便于 next_id = max+1）。
        """

        max_id = 200000
        for tid in self._by_id.keys():
            if 200001 <= tid <= 300000:
                if tid > max_id:
                    max_id = tid
        return max_id

    def list_reward_types(self) -> set[str]:
        return set(self._reward_types)

