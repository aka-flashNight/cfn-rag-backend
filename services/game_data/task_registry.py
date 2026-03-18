from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import Task
from .parsers import discover_list_entries, parse_json
from .reward_utils import parse_name_count


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
        self._submit_items: set[str] = set()
        # 提交/持有物品数量统计：用于 agent 草案校验的“数量合理性”
        self._submit_stats: dict[str, tuple[int, int]] = {}
        self._contain_stats: dict[str, tuple[int, int]] = {}
        # 奖励物品名 -> (min_qty, max_qty)
        self._reward_stats: dict[str, tuple[int, int]] = {}

    def load(self) -> None:
        list_xml = self.task_root / "list.xml"
        if not list_xml.exists():
            raise FileNotFoundError(f"未找到 task/list.xml: {list_xml}")

        entries = discover_list_entries(list_xml, tags={"task"})
        tasks: list[Task] = []
        for filename in entries:
            if not filename.lower().endswith(".json"):
                continue
            # 排除 preview_tasks.json （仅作展示/预览用） bonus_tasks.json（彩蛋任务）
            if filename.lower() == "preview_tasks.json":
                continue
            if filename.lower() == "bonus_tasks.json":
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
        self._submit_items = set()
        self._submit_stats = {}
        self._contain_stats = {}
        self._reward_stats = {}

        for t in tasks:
            self._by_id[t.id] = t

            if t.get_npc:
                self._by_npc.setdefault(t.get_npc, []).append(t)

            # 提交物品池：仅记录物品名集合
            for expr in t.finish_submit_items or []:
                name, _ = parse_name_count(expr)
                if name:
                    self._submit_items.add(name)

            # 提交物品数量统计（finish_submit_items）
            for expr in t.finish_submit_items or []:
                name, count = parse_name_count(expr)
                if not name or count <= 0:
                    continue
                mn, mx = self._submit_stats.get(name, (None, None))  # type: ignore
                if mn is None or count < mn:
                    mn = count
                if mx is None or count > mx:
                    mx = count
                self._submit_stats[name] = (int(mn), int(mx))  # type: ignore

            # 持有物品数量统计（finish_contain_items）
            for expr in t.finish_contain_items or []:
                name, count = parse_name_count(expr)
                if not name or count <= 0:
                    continue
                mn, mx = self._contain_stats.get(name, (None, None))  # type: ignore
                if mn is None or count < mn:
                    mn = count
                if mx is None or count > mx:
                    mx = count
                self._contain_stats[name] = (int(mn), int(mx))  # type: ignore

            # 奖励池：记录物品名集合 + 数量区间
            for reward in t.rewards or []:
                name, count = parse_name_count(reward)
                if not name:
                    continue
                self._reward_types.add(name)
                if count <= 0:
                    continue
                mn, mx = self._reward_stats.get(name, (None, None))  # type: ignore
                if mn is None or count < mn:
                    mn = count
                if mx is None or count > mx:
                    mx = count
                self._reward_stats[name] = (int(mn), int(mx))  # type: ignore

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
        """
        注意：此方法历史命名沿用，但实际返回的是“已有任务 rewards 中出现过的物品名集合”，
        不是“奖励类型集合”（例如 '药剂' / '武器' 这种 item.type）。
        """
        return set(self._reward_types)

    def list_reward_item_names(self) -> set[str]:
        """
        兼容别名：返回“已有任务 rewards 中出现过的物品名集合”。
        """
        return self.list_reward_types()

    def list_submit_items(self) -> set[str]:
        """
        所有任务出现过的提交物品名集合（来自 finish_submit_items）。
        """

        return set(self._submit_items)

    def get_reward_stats(self) -> dict[str, tuple[int, int]]:
        """
        奖励物品统计：
        - key: 物品名
        - value: (min_qty, max_qty) —— 在所有任务 rewards 中出现的数量区间
        """

        return dict(self._reward_stats)

    def get_submit_stats(self) -> dict[str, tuple[int, int]]:
        """
        提交物品数量统计：
        - key: 物品名
        - value: (min_qty, max_qty) —— 在所有任务 finish_submit_items 中出现的数量区间
        """

        return dict(self._submit_stats)

    def get_contain_stats(self) -> dict[str, tuple[int, int]]:
        """
        持有物品数量统计：
        - key: 物品名
        - value: (min_qty, max_qty) —— 在所有任务 finish_contain_items 中出现的数量区间
        """

        return dict(self._contain_stats)

    def list_agent_tasks(self) -> list[Task]:
        """Agent 生成的任务（ID 区间 200001–300000）。"""

        return [t for tid, t in self._by_id.items() if 200001 <= tid <= 300000]

