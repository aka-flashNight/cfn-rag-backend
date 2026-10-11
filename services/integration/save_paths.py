"""按存档增量文件的路径约定（B2-2/B2-4/B2-7 的单一真源）。

落点：`data/rag/agent_saves/<slotKey>/`（整体 gitignore，不与正式数据混放）；
槽目录下文件平铺——阶段二先两个：`agent_tasks.json`、`agent_text.json`；
阶段四的 `task_text_overrides.json` / `dialogue_overrides.json` 同目录加入。
共享侧（standalone）对照落点：`data/task/agent_tasks.json` + `data/task/text/agent_text.json`。

data_root 口径与 `GameDataRegistry.data_root` 一致＝`<resources>/data`。
"""

from __future__ import annotations

from pathlib import Path

from services.integration.state import get_integration_state, is_valid_slot_key

#: 按存档目录名（位于 data/rag/ 下；冻结，禁止改名）
AGENT_SAVES_DIRNAME = "agent_saves"


def shared_task_paths(data_root: Path) -> tuple[Path, Path]:
    """共享任务文件（standalone 落点）：(agent_tasks.json, agent_text.json)。"""
    root = Path(data_root).resolve()
    return (
        (root / "task" / "agent_tasks.json").resolve(),
        (root / "task" / "text" / "agent_text.json").resolve(),
    )


def agent_save_dir(data_root: Path, slot_key: str) -> Path:
    """槽位目录 data/rag/agent_saves/<slotKey>/（不校验键、不保证存在）。"""
    return (Path(data_root).resolve() / "rag" / AGENT_SAVES_DIRNAME / slot_key).resolve()


def agent_save_task_paths(data_root: Path, slot_key: str) -> tuple[Path, Path]:
    """槽位任务文件（平铺）：(agent_tasks.json, agent_text.json)。"""
    slot_dir = agent_save_dir(data_root, slot_key)
    return (
        (slot_dir / "agent_tasks.json").resolve(),
        (slot_dir / "agent_text.json").resolve(),
    )


def resolve_task_write_paths(data_root: Path) -> tuple[Path, Path, str | None]:
    """按当前绑定解析任务写盘落点（B2-2）：bind→槽目录两文件（文件级隔离）；
    未 bind→共享文件（现状）。返回 (tasks 路径, text 路径, slot_key|None)。

    绑定了但 slot_key 非法（含路径分隔符等）→ ValueError，拒绝写盘（fail-closed），
    绝不回退到共享文件——那会把某存档的任务写进全存档可见的共享数据。
    """
    slot_key = get_integration_state().slot_key
    if slot_key is None:
        tasks_path, text_path = shared_task_paths(data_root)
        return tasks_path, text_path, None
    if not is_valid_slot_key(slot_key):
        raise ValueError(f"存档槽位键非法，拒绝写盘: {slot_key!r}")
    tasks_path, text_path = agent_save_task_paths(data_root, slot_key)
    return tasks_path, text_path, slot_key
