"""按存档增量文件的路径约定（B2-2/B2-4/B2-7 的单一真源）。

落点：`data/rag/agent_saves/<slotKey>/`（整体 gitignore，不与正式数据混放）；
槽目录下文件平铺——阶段二两个：`agent_tasks.json`、`agent_text.json`；
阶段四两个覆盖层：`dialogue_overrides.json`、`task_text_overrides.json`。
共享侧（standalone）对照落点：`data/task/agent_tasks.json` + `data/task/text/agent_text.json`。
覆盖层**没有**共享侧落点（未绑定存档即无处可写，见 resolve_override_paths）。

data_root 口径与 `GameDataRegistry.data_root` 一致＝`<resources>/data`。
"""

from __future__ import annotations

from pathlib import Path

from services.integration.state import get_integration_state, is_valid_slot_key

#: 按存档目录名（位于 data/rag/ 下；冻结，禁止改名）
AGENT_SAVES_DIRNAME = "agent_saves"

#: 阶段四两个覆盖层文件名（冻结，禁止改名；见开发计划 §2.0.2／附录 B-5）
DIALOGUE_OVERRIDES_FILENAME = "dialogue_overrides.json"
TASK_TEXT_OVERRIDES_FILENAME = "task_text_overrides.json"


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


def agent_save_override_paths(data_root: Path, slot_key: str) -> tuple[Path, Path]:
    """槽位覆盖层文件（平铺）：(dialogue_overrides.json, task_text_overrides.json)。"""
    slot_dir = agent_save_dir(data_root, slot_key)
    return (
        (slot_dir / DIALOGUE_OVERRIDES_FILENAME).resolve(),
        (slot_dir / TASK_TEXT_OVERRIDES_FILENAME).resolve(),
    )


def resolve_override_paths(data_root: Path) -> tuple[Path, Path, str]:
    """按当前绑定解析覆盖层落点，返回 (对话覆盖层, 任务文本覆盖层, slot_key)。

    覆盖层是**按存档**数据，没有共享落点：未绑定存档时无处可写 → ValueError
    （fail-closed，与 resolve_task_write_paths 同一口径，绝不回退到全存档可见的位置）。
    """
    slot_key = get_integration_state().slot_key
    if slot_key is None:
        raise ValueError("未绑定存档，对话覆盖层无处落盘")
    if not is_valid_slot_key(slot_key):
        raise ValueError(f"存档槽位键非法，拒绝写盘: {slot_key!r}")
    dialogue_path, text_path = agent_save_override_paths(data_root, slot_key)
    return dialogue_path, text_path, slot_key


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
