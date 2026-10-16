"""NPC 状态管理：进程内单例 + asyncio.Lock + 防抖落盘（修 E1 并发丢失更新）。

对应 docs/v3-developer/06-存储启动与打包瘦身.md §1.1：
- 启动时一次性读 npc_state_db.json 入内存 dict（替代旧「每请求全量读 JSON」）；
- 读写经 asyncio.Lock（读多写少，单锁即可）；
- update_favorability 只改内存 + 置 dirty；后台 5s 防抖落盘（tmp+replace 原子写）；
- 进程退出前 flush；
- 条目新增可选字段 appearance（形象描述，07 使用），读取缺省容忍，
  写回保留未知字段（forward-compatible）。

按存档作用域（B3-1）：文件结构为 {"global": {…}, "saves": {"<slotKey>": {…}}}。
静态身份字段（sex/faction/titles/emotions/challenge/appearance 与未知字段）只存
global——它们是 NPC 设定，不随存档变化，整条复制到每个存档会让新存档里的 NPC
丢掉阵营与头衔（同阵营提示词块与情绪白名单都会跟着退化）。只有 PER_SAVE_FIELDS
（好感度与关系等级）按存档隔离：绑定存档时写入 saves.<slotKey>，未绑定时写 global。
读取时合并——绑定存档下某 NPC 没有存档内记录时直接沿用 global 的那份，
那是设定里的初始好感（A 兵团 25/熟悉、其余 10/陌生），归零等于每个新存档都从零开始。
加载时若检测到旧版平铺结构（顶层直接是 NPC 名），整体视为 global 读入，
下一次落盘自然写成新结构，不做额外迁移/回填。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from services.game_data.paths import find_resources_directory
from services.integration.state import get_integration_state

logger = logging.getLogger(__name__)

AUTOSAVE_DEBOUNCE_S = 5.0

#: 文件结构的两个顶层键
GLOBAL_KEY = "global"
SAVES_KEY = "saves"

#: 按存档隔离的字段；其余字段一律只在 global 里维护
PER_SAVE_FIELDS = ("favorability", "relationship_level")


def _empty_state() -> "NPCState":
    """作用域内查无此 NPC 时的初始状态（好感 0／陌生／只允许「普通」情绪）。"""
    return NPCState(favorability=0, relationship_level="陌生", emotions=["普通"])


def get_npc_state_path(resources_dir: Path | None = None) -> Path:
    """npc_state_db.json 路径：<resources>/data/rag/npc_state_db.json。"""
    root = Path(resources_dir) if resources_dir else find_resources_directory()
    return root / "data" / "rag" / "npc_state_db.json"


@dataclass
class NPCState:
    favorability: int = 0
    relationship_level: str = "陌生"
    sex: str | None = None
    # 可选：当前 NPC 的切磋关卡名（npc_state_db.json 的 challenge 字段）
    challenge: str | None = None
    emotions: list[str] = field(default_factory=list)
    faction: str | None = None
    titles: list[str] = field(default_factory=list)
    # 形象描述文本（07 立绘与多模态使用；缺省容忍）
    appearance: str | None = None
    # npc_state_db.json 中的未知字段原样保留（forward-compatible 写回）
    extra: dict[str, Any] = field(default_factory=dict)


class NPCManager:
    """NPC 好感度与关系等级的本地状态（单例 + 单锁 + 按存档作用域）。"""

    def __init__(
        self,
        state: dict[str, NPCState],
        state_path: Path | None = None,
        saves: dict[str, dict[str, NPCState]] | None = None,
    ) -> None:
        #: global 作用域：NPC 全量设定 ＋ 未绑定存档时的好感度
        self._global: dict[str, NPCState] = state
        #: saves.<slotKey> 作用域：只含 PER_SAVE_FIELDS
        self._saves: dict[str, dict[str, NPCState]] = dict(saves or {})
        self._path = state_path or get_npc_state_path()
        self._lock = asyncio.Lock()
        self._dirty = False
        self._autosave_task: asyncio.Task | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    @classmethod
    async def load(cls, state_path: Path | None = None) -> "NPCManager":
        """从本地 JSON 初始化（一次全量读入内存）。文件不存在时从对话数据初始化 NPC 列表。"""
        path = state_path or get_npc_state_path()
        raw_global: dict[str, dict[str, Any]] = {}
        raw_saves: dict[str, dict[str, dict[str, Any]]] = {}
        if path.exists():
            try:
                text = await asyncio.to_thread(path.read_text, "utf-8")
                parsed = json.loads(text) if text.strip() else {}
                if isinstance(parsed, dict):
                    raw_global, raw_saves = cls._split_scopes(parsed)
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("npc_state_db.json 读取失败，按空状态启动: %s", exc)

        state = {name: cls._state_from_item(item) for name, item in raw_global.items()}
        saves = {
            slot: {name: cls._state_from_item(item) for name, item in entries.items()}
            for slot, entries in raw_saves.items()
        }

        manager = cls(state=state, state_path=path, saves=saves)
        if not path.exists():
            await manager._init_from_dialogues()
            await manager.flush()
        return manager

    @staticmethod
    def _split_scopes(
        parsed: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
        """拆出 (global 条目, saves 条目)。

        顶层带 global/saves 键＝新结构；否则是旧版平铺结构（顶层直接是 NPC 名），
        整体视为 global 读入，下次落盘自然写成新结构。
        """
        def _entries(raw: Any) -> dict[str, dict[str, Any]]:
            return {k: v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}

        if isinstance(parsed.get(GLOBAL_KEY), dict) or isinstance(parsed.get(SAVES_KEY), dict):
            saves_raw = parsed.get(SAVES_KEY)
            saves = (
                {str(slot): _entries(entries) for slot, entries in saves_raw.items()}
                if isinstance(saves_raw, dict) else {}
            )
            return _entries(parsed.get(GLOBAL_KEY)), saves
        return _entries(parsed), {}

    @staticmethod
    def _state_from_item(item: dict[str, Any]) -> NPCState:
        favorability = int(item.get("favorability", 0) or 0)
        level = str(item.get("relationship_level") or "").strip() or NPCManager._compute_relationship_level(favorability)

        def _str_list(key: str) -> list[str]:
            raw = item.get(key)
            return [v.strip() for v in raw if isinstance(v, str) and v.strip()] if isinstance(raw, list) else []

        def _opt_str(key: str) -> str | None:
            v = str(item.get(key) or "").strip()
            return v or None

        emotions = _str_list("emotions") or ["普通"]
        modeled = {"favorability", "relationship_level", "sex", "challenge", "emotions", "faction", "titles", "appearance"}
        extra = {k: v for k, v in item.items() if k not in modeled}
        return NPCState(
            favorability=favorability,
            relationship_level=level,
            sex=_opt_str("sex"),
            challenge=_opt_str("challenge"),
            emotions=emotions,
            faction=_opt_str("faction"),
            titles=_str_list("titles"),
            appearance=_opt_str("appearance"),
            extra=extra,
        )

    async def _init_from_dialogues(self) -> None:
        """文件缺失时从对话数据提取 NPC 名称初始化状态（不阻断启动）。"""
        try:
            dialogues_dir = find_resources_directory() / "data" / "dialogues"
            list_path = dialogues_dir / "list.xml"
            if not list_path.exists():
                logger.warning("未找到对话列表文件: %s", list_path)
                return

            def _collect() -> set[str]:
                names: set[str] = set()
                for name in [
                    (elem.text or "").strip()
                    for elem in ET.parse(list_path).getroot().findall(".//items")
                    if (elem.text or "").strip()
                ]:
                    file_path = dialogues_dir / name
                    if not file_path.exists():
                        continue
                    xml_root = ET.parse(file_path).getroot()
                    file_level = xml_root.find(".//Dialogues/Name")
                    file_level_name = file_level.text.strip() if file_level is not None and file_level.text else None
                    for sub in xml_root.findall(".//SubDialogue"):
                        char_elem = sub.find("Char")
                        name_elem = sub.find("Name")
                        char_name = None
                        if char_elem is not None and (char_elem.text or "").strip():
                            char_name = char_elem.text.strip()
                        elif name_elem is not None and (name_elem.text or "").strip():
                            char_name = name_elem.text.strip()
                        elif file_level_name:
                            char_name = file_level_name
                        # 跳过玩家占位符（$PC 等）
                        if char_name and not char_name.startswith("$"):
                            names.add(char_name)
                return names

            npc_names = await asyncio.to_thread(_collect)
            for npc_name in sorted(npc_names):
                self._global.setdefault(npc_name, _empty_state())
            logger.info("从对话数据初始化了 %d 个 NPC", len(npc_names))
        except Exception as exc:
            logger.error("从对话数据初始化 NPC 失败（不影响启动）: %s", exc)

    # ------------------------------------------------------------------
    # 作用域（B3-1）
    # ------------------------------------------------------------------

    @staticmethod
    def _current_slot() -> str | None:
        """当前存档槽位；未绑定＝None（读写都落 global 作用域）。"""
        return get_integration_state().slot_key

    def _scoped_locked(self, npc_name: str, slot: str | None) -> NPCState:
        """合并出该作用域下的 NPC 状态：设定取 global，好感度取作用域内记录。

        绑定存档但作用域内没有该 NPC 的记录＝这个存档还没动过这份好感，
        直接返回 global 那一份：那是 NPC 设定里的初始好感（A 兵团 25/熟悉、
        其余 10/陌生），归零会让每个新存档都从「谁都不认识」开始。
        """
        base = self._global.get(npc_name)
        if slot is None:
            return base if base is not None else _empty_state()
        override = self._saves.get(slot, {}).get(npc_name)
        if base is None:
            return override if override is not None else _empty_state()
        if override is None:
            return base
        return replace(
            base,
            favorability=override.favorability,
            relationship_level=override.relationship_level,
        )

    # ------------------------------------------------------------------
    # 读取 / 更新（全部经 asyncio.Lock）
    # ------------------------------------------------------------------

    def peek_static(self, npc_name: str) -> NPCState | None:
        """同步只读该 NPC 的静态设定（单例内查无此 NPC＝None）。

        对话写盘链路（``agent_tools.task_tools`` / ``dialogue.editor``）按
        ``BaseTool.run`` 契约是同步的，等不到 await；而 titles 与 emotions 属加载后
        不再变动的设定字段（改好感的 ``replace`` 只作用于 saves 作用域的副本），
        故直接读 global 作用域，不经 asyncio 锁也不触发任何写。
        """
        return self._global.get((npc_name or "").strip())

    async def get(self, npc_name: str) -> NPCState:
        async with self._lock:
            return self._scoped_locked(npc_name.strip(), self._current_slot())

    async def all_states(self) -> dict[str, NPCState]:
        """当前作用域下的全量 NPC 状态（设定来自 global，好感度按作用域合并）。"""
        async with self._lock:
            slot = self._current_slot()
            return {name: self._scoped_locked(name, slot) for name in self._global}

    async def update_favorability(self, npc_name: str, change_value: int) -> NPCState:
        """更新好感度（内存 + 置 dirty，防抖落盘）；区间映射自动刷新关系等级。

        - 0-20 陌生 / 20-50 熟悉 / 50-80 朋友 / 80-100 生死之交

        写入落到当前作用域：绑定存档→saves.<slotKey>（只存好感度与关系等级），
        未绑定→global（整条设定一起写回）。
        """
        npc_name = npc_name.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")
        async with self._lock:
            slot = self._current_slot()
            current = self._scoped_locked(npc_name, slot)
            new_fav = max(0, min(100, current.favorability + int(change_value)))
            new_level = self._compute_relationship_level(new_fav)
            updated = replace(current, favorability=new_fav, relationship_level=new_level)
            if slot is None:
                self._global[npc_name] = updated
            else:
                self._saves.setdefault(slot, {})[npc_name] = NPCState(
                    favorability=new_fav, relationship_level=new_level,
                )
            self._mark_dirty_locked()
            return updated

    @staticmethod
    def _compute_relationship_level(favorability: int) -> str:
        if favorability < 20:
            return "陌生"
        if favorability < 50:
            return "熟悉"
        if favorability < 80:
            return "朋友"
        return "生死之交"

    # ------------------------------------------------------------------
    # 防抖落盘
    # ------------------------------------------------------------------

    def _mark_dirty_locked(self) -> None:
        self._dirty = True
        if self._autosave_task is None or self._autosave_task.done():
            loop = asyncio.get_running_loop()
            self._autosave_task = loop.create_task(self._autosave_after_debounce())

    async def _autosave_after_debounce(self) -> None:
        """防抖窗口（5s）内无新写入才落盘；窗口内有新写入则继续顺延。"""
        try:
            while True:
                await asyncio.sleep(AUTOSAVE_DEBOUNCE_S)
                async with self._lock:
                    if not self._dirty:
                        return
                    self._dirty = False
                await self._write_file()
        except asyncio.CancelledError:
            raise

    async def flush(self) -> None:
        """立即落盘（进程退出前 / 关键节点调用）。"""
        async with self._lock:
            self._dirty = False
        await self._write_file()

    async def _write_file(self) -> None:
        async with self._lock:
            snapshot: dict[str, Any] = {
                GLOBAL_KEY: {
                    name: self._serialize_item(state) for name, state in self._global.items()
                },
                SAVES_KEY: {
                    slot: {
                        name: {key: getattr(state, key) for key in PER_SAVE_FIELDS}
                        for name, state in entries.items()
                    }
                    for slot, entries in self._saves.items()
                },
            }
        await asyncio.to_thread(self._atomic_write, snapshot)

    @staticmethod
    def _serialize_item(state: NPCState) -> dict[str, Any]:
        item: dict[str, Any] = {
            "favorability": state.favorability,
            "relationship_level": state.relationship_level,
        }
        if state.sex:
            item["sex"] = state.sex
        if state.challenge:
            item["challenge"] = state.challenge
        if state.emotions:
            item["emotions"] = list(state.emotions)
        if state.faction:
            item["faction"] = state.faction
        if state.titles:
            item["titles"] = list(state.titles)
        if state.appearance:
            item["appearance"] = state.appearance
        # 未知字段原样写回（forward-compatible）
        item.update(state.extra)
        return item

    def _atomic_write(self, snapshot: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp_name, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self._path)
        except Exception:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise

    async def stop(self) -> None:
        """停止防抖任务并 flush（应用关闭时调用）。"""
        if self._autosave_task is not None and not self._autosave_task.done():
            self._autosave_task.cancel()
            try:
                await self._autosave_task
            except asyncio.CancelledError:
                pass
        await self.flush()


_NPC_MANAGER: NPCManager | None = None


async def get_npc_manager() -> NPCManager:
    """全局单例（startup 初始化；未初始化时懒加载兜底）。"""
    global _NPC_MANAGER
    if _NPC_MANAGER is not None:
        return _NPC_MANAGER
    manager = await NPCManager.load()
    _NPC_MANAGER = manager
    return manager


def set_npc_manager(manager: NPCManager | None) -> None:
    global _NPC_MANAGER
    _NPC_MANAGER = manager


def peek_static(npc_name: str) -> NPCState | None:
    """同步取该 NPC 的静态设定；单例未初始化（未走 startup 的脚本/测试）＝None。"""
    return _NPC_MANAGER.peek_static(npc_name) if _NPC_MANAGER is not None else None
