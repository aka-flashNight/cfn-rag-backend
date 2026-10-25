"""NPC 日常对话基础数据（``data/dialogues/*.xml``）的内存视图。

按 ``list.xml`` 登记项读取，只认 ``<Dialogues><Name>`` 作为 NPC 键——游戏侧
``DataQueryTask.QueryNpcDialogue`` 就是用这个名字查表（``XmlDataLoader.cs:83-93``），
而检索语料里的 ``character`` 取的是组内首个非 ``$PC`` 的 ``<Name>``，实测 370 组中
有 14 组两者不一致，故覆盖层一律按块名定位（开发计划 §2.0.4）。

工具执行是同步 CPU 操作（``BaseTool.run`` 契约），所以这里按 (mtime, size) 做
整目录缓存：13 个 XML 共约 330KB，只在文件真变化时重解析。
"""

from __future__ import annotations

import logging
import threading
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from services.dialogue.models import DialogueGroup, DialogueLine, sanitize_text

logger = logging.getLogger(__name__)


def _stat_or_none(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


class DialogueSources:
    """日常对话基础数据读取器（实例级缓存，跨调用复用）。"""

    def __init__(self, data_root: Path) -> None:
        self.dialogues_dir = (Path(data_root).resolve() / "dialogues").resolve()
        self._fingerprint: tuple[tuple[str, tuple[int, int] | None], ...] | None = None
        self._by_npc: dict[str, list[DialogueGroup]] = {}
        self._order: list[str] = []
        #: 说话人名 → 基础数据里出现过的 Title（按出现序，写盘行补 Title 用）
        self._titles: dict[str, list[str]] = {}
        # 实例经 shared_sources() 进程级共享，可能被 to_thread 的调用方并发触碰
        self._lock = threading.RLock()

    # ------------------------------------------------------------------

    def _list_filenames(self) -> list[str]:
        list_path = self.dialogues_dir / "list.xml"
        if not list_path.is_file():
            logger.warning("未找到对话列表文件: %s", list_path)
            return []
        try:
            root = ET.parse(list_path).getroot()
        except (OSError, ET.ParseError) as exc:
            logger.error("解析 %s 失败: %s", list_path, exc)
            return []
        return [
            (elem.text or "").strip()
            for elem in root.findall(".//items")
            if (elem.text or "").strip()
        ]

    def _fingerprint_of(self, filenames: list[str]) -> tuple[tuple[str, tuple[int, int] | None], ...]:
        return tuple(
            (name, _stat_or_none(self.dialogues_dir / name)) for name in filenames
        )

    def _ensure_locked(self) -> None:
        """重解析基础数据；调用方须持有 self._lock。"""
        filenames = self._list_filenames()
        fingerprint = self._fingerprint_of(filenames)
        if fingerprint == self._fingerprint:
            return

        by_npc: dict[str, list[DialogueGroup]] = {}
        order: list[str] = []
        for name in filenames:
            path = self.dialogues_dir / name
            if not path.is_file():
                continue
            try:
                root = ET.parse(path).getroot()
            except (OSError, ET.ParseError) as exc:
                # 单文件坏了不拖垮整份基础数据：记错并跳过（fail-closed 口径）
                logger.error("解析对话文件失败，跳过 %s: %s", name, exc)
                continue
            for block in root.findall(".//Dialogues"):
                npc = sanitize_text(block.findtext("Name"))
                if not npc:
                    continue
                groups = by_npc.setdefault(npc, [])
                if npc not in order:
                    order.append(npc)
                for elem in block.findall("Dialogue"):
                    group = self._parse_group(npc, elem, name)
                    if group is not None:
                        groups.append(group)

        titles: dict[str, list[str]] = {}
        for groups in by_npc.values():
            for group in groups:
                for line in group.lines:
                    # 玩家行的 Title 是 $PC_TITLE 占位符，不是称号，不进表
                    if line.name and line.title and not line.is_player:
                        titles.setdefault(line.name, []).append(line.title)

        self._by_npc = by_npc
        self._order = order
        self._titles = titles
        self._fingerprint = fingerprint

    @staticmethod
    def _parse_group(npc: str, elem: ET.Element, source_file: str) -> DialogueGroup | None:
        group_id = (elem.get("id") or "").strip()
        if not group_id:
            logger.warning("对话组缺 id 属性，跳过（%s / %s）", npc, source_file)
            return None
        try:
            requirement = int((elem.findtext("TaskRequirement") or "0").strip() or 0)
        except ValueError:
            requirement = 0

        lines: list[DialogueLine] = []
        for sub in elem.findall("SubDialogue"):
            text = sanitize_text(sub.findtext("Text"))
            if not text:
                continue
            lines.append(DialogueLine(
                name=sanitize_text(sub.findtext("Name")),
                title=sanitize_text(sub.findtext("Title")),
                char=sanitize_text(sub.findtext("Char")),
                text=text,
            ))
        if not lines:
            return None
        return DialogueGroup(
            npc=npc,
            group_id=group_id,
            lines=lines,
            task_requirement=requirement,
            source_file=source_file,
        )

    # ------------------------------------------------------------------

    def npc_names(self) -> list[str]:
        """全部日常对话 NPC 名（按 list.xml 原序，同 NPC 多块只出现一次）。"""
        with self._lock:
            self._ensure_locked()
            return list(self._order)

    def groups(self, npc: str) -> list[DialogueGroup]:
        """该 NPC 的基础对话组副本（按文件内原序）；无此 NPC 返回空表。"""
        with self._lock:
            self._ensure_locked()
            return [g.copy() for g in self._by_npc.get((npc or "").strip(), [])]

    def group(self, npc: str, group_id: str) -> DialogueGroup | None:
        """按 (NPC名, 组id) 取基础组副本。"""
        target = str(group_id)
        for group in self.groups(npc):
            if group.group_id == target:
                return group
        return None

    def iter_all_groups(self):
        """遍历全部基础组（关键词搜索用；产出副本）。"""
        with self._lock:
            self._ensure_locked()
            snapshot = [g for npc in self._order for g in self._by_npc.get(npc, [])]
        for group in snapshot:
            yield group.copy()

    def speaker_title(self, speaker: str) -> str:
        """该说话人在基础对话数据里的 Title（同人有多个时取出现最多的）；查不到＝空串。"""
        key = (speaker or "").strip()
        if not key:
            return ""
        with self._lock:
            self._ensure_locked()
            seen = list(self._titles.get(key, ()))
        if not seen:
            return ""
        return Counter(seen).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# 写盘行的 Title / 情绪白名单（日常对话与任务文本两条写入路径共用同一口径）
# ---------------------------------------------------------------------------

def resolve_speaker_title(
    speaker: str,
    sources: DialogueSources | None = None,
    *,
    own: str = "",
) -> str:
    """一行台词的 Title 取值（四级回落；玩家行不走这里，走 ``player_defaults``）。

    1. ``own``：本次改动范围内该说话人已有的 Title——游戏里真显示过的那个，最权威；
    2. 基础日常对话数据里该说话人的 Title；
    3. ``npc_state_db.json`` 里该 NPC 的第一个称号；
    4. 说话人名。

    第 2 级排在第 3 级之前，是因为实测两者并不总相等：92 组 (Name, Title) 里有 8 组
    状态库首称号与基础数据不同（如「前治安官」基础数据用「前线统筹」、状态库首称号是
    「警长」），而玩家看到的铭牌取自基础数据。第 4 级只是不留空的保底——基础数据
    693 行里 Title 等于 Name 的是 0 行，拿名字当称号是凭空造一个游戏里没有的显示值。
    """
    name = (speaker or "").strip()
    own_title = (own or "").strip()
    if own_title:
        return own_title
    if sources is not None:
        found = sources.speaker_title(name)
        if found:
            return found
    from services.npc.manager import peek_static

    for title in getattr(peek_static(name), "titles", None) or []:
        if str(title or "").strip():
            return str(title).strip()
    return name


def speaker_emotions(speaker: str) -> list[str]:
    """该说话人在 NPC 设定里的情绪白名单；查不到＝空表（情绪按短标签放行）。"""
    from services.npc.manager import peek_static

    raw = getattr(peek_static((speaker or "").strip()), "emotions", None) or []
    return [str(item).strip() for item in raw if str(item or "").strip()]
