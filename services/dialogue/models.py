"""对话修改（阶段四）的数据模型。

两个覆盖层的行结构不同，但共用一个 ``DialogueLine``：

- 日常对话覆盖层 ``dialogue_overrides.json`` 的 ``subdialogues`` 用 **PascalCase**
  键（``Name/Title/Char/Text``），与基础 XML 一致，由宿主照 ``XmlDataLoader``
  的既有映射转成运行时的小写键（附录 B-5 冻结口径）；
- 任务文本覆盖层 ``task_text_overrides.json`` 是平铺的 ``$KEY → 数组``，数组项
  用**小写**键（``name/title/char/text``，可带 ``imageurl``），与
  ``data/task/text/*.json`` 逐字一致，靠既有 dictMerge 语义生效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: 玩家台词的 Name 占位符（对话文本里成对出现 Title=$PC_TITLE、Char=$PC_CHAR）
PLAYER_NAME = "$PC"
#: 玩家台词的 Title / Char 占位符：AS2 `getDialogueSpecialString` 只认这三个字面值
#: （``$PC``→角色名、``$PC_TITLE``→玩家称号、``$PC_CHAR``→「玩家」），写成别的即原样显示。
PLAYER_TITLE = "$PC_TITLE"
PLAYER_CHAR = "$PC_CHAR"


def player_defaults() -> tuple[str, str]:
    """玩家行缺省的 (Title, Char)。"""
    return PLAYER_TITLE, PLAYER_CHAR


#: 情绪标签长度上限（基础数据里的情绪都是两三字的短词）
MAX_EMOTION_CHARS = 12


def normalize_emotion(raw: Any, allowed: Iterable[str] | None = None) -> str:
    """归一模型给的情绪标签；不合法＝空串（该行 Char 不带表情后缀）。

    模型可能把整串 Char（``名#情绪``）或一整句话塞进 emotion，直接拼会得到
    ``名#名#情绪`` 这类 AS2 ``split("#")`` 认不出的帧标签，故只取末段并卡长度；
    给了白名单（NPC 设定里的 ``emotions``）就只认白名单里的标签——查不到白名单时
    放行短标签，免得把没登记情绪的 NPC 的台词全削成无表情。
    """
    text = str(raw or "").strip().rsplit("#", 1)[-1].strip()
    if not text or len(text) > MAX_EMOTION_CHARS:
        return ""
    pool = {str(item).strip() for item in (allowed or []) if str(item or "").strip()}
    return text if not pool or text in pool else ""


def compose_char(name: str, emotion: str) -> str:
    """Char 拼接：有情绪＝``名#情绪``，没情绪＝纯名字（基础数据里的主要写法）。"""
    return f"{name}#{emotion}" if emotion else name


#: 单组行数上限（基础数据实测最多 10 行，留足新增余量）
MAX_LINES_PER_GROUP = 24
#: 单行台词字符上限（基础数据实测最长 159 字）
MAX_LINE_TEXT_CHARS = 400


def sanitize_text(value: Any) -> str:
    """清掉模型可能带出的控制字符与首尾空白。

    XML 1.0 非法字符（``\\x00-\\x08`` 等）会让宿主的 XmlDocument 解析失败，
    而对话覆盖层最终要经宿主读进游戏，故在写盘前统一滤掉；制表/换行保留。
    """
    text = str(value or "")
    cleaned = "".join(
        ch for ch in text
        if ch in "\t\n\r" or ord(ch) >= 0x20
    )
    return cleaned.strip()


@dataclass(frozen=True)
class DialogueLine:
    """一行台词。``extra`` 装 imageurl 之类的附加字段，读写全程原样保留。"""

    name: str
    title: str
    char: str
    text: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_player(self) -> bool:
        return self.name == PLAYER_NAME or self.char.startswith("$PC")

    def copy(self) -> "DialogueLine":
        return DialogueLine(
            name=self.name,
            title=self.title,
            char=self.char,
            text=self.text,
            extra=dict(self.extra),
        )

    def to_subdialogue(self) -> dict[str, Any]:
        """日常对话覆盖层条目（PascalCase；基础 XML 无 Target/ImageUrl 字段，故不输出）。"""
        return {
            "Name": self.name,
            "Title": self.title,
            "Char": self.char,
            "Text": self.text,
        }

    def to_task_text_item(self) -> dict[str, Any]:
        """任务文本条目（小写；键序与基础文件一致，附加字段跟在后面）。"""
        item: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "char": self.char,
            "text": self.text,
        }
        item.update(self.extra)
        return item

    @classmethod
    def from_subdialogue(cls, raw: Any) -> "DialogueLine | None":
        """解析 PascalCase 条目；缺文本或不是对象时返回 None。"""
        if not isinstance(raw, dict):
            return None
        text = sanitize_text(raw.get("Text"))
        if not text:
            return None
        return cls(
            name=str(raw.get("Name") or "").strip(),
            title=str(raw.get("Title") or "").strip(),
            char=str(raw.get("Char") or "").strip(),
            text=text,
        )

    @classmethod
    def from_task_text_item(cls, raw: Any) -> "DialogueLine | None":
        """解析小写条目；缺文本或不是对象时返回 None。"""
        if not isinstance(raw, dict):
            return None
        text = sanitize_text(raw.get("text"))
        if not text:
            return None
        extra = {
            k: v for k, v in raw.items()
            if k not in ("name", "title", "char", "text")
        }
        return cls(
            name=str(raw.get("name") or "").strip(),
            title=str(raw.get("title") or "").strip(),
            char=str(raw.get("char") or "").strip(),
            text=text,
            extra=extra,
        )


@dataclass
class DialogueGroup:
    """一组连续对话（一个 ``<Dialogue>`` 或一个任务 ``$KEY`` 指向的数组）。"""

    #: 日常对话＝``<Dialogues><Name>``；任务对话＝归属任务 id（仅用于展示与去重）
    npc: str
    #: 日常对话＝``<Dialogue id>`` 或覆盖层分配的 ``agent_<n>``；任务对话＝文本 key
    group_id: str
    lines: list[DialogueLine] = field(default_factory=list)
    #: 主线进度门槛（``<TaskRequirement>``，缺省 0＝始终可用）；任务对话恒 0
    task_requirement: int = 0
    #: 该组是否来自覆盖层（区分基础数据与已改写内容）
    from_overlay: bool = False
    #: 基础文件名，仅用于日志与排查
    source_file: str = ""

    def copy(self) -> "DialogueGroup":
        return DialogueGroup(
            npc=self.npc,
            group_id=self.group_id,
            lines=[line.copy() for line in self.lines],
            task_requirement=self.task_requirement,
            from_overlay=self.from_overlay,
            source_file=self.source_file,
        )

    def speaks(self, npc_name: str) -> bool:
        """组内是否有该 NPC 说话（候选门槛：没有就不进候选）。"""
        target = (npc_name or "").strip()
        if not target:
            return False
        return any(line.name == target for line in self.lines)
