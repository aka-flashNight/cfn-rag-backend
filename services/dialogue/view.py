"""对话修改子 Agent 的候选视图：四种查询范围、编号与文本渲染。

范围口径（用户裁定）。**这里是多选，不是单选**——一次查询可以同时给多个范围，
结果按范围分桶拼接后统一编号；给模型的参数说明里必须写明「多选」。

- ``daily_all`` 日常全量（推荐）：该 NPC 的全部日常对话组（基础 ∪ 本存档覆盖层）；
- ``task_recent`` 任务近期（推荐）：未完成、非 Agent 的任务里，最近的**主线 5 组＋支线 5 组**；
- ``keyword`` 关键词（可选）：在**未被上面选中范围覆盖的**组里逐个搜关键词取并集，上限 30 组；
- ``task_all`` 任务全量（**不推荐**）：该 NPC 说话的全部未完成任务对话组。
  ⚠ 与 ``task_recent`` 冲突，两者都选时采用 ``task_recent``。
  ⚠ 给模型的注释**必须连理由一起写全，不能只留「不推荐」三个字**（用户裁定 2026-10-13：
  光写不推荐会让模型觉得这个选项莫名其妙）。逐字口径：
  「不推荐。除非是非常重大的关系变化、导致所有对话都必须全部修改时，才选中任务全量；否则不建议选。」

两条硬门槛（对所有范围一致）：

1. **组内必须有该 NPC 说话**才算候选——没有就不在（用户裁定）；
2. **已完成任务与 Agent 任务全部排除**（改了玩家也看不到，且 Agent 任务的对话
   只在发布时生成一次，不进后续修改范围）。

⚠ NPC 身份一律用 ``<Dialogues><Name>``（块名），不用检索语料里的 ``character``：
实测 370 组里有 14 组两者不一致，而游戏侧 ``QueryNpcDialogue`` 是按块名查表的。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from services.dialogue.models import DialogueGroup, DialogueLine, PLAYER_NAME
from services.dialogue import overrides as overlay
from services.dialogue import task_text as text_overlay
from services.dialogue.sources import DialogueSources

logger = logging.getLogger(__name__)

SCOPE_DAILY_ALL = "daily_all"
SCOPE_TASK_RECENT = "task_recent"
SCOPE_KEYWORD = "keyword"
SCOPE_TASK_ALL = "task_all"

ALL_SCOPES: tuple[str, ...] = (SCOPE_DAILY_ALL, SCOPE_TASK_RECENT, SCOPE_KEYWORD, SCOPE_TASK_ALL)

#: 单次查询返回的台词总字符预算（用户裁定：15000 差不多，放宽到 30000 也行）
CANDIDATE_TEXT_BUDGET = 30000
#: 关键词命中组数上限
KEYWORD_GROUP_CAP = 30
#: 任务近期的主线／支线组数
TASK_RECENT_MAIN_GROUPS = 5
TASK_RECENT_SIDE_GROUPS = 5

#: Agent 任务 id 区间（与 TaskRegistry.list_agent_tasks 同口径）
AGENT_TASK_ID_RANGE = (200001, 300000)

_CHAIN_MAIN = "主线"
_CHAIN_SIDE = "支线"
#: 无链序号时排到最后（同前缀内的兜底序）
_CHAIN_INDEX_LAST = 1 << 30


def is_agent_task_id(task_id: Any) -> bool:
    """是否 Agent 生成的委托（id 落在 200001–300000）。"""
    try:
        tid = int(task_id)
    except (TypeError, ValueError):
        return False
    low, high = AGENT_TASK_ID_RANGE
    return low <= tid <= high


def normalize_finished_ids(raw: Any) -> set[int]:
    """存档投影的 ``tasks_finished`` → 已完成任务 id 集合。

    投影口径是「完成 id 的整数数组」（通信_aka_Agent联动.as:523-557）；同时容忍
    「id→完成次数」表（值为次数非布尔，显式 >=1，禁 ==true）。
    """
    out: set[int] = set()
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                if float(value) >= 1:
                    out.add(int(key))
            except (TypeError, ValueError):
                continue
        return out
    for item in raw if isinstance(raw, (list, tuple, set)) else []:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class CandidateGroup:
    """一个可修改的对话组（编号从 1 起；0 保留给「新增一组日常对话」）。"""

    number: int
    kind: str                 # "daily" | "task"
    npc: str                  # 该组归属的 NPC（块名）
    key: str                  # 日常＝组 id；任务＝文本 key（$MAIN_GET_0）
    task_id: int | None = None
    task_kind: str = ""       # "get"（接取）| "finish"（完成）
    heading: str = ""         # 组说明行（不含编号前缀）
    body: str = ""            # 台词行（不含组说明）
    group: DialogueGroup = field(default_factory=lambda: DialogueGroup(npc="", group_id=""))

    @property
    def render_size(self) -> int:
        """拼给模型时这一组占的字符数（编号前缀按固定 8 字估）。"""
        return len(self.heading) + len(self.body) + 8

    def to_index_entry(self) -> dict[str, Any]:
        """编号映射表条目（跨轮传递用，纯数据不含对象引用）。"""
        return {
            "number": self.number,
            "kind": self.kind,
            "npc": self.npc,
            "key": self.key,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
        }


@dataclass
class CandidateSelection:
    """一次查询的结果：已编号的组 + 截断说明。"""

    npc: str
    groups: list[CandidateGroup] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def index_entries(self) -> list[dict[str, Any]]:
        return [g.to_index_entry() for g in self.groups]

    def render(self) -> str:
        """拼给模型的候选清单（编号在前，台词按行序；Title 不拼、也不允许模型写）。"""
        if not self.groups:
            head = f"未找到「{self.npc}」说话的可修改对话组。"
            return "\n".join([head, *self.notes]) if self.notes else head

        blocks = [f"以下是「{self.npc}」可修改的对话组，共 {len(self.groups)} 组，均已编号："]
        blocks.extend(
            f"【编号 {cand.number}】{cand.heading}\n{cand.body}" for cand in self.groups
        )
        blocks.extend(self.notes)
        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 视图
# ---------------------------------------------------------------------------

class DialogueView:
    """按 (data_root, 存档槽位, 已完成任务) 给出的对话合并视图与候选筛选。"""

    def __init__(
        self,
        *,
        data_root: Path,
        game_data: Any = None,
        slot_key: str | None = None,
        finished_ids: Iterable[int] | None = None,
        sources: DialogueSources | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.game_data = game_data
        self.slot_key = slot_key
        self.finished_ids = normalize_finished_ids(finished_ids)
        self.sources = sources if sources is not None else DialogueSources(self.data_root)

    # --- 覆盖层路径 -----------------------------------------------------

    def _override_paths(self) -> tuple[Path | None, Path | None]:
        """读侧路径：未绑定存档＝没有覆盖层（两个 None，静默按基础数据走）。"""
        if not self.slot_key:
            return None, None
        from services.integration.save_paths import agent_save_override_paths

        dialogue_path, text_path = agent_save_override_paths(self.data_root, self.slot_key)
        return dialogue_path, text_path

    def override_paths(self) -> tuple[Path, Path]:
        """写侧路径：与读侧同一个槽位；没有槽位 → ValueError（覆盖层无共享落点）。

        必须用视图自己的 ``slot_key`` 而不是重读全局绑定：对话修改子 Agent 最长跑
        900 秒，期间玩家换档会把 A 档的改动写进 B 档的覆盖层（静默串档）。
        """
        from services.integration.save_paths import agent_save_override_paths
        from services.integration.state import is_valid_slot_key

        if not self.slot_key:
            raise ValueError("未绑定存档，对话覆盖层无处落盘")
        if not is_valid_slot_key(self.slot_key):
            raise ValueError(f"存档槽位键非法，拒绝写盘: {self.slot_key!r}")
        dialogue_path, text_path = agent_save_override_paths(self.data_root, self.slot_key)
        return dialogue_path, text_path

    # --- 日常对话 -------------------------------------------------------

    def daily_overrides(self) -> list[overlay.DialogueOverride]:
        path, _ = self._override_paths()
        return overlay.load_overrides(path) if path is not None else []

    def daily_groups(self, npc: str) -> list[DialogueGroup]:
        """该 NPC 的日常对话合并视图（基础组原序，新增组追加在末尾）。"""
        name = (npc or "").strip()
        base = self.sources.groups(name)
        return overlay.merge_groups(base, self.daily_overrides(), name)

    def npc_line_samples(self, npc: str, *, task_limit: int = 6) -> list[str]:
        """该 NPC 自己的台词样本：全部日常台词 + 至多 ``task_limit`` 条任务台词。

        日常对话总量一般不多，全量取；任务对话数量大，按候选顺序取前若干条。
        任务侧含已完成任务（「过往台词」本就该看已经发生过的对话），但仍排除
        Agent 任务；只取该 NPC 本人说的行（玩家行与同组其他 NPC 的行都跳过）。
        供 prompt 给别的 NPC 写台词时参考其说话风格；无此 NPC 或无台词返回空表。
        """
        name = (npc or "").strip()
        if not name:
            return []
        out: list[str] = []
        for group in self.daily_groups(name):
            out.extend(_own_texts(group, name))
        if task_limit <= 0:
            return out
        task_lines: list[str] = []
        for cand in self.task_candidates(name, include_finished=True):
            task_lines.extend(_own_texts(cand.group, name))
            if len(task_lines) >= task_limit:
                break
        return out + task_lines[:task_limit]

    # --- 任务对话 -------------------------------------------------------

    def _base_task_texts(self) -> dict[str, Any]:
        task_texts = getattr(self.game_data, "task_texts", None)
        if task_texts is None:
            return {}
        try:
            return task_texts.all()
        except Exception as exc:  # pragma: no cover - 注册表未加载时的降级
            logger.warning("读取任务文本注册表失败，任务对话候选降级为空: %s", exc)
            return {}

    def task_text_overrides(self) -> dict[str, list[dict[str, Any]]]:
        _, path = self._override_paths()
        return text_overlay.load_task_text_overrides(path) if path is not None else {}

    def _resolve_title(self, raw_title: Any, texts: dict[str, Any]) -> str:
        key = str(raw_title or "").strip()
        if not key:
            return ""
        value = texts.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return key

    def task_candidates(self, npc: str, *, include_finished: bool = False) -> list[TaskGroupCandidate]:
        """「该 NPC 说话」的任务对话组，按（链前缀, 链序号, 任务id, get/finish）排。

        默认排除已完成任务（修改候选口径）；``include_finished=True`` 仅供台词
        样本读取，把已完成任务也纳入。Agent 任务在任何情况下都排除。
        """
        name = (npc or "").strip()
        tasks_registry = getattr(self.game_data, "tasks", None)
        if not name or tasks_registry is None:
            return []

        texts = self._base_task_texts()
        text_overrides = self.task_text_overrides()

        out: list[TaskGroupCandidate] = []
        for task in tasks_registry.list_all_tasks():
            task_id = getattr(task, "id", None)
            if is_agent_task_id(task_id):
                continue
            if not include_finished and task_id in self.finished_ids:
                continue
            title = self._resolve_title(getattr(task, "title", None), texts)
            prefix, chain_index = _parse_chain(getattr(task, "chain", None))
            for kind, key_field, npc_field in _TASK_CONVERSATION_FIELDS:
                key = str(getattr(task, key_field, None) or "").strip()
                if not key:
                    continue
                value = text_overlay.merged_value(key, texts.get(key), text_overrides)
                lines = text_overlay.lines_from_items(value)
                if not lines:
                    continue
                group = DialogueGroup(
                    npc=str(getattr(task, npc_field, "") or "").strip(),
                    group_id=key,
                    lines=lines,
                )
                if not group.speaks(name):
                    continue
                kind_label = "接取对话" if kind == "get" else "完成对话"
                out.append(TaskGroupCandidate(
                    task_id=task_id,
                    title=title,
                    chain_prefix=prefix,
                    chain_index=chain_index,
                    kind=kind,
                    text_key=key,
                    heading=(
                        f"任务《{title or task_id}》{kind_label}"
                        f"（任务 id {task_id}，文本 key {key}，共 {len(lines)} 行）"
                    ),
                    group=group,
                ))

        out.sort(key=lambda item: item.sort_key)
        return out

    # --- 候选筛选 -------------------------------------------------------

    def select(
        self,
        npc: str,
        scopes: Iterable[str],
        keywords: Iterable[str] | None = None,
    ) -> CandidateSelection:
        """按范围取候选组、编号、按文本预算截断。

        拼接顺序＝日常全量 → 任务近期 → 关键词命中 → 任务全量，故预算不够时
        先被砍掉的是最不推荐的「任务全量」尾巴。
        """
        name = (npc or "").strip()
        requested = normalize_scopes(scopes)
        notes: list[str] = []
        if not requested:
            return CandidateSelection(npc=name, notes=["未指定任何查询范围。"])

        daily_pool = [g for g in self.daily_groups(name) if g.speaks(name)]
        task_pool = self.task_candidates(name)

        buckets: list[tuple[str, list[_Pick]]] = []
        if SCOPE_DAILY_ALL in requested:
            buckets.append((
                "日常全量",
                [_Pick("daily", g, None) for g in daily_pool],
            ))
        if SCOPE_TASK_RECENT in requested:
            buckets.append((
                "任务近期",
                [_Pick("task", c.group, c) for c in _recent_task_groups(task_pool, name, notes)],
            ))
        taken = {(p.kind, p.group.group_id) for _, picks in buckets for p in picks}
        if SCOPE_KEYWORD in requested:
            words = [w.strip() for w in (keywords or []) if str(w or "").strip()]
            if not words:
                notes.append("选了关键词范围但没给关键词，本次未做关键词搜索。")
            else:
                universe = [_Pick("daily", g, None) for g in daily_pool]
                universe += [_Pick("task", c.group, c) for c in task_pool]
                rest = [p for p in universe if (p.kind, p.group.group_id) not in taken]
                hits = [p for p in rest if _matches_keywords(p, words)]
                if len(hits) > KEYWORD_GROUP_CAP:
                    notes.append(
                        f"关键词命中 {len(hits)} 组，超过上限 {KEYWORD_GROUP_CAP}，只取前 {KEYWORD_GROUP_CAP} 组；"
                        "请用更精确的关键词或追加关键词重查。"
                    )
                    hits = hits[:KEYWORD_GROUP_CAP]
                buckets.append((f"关键词（{'、'.join(words)}）", hits))
        if SCOPE_TASK_ALL in requested:
            buckets.append((
                "任务全量",
                [_Pick("task", c.group, c) for c in task_pool],
            ))

        ordered: list[_Pick] = []
        seen: set[tuple[str, str]] = set()
        for _, picks in buckets:
            for pick in picks:
                sig = (pick.kind, pick.group.group_id)
                if sig in seen:
                    continue
                seen.add(sig)
                ordered.append(pick)

        candidates = [
            CandidateGroup(
                number=0,
                kind=pick.kind,
                npc=name,
                key=pick.group.group_id,
                task_id=pick.task.task_id if pick.task else None,
                task_kind=pick.task.kind if pick.task else "",
                heading=_heading(pick),
                body=_body(pick.group),
                group=pick.group,
            )
            for pick in ordered
        ]
        kept, omitted = _apply_budget(candidates)
        for number, cand in enumerate(kept, start=1):
            cand.number = number
        if omitted:
            notes.append(
                f"超出单次文本预算（{CANDIDATE_TEXT_BUDGET} 字符），另有 {omitted} 组未列出；"
                "请改用关键词范围缩小后重查。"
            )

        return CandidateSelection(npc=name, groups=kept, notes=notes)


@dataclass
class _Pick:
    """筛选过程中的一条候选（尚未编号，编号在预算截断后才分配）。"""

    kind: str
    group: DialogueGroup
    task: TaskGroupCandidate | None


def normalize_scopes(scopes: Iterable[str]) -> list[str]:
    """归一化范围列表：去重保序、丢弃未知项；任务近期与任务全量冲突时取任务近期。"""
    out: list[str] = []
    for raw in scopes or []:
        scope = str(raw or "").strip()
        if scope in ALL_SCOPES and scope not in out:
            out.append(scope)
    if SCOPE_TASK_RECENT in out and SCOPE_TASK_ALL in out:
        out.remove(SCOPE_TASK_ALL)
    return out


def _recent_task_groups(
    task_pool: list[TaskGroupCandidate],
    npc: str,
    notes: list[str],
) -> list[TaskGroupCandidate]:
    """任务近期＝未完成的主线最近 5 组＋支线最近 5 组（task_pool 已按链序排好）。"""
    picked: list[TaskGroupCandidate] = []
    for prefix, quota in ((_CHAIN_MAIN, TASK_RECENT_MAIN_GROUPS), (_CHAIN_SIDE, TASK_RECENT_SIDE_GROUPS)):
        matched = [c for c in task_pool if c.chain_prefix == prefix][:quota]
        picked.extend(matched)
        if not matched:
            notes.append(f"未完成的{prefix}任务里没有「{npc}」说话的对话组。")
    return picked


def _matches_keywords(pick: _Pick, words: list[str]) -> bool:
    """关键词命中＝任一关键词出现在台词正文、说话人名或任务标题里（子串，忽略大小写）。"""
    haystack = " ".join(
        [line.text for line in pick.group.lines]
        + [line.name for line in pick.group.lines]
        + ([pick.task.title] if pick.task else [])
    ).lower()
    return any(word.lower() in haystack for word in words)


def _own_texts(group: DialogueGroup, npc: str) -> list[str]:
    """组内该 NPC 本人说的非空台词（玩家行与其他说话人的行跳过）。"""
    out: list[str] = []
    for line in group.lines:
        if line.is_player or line.name != npc:
            continue
        text = (line.text or "").strip()
        if text:
            out.append(text)
    return out


def _body(group: DialogueGroup) -> str:
    """台词行（不含组说明与编号）；Title 不拼给模型。"""
    lines = []
    for index, line in enumerate(group.lines, start=1):
        # 玩家行直接按落盘占位符显示：模型「原样抄回」就会照抄 $PC，
        # 写成中文「玩家」只能靠后端改写，等于把正确性押在兜底上。
        speaker = PLAYER_NAME if line.is_player else (line.name or "旁白")
        lines.append(f"  {index}) {speaker}：{line.text}")
    return "\n".join(lines)


def _apply_budget(candidates: list[CandidateGroup]) -> tuple[list[CandidateGroup], int]:
    """按渲染后字符预算从头截断，返回（保留的候选, 被砍掉的组数）。"""
    kept: list[CandidateGroup] = []
    used = 0
    for index, cand in enumerate(candidates):
        cost = cand.render_size
        if kept and used + cost > CANDIDATE_TEXT_BUDGET:
            return kept, len(candidates) - index
        kept.append(cand)
        used += cost
    return kept, 0


def _heading(pick: _Pick) -> str:
    group = pick.group
    if pick.kind == "task":
        return f"{pick.task.heading}【任务对话：只能改本组台词或在组尾加台词，不能新增组】"
    origin = "已改写过" if group.from_overlay else "基础数据"
    return (
        f"日常对话 组 id {group.group_id}（{origin}，共 {len(group.lines)} 行）"
        f"【日常对话：可改本组台词、可在组尾加台词】"
    )


#: (kind, 任务对象上的对话 key 字段, 任务对象上的 NPC 字段)
_TASK_CONVERSATION_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("get", "get_conversation", "get_npc"),
    ("finish", "finish_conversation", "finish_npc"),
)


@dataclass
class TaskGroupCandidate:
    """一个任务对话组＋它的排序依据。"""

    task_id: int | None
    title: str
    chain_prefix: str
    chain_index: int
    kind: str
    text_key: str
    heading: str
    group: DialogueGroup

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        """主线在前、支线次之、其它链最后；同前缀按链序号与任务 id 升序，接取先于完成。"""
        return (
            _chain_prefix_rank(self.chain_prefix),
            self.chain_index,
            int(self.task_id or 0),
            0 if self.kind == "get" else 1,
        )


def _parse_chain(raw: Any) -> tuple[str, int]:
    """``"主线#12"`` → ("主线", 12)；无序号时序号排到最后。"""
    text = str(raw or "").strip()
    if not text:
        return ("", _CHAIN_INDEX_LAST)
    prefix, sep, index_text = text.partition("#")
    prefix = prefix.strip()
    if not sep:
        return (prefix, _CHAIN_INDEX_LAST)
    try:
        return (prefix, int(index_text.strip()))
    except ValueError:
        return (prefix, _CHAIN_INDEX_LAST)


def _chain_prefix_rank(prefix: str) -> int:
    if prefix == _CHAIN_MAIN:
        return 0
    if prefix == _CHAIN_SIDE:
        return 1
    return 2


# ---------------------------------------------------------------------------
# 构造入口
# ---------------------------------------------------------------------------

#: 基础对话数据按 data_root 共享（13 个 XML 约 330KB，每次查询都重解析不划算）
_SOURCES_CACHE: dict[str, DialogueSources] = {}


def shared_sources(data_root: Path) -> DialogueSources:
    """按 data_root 取共享的日常对话读取器（内含 mtime 缓存）。"""
    key = str(Path(data_root).resolve())
    sources = _SOURCES_CACHE.get(key)
    if sources is None:
        sources = DialogueSources(data_root)
        _SOURCES_CACHE[key] = sources
    return sources


def reset_sources_cache() -> None:
    """测试用：清空共享读取器。"""
    _SOURCES_CACHE.clear()


def build_view(game_data: Any = None, *, slot_key: str | None = None) -> DialogueView:
    """构造视图；不给 slot_key＝按当前绑定存档（未绑定＝只读基础数据，写入被拒）。

    ``slot_key`` 把一次委派钉死在受理当刻的存档上：对话修改子 Agent 最长跑 900 秒，
    期间玩家换档时覆盖层的读与写都必须仍指向原存档，否则改动会串进别的存档。
    已完成任务投影只有当前绑定存档那一份，钉死场景下沿用当前投影——它只影响候选
    筛选（哪些任务对话可改），不影响写盘落点。
    """
    from services.integration.state import get_integration_state

    data_root = getattr(game_data, "data_root", None)
    if data_root is None:
        from services.game_data.paths import get_game_data_root

        data_root = get_game_data_root()

    snapshot = get_integration_state().snapshot()
    progress = snapshot.get("progress")
    return DialogueView(
        data_root=Path(data_root),
        game_data=game_data,
        slot_key=(slot_key or "").strip() or snapshot.get("bound_slot"),
        finished_ids=progress.get("tasks_finished") if isinstance(progress, dict) else None,
        sources=shared_sources(Path(data_root)),
    )
