"""会话存储（SQLite 单连接 + WAL，修 E4）：消息 / 会话 / 摘要 / 任务草案 / ask 计数 / 长期记忆。

对应 docs/v3-developer/06-存储启动与打包瘦身.md §1.2：
- 单连接 sqlite3.connect(check_same_thread=False) + threading.Lock + WAL，不再每操作新建连接；
- 统一返回 dataclass（ChatMessage），消灭「dict 列表 vs dataclass 双形状」；
- 草案过期：连续 N 次 ask 未触碰任务工具即删除，N 由 draft_keep_turns 配置；
- bargain_count 独立成列，不再混进草案 JSON（修 D5），内部字段 _draft_commit_valid 同样不入库；
- sessions.slot_key 存档作用域列（B2-5）：创建时记录当前绑定槽位，缺省 NULL＝standalone，不做迁移回填。

长期记忆（B3-2）：long_term_memory 表按 (slot_key, npc) 作用域存两类条目——
kind='base'（每个作用域唯一一条基础长期记忆，entry_no 为 NULL）与
kind='fact'（追加的关键记录，entry_no 是作用域内单调递增编号，永不复用或重排）。
未绑定存档时用固定作用域键 STANDALONE_SCOPE，与各存档互不混淆。

对话修改委派（阶段四 B4-1）：dialogue_jobs 表记一条委派的全生命周期状态机，
并按 (slot_key, task_id) 建部分唯一索引做**硬幂等**——重要任务只在首次收到完成
事件时委派一次，重复事件在数据库层就被挡掉，不依赖调用方自查。
session_dialogue_hints 表存好感度跨档后的待提示文案与剩余轮数（触发二，跨轮持续）。
session_published_task_highlights 表存「刚发布的委托」的达成要点供给期（单会话单条）：
发布轮记 rounds_left＝published_task_highlight_turns-1，之后每轮装配时递减，归零即删；
该表只服务提示词装配，不落 npc_state_db / status / progress。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from services.game_data.paths import (
    RESOURCE_FOLDER_NAMES,
    pick_existing_or_default_resource_root,
)

logger = logging.getLogger(__name__)

# 摘要触发间隔（消息条数）
SUMMARIZE_INTERVAL = 30

# 精确短期记忆长度档位白名单（消息条数）：前端 短/中/长/几乎无限。
# 与 SUMMARIZE_INTERVAL 是两回事——档位决定「多少条近期对话逐字进 prompt」，
# 摘要间隔决定「多久把近期对话压成一段永久模糊记忆」，后者不随档位变化。
PRECISE_MEMORY_OPTIONS = (10, 30, 100, 500)

# 长期记忆（B3-2）：未绑定存档时的独立作用域键（与各存档互不混淆）
STANDALONE_SCOPE = "__standalone__"

# 追加长期记忆（kind='fact'）条数上限；达到上限后聊天侧只保留调整能力
LTM_MAX_FACT_ENTRIES = 20

# 单条长期记忆入库长度上限（超出截断）；提示词侧的期望长度更短，见 long_term.py
LTM_MAX_CHARS = 500

LTM_KIND_BASE = "base"
LTM_KIND_FACT = "fact"

# 对话修改委派记录（阶段四 B4-1）的状态机。链路：
#   received（收到信号）→ judging（初判中）→ delegated（已入队，delegated_at 落时戳）
#                                        ↘ dismissed（不够格/非绑定槽位/频率限制/重复）
#   delegated → generating（worker 开始执行）→ written（覆盖层已落盘）
#                                            ↘ skipped（子 Agent 判断不动或轮限耗尽）
#                                            ↘ failed（异常/超时）
DIALOGUE_JOB_RECEIVED = "received"
DIALOGUE_JOB_JUDGING = "judging"
DIALOGUE_JOB_DELEGATED = "delegated"
DIALOGUE_JOB_DISMISSED = "dismissed"
DIALOGUE_JOB_GENERATING = "generating"
DIALOGUE_JOB_WRITTEN = "written"
DIALOGUE_JOB_SKIPPED = "skipped"
DIALOGUE_JOB_FAILED = "failed"

DIALOGUE_JOB_STATUSES = (
    DIALOGUE_JOB_RECEIVED,
    DIALOGUE_JOB_JUDGING,
    DIALOGUE_JOB_DELEGATED,
    DIALOGUE_JOB_DISMISSED,
    DIALOGUE_JOB_GENERATING,
    DIALOGUE_JOB_WRITTEN,
    DIALOGUE_JOB_SKIPPED,
    DIALOGUE_JOB_FAILED,
)

# 「占位」状态：这条记录已经把该存档该任务的委派名额用掉了（派出去过或正在派出途中）。
# 刻意不含 dismissed（不够格，从没派出过）与 failed（没派出去 / 派出后跑坏）——
# 两者都不该挡掉后来的重试机会：漏标记的任务要靠聊天侧兜底调用改写，跑坏的任务要能重派。
# 重试成本由冷却窗（同存档同 NPC）与每会话上限封住，两者都只数 delegated_at，dismissed 不计。
DIALOGUE_JOB_CLAIMED_STATUSES = (
    DIALOGUE_JOB_RECEIVED,
    DIALOGUE_JOB_JUDGING,
    DIALOGUE_JOB_DELEGATED,
    DIALOGUE_JOB_GENERATING,
    DIALOGUE_JOB_WRITTEN,
    DIALOGUE_JOB_SKIPPED,
)

#: 占位条件的 SQL 片段：部分唯一索引的 WHERE 谓词与插入前预检查都从这里拼，两处不会漂移
DIALOGUE_JOB_CLAIMED_SQL = (
    "status IN (" + ", ".join(f"'{status}'" for status in DIALOGUE_JOB_CLAIMED_STATUSES) + ")"
)

# 不允许混入草案 JSON 的内部/业务分离字段（修 D5）
_DRAFT_INTERNAL_FIELDS = ("bargain_count", "_draft_commit_valid")


def get_db_path() -> Path:
    """memory.db 路径（resources/tools/memory.db，开发环境与 exe 运行时同规则）。"""
    if hasattr(sys, "_MEIPASS") or getattr(sys, "frozen", False):
        base_dir = Path(os.path.dirname(sys.executable))
    else:
        base_dir = Path(__file__).resolve().parent.parent.parent

    for name in RESOURCE_FOLDER_NAMES:  # 情况1: base_dir/<resources>/tools/memory.db
        db_path = base_dir / name / "tools" / "memory.db"
        if db_path.parent.exists():
            return db_path

    cwd = Path(os.getcwd()).resolve()  # 情况2: 当前工作目录
    for name in RESOURCE_FOLDER_NAMES:
        db_path = cwd / name / "tools" / "memory.db"
        if db_path.parent.exists():
            return db_path

    for name in RESOURCE_FOLDER_NAMES:  # 情况3: 工作目录父目录
        db_path = cwd.parent / name / "tools" / "memory.db"
        if db_path.parent.exists():
            return db_path

    root = pick_existing_or_default_resource_root(base_dir)
    db_path = root / "tools" / "memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@dataclass
class ChatMessage:
    """统一消息形状（API 层与实现一致）。"""

    id: int
    role: str
    content: str
    timestamp: float


@dataclass
class TaskDraftRow:
    session_id: str
    draft_id: str
    npc_name: str
    draft: dict[str, Any]
    bargain_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class SessionInfo:
    session_id: str
    npc_name: str
    title: str
    created_at: float
    slot_key: str | None = None


@dataclass
class LongTermEntry:
    """一条长期记忆。kind='base' 时 entry_no 为 None；kind='fact' 时是作用域内编号。"""

    id: int
    slot_key: str
    npc: str
    kind: str
    content: str
    entry_no: int | None = None
    source_session_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class DialogueJobRow:
    """一条对话修改委派记录。task_id 为 None＝非重要任务触发（不参与幂等）。"""

    job_id: str
    slot_key: str
    npc: str
    trigger: str
    status: str
    task_id: int | None = None
    session_id: str | None = None
    reason: str = ""
    instruction: str = ""
    detail: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    delegated_at: float | None = None


@dataclass
class DialogueHint:
    """好感度跨档后待提示的对话修改建议（阶段四触发二，持续 rounds_left 轮）。"""

    session_id: str
    hint: str
    rounds_left: int = 0


@dataclass
class PublishedTaskHighlight:
    """刚发布的委托的达成要点供给记录（单会话单条，同一时间只跟踪最近发布的一份）。"""

    session_id: str
    task_id: int
    rounds_left: int = 0


@dataclass
class DialogueLedgerRow:
    """对话内嵌插话的已播台账行：seq 与 chat_history.message_id 一一对应。"""

    seq: int
    role: str
    content: str
    message_id: int


class MemoryStore:
    """SQLite 存储访问层（同步实现 + asyncio.to_thread 包装）。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else get_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    # ------------------------------------------------------------------
    # 表结构
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_session_time ON chat_history(session_id, timestamp)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    npc_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    slot_key TEXT,
                    dialogue_instance_id TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS dialogue_interject_ledger (
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, seq)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    summary TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_task_drafts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    draft_id TEXT NOT NULL,
                    npc_name TEXT NOT NULL DEFAULT '',
                    draft_json TEXT NOT NULL,
                    bargain_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_ask_counters (
                    session_id TEXT NOT NULL PRIMARY KEY,
                    rounds_without_task INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slot_key TEXT NOT NULL,
                    npc TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    entry_no INTEGER,
                    content TEXT NOT NULL,
                    source_session_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_scope ON long_term_memory(slot_key, npc, kind)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS dialogue_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    slot_key TEXT NOT NULL,
                    npc TEXT NOT NULL,
                    trigger TEXT NOT NULL DEFAULT '',
                    task_id INTEGER,
                    session_id TEXT,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    instruction TEXT NOT NULL DEFAULT '',
                    detail TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    delegated_at REAL
                )
                """
            )
            # 部分唯一索引＝重要任务的硬幂等：同存档同任务只允许一条「占位」记录
            # （派出去过或正在派出途中）。task_id 为 NULL 的记录（关系/态度/关键信息触发）
            # 与 dismissed / failed 的留痕都不进索引，因此不会挡掉后来的委派机会。
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_dialogue_job_task"
                " ON dialogue_jobs(slot_key, task_id)"
                f" WHERE task_id IS NOT NULL AND {DIALOGUE_JOB_CLAIMED_SQL}"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dialogue_job_scope"
                " ON dialogue_jobs(slot_key, npc, delegated_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dialogue_job_session"
                " ON dialogue_jobs(session_id)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_dialogue_hints (
                    session_id TEXT NOT NULL PRIMARY KEY,
                    hint TEXT NOT NULL DEFAULT '',
                    rounds_left INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            # 刚发布的委托的达成要点供给期（单会话单条；轮次只存在于库里，不进提示词）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_published_task_highlights (
                    session_id TEXT NOT NULL PRIMARY KEY,
                    task_id INTEGER NOT NULL,
                    rounds_left INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._migrate_legacy_schema(cur)
            self._conn.commit()
    def _migrate_legacy_schema(self, cur: sqlite3.Cursor) -> None:
        """旧库升级：草案补 bargain_count 独立列（修 D5）＋ sessions 补 slot_key（B2-5）
        ＋ sessions 补 dialogue_instance_id（对话内嵌插话，P0）。"""
        self._migrate_draft_bargain_count(cur)
        cur.execute("PRAGMA table_info(sessions)")
        cols = {str(r[1]) for r in cur.fetchall()}
        if cols and "slot_key" not in cols:
            cur.execute("ALTER TABLE sessions ADD COLUMN slot_key TEXT")
            logger.info("旧库迁移完成：sessions 补 slot_key 列（缺省 NULL＝standalone，不回填）")
        if cols and "dialogue_instance_id" not in cols:
            cur.execute("ALTER TABLE sessions ADD COLUMN dialogue_instance_id TEXT")
            logger.info("旧库迁移完成：sessions 补 dialogue_instance_id 列（对话内嵌插话）")
        # 一组对话一个 session：dialogue_instance_id 非空时唯一（普通终端会话该列为 NULL）
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_dialogue_instance"
            " ON sessions(dialogue_instance_id) WHERE dialogue_instance_id IS NOT NULL"
        )

    def _migrate_draft_bargain_count(self, cur: sqlite3.Cursor) -> None:
        """v3 前的旧库升级：session_task_drafts 补 bargain_count 独立列（修 D5），
        并把旧草案 JSON 内混入的 bargain_count / _draft_commit_valid 剥离。"""
        cur.execute("PRAGMA table_info(session_task_drafts)")
        cols = {str(r[1]) for r in cur.fetchall()}
        if not cols or "bargain_count" in cols:
            return
        cur.execute(
            "ALTER TABLE session_task_drafts ADD COLUMN bargain_count INTEGER NOT NULL DEFAULT 0"
        )
        rows = cur.execute("SELECT id, draft_json FROM session_task_drafts").fetchall()
        for row_id, draft_json in rows:
            try:
                draft = json.loads(draft_json) if draft_json else {}
            except json.JSONDecodeError:
                continue
            if isinstance(draft, dict) and draft.get("bargain_count") is not None:
                try:
                    bc = int(draft["bargain_count"])
                except (TypeError, ValueError):
                    bc = 0
                cur.execute(
                    "UPDATE session_task_drafts SET bargain_count = ? WHERE id = ?", (bc, row_id)
                )
        # 剥离旧草案 JSON 中的内部字段
        cur.execute(
            "UPDATE session_task_drafts SET draft_json = REPLACE(draft_json, '\"_draft_commit_valid\": true, ', '')"
        )
        cur.execute(
            "UPDATE session_task_drafts SET draft_json = REPLACE(draft_json, ', \"_draft_commit_valid\": false', '')"
        )
        logger.info("旧库迁移完成：session_task_drafts 补 bargain_count 列（%d 行回填）", len(rows))

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------

    def add_message_sync(self, session_id: str, role: str, content: str) -> int:
        """新增一条消息，返回该会话消息总数（调用方据此触发摘要）。"""
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if role not in {"user", "assistant"}:
            raise ValueError("role 必须是 'user' 或 'assistant'。")
        ts = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, ts),
            )
            cur.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (session_id,))
            count = int(cur.fetchone()[0])
            self._conn.commit()
            return count

    def count_messages_sync(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (session_id.strip(),))
            return int(cur.fetchone()[0])

    def get_history_sync(
        self,
        session_id: str,
        limit: int = SUMMARIZE_INTERVAL,
        offset: int = 0,
        order: Literal["asc", "desc"] = "asc",
    ) -> list[ChatMessage]:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if limit <= 0:
            return []
        offset = max(0, offset)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, role, content, timestamp FROM chat_history
                WHERE session_id = ? ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            )
            rows = cur.fetchall()
        messages = [
            ChatMessage(id=int(r["id"]), role=str(r["role"]), content=str(r["content"]), timestamp=float(r["timestamp"]))
            for r in rows
        ]
        return list(reversed(messages)) if order == "asc" else messages

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    def get_summary_sync(self, session_id: str) -> str | None:
        session_id = session_id.strip()
        if not session_id:
            return None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
        return str(row["summary"]) if row else None

    def get_summary_message_count_sync(self, session_id: str) -> int:
        """摘要覆盖到第几条消息（0＝没有摘要）；「摘要之后的原文」据此切片。"""
        session_id = session_id.strip()
        if not session_id:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT message_count FROM session_summaries WHERE session_id = ?", (session_id,),
            )
            row = cur.fetchone()
        return int(row["message_count"]) if row else 0

    def save_summary_sync(self, session_id: str, summary: str, message_count: int) -> None:
        ts = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_summaries (session_id, summary, message_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    message_count = excluded.message_count,
                    updated_at = excluded.updated_at
                """,
                (session_id.strip(), summary, message_count, ts),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    def create_session_sync(self, npc_name: str, title: str, slot_key: str | None = None) -> SessionInfo:
        """建会话并记当前存档作用域（B2-5）；slot_key=None＝standalone。"""
        npc_name = npc_name.strip()
        title = title.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")
        if not title:
            raise ValueError("title 不能为空。")
        slot_key = str(slot_key).strip() if slot_key is not None else None
        slot_key = slot_key or None
        info = SessionInfo(
            session_id=str(uuid.uuid4()), npc_name=npc_name, title=title,
            created_at=time.time(), slot_key=slot_key,
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO sessions (session_id, npc_name, title, created_at, slot_key) VALUES (?, ?, ?, ?, ?)",
                (info.session_id, info.npc_name, info.title, info.created_at, info.slot_key),
            )
            self._conn.commit()
        return info

    def list_sessions_sync(self, slot_key: str | None = None) -> list[SessionInfo]:
        """按存档作用域过滤（B2-5）：slot_key=None→只回 standalone（NULL）会话。"""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT session_id, npc_name, title, created_at, slot_key FROM sessions
                WHERE slot_key IS ? ORDER BY created_at DESC, session_id ASC
                """,
                (slot_key,),
            )
            rows = cur.fetchall()
        return [
            SessionInfo(
                session_id=str(r["session_id"]), npc_name=str(r["npc_name"]), title=str(r["title"]),
                created_at=float(r["created_at"]),
                slot_key=str(r["slot_key"]) if r["slot_key"] is not None else None,
            )
            for r in rows
        ]

    def update_session_title_sync(self, session_id: str, title: str) -> SessionInfo:
        session_id = session_id.strip()
        title = title.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not title:
            raise ValueError("title 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT session_id, npc_name, title, created_at, slot_key FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"会话 '{session_id}' 不存在。")
            cur.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id))
            self._conn.commit()
        return SessionInfo(
            session_id=str(row["session_id"]),
            npc_name=str(row["npc_name"]),
            title=title,
            created_at=float(row["created_at"]),
            slot_key=str(row["slot_key"]) if row["slot_key"] is not None else None,
        )

    def delete_session_sync(self, session_id: str) -> None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            if cur.fetchone() is None:
                raise ValueError(f"会话 '{session_id}' 不存在。")
            cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM session_task_drafts WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM session_ask_counters WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM session_dialogue_hints WHERE session_id = ?", (session_id,))
            cur.execute(
                "DELETE FROM session_published_task_highlights WHERE session_id = ?", (session_id,),
            )
            cur.execute("DELETE FROM dialogue_interject_ledger WHERE session_id = ?", (session_id,))
            self._conn.commit()

    # ------------------------------------------------------------------
    # 对话内嵌插话：session 映射与已播台账（P0）
    # ------------------------------------------------------------------

    _SESSION_SELECT_COLS = "session_id, npc_name, title, created_at, slot_key"

    @classmethod
    def _session_row_to_info(cls, row: sqlite3.Row) -> SessionInfo:
        return SessionInfo(
            session_id=str(row["session_id"]),
            npc_name=str(row["npc_name"]),
            title=str(row["title"]),
            created_at=float(row["created_at"]),
            slot_key=str(row["slot_key"]) if row["slot_key"] is not None else None,
        )

    def dialogue_session_get_sync(self, dialogue_instance_id: str) -> SessionInfo | None:
        """按 dialogue_instance_id 取插话 session；未建立＝None。"""
        key = (dialogue_instance_id or "").strip()
        if not key:
            return None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT {self._SESSION_SELECT_COLS} FROM sessions WHERE dialogue_instance_id = ?",
                (key,),
            )
            row = cur.fetchone()
        return self._session_row_to_info(row) if row else None

    def dialogue_session_ensure_sync(
        self,
        dialogue_instance_id: str,
        npc_name: str,
        title: str,
        slot_key: str | None = None,
    ) -> tuple[SessionInfo, bool]:
        """按 dialogue_instance_id 建/取 session；返回 (会话, 是否本次新建)。

        一组对话一个 session（协议 D-04 / 6.1）：同 id 重复请求只复用，不新建。
        并发/跨进程重复创建由 dialogue_instance_id 唯一索引兜底，读回已存在那条。
        """
        key = (dialogue_instance_id or "").strip()
        if not key:
            raise ValueError("dialogue_instance_id 不能为空。")
        npc_name = (npc_name or "").strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")
        title = (title or "").strip() or f"对话插话 · {npc_name}"
        slot_key = str(slot_key).strip() if slot_key is not None else None
        slot_key = slot_key or None

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT {self._SESSION_SELECT_COLS} FROM sessions WHERE dialogue_instance_id = ?",
                (key,),
            )
            row = cur.fetchone()
            if row is not None:
                return self._session_row_to_info(row), False
            info = SessionInfo(
                session_id=str(uuid.uuid4()), npc_name=npc_name, title=title,
                created_at=time.time(), slot_key=slot_key,
            )
            try:
                cur.execute(
                    "INSERT INTO sessions"
                    " (session_id, npc_name, title, created_at, slot_key, dialogue_instance_id)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (info.session_id, info.npc_name, info.title, info.created_at, info.slot_key, key),
                )
            except sqlite3.IntegrityError:
                # 另一个进程已抢先写入同一 dialogue_instance_id：读回那条
                self._conn.rollback()
                cur.execute(
                    f"SELECT {self._SESSION_SELECT_COLS} FROM sessions WHERE dialogue_instance_id = ?",
                    (key,),
                )
                row = cur.fetchone()
                if row is None:
                    raise
                return self._session_row_to_info(row), False
            self._conn.commit()
        return info, True

    def dialogue_ledger_read_sync(self, session_id: str) -> list[DialogueLedgerRow]:
        """读取某插话 session 的已播台账（按 seq 升序）。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return []
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT seq, role, content, message_id FROM dialogue_interject_ledger"
                " WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            )
            rows = cur.fetchall()
        return [
            DialogueLedgerRow(
                seq=int(r["seq"]),
                role=str(r["role"]),
                content=str(r["content"]),
                message_id=int(r["message_id"]),
            )
            for r in rows
        ]

    def dialogue_played_append_sync(
        self, session_id: str, rows: list[dict[str, Any]],
    ) -> list[int]:
        """按 seq 幂等追加已播行：只写缺口；同 seq 内容变化时更新原消息，不重复插入。

        `rows`：`[{"seq": int, "role": "user"|"assistant", "content": str}, ...]`。
        返回本次**新增**的 seq 列表（内容修正不计入新增）。
        """
        session_id = (session_id or "").strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        appended: list[int] = []
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            for index, row in enumerate(sorted(rows, key=lambda item: int(item["seq"]))):
                seq = int(row["seq"])
                role = str(row["role"])
                content = str(row["content"])
                if role not in {"user", "assistant"}:
                    raise ValueError("role 必须是 'user' 或 'assistant'。")
                if not content.strip():
                    continue
                cur.execute(
                    "SELECT role, content, message_id FROM dialogue_interject_ledger"
                    " WHERE session_id = ? AND seq = ?",
                    (session_id, seq),
                )
                existing = cur.fetchone()
                if existing is None:
                    # 同一批写入用微秒级递增时间戳，保证会话历史顺序与 seq 一致
                    ts = now + index * 1e-6
                    cur.execute(
                        "INSERT INTO chat_history (session_id, role, content, timestamp)"
                        " VALUES (?, ?, ?, ?)",
                        (session_id, role, content, ts),
                    )
                    message_id = int(cur.lastrowid)
                    cur.execute(
                        "INSERT INTO dialogue_interject_ledger"
                        " (session_id, seq, role, content, message_id, updated_at)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, seq, role, content, message_id, ts),
                    )
                    appended.append(seq)
                elif str(existing["content"]) != content or str(existing["role"]) != role:
                    # Host 修正了该 seq 的已播内容：以现场为准，改原消息而不是再插一条
                    message_id = int(existing["message_id"])
                    cur.execute(
                        "UPDATE chat_history SET role = ?, content = ? WHERE id = ?",
                        (role, content, message_id),
                    )
                    cur.execute(
                        "UPDATE dialogue_interject_ledger SET role = ?, content = ?, updated_at = ?"
                        " WHERE session_id = ? AND seq = ?",
                        (role, content, now, session_id, seq),
                    )
            self._conn.commit()
        return appended

    # ------------------------------------------------------------------
    # 任务草案（bargain_count 独立列，修 D5）
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_draft(draft: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in draft.items() if k not in _DRAFT_INTERNAL_FIELDS}

    def upsert_draft_sync(
        self, session_id: str, draft: dict[str, Any], *, bargain_count: int | None = None
    ) -> str:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not isinstance(draft, dict):
            raise ValueError("draft 必须是 dict。")
        draft = self._clean_draft(dict(draft))
        draft_id = str(draft.get("draft_id") or "").strip() or str(int(time.time() * 1000))
        draft.setdefault("draft_id", draft_id)
        npc_name = str(draft.get("npc_name") or "").strip()
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            if bargain_count is None:
                cur.execute(
                    "SELECT bargain_count FROM session_task_drafts WHERE session_id = ?", (session_id,)
                )
                row = cur.fetchone()
                bc = int(row["bargain_count"]) if row else 0
            else:
                bc = int(bargain_count)
            cur.execute(
                """
                INSERT INTO session_task_drafts
                    (session_id, draft_id, npc_name, draft_json, bargain_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    draft_id = excluded.draft_id,
                    npc_name = excluded.npc_name,
                    draft_json = excluded.draft_json,
                    bargain_count = excluded.bargain_count,
                    updated_at = excluded.updated_at
                """,
                (session_id, draft_id, npc_name, json.dumps(draft, ensure_ascii=False), bc, now, now),
            )
            self._conn.commit()
        return draft_id

    def get_draft_sync(self, session_id: str) -> TaskDraftRow | None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT session_id, draft_id, npc_name, draft_json, bargain_count, created_at, updated_at
                FROM session_task_drafts WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        draft_json = row["draft_json"]
        try:
            draft = json.loads(draft_json) if draft_json else {}
        except json.JSONDecodeError:
            draft = {"_draft_json_raw": draft_json}
        if isinstance(draft, dict):
            draft = self._clean_draft(draft)  # 旧库可能残留内部字段
        return TaskDraftRow(
            session_id=str(row["session_id"]),
            draft_id=str(row["draft_id"]),
            npc_name=str(row["npc_name"] or ""),
            draft=draft if isinstance(draft, dict) else {},
            bargain_count=int(row["bargain_count"] or 0),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def update_partial_sync(
        self,
        session_id: str,
        draft_id: str | None,
        modify_fields: dict[str, Any],
        *,
        bargain_count: int | None = None,
    ) -> TaskDraftRow | None:
        """局部更新草案：顶层 merge modify_fields；draft_id 不匹配返回 None。"""
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        if not isinstance(modify_fields, dict):
            raise ValueError("modify_fields 必须是 dict。")
        modify_fields = self._clean_draft(dict(modify_fields))
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT session_id, draft_id, npc_name, draft_json, bargain_count, created_at
                FROM session_task_drafts WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            current_draft_id = str(row["draft_id"] or "").strip()
            if draft_id is not None and str(draft_id).strip() and str(draft_id).strip() != current_draft_id:
                return None
            try:
                current_draft = json.loads(row["draft_json"]) if row["draft_json"] else {}
            except json.JSONDecodeError:
                current_draft = {}
            if not isinstance(current_draft, dict):
                current_draft = {}
            current_draft.update(modify_fields)
            current_draft = self._clean_draft(current_draft)
            new_draft_id = str(current_draft.get("draft_id") or current_draft_id).strip()
            new_npc_name = str(current_draft.get("npc_name") or row["npc_name"] or "").strip()
            bc = int(row["bargain_count"] or 0) if bargain_count is None else int(bargain_count)
            cur.execute(
                """
                UPDATE session_task_drafts
                SET draft_id = ?, npc_name = ?, draft_json = ?, bargain_count = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    new_draft_id,
                    new_npc_name,
                    json.dumps(current_draft, ensure_ascii=False),
                    bc,
                    now,
                    session_id,
                ),
            )
            self._conn.commit()
            return TaskDraftRow(
                session_id=session_id,
                draft_id=new_draft_id,
                npc_name=new_npc_name,
                draft=current_draft,
                bargain_count=bc,
                created_at=float(row["created_at"]),
                updated_at=now,
            )

    def delete_draft_sync(self, session_id: str, *, draft_id: str | None = None) -> bool:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        with self._lock:
            cur = self._conn.cursor()
            if draft_id is not None and str(draft_id).strip():
                cur.execute(
                    "DELETE FROM session_task_drafts WHERE session_id = ? AND draft_id = ?",
                    (session_id, str(draft_id).strip()),
                )
            else:
                cur.execute("DELETE FROM session_task_drafts WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # 草案过期计数（连续 N 次 ask 未触碰任务工具即删除）
    # ------------------------------------------------------------------

    def get_rounds_without_task_sync(self, session_id: str) -> int:
        session_id = session_id.strip()
        if not session_id:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT rounds_without_task FROM session_ask_counters WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def increment_rounds_without_task_sync(self, session_id: str) -> int:
        session_id = session_id.strip()
        if not session_id:
            return 0
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_ask_counters (session_id, rounds_without_task, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    rounds_without_task = rounds_without_task + 1,
                    updated_at = excluded.updated_at
                """,
                (session_id, now),
            )
            self._conn.commit()
            cur.execute(
                "SELECT rounds_without_task FROM session_ask_counters WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
        return int(row[0]) if row else 1

    def reset_rounds_without_task_sync(self, session_id: str) -> None:
        session_id = session_id.strip()
        if not session_id:
            return
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_ask_counters (session_id, rounds_without_task, updated_at)
                VALUES (?, 0, ?)
                ON CONFLICT(session_id) DO UPDATE SET rounds_without_task = 0, updated_at = excluded.updated_at
                """,
                (session_id, now),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 长期记忆（B3-2）
    # ------------------------------------------------------------------

    @staticmethod
    def _ltm_row_to_entry(row: sqlite3.Row) -> LongTermEntry:
        return LongTermEntry(
            id=int(row["id"]),
            slot_key=str(row["slot_key"]),
            npc=str(row["npc"]),
            kind=str(row["kind"]),
            content=str(row["content"]),
            entry_no=int(row["entry_no"]) if row["entry_no"] is not None else None,
            source_session_id=(
                str(row["source_session_id"]) if row["source_session_id"] is not None else None
            ),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    _LTM_SELECT_COLS = (
        "id, slot_key, npc, kind, entry_no, content, source_session_id, created_at, updated_at"
    )

    def ltm_list_sync(self, slot_key: str, npc: str) -> list[LongTermEntry]:
        """作用域内全部长期记忆：基础记忆在前，追加记忆按编号升序。"""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT {self._LTM_SELECT_COLS} FROM long_term_memory
                WHERE slot_key = ? AND npc = ?
                ORDER BY CASE kind WHEN 'base' THEN 0 ELSE 1 END, entry_no ASC, id ASC
                """,
                (slot_key, npc),
            )
            rows = cur.fetchall()
        return [self._ltm_row_to_entry(r) for r in rows]

    def ltm_get_base_sync(self, slot_key: str, npc: str) -> str | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT content FROM long_term_memory WHERE slot_key = ? AND npc = ? AND kind = 'base'",
                (slot_key, npc),
            )
            row = cur.fetchone()
        return str(row["content"]) if row else None

    def ltm_fact_count_sync(self, slot_key: str, npc: str) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM long_term_memory WHERE slot_key = ? AND npc = ? AND kind = 'fact'",
                (slot_key, npc),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def ltm_save_base_sync(
        self, slot_key: str, npc: str, content: str, source_session_id: str | None = None,
    ) -> LongTermEntry:
        """写入/覆盖基础长期记忆（每个作用域唯一一条）。"""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE long_term_memory
                SET content = ?, source_session_id = ?, updated_at = ?
                WHERE slot_key = ? AND npc = ? AND kind = 'base'
                """,
                (content, source_session_id, now, slot_key, npc),
            )
            if cur.rowcount == 0:
                cur.execute(
                    """
                    INSERT INTO long_term_memory
                        (slot_key, npc, kind, entry_no, content, source_session_id, created_at, updated_at)
                    VALUES (?, ?, 'base', NULL, ?, ?, ?, ?)
                    """,
                    (slot_key, npc, content, source_session_id, now, now),
                )
            self._conn.commit()
            cur.execute(
                f"SELECT {self._LTM_SELECT_COLS} FROM long_term_memory"
                " WHERE slot_key = ? AND npc = ? AND kind = 'base'",
                (slot_key, npc),
            )
            row = cur.fetchone()
        return self._ltm_row_to_entry(row)

    def ltm_append_fact_sync(
        self, slot_key: str, npc: str, content: str, source_session_id: str | None = None,
    ) -> LongTermEntry:
        """追加一条关键长期记忆；编号＝作用域内最大值 +1（永不复用、不重排）。"""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(entry_no), 0) FROM long_term_memory"
                " WHERE slot_key = ? AND npc = ? AND kind = 'fact'",
                (slot_key, npc),
            )
            row = cur.fetchone()
            entry_no = int(row[0]) + 1
            cur.execute(
                """
                INSERT INTO long_term_memory
                    (slot_key, npc, kind, entry_no, content, source_session_id, created_at, updated_at)
                VALUES (?, ?, 'fact', ?, ?, ?, ?, ?)
                """,
                (slot_key, npc, entry_no, content, source_session_id, now, now),
            )
            new_id = cur.lastrowid
            self._conn.commit()
            cur.execute(
                f"SELECT {self._LTM_SELECT_COLS} FROM long_term_memory WHERE id = ?",
                (new_id,),
            )
            inserted = cur.fetchone()
        return self._ltm_row_to_entry(inserted)

    def ltm_update_fact_sync(
        self,
        slot_key: str,
        npc: str,
        entry_no: int,
        content: str,
        source_session_id: str | None = None,
    ) -> LongTermEntry | None:
        """按编号调整一条关键长期记忆；编号不存在＝None（不改任何数据）。"""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE long_term_memory
                SET content = ?, source_session_id = ?, updated_at = ?
                WHERE slot_key = ? AND npc = ? AND kind = 'fact' AND entry_no = ?
                """,
                (content, source_session_id, now, slot_key, npc, entry_no),
            )
            if cur.rowcount == 0:
                self._conn.commit()
                return None
            self._conn.commit()
            cur.execute(
                f"SELECT {self._LTM_SELECT_COLS} FROM long_term_memory"
                " WHERE slot_key = ? AND npc = ? AND kind = 'fact' AND entry_no = ?",
                (slot_key, npc, entry_no),
            )
            row = cur.fetchone()
        return self._ltm_row_to_entry(row)

    # ------------------------------------------------------------------
    # 对话修改委派记录（阶段四 B4-1）
    # ------------------------------------------------------------------

    _DIALOGUE_JOB_COLS = (
        "job_id, slot_key, npc, trigger, task_id, session_id, status,"
        " reason, instruction, detail, created_at, updated_at, delegated_at"
    )

    @classmethod
    def _dialogue_job_row(cls, row: sqlite3.Row) -> DialogueJobRow:
        return DialogueJobRow(
            job_id=str(row["job_id"]),
            slot_key=str(row["slot_key"]),
            npc=str(row["npc"]),
            trigger=str(row["trigger"] or ""),
            task_id=int(row["task_id"]) if row["task_id"] is not None else None,
            session_id=str(row["session_id"]) if row["session_id"] is not None else None,
            status=str(row["status"]),
            reason=str(row["reason"] or ""),
            instruction=str(row["instruction"] or ""),
            detail=str(row["detail"] or ""),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            delegated_at=float(row["delegated_at"]) if row["delegated_at"] is not None else None,
        )

    def dialogue_job_create_sync(
        self,
        *,
        slot_key: str,
        npc: str,
        trigger: str,
        reason: str = "",
        instruction: str = "",
        task_id: int | None = None,
        session_id: str | None = None,
        status: str = DIALOGUE_JOB_RECEIVED,
        detail: str = "",
    ) -> DialogueJobRow | None:
        """新建一条委派记录。

        带 task_id 时按 (slot_key, task_id) 硬幂等：**已有占位记录**（派出去过或正在派出
        途中）＝返回 None，调用方据此「静默跳过」。既有记录都是 dismissed / failed 时
        不占位，本次照常新建——漏标记的任务留给聊天侧兜底，跑坏的任务允许重派。
        """
        if status not in DIALOGUE_JOB_STATUSES:
            raise ValueError(f"未知的委派状态：{status}")
        now = time.time()
        job_id = uuid.uuid4().hex
        with self._lock:
            cur = self._conn.cursor()
            if task_id is not None:
                existing = self._dialogue_job_find(cur, slot_key, task_id, claimed_only=True)
                if existing is not None:
                    return None
            try:
                cur.execute(
                    """
                    INSERT INTO dialogue_jobs
                        (job_id, slot_key, npc, trigger, task_id, session_id, status,
                         reason, instruction, detail, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, slot_key, npc, trigger, task_id, session_id, status,
                     reason, instruction, detail, now, now),
                )
            except sqlite3.IntegrityError:
                # 另一个进程（开发服务与 exe 同用一个 memory.db）已抢先写入同一任务
                self._conn.rollback()
                return None
            self._conn.commit()
            cur.execute(
                f"SELECT {self._DIALOGUE_JOB_COLS} FROM dialogue_jobs WHERE job_id = ?",
                (job_id,),
            )
            row = cur.fetchone()
        return self._dialogue_job_row(row)

    def _dialogue_job_find(
        self,
        cur: sqlite3.Cursor,
        slot_key: str,
        task_id: int,
        *,
        claimed_only: bool,
    ) -> sqlite3.Row | None:
        """按 (存档, 任务) 查既有委派记录。

        ``claimed_only=True`` 只看占位记录（幂等判定用）；``False`` 时任意状态都算
        （事后核查「为什么没改」要能读到 dismissed / failed 的留痕）。
        """
        sql = (
            f"SELECT {self._DIALOGUE_JOB_COLS} FROM dialogue_jobs"
            " WHERE slot_key = ? AND task_id = ?"
        )
        if claimed_only:
            sql = f"{sql} AND {DIALOGUE_JOB_CLAIMED_SQL}"
        cur.execute(sql, (slot_key, task_id))
        return cur.fetchone()

    def dialogue_job_find_by_task_sync(self, slot_key: str, task_id: int) -> DialogueJobRow | None:
        """查某存档某任务的既有委派记录（任意状态，含驳回与失败留痕）。"""
        with self._lock:
            row = self._dialogue_job_find(
                self._conn.cursor(), slot_key, task_id, claimed_only=False,
            )
        return self._dialogue_job_row(row) if row else None

    def dialogue_job_set_status_sync(
        self, job_id: str, status: str, *, detail: str = "",
    ) -> DialogueJobRow | None:
        """推进状态机；job_id 不存在＝None。转入 delegated 时落一次 delegated_at。"""
        if status not in DIALOGUE_JOB_STATUSES:
            raise ValueError(f"未知的委派状态：{status}")
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE dialogue_jobs
                SET status = ?, detail = ?, updated_at = ?,
                    delegated_at = COALESCE(delegated_at, ?)
                WHERE job_id = ?
                """,
                (status, detail, now, now if status == DIALOGUE_JOB_DELEGATED else None, job_id),
            )
            if cur.rowcount == 0:
                self._conn.commit()
                return None
            self._conn.commit()
            cur.execute(
                f"SELECT {self._DIALOGUE_JOB_COLS} FROM dialogue_jobs WHERE job_id = ?",
                (job_id,),
            )
            row = cur.fetchone()
        return self._dialogue_job_row(row)

    def dialogue_job_delegated_since_sync(self, slot_key: str, npc: str, since: float) -> int:
        """冷却窗口内该存档该 NPC 已委派的次数（频率限制口径，dismissed 不计）。"""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM dialogue_jobs"
                " WHERE slot_key = ? AND npc = ? AND delegated_at IS NOT NULL AND delegated_at >= ?",
                (slot_key, npc, since),
            )
            return int(cur.fetchone()[0])

    def dialogue_job_session_delegated_sync(self, session_id: str) -> int:
        """该会话累计委派次数（每会话上限口径，dismissed 不计）。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM dialogue_jobs"
                " WHERE session_id = ? AND delegated_at IS NOT NULL",
                (session_id,),
            )
            return int(cur.fetchone()[0])

    # ------------------------------------------------------------------
    # 好感度跨档提示（阶段四触发二）
    # ------------------------------------------------------------------

    def dialogue_hint_read_sync(self, session_id: str) -> DialogueHint | None:
        """待提示的跨档文案；没有该会话、轮数用完或文案为空＝None。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT session_id, hint, rounds_left FROM session_dialogue_hints"
                " WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        hint = str(row["hint"] or "").strip()
        rounds_left = int(row["rounds_left"])
        if not hint or rounds_left <= 0:
            return None
        return DialogueHint(session_id=str(row["session_id"]), hint=hint, rounds_left=rounds_left)

    def dialogue_hint_write_sync(self, session_id: str, hint: str, rounds_left: int) -> None:
        """写入/更新跨档提示；rounds_left<=0 或 hint 为空＝删除该行（提示已用完或已调用）。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            if rounds_left <= 0 or not (hint or "").strip():
                cur.execute(
                    "DELETE FROM session_dialogue_hints WHERE session_id = ?", (session_id,),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO session_dialogue_hints (session_id, hint, rounds_left, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        hint = excluded.hint,
                        rounds_left = excluded.rounds_left,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, hint.strip(), rounds_left, now),
                )
            self._conn.commit()

    def dialogue_hint_clear_sync(self, session_id: str) -> None:
        """清掉跨档提示：模型已经调用过委派，不再提示（§4.0.2-2）。"""
        self.dialogue_hint_write_sync(session_id, "", 0)

    # ------------------------------------------------------------------
    # 刚发布委托的达成要点供给期
    # ------------------------------------------------------------------

    def published_task_highlight_read_sync(
        self, session_id: str,
    ) -> PublishedTaskHighlight | None:
        """当前会话正在供给的委托记录；没有记录、轮数用完＝None。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT session_id, task_id, rounds_left FROM session_published_task_highlights"
                " WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        rounds_left = int(row["rounds_left"])
        if rounds_left <= 0:
            return None
        return PublishedTaskHighlight(
            session_id=str(row["session_id"]),
            task_id=int(row["task_id"]),
            rounds_left=rounds_left,
        )

    def published_task_highlight_write_sync(
        self, session_id: str, task_id: int, rounds_left: int,
    ) -> None:
        """写入/覆盖该会话的记录；rounds_left<=0＝删除（发布新委托时覆盖旧记录）。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return
        try:
            tid = int(task_id)
        except (TypeError, ValueError):
            return
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            if rounds_left <= 0:
                cur.execute(
                    "DELETE FROM session_published_task_highlights WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO session_published_task_highlights
                        (session_id, task_id, rounds_left, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        task_id = excluded.task_id,
                        rounds_left = excluded.rounds_left,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, tid, int(rounds_left), now),
                )
            self._conn.commit()

    def published_task_highlight_clear_sync(self, session_id: str) -> None:
        """撤掉记录（任务已完成 / 轮次用尽 / 会话删除）。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM session_published_task_highlights WHERE session_id = ?", (session_id,),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

# ---------------------------------------------------------------------------
# asyncio 包装（API 层 await 调用）
# ---------------------------------------------------------------------------

def _make_async(attr: str):
    async def _call(self: MemoryStore, *args, **kwargs):
        return await asyncio.to_thread(getattr(self, attr), *args, **kwargs)
    _call.__name__ = attr.removesuffix("_sync")
    _call.__doc__ = f"async 版本：{attr}"
    return _call


for _name in [
    "add_message",
    "count_messages",
    "get_history",
    "get_summary",
    "save_summary",
    "get_summary_message_count",
    "create_session",
    "list_sessions",
    "update_session_title",
    "delete_session",
    "dialogue_session_get",
    "dialogue_session_ensure",
    "dialogue_ledger_read",
    "dialogue_played_append",
    "upsert_draft",
    "get_draft",
    "update_partial",
    "delete_draft",
    "get_rounds_without_task",
    "increment_rounds_without_task",
    "reset_rounds_without_task",
    "ltm_list",
    "ltm_get_base",
    "ltm_fact_count",
    "ltm_save_base",
    "ltm_append_fact",
    "ltm_update_fact",
    "dialogue_job_create",
    "dialogue_job_find_by_task",
    "dialogue_job_set_status",
    "dialogue_job_delegated_since",
    "dialogue_job_session_delegated",
    "dialogue_hint_read",
    "dialogue_hint_write",
    "dialogue_hint_clear",
    "published_task_highlight_read",
    "published_task_highlight_write",
    "published_task_highlight_clear",
]:
    setattr(MemoryStore, _name, _make_async(f"{_name}_sync"))


_MEMORY_STORE: MemoryStore | None = None
_STORE_LOCK = threading.Lock()


def get_memory_store() -> MemoryStore:
    """全局单例（startup 初始化；未初始化时懒加载兜底）。"""
    global _MEMORY_STORE
    if _MEMORY_STORE is not None:
        return _MEMORY_STORE
    with _STORE_LOCK:
        if _MEMORY_STORE is None:
            _MEMORY_STORE = MemoryStore()
        return _MEMORY_STORE


def set_memory_store(store: MemoryStore | None) -> None:
    global _MEMORY_STORE
    with _STORE_LOCK:
        _MEMORY_STORE = store
