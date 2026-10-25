"""对话内嵌插话（`services/dialogue_interject/`）。

独立编排层：不塞进 TurnOrchestrator，复用底层 LLM/工具/记忆/NPC/检索服务。
真源：《协议》与共享 fixture（游戏仓 docs/，只读）。

B0–B4 已交付：schemas / parser / plan / sessions / context / prompts /
真 LLM 日常流 + 任务流（含工具轮、草案 confirm 同源文案、委派降权、status 旁路）/
离线假流端点。任务模式只开检索与长期记忆，禁任务工具与委派。
"""

from services.dialogue_interject.context import (
    DailyBlocks,
    assemble_daily_blocks,
    assemble_task_blocks,
    build_user_context,
)
from services.dialogue_interject.parser import (
    JSONLStreamParser,
    ParseIssue,
    ParseOptions,
    ParsedEntry,
    TextSegment,
    clean_text,
    resolve_emotion,
    split_text,
    strip_brackets,
)
from services.dialogue_interject.plan import (
    NormalizedPlan,
    PlanRejection,
    normalize_task_plan,
)
from services.dialogue_interject.schemas import (
    DialogueBlock,
    FinalizeRequest,
    FinalizeResponse,
    InterjectRequest,
    Limits,
    PlayedLine,
    ReferenceLine,
    RemainingLine,
    SSEEvent,
    TaskContext,
    TaskSummary,
)
from services.dialogue_interject.sessions import (
    ConcurrentFinalizeError,
    SessionBinding,
    default_session_title,
    ensure_session,
    finalize,
    map_played_line,
)

__all__ = [
    "DailyBlocks",
    "assemble_daily_blocks",
    "assemble_task_blocks",
    "build_user_context",
    "JSONLStreamParser",
    "ParseIssue",
    "ParseOptions",
    "ParsedEntry",
    "TextSegment",
    "clean_text",
    "resolve_emotion",
    "split_text",
    "strip_brackets",
    "NormalizedPlan",
    "PlanRejection",
    "normalize_task_plan",
    "DialogueBlock",
    "FinalizeRequest",
    "FinalizeResponse",
    "InterjectRequest",
    "Limits",
    "PlayedLine",
    "ReferenceLine",
    "RemainingLine",
    "SSEEvent",
    "TaskContext",
    "TaskSummary",
    "SessionBinding",
    "ConcurrentFinalizeError",
    "default_session_title",
    "ensure_session",
    "finalize",
    "map_played_line",
]
