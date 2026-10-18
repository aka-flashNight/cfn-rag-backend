"""对话内嵌插话（`services/dialogue_interject/`）。

独立编排层：不塞进 TurnOrchestrator，复用底层 LLM/工具/记忆/NPC/检索服务。
真源：《协议》与共享 fixture（游戏仓 docs/，只读）。

P0 已交付：schemas / parser / plan / sessions / context / prompts 骨架 /
假流端点；真 LLM、工具矩阵与提示词全量在 P1/P2/P4。
"""

from services.dialogue_interject.context import (
    DailyBlocks,
    assemble_daily_blocks,
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
