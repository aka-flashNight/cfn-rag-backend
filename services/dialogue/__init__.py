"""对话修改（阶段四）的数据层：基础数据、覆盖层读写、合并视图与候选筛选。

三方职责：

- ``sources`` 读 ``data/dialogues/*.xml``（mtime 缓存，工具执行期不跑 IO 契约下可复用）；
- ``overrides`` / ``task_text`` 读写 ``data/rag/agent_saves/<slotKey>/`` 下的两个覆盖层；
- ``view`` 给出合并视图与四种查询范围的候选筛选，``editor`` 是写入侧唯一入口。

基础数据永不改写（FR-8）；未绑定存档＝只读基础数据，写入 fail-closed 被拒。
"""

from __future__ import annotations

from services.dialogue.editor import DialogueEditor, EditOutcome
from services.dialogue.models import DialogueGroup, DialogueLine
from services.dialogue.sources import DialogueSources
from services.dialogue.view import (
    ALL_SCOPES,
    CANDIDATE_TEXT_BUDGET,
    CandidateGroup,
    CandidateSelection,
    DialogueView,
    KEYWORD_GROUP_CAP,
    SCOPE_DAILY_ALL,
    SCOPE_KEYWORD,
    SCOPE_TASK_ALL,
    SCOPE_TASK_RECENT,
    build_view,
    is_agent_task_id,
    normalize_finished_ids,
    shared_sources,
)

__all__ = [
    "ALL_SCOPES",
    "CANDIDATE_TEXT_BUDGET",
    "CandidateGroup",
    "CandidateSelection",
    "DialogueEditor",
    "DialogueGroup",
    "DialogueLine",
    "DialogueSources",
    "DialogueView",
    "EditOutcome",
    "KEYWORD_GROUP_CAP",
    "SCOPE_DAILY_ALL",
    "SCOPE_KEYWORD",
    "SCOPE_TASK_ALL",
    "SCOPE_TASK_RECENT",
    "build_view",
    "is_agent_task_id",
    "normalize_finished_ids",
    "shared_sources",
]
