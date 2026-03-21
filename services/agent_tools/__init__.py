"""
Agent 工具层：面向 LLM 的 Function Calling schema 与任务草案校验逻辑。

当前阶段（Phase 2）主要落地：
- ``schemas.py``：OpenAI tools/function 的参数 schema 定义（prepare/draft/update）
- ``validator.py``：Validation Pipeline（V1-V10）
"""

from .schemas import (  # noqa: F401
    CONFIRM_AGENT_TASK_TOOL,
    PREPARE_TASK_CONTEXT_TOOL,
    SEARCH_KNOWLEDGE_TOOL,
    DRAFT_AGENT_TASK_TOOL,
    UPDATE_TASK_DRAFT_TOOL,
    UPDATE_NPC_MOOD_TOOL,
)

# validator 依赖 game_data 的 pydantic 数据模型；为避免在缺少可选依赖时 import 失败，
# 这里不在包初始化阶段直接导入。
#
# 使用方需要时可直接从 `services.agent_tools.validator` 导入：
# - `DraftValidationContext`
# - `validate_task_draft`         (V1-V10 完整管线)
# - `validate_task_draft_v1_v6`   (向后兼容，仅 V1-V6)
