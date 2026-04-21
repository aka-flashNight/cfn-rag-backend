"""
Agent 工具层：任务草案校验与上下文构建（validator / context_builder / task_tools）。

面向 LLM 的 **Function Calling 工具定义** 已迁移至 ``services/skills/``（SkillRegistry）。
``schemas.py`` 仍保留参数 JSON Schema 常量与 TypedDict，供 skills 与各校验逻辑复用。
"""

# validator 依赖 game_data 的 pydantic 数据模型；为避免在缺少可选依赖时 import 失败，
# 这里不在包初始化阶段直接导入。
#
# 使用方需要时可直接从 `services.agent_tools.validator` 导入：
# - `DraftValidationContext`
# - `validate_task_draft`         (V1-V10 完整管线)
# - `validate_task_draft_v1_v6`   (向后兼容，仅 V1-V6)
