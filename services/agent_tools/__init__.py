"""
Agent 工具层：任务草案校验与上下文构建（validator / context_builder / task_tools / handlers）。

面向 LLM 的 **Function Calling 工具定义** 现在在 ``services/tools/<category>/<name>.py``
（基于 ``services.tools.base.BaseTool`` / ``ToolRegistry``）。
``handlers.py`` 提供纯业务函数，供新 tools 的 ``run()`` 方法复用。
``schemas.py`` 保留参数 JSON Schema 常量与 TypedDict。
``services/skills/`` 现在按 Anthropic 2026 规范存放 YAML frontmatter + Markdown body 格式的 SKILL.md 文档，
只提供 **流程/领域知识**（不再是可执行函数），由 ``list_skills / read_skill / read_skill_file`` 三个元工具按需加载。
"""

# validator 依赖 game_data 的 pydantic 数据模型；为避免在缺少可选依赖时 import 失败，
# 这里不在包初始化阶段直接导入。
#
# 使用方需要时可直接从 `services.agent_tools.validator` 导入：
# - `DraftValidationContext`
# - `validate_task_draft`         (V1-V10 完整管线)
# - `validate_task_draft_v1_v6`   (向后兼容，仅 V1-V6)
