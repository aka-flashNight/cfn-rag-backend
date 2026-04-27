"""
Atomic tool registry (OpenAI function calling).

每个 tool 一个模块，顶层导出 ``tool = XxxTool()``；
``ToolRegistry.discover()`` 递归扫描 ``services/tools/<category>/<name>.py`` 自动注册。

Tools 只负责"执行原子操作 + 生成 OpenAI function schema"，
复合流程、NPC 指引、渐进式披露内容全部走 ``services/skills/``（YAML frontmatter + Markdown body）。
"""

from __future__ import annotations

from services.tools.base import (
    BaseTool,
    Category,
    ToolContext,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
)

__all__ = [
    "BaseTool",
    "Category",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "get_tool_registry",
]
