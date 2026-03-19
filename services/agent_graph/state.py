"""
LangGraph 图状态定义（AgentState）。

对应文档 6.2.2，管理整个三阶段管线的共享状态。
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ── LangGraph 消息列表（自动累加） ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 上下文（prepare_context 节点填充） ──
    npc_name: str
    player_progress: int           # 1-6
    npc_affinity: int              # 0-100
    npc_relationship_level: str    # 陌生/熟悉/朋友/生死之交
    npc_faction: str
    npc_titles: list[str]
    npc_sex: str
    npc_challenge: Optional[str]   # NPC 的切磋关卡名
    npc_emotions: list[str]
    session_id: str
    retrieved_context: str         # RAG 检索结果文本

    # ── LLM 调用参数 ──
    api_key: str
    api_base: str
    model_name: str
    image_path: Optional[str]
    image_description: Optional[str]
    emotion_hint: str

    # ── 任务协商状态 ──
    pending_task_draft: Optional[dict[str, Any]]
    task_confirmed: bool
    task_write_result: Optional[str]

    # ── 控制信号 ──
    tool_call_round: int           # 当前决策-执行循环轮次（安全上限 5）
    has_tool_calls: bool           # 决策轮是否产出 tool_calls

    # ── 输出（generate_response / post_process 节点填充） ──
    final_reply: str               # 完整的 NPC 回复文本
    emotion: str                   # 解析出的情绪标签
    favorability_change: int       # 解析出的好感度变化

    # ── 流式回调（由调用方在 config 中注入） ──
    # streaming_callback 不放在 state 中，通过 config 的 callbacks 传递

    # ── 原始 payload 透传（后处理需要） ──
    payload_dict: dict[str, Any]
    effective_summarize_interval: int

    # ── 节点间内部传递（以下划线开头，不暴露给外部） ──
    _system_prompt: str
    _user_prompt: str
    _tool_messages: list[dict[str, str]]
    _pending_tool_calls: list[dict[str, Any]]
    _mood_tool_calls: list[dict[str, Any]]
    _decision_reply: str
    # SSE: 工具调用/关键阶段的 UI 事件（由 tool_executor_node 填充）
    _ui_events: list[dict[str, Any]]
