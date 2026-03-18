"""
LangGraph 三阶段管线：组装与编译。

图结构（对应文档 6.2.8）：
  START
    -> prepare_context
    -> decision
    -> (should_continue?)
        -> has_tool_calls & round < 5: tool_executor -> decision (循环)
        -> otherwise: generate_response -> parse_mood -> post_process -> END
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from .state import AgentState
from .nodes import (
    prepare_context_node,
    decision_node,
    tool_executor_node,
    generate_response_node,
    parse_mood_node,
    post_process_node,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5


def _should_continue(state: dict[str, Any]) -> Literal["tool_executor", "generate_response"]:
    """
    条件路由：决定 decision 之后走工具执行还是生成回复。
    """
    has_tools = state.get("has_tool_calls", False)
    round_num = state.get("tool_call_round", 0)

    if has_tools and round_num < MAX_TOOL_ROUNDS:
        return "tool_executor"
    return "generate_response"


def build_agent_graph() -> StateGraph:
    """
    构建并返回未编译的 StateGraph。
    调用方可以 .compile() 后使用。
    """
    graph = StateGraph(AgentState)

    graph.add_node("prepare_context", prepare_context_node)
    graph.add_node("decision", decision_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("parse_mood", parse_mood_node)
    graph.add_node("post_process", post_process_node)

    graph.set_entry_point("prepare_context")

    graph.add_edge("prepare_context", "decision")

    graph.add_conditional_edges(
        "decision",
        _should_continue,
        {
            "tool_executor": "tool_executor",
            "generate_response": "generate_response",
        },
    )

    graph.add_edge("tool_executor", "decision")

    graph.add_edge("generate_response", "parse_mood")
    graph.add_edge("parse_mood", "post_process")
    graph.add_edge("post_process", END)

    return graph


def compile_agent_graph():
    """
    编译完整的 agent 图（用于 ask 非流式接口）。
    """
    graph = build_agent_graph()
    return graph.compile()


def build_decision_loop_graph() -> StateGraph:
    """
    构建仅包含 prepare_context -> decision <-> tool_executor 的子图。
    当 decision 判断无需工具时以 END 退出，供 ask_stream 手动衔接流式 generate。
    """
    graph = StateGraph(AgentState)

    graph.add_node("prepare_context", prepare_context_node)
    graph.add_node("decision", decision_node)
    graph.add_node("tool_executor", tool_executor_node)

    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "decision")

    def _loop_or_exit(state: dict[str, Any]) -> Literal["tool_executor", "__end__"]:
        has_tools = state.get("has_tool_calls", False)
        round_num = state.get("tool_call_round", 0)
        if has_tools and round_num < MAX_TOOL_ROUNDS:
            return "tool_executor"
        return "__end__"

    graph.add_conditional_edges(
        "decision",
        _loop_or_exit,
        {
            "tool_executor": "tool_executor",
            "__end__": END,
        },
    )
    graph.add_edge("tool_executor", "decision")

    return graph


def compile_decision_loop():
    """
    编译决策循环子图（用于 ask_stream 流式接口）。
    """
    graph = build_decision_loop_graph()
    return graph.compile()


# ---------------------------------------------------------------------------
# 预编译的全局图实例（惰性初始化）
# ---------------------------------------------------------------------------

_FULL_GRAPH = None
_DECISION_LOOP = None


def get_full_graph():
    global _FULL_GRAPH
    if _FULL_GRAPH is None:
        _FULL_GRAPH = compile_agent_graph()
    return _FULL_GRAPH


def get_decision_loop():
    global _DECISION_LOOP
    if _DECISION_LOOP is None:
        _DECISION_LOOP = compile_decision_loop()
    return _DECISION_LOOP
