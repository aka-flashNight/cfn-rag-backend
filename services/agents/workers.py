"""
Worker 子图实现（Query / Task / Dialogue）。

复用 ``services.agent_graph.nodes`` 的 ``decision_node / tool_executor_node / generate_response_node / parse_mood_node``，
通过在入口节点设置 ``_active_worker`` 与 ``_active_worker_tool_names`` 让决策轮只看到白名单工具。

每个 worker 退出前会产出一段 ``last_worker_summary`` 写回主图，供下一次 supervisor 决策参考。

- QueryAgent：tools=query+system；只做检索；不生成最终对白（对白由 DialogueAgent 做）。
- TaskAgent：tools=task+search+system；可产出 draft / confirm / cancel；confirm 时触发 HITL（由主图 ``interrupt_before`` 控制）。
- DialogueAgent：tools=mood+system；生成最终回复 + mood 上报。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from services.agent_graph.nodes import (
    decision_node,
    generate_response_node,
    parse_mood_node,
    tool_executor_node,
)
from services.agent_graph.prompts import build_agent_tail, build_in_turn_history_appendix
from services.agent_graph.state import AgentState
from services.agents.tool_scopes import tools_for_worker

logger = logging.getLogger(__name__)

MAX_WORKER_TOOL_ROUNDS = 4

# task_worker 在这些 tool_name → status 组合出现时**立即退出**决策环，
# 避免模型在同一用户消息内一气呵成地 draft→confirm（应该把草案交给玩家先看）。
_TASK_TERMINAL_STATUSES: dict[str, frozenset[str]] = {
    "draft_agent_task": frozenset({"draft_created", "draft_updated"}),
    "update_task_draft": frozenset({"draft_updated", "draft_created"}),
    "confirm_agent_task": frozenset({"confirmed"}),
    "cancel_agent_task": frozenset({"cancelled"}),
}


# ---------------------------------------------------------------------------
# Worker-entry nodes（只做 state 预处理，不调 LLM）
# ---------------------------------------------------------------------------

def _make_entry_node(worker_name: str):
    tool_names = list(tools_for_worker(worker_name))
    # agent tail 在进程启动时预构建（不含 NPC 可变内容），与 _prompt_base 拼接后得到最终 system。
    agent_tail = build_agent_tail(worker_name)  # type: ignore[arg-type]

    async def _entry(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        # 组装本 worker 的 system prompt：
        #   prefix = state["_prompt_base"]（所有 agent 一致，prefix cache 命中）
        #   appendix = 本轮 supervisor 已说出的新增对白（若有）
        #   tail   = 本 worker 专属指令（工具使用纲领 / 对话规则 等）
        base = state.get("_prompt_base") or state.get("_system_prompt") or ""
        appendix = build_in_turn_history_appendix(
            npc_name=state.get("npc_name", ""),
            interim_reply=state.get("interim_reply") or "",
        )
        if base and appendix:
            system_prompt = f"{base}\n\n{appendix}\n\n{agent_tail}"
        elif base:
            system_prompt = f"{base}\n\n{agent_tail}"
        elif appendix:
            system_prompt = f"{appendix}\n\n{agent_tail}"
        else:
            system_prompt = agent_tail
        return {
            "_active_worker": worker_name,
            "_active_worker_tool_names": tool_names,
            "_system_prompt": system_prompt,
            # 进入 worker 时本 worker 的本地循环计数归零，避免跨 supervisor 跳累加
            "tool_call_round": 0,
            "has_tool_calls": False,
            "_pending_tool_calls": [],
            "_tool_messages": [],
            # 更新 last_worker_name，便于 supervisor 下一轮看到"谁刚跑完"
            "last_worker_name": worker_name,
        }

    _entry.__name__ = f"{worker_name}_entry"
    return _entry


# ---------------------------------------------------------------------------
# 通用 worker 退出节点：汇总本轮 worker 成果
# ---------------------------------------------------------------------------

def _summarize_tool_messages(tool_messages: list[dict[str, str]]) -> str:
    """把本轮 worker 的 tool 执行结果摘要化（给 supervisor 下一跳看，≤180 字）。"""
    if not tool_messages:
        return "（本轮 worker 未调用任何工具）"
    parts: list[str] = []
    for tm in tool_messages[-3:]:  # 只看最后 3 条，避免 supervisor prompt 过长
        name = tm.get("tool_name") or ""
        try:
            parsed = json.loads(tm.get("result") or "{}")
            status = parsed.get("status") or ""
            msg = parsed.get("message") or ""
        except Exception:
            status = ""
            msg = (tm.get("result") or "")[:40]
        parts.append(f"{name}={status or 'ok'}{('/' + msg) if msg else ''}")
    return "; ".join(parts)[:180]


async def _worker_exit_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """worker 退出：清除 per-worker 白名单，写回 summary 与 counters。"""
    worker_name = state.get("_active_worker") or ""
    tool_messages = state.get("_tool_messages") or []
    ok = True
    for tm in tool_messages:
        try:
            parsed = json.loads(tm.get("result") or "{}")
            if parsed.get("status") == "error":
                ok = False
                break
        except Exception:
            continue

    consecutive: dict[str, int] = dict(state.get("agent_consecutive_failures") or {})
    blacklist = list(state.get("agent_blacklist") or [])
    if ok:
        consecutive[worker_name] = 0
    else:
        consecutive[worker_name] = consecutive.get(worker_name, 0) + 1
        if consecutive[worker_name] >= 2 and worker_name not in blacklist:
            blacklist.append(worker_name)

    # HITL 状态机：
    # - draft_created / draft_updated → 等玩家确认（awaiting=True）
    # - confirmed / cancelled          → 链路结束（awaiting=False，清 draft_id）
    awaiting = state.get("awaiting_confirmation", False)
    confirmation_draft_id = state.get("confirmation_draft_id")
    if worker_name == "task" and tool_messages:
        last = tool_messages[-1]
        last_tool = last.get("tool_name") or ""
        try:
            parsed = json.loads(last.get("result") or "{}")
        except Exception:
            parsed = {}
        status = parsed.get("status") if isinstance(parsed, dict) else None
        if last_tool in ("draft_agent_task", "update_task_draft") and status in ("draft_created", "draft_updated"):
            awaiting = True
            confirmation_draft_id = parsed.get("draft_id") or confirmation_draft_id
        elif last_tool in ("confirm_agent_task", "cancel_agent_task") and status in ("confirmed", "cancelled"):
            awaiting = False
            confirmation_draft_id = None

    summary = _summarize_tool_messages(tool_messages)

    # 注意：不再发独立的 pending_confirmation 事件——`{任务草案拟定完成}` 已经以 `{...}`
    # 形式拼进 final_reply（由 tool_executor_node 追加到 _system_prefix_text）；
    # awaiting_confirmation / confirmation_draft_id 仍会出现在 done 事件中，供前端打标。
    return {
        "last_worker_name": worker_name,
        "last_worker_ok": ok,
        "last_worker_summary": summary,
        "agent_consecutive_failures": consecutive,
        "agent_blacklist": blacklist,
        "awaiting_confirmation": awaiting,
        "confirmation_draft_id": confirmation_draft_id,
        # 清除 worker 私有字段
        "_active_worker": "",
        "_active_worker_tool_names": [],
    }


# ---------------------------------------------------------------------------
# Query / Task sub-graphs（仅工具循环，不做最终对白）
# ---------------------------------------------------------------------------

def _last_tool_terminal(
    state: dict[str, Any],
    worker_name: str,
) -> bool:
    """检查最近一次工具执行结果是否是"本 worker 应当结束"的终态。

    目前只有 task 场景需要硬退出 —— 草案/发布/取消任一成功状态出现后继续调 LLM 会:
      1) 同一 turn 里把 draft 和 confirm 连着做（用户根本没看到草案就发布了）；
      2) 产生不必要的 LLM 调用。
    """
    if worker_name != "task":
        return False
    tool_messages = state.get("_tool_messages") or []
    if not tool_messages:
        return False
    last = tool_messages[-1]
    tool_name = last.get("tool_name") or ""
    statuses = _TASK_TERMINAL_STATUSES.get(tool_name)
    if not statuses:
        return False
    try:
        parsed = json.loads(last.get("result") or "{}")
    except Exception:
        return False
    return isinstance(parsed, dict) and parsed.get("status") in statuses


def _make_should_continue(worker_name: str):
    def _should_continue(state: dict[str, Any]) -> Literal["tool_executor", "exit"]:
        has_tools = state.get("has_tool_calls", False)
        round_num = state.get("tool_call_round", 0)
        if _last_tool_terminal(state, worker_name):
            # 终态出现 —— 不管 LLM 想不想再调工具，直接收工。
            return "exit"
        if has_tools and round_num < MAX_WORKER_TOOL_ROUNDS:
            return "tool_executor"
        return "exit"

    _should_continue.__name__ = f"{worker_name}_should_continue"
    return _should_continue


def _build_tool_loop_worker(worker_name: str) -> StateGraph:
    """Query / Task 这类 worker：decision ↔ tool_executor 直到收敛，然后退出。"""
    graph = StateGraph(AgentState)
    graph.add_node(f"{worker_name}_entry", _make_entry_node(worker_name))
    graph.add_node(f"{worker_name}_decision", decision_node)
    graph.add_node(f"{worker_name}_tool_executor", tool_executor_node)
    graph.add_node(f"{worker_name}_exit", _worker_exit_node)

    graph.set_entry_point(f"{worker_name}_entry")
    graph.add_edge(f"{worker_name}_entry", f"{worker_name}_decision")
    graph.add_conditional_edges(
        f"{worker_name}_decision",
        _make_should_continue(worker_name),
        {
            "tool_executor": f"{worker_name}_tool_executor",
            "exit": f"{worker_name}_exit",
        },
    )
    # tool_executor 出口也判一次终态；命中则跳过再一次 decision LLM 调用，直达 exit。
    graph.add_conditional_edges(
        f"{worker_name}_tool_executor",
        _make_should_continue(worker_name),
        {
            "tool_executor": f"{worker_name}_decision",
            "exit": f"{worker_name}_exit",
        },
    )
    graph.add_edge(f"{worker_name}_exit", END)
    return graph


def build_query_worker() -> StateGraph:
    return _build_tool_loop_worker("query")


def build_task_worker() -> StateGraph:
    return _build_tool_loop_worker("task")


# ---------------------------------------------------------------------------
# Dialogue sub-graph（最终对白 + mood）
# ---------------------------------------------------------------------------

async def _dialogue_finalize(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Dialogue worker 产出最终 final_reply 后，设置标志让主图走到 END。"""
    return {
        "routing_decision": "end",
        "routing_reason": "dialogue worker 已生成最终对白",
    }


def build_dialogue_worker() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("dialogue_entry", _make_entry_node("dialogue"))
    graph.add_node("dialogue_generate", generate_response_node)
    graph.add_node("dialogue_parse_mood", parse_mood_node)
    graph.add_node("dialogue_exit", _worker_exit_node)
    graph.add_node("dialogue_finalize", _dialogue_finalize)

    graph.set_entry_point("dialogue_entry")
    graph.add_edge("dialogue_entry", "dialogue_generate")
    graph.add_edge("dialogue_generate", "dialogue_parse_mood")
    graph.add_edge("dialogue_parse_mood", "dialogue_exit")
    graph.add_edge("dialogue_exit", "dialogue_finalize")
    graph.add_edge("dialogue_finalize", END)
    return graph


__all__ = [
    "build_query_worker",
    "build_task_worker",
    "build_dialogue_worker",
    "MAX_WORKER_TOOL_ROUNDS",
]
