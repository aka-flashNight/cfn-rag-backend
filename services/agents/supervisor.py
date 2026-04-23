"""
Supervisor 节点（路由 + 可选 interim_reply + 四重防护）。

职责：
    1. 根据当前 state（用户 query / worker 结果 / pending_draft 等）决定下一步路由到哪个 worker。
    2. （可选）产生一段 ≤80 字的 interim_reply，供前端做"任务准备中"的伪流式穿插对白。
    3. 在每轮回环前检查：
        - 总 hop 次数 ≤ MAX_HOPS
        - 该 worker 调用次数 ≤ MAX_PER_WORKER_CALLS
        - 该 worker 连续失败次数 < MAX_CONSECUTIVE_FAILURES（否则加入黑名单）
        - token_budget_spent ≤ TOKEN_BUDGET_HARD_CAP（否则直接 end）

Supervisor 使用 **强约束 JSON 输出**（尝试 response_format=json_object；若不支持则回退正则兜底）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from services.agent_graph.prompts import (
    build_prompt_base,
    build_agent_tail,
    compose_agent_user,
    format_player_utterance,
)
from services.llm_client import call_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety guards (envelope defaults; can be overridden via env)
# ---------------------------------------------------------------------------

MAX_HOPS = int(os.environ.get("CFN_SUPERVISOR_MAX_HOPS") or 6)
MAX_PER_WORKER_CALLS = int(os.environ.get("CFN_SUPERVISOR_MAX_PER_WORKER") or 3)
MAX_CONSECUTIVE_FAILURES = int(
    os.environ.get("CFN_SUPERVISOR_MAX_CONSECUTIVE_FAILURES") or 2
)
TOKEN_BUDGET_HARD_CAP = int(
    os.environ.get("CFN_SUPERVISOR_TOKEN_BUDGET") or 40_000
)

# supervisor JSON schema hint (用在 prompt 中，而非 response_format JSON Schema)
_SUPERVISOR_OUTPUT_HINT = (
    '只输出一个 JSON 对象，字段如下：\n'
    '{\n'
    '  "route": "query" | "task" | "dialogue" | "end",        // 必填\n'
    '  "reason": "一句话理由（用于 debug；≤40字）",           // 必填\n'
    '  "interim_reply": "",                                    // 可选，≤80 字 NPC 口吻短句；无话可说则留空字符串\n'
    '  "mood": {                                               // route="dialogue" 或 "end" 时必填；其它情况留 null\n'
    '    "emotion": "<从可用情绪列表中选；若列表里有「普通」则未决定时填「普通」>",\n'
    '    "favorability_change": 0                              // 整数，范围 -5~5，常规为 0\n'
    '  }\n'
    '}\n'
    '不要包含任何其它说明文字或 markdown 代码块。'
)


_SUPERVISOR_ROUTING_GUIDE = """\
【Supervisor 路由规则】（你的判断是**本轮唯一一次**路由，worker 完成后直接接 dialogue，不会再回到你）
- 玩家消息**索要任务/奖励/关卡**，且无 pending_draft：route=task（task_worker 会走 prepare→draft 两步，停在 draft_created 等玩家确认）。
- pending_task_draft=是，且玩家消息是"接受/同意/确认/发布/拒绝/不要/取消/算了/讨价还价/改一下…"等
  **指向草案本身**的意图：route=task（task_worker 会走 confirm/cancel/update）。
- 特别注意：像"我接了 / 接 / 好 / 行 / 可以 / 确认 / 就这个 / 那就这样"这类**短确认语**，
  在 pending_task_draft=是 时，默认都视为**针对草案本身的接受**，应 route=task，不要误判成普通闲聊。
- pending_task_draft=是，但玩家在聊别的（天气/闲聊/其它问题）：route=dialogue，草案先搁置，下轮再议。
- 玩家消息**询问世界观/NPC/关卡/物品等知识细节**且上下文没答案：route=query；
  若 retrieved_context 已足够回答：route=dialogue。
- 普通对话 / 情感交流：route=dialogue。
- 若 blacklist 中有 worker，该 worker 本轮不可再选；若唯一可用 worker 被拉黑，选 dialogue 兜底。
- 几乎**不需要**选 route=end —— 只有当图已异常循环且无法给出对白时才用。

【interim_reply 生成规则】
- 仅在 route=task **且** 本轮是首次 task 路由（agent_call_counts.task == 0）时考虑产出；
  其他情况 interim_reply=""。
- interim_reply 用 NPC 自己的口吻（≤80 字），表示"我先想想 / 让我看看…"意味，不能剧透任务细节，
  不要涉及具体地区，不能和后续 worker 可能产出的任务内容冲突；不确定就留空。

【mood 字段规则】（等价于一次 update_npc_mood 工具调用——**本轮务必填写**，因为 dialogue_worker 一定会跑）
- 任何 route 都必须产出 mood（包括 route=query/task/dialogue/end）。dialogue_worker 会依赖它。
- emotion：从下方"可用情绪"清单中选择；不确定时若列表含"普通"则填"普通"，否则挑列表首项。
- favorability_change：整数，**严格 -5 ~ 5**。常规对话 0；玩家礼貌/示好 +1~+3；玩家协助/贴心 +3~+5；
  玩家冒犯/冷淡 -1~-3；玩家侮辱/敌意 -3~-5。不要写 -100 或 +10 这种越界值。
- 不要附加 title / reason / favorability_delta 等**任何**其它字段——schema 之外的字段一律忽略。
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_supervisor_output(text: str) -> dict[str, Any]:
    """
    尽量宽容地把模型输出解析成 {route, reason, interim_reply, mood}。
    任何解析失败 → 回退 route=dialogue，interim_reply="", mood=None（由节点层填默认值）。
    """
    fallback = {"route": "dialogue", "reason": "supervisor 空输出兜底", "interim_reply": "", "mood": None}
    if not text:
        return fallback
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    try:
        obj = json.loads(s)
    except Exception:
        m = _JSON_RE.search(s)
        if not m:
            return {**fallback, "reason": "JSON 解析失败兜底"}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {**fallback, "reason": "JSON 子串解析失败"}

    if not isinstance(obj, dict):
        return {**fallback, "reason": "非对象输出兜底"}

    route = str(obj.get("route") or "dialogue").strip().lower()
    if route not in ("query", "task", "dialogue", "end"):
        route = "dialogue"
    reason = str(obj.get("reason") or "")[:80]
    interim = str(obj.get("interim_reply") or "").strip()
    if len(interim) > 80:
        interim = interim[:80]

    mood_obj = obj.get("mood")
    mood: Optional[dict[str, Any]] = None
    if isinstance(mood_obj, dict):
        try:
            emo = str(mood_obj.get("emotion") or "").strip()
            fc_raw = mood_obj.get("favorability_change", 0)
            fc = int(fc_raw) if fc_raw is not None else 0
            if fc < -5:
                fc = -5
            if fc > 5:
                fc = 5
            mood = {"emotion": emo, "favorability_change": fc}
        except Exception:
            mood = None

    return {"route": route, "reason": reason, "interim_reply": interim, "mood": mood}


def _resolve_mood(
    mood_obj: Optional[dict[str, Any]],
    allowed_emotions: list[str],
) -> dict[str, Any]:
    """把 supervisor 输出的 mood 规范到合法 emotion + 合法 favorability_change 区间。"""
    default_emotion = "普通" if "普通" in allowed_emotions else (
        allowed_emotions[0] if allowed_emotions else "普通"
    )
    if not mood_obj:
        return {"emotion": default_emotion, "favorability_change": 0}
    emo = (mood_obj.get("emotion") or "").strip()
    if not emo or (allowed_emotions and emo not in allowed_emotions):
        emo = default_emotion
    try:
        fc = int(mood_obj.get("favorability_change", 0))
    except Exception:
        fc = 0
    fc = max(-5, min(5, fc))
    return {"emotion": emo, "favorability_change": fc}


def _apply_blacklist(route: str, blacklist: list[str]) -> str:
    """如果路由到黑名单 worker，强制降级为 dialogue。"""
    if route in blacklist:
        return "dialogue" if "dialogue" not in blacklist else "end"
    return route


def _default_interim_reply(route: str) -> str:
    """当 supervisor 没产出 interim_reply 时的兜底短句。

    目前只对 task 路由强制兜底。这里的文案必须：
    - 保持 NPC 口吻，但不剧透任务细节；
    - 足够短，便于前端伪流式；
    - 与后续 dialogue 正文不冲突。
    """
    if route == "task":
        return "好，我想想手头有什么事正适合交给你。"
    return ""


def _normalize_user_text(text: str) -> str:
    s = (text or "").strip().lower()
    return re.sub(r"\s+", "", s)


def _rule_route_pending_draft(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    """待确认草案场景的规则短路。

    目标：
    - 避免把“我接了 / 好 / 行 / 确认”这类高置信确认语句再交给 supervisor LLM，
      减少一次调用，也避免模型把它误判成普通对话；
    - 只覆盖**高置信**的 task 相关意图，低置信/歧义输入仍交给 LLM 决策。
    """
    pending_draft = state.get("pending_task_draft")
    if not pending_draft:
        return None

    query = _normalize_user_text((state.get("payload_dict") or {}).get("query") or "")
    if not query:
        return None

    # 议价 / 修改：优先级最高，避免“好，奖励再加点”被误判为接受
    bargain_keywords = (
        "加点", "加一些", "多一点", "再多", "提高", "增加", "改一下", "修改",
        "换一个", "换成", "奖励", "金币", "报酬", "酬劳", "难度", "条件",
        "内容", "细节", "地点", "目标", "要求",
    )
    if any(k in query for k in bargain_keywords):
        return {
            "route": "task",
            "reason": "规则直连：待确认草案下，玩家在改任务/议价",
            "interim_reply": "",
            "mood": None,
        }

    reject_patterns = (
        r"^(不|不要|不接|不做|不行|不去|不想|不干)(了|啦|吧)?$",
        r"^(算了|取消|作废|先不了|先不接|我不接|我不要|我拒绝)$",
    )
    if any(re.match(p, query) for p in reject_patterns):
        return {
            "route": "task",
            "reason": "规则直连：待确认草案下，玩家拒绝/取消",
            "interim_reply": "",
            "mood": None,
        }

    accept_patterns = (
        r"^(好|好的|好吧|行|行吧|可以|可|没问题|成交|就这?个|就它|就这样|那就这样|来吧)$",
        r"^(接|接了|我接|我接了|接吧|接受|我接受|确认|我确认|发布|我发布|确定|我确定)$",
        r"^(我接了这个任务|我接受这个任务|那我接了|那就接了)$",
    )
    if any(re.match(p, query) for p in accept_patterns):
        return {
            "route": "task",
            "reason": "规则直连：待确认草案下，玩家确认接受",
            "interim_reply": "",
            "mood": None,
        }

    return None


def _hard_guard(state: dict[str, Any]) -> Optional[str]:
    """返回硬熔断路由（"end"）或 None（放行由模型决定）。"""
    if state.get("worker_hops", 0) >= MAX_HOPS:
        return "end"
    if state.get("token_budget_spent", 0) >= TOKEN_BUDGET_HARD_CAP:
        return "end"
    return None


def _soft_guard(state: dict[str, Any], proposed_route: str) -> str:
    """当模型提议某 worker 时，检查 per-agent 上限，必要时降级。"""
    if proposed_route == "end":
        return "end"
    counts: dict[str, int] = state.get("agent_call_counts", {}) or {}
    if counts.get(proposed_route, 0) >= MAX_PER_WORKER_CALLS:
        # 已超过该 worker 调用上限
        if proposed_route != "dialogue":
            return "dialogue"
        return "end"
    return proposed_route


def _build_supervisor_prompt(state: dict[str, Any]) -> tuple[str, str]:
    """构造 supervisor 的 system + user prompt。

    - system：仅 static L1+L2+tagline（= ``state['_prompt_base']``）
    - user：共享当轮上下文 + supervisor 路由/JSON 规范 + 状态摘要 +（最后）玩家原话
    """
    prompt_base = state.get("_prompt_base") or ""
    if not prompt_base:
        # 极少数路径绕开 prepare_context_node 时的兜底：当场构建一次 base。
        prompt_base = build_prompt_base(
            npc_name=state.get("npc_name", ""),
            sex=state.get("npc_sex", "") or "",
            faction=state.get("npc_faction", "") or "",
            titles=state.get("npc_titles") or [],
            emotions=state.get("npc_emotions") or [],
        )

    allowed_emotions = state.get("npc_emotions") or []
    emotions_line = (
        "可用情绪（emotion 只能从中取）：[" + "、".join(allowed_emotions) + "]"
        if allowed_emotions
        else "可用情绪：[普通]"
    )

    sup_tail = build_agent_tail("supervisor")
    system_prompt = prompt_base

    pending_draft = state.get("pending_task_draft")
    has_pending = bool(pending_draft)
    counts: dict[str, int] = state.get("agent_call_counts", {}) or {}
    blacklist: list[str] = state.get("agent_blacklist", []) or []
    last_worker = state.get("last_worker_name") or ""
    last_summary = state.get("last_worker_summary") or ""

    status_parts = [
        f"pending_task_draft 存在：{'是' if has_pending else '否'}",
        f"worker 调用次数：{counts or {}}",
    ]
    if blacklist:
        status_parts.append(f"已熔断 worker：{blacklist}")
    if last_worker:
        status_parts.append(f"上一跳 worker: {last_worker}；摘要: {last_summary[:180]}")

    user_shared = (state.get("_user_shared") or state.get("_user_prompt") or "").strip()
    pline = format_player_utterance(
        user_query=(state.get("_user_query") or (state.get("payload_dict") or {}).get("query") or ""),
    )
    user_prompt = compose_agent_user(
        user_shared,
        _SUPERVISOR_ROUTING_GUIDE,
        sup_tail,
        _SUPERVISOR_OUTPUT_HINT,
        emotions_line,
        "\n".join(status_parts) + "\n\n请按约束输出 JSON。",
        pline,
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Supervisor node (async)
# ---------------------------------------------------------------------------

async def supervisor_node(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    单次 supervisor 调用：路由 + 可选 interim_reply + 防护。
    """
    hops = int(state.get("worker_hops", 0))
    counts: dict[str, int] = dict(state.get("agent_call_counts", {}) or {})
    consecutive_failures: dict[str, int] = dict(
        state.get("agent_consecutive_failures", {}) or {}
    )
    blacklist: list[str] = list(state.get("agent_blacklist", []) or [])
    ui_events: list[dict[str, Any]] = list(state.get("_ui_events", []) or [])

    # 1. hard-guard：hops/token 预算
    hard = _hard_guard(state)
    if hard == "end":
        logger.info(
            "supervisor hard-guard triggered: hops=%s token_spent=%s",
            hops, state.get("token_budget_spent", 0),
        )
        ui_events.append({
            "event_type": "agent_status",
            "agent": "supervisor",
            "text": "已达到本轮最大调用上限，直接生成最终回复。",
        })
        return {
            "routing_decision": "dialogue",
            "routing_reason": "hard-guard 熔断：改为对话兜底",
            "interim_reply": "",
            "worker_hops": hops + 1,
            "_ui_events": ui_events,
        }

    # 2. 待确认草案的规则短路：
    #    高置信 accept / reject / bargain 不再进 supervisor LLM，直接进 task，
    #    既减少一次调用，也避免“我接了”这类短句被误判成普通对话。
    shortcut = _rule_route_pending_draft(state)
    if shortcut is not None:
        route_raw = shortcut["route"]
        reason = shortcut["reason"]
        interim = shortcut["interim_reply"]
        mood_raw = shortcut.get("mood")
    else:
        # 3. 其余情况交给 LLM 决策
        system_prompt, user_prompt = _build_supervisor_prompt(state)
        try:
            reply_text, _tcs = await call_llm(
                api_key=state.get("api_key"),
                api_base=state.get("api_base"),
                model_name=state.get("model_name"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=None,  # supervisor 自己不用 function calling；纯文本 JSON 输出
            )
        except Exception as e:
            logger.warning("supervisor LLM 失败，回退 dialogue: %s", e)
            reply_text = ""

        parsed = _parse_supervisor_output(reply_text or "")
        route_raw = parsed["route"]
        reason = parsed["reason"]
        interim = parsed["interim_reply"]
        mood_raw = parsed.get("mood")

    # 4. 应用 blacklist / per-worker soft guard
    route = _apply_blacklist(route_raw, blacklist)
    route = _soft_guard(state, route)

    # 5. 首次 task 时要求 supervisor 一定先说一句；其他路由不发。
    if route == "task" and counts.get("task", 0) == 0:
        interim = interim or _default_interim_reply(route)
    else:
        interim = ""

    # 6. 累计计数
    if route in ("query", "task", "dialogue"):
        counts[route] = counts.get(route, 0) + 1

    # 7. 组装 SSE 事件
    #    - agent_status：debug 用，保留（前端目前不显示，但后续可打开）
    #    - interim_reply：**不再发独立事件**，文本存在 state["interim_reply"] 里，
    #      由 _ask_stream_with_graph 直接拼入 content 流，并最终进 final_reply。
    #    - mood：发一条 mood_update 早推事件；**不再发 update_npc_mood 的 tool_status**
    #      （这是内部工具，不需要给玩家看工具名称）。
    if route in ("query", "task", "dialogue"):
        ui_events.append({
            "event_type": "agent_status",
            "agent": route,
            "text": f"路由到 {route}：{reason}" if reason else f"路由到 {route}",
        })

    # 8. mood：任何路由分支都解析一次（新拓扑下 dialogue_worker 始终会执行），
    #    等价于"supervisor 在分发前一次性执行 update_npc_mood"。
    updates: dict[str, Any] = {
        "routing_decision": route,
        "routing_reason": reason,
        "interim_reply": interim,
        "interim_reply_emitted": bool(interim) or state.get("interim_reply_emitted", False),
        "worker_hops": hops + 1,
        "agent_call_counts": counts,
        "agent_consecutive_failures": consecutive_failures,
        "agent_blacklist": blacklist,
        "_ui_events": ui_events,
    }

    if not state.get("_mood_resolved_by_supervisor"):
        allowed_emotions = state.get("npc_emotions") or []
        resolved = _resolve_mood(mood_raw, allowed_emotions)
        ui_events.append({
            "event_type": "mood_update",
            "emotion": resolved["emotion"],
            "favorability_change": resolved["favorability_change"],
        })
        updates.update({
            "emotion": resolved["emotion"],
            "favorability_change": resolved["favorability_change"],
            "_mood_resolved_by_supervisor": True,
            "_mood_event_emitted": True,
            "_ui_events": ui_events,
        })

    return updates


__all__ = [
    "supervisor_node",
    "MAX_HOPS",
    "MAX_PER_WORKER_CALLS",
    "MAX_CONSECUTIVE_FAILURES",
    "TOKEN_BUDGET_HARD_CAP",
]
