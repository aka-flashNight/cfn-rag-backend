"""TurnOrchestrator：回合状态机（对应 docs/v3-developer/03 §2，SSE 契约见 events.py）。

一个玩家消息 = 一个 Turn（一个 orchestrator 实例）：

    IDLE → BUILD_CONTEXT → BURST_STREAMING（调用 #1：meta 行 → 台词 或 路由工具调用）
         → SYNC_ACTION（confirm/cancel + 动作后回复）或 启动子 Agent
         → MERGE_WAIT（宽限/过渡语/汇合 #2）→ POST_PROCESS → DONE

v3 再次重构（路由工具化）：聊天主 Agent 的工具表就是它的路由表——只有
services/tools/route/ 下的五个路由声明，不再持有 prepare_task_context。meta 行
只剩情绪与好感，委派意图改由路由工具承载；路由轮不输出台词，后台办完结果交回来
再让 NPC 开口。任务选型（task_type / reward_types）整体移交任务 Agent。

硬约束：
- 情绪先于正文（meta 行解析瞬间发 meta 事件）；
- 过渡语只说一次、只说方向不报数字（merge.py）；
- confirm 失败 → system_notice 真实原因 + 1 次补救回复；
- 同会话并发请求经会话级锁串行；新回合取消旧子 Agent（fire-and-steer）；
- LLM 预算：聊天主 Agent ≤3 次（#1 + 过渡/动作后回复 + 汇合 #2；纯聊天轮台词为空时
  的兜底调用占「+1」那一格），
  TaskRunner ≤4 轮、SearchRunner ≤3 轮。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from services.agent_tools.draft_formatting import _detailed_draft_summary
from services.agent_tools.handlers import (
    coerce_return_for_reward,
    execute_cancel_agent_task,
    execute_confirm_agent_task,
    validate_confirm_dialogues,
)
from services.agent_tools.task_tools import resolve_task_npcs
from services.dialogue.delegation import (
    DELEGATE_TOOL_NAME,
    RELATIONSHIP_HINT_ROUNDS,
    apply_delegate_calls,
    build_relationship_hint,
    delegate_tool_or_none,
)
from services.dialogue.models import PLAYER_NAME
from services.game_data.map_registry import _norm
from services.llm import (
    ChatRequest,
    LLMClient,
    LLMConfig,
    Meta,
    split_meta_events,
)
from services.llm.errors import LLMError
from services.memory.long_term import (
    CHAT_TOOL_NAME,
    apply_chat_tool,
    build_chat_tool,
    format_block as format_long_term_block,
    normalize_content,
    parse_arguments,
    tool_call_name,
)
from services.memory.store import MemoryStore, PRECISE_MEMORY_OPTIONS, TaskDraftRow
from services.memory.summarize import SummaryRequest, get_summary_worker, should_summarize
from services.npc.manager import NPCManager
from services.npc.task_publish_policy import is_task_publish_disabled
from services.orchestrator.context import TurnContext, assemble_context
from services.portraits import build_image_message_content
from services.orchestrator.events import (
    SSEEvent,
    accumulate_usage,
    agent_status_event,
    content_event,
    done_event,
    error_event,
    meta_event,
    system_notice_event,
    tool_status_event,
)
from services.orchestrator.merge import MergeCoordinator
from services.orchestrator.prompts import (
    CONFIRM_FAIL_INSTRUCTION,
    EMPTY_REPLY_FALLBACK_TEXT,
    SEARCH_MERGE_INSTRUCTION,
    TASK_CANCEL_MERGE_INSTRUCTION,
    TASK_CANCEL_NOOP_INSTRUCTION,
    TASK_CONFIRM_MERGE_INSTRUCTION,
    build_confirm_args_user_prompt,
    build_confirm_retry_prompt,
    build_empty_reply_user_prompt,
    build_finish_npc_reference,
    build_finished_task_block,
    build_player_message,
    build_merge_user_prompt,
    build_static_system,
    build_user_shared_core,
    confirm_args_system,
    fallback_confirm_args,
    parse_confirm_args_json as parse_confirm_args,
)
from services.subagents import (
    SearchRunner,
    SessionSubagents,
    TaskRunner,
    get_session_subagents,
)
from services.tools.base import (
    ROUTE_TOOLS_DRAFT,
    ROUTE_TOOLS_NEW,
    ToolRegistry,
    get_tool_registry,
)

logger = logging.getLogger(__name__)

# 子 Agent（任务/检索）的近期对话预算：固定 8 条（4 轮），不随玩家的精确短期
# 记忆档位变化——子 Agent 只需要「刚才谈到哪」，把 500 条历史灌进去只会挤占
# 它自己的工具循环预算。玩家档位只作用于聊天主 Agent 的 prompt。
SUBAGENT_RECENT_MESSAGES = 8

# 路由声明工具名的公共前缀：去掉后即后台分派用的 kind（task_draft/search/...）
_ROUTE_PREFIX = "route_"
_ROUTE_NOTE_MAX_CHARS = 60
_ROUTE_QUERY_MAX_CHARS = 40
#: 路由工具自带的进度提示（给玩家看的等待文案）：与子 Agent 的 ui_hint 同口径
_ROUTE_UI_HINT_MAX_CHARS = 12
_ROUTE_KEYWORDS_MAX_CHARS = 80


@dataclass(frozen=True)
class Route:
    """聊天主 Agent 本轮的路由声明（从 tool_calls 解析；None 表示纯聊天）。"""

    kind: str
    note: str = ""
    query: str = ""
    keywords: str = ""
    sources: tuple[str, ...] = ()
    deep_search: bool = False
    #: 模型在调用时自带的进度提示（≤12 字）；空＝由后台回落固定文案
    ui_hint: str = ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _parse_route(tool_calls: list[dict[str, Any]]) -> Optional[Route]:
    """取第一个路由声明工具调用；参数解析失败按空载荷处理（路由本身仍生效）。"""
    for tc in tool_calls or []:
        func = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = str(func.get("name", "") or "")
        if not name.startswith(_ROUTE_PREFIX):
            continue
        raw = func.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("路由参数解析失败，按空载荷处理: %s %.200s", name, str(raw))
            args = {}
        if not isinstance(args, dict):
            args = {}
        note = str(args.get("note") or "").strip()[:_ROUTE_NOTE_MAX_CHARS]
        query = str(args.get("query") or "").strip()[:_ROUTE_QUERY_MAX_CHARS]
        keywords = str(args.get("keywords") or "").strip()[:_ROUTE_KEYWORDS_MAX_CHARS]
        ui_hint = str(args.get("ui_hint") or "").strip()[:_ROUTE_UI_HINT_MAX_CHARS]
        raw_sources = args.get("sources")
        if isinstance(raw_sources, str):
            sources = (raw_sources.strip().lower(),) if raw_sources.strip() else ()
        elif isinstance(raw_sources, (list, tuple)):
            sources = tuple(
                str(item).strip().lower() for item in raw_sources if str(item).strip()
            )
        else:
            sources = ()
        return Route(
            kind=name[len(_ROUTE_PREFIX):],
            note=note,
            query=query,
            keywords=keywords,
            sources=sources,
            deep_search=_as_bool(args.get("deep_search")),
            ui_hint=ui_hint,
        )
    return None


def _long_term_record_line(result: str, tool_call: dict[str, Any]) -> str:
    """落库结果 + 真正写进去的内容（兜底轮要告诉 NPC 它刚刚记下了什么）。"""
    args = parse_arguments(tool_call)
    content = normalize_content(args.get("content"))
    base_content = normalize_content(args.get("base_content"))
    line = f"{result}「{content}」" if content else result
    if base_content:
        line += f"，基础印象「{base_content}」"
    return line


# ---------------------------------------------------------------------------
# confirm 发布文本：归一化 / 降级 / 生成（终端的插话共用同一份实现）
# ---------------------------------------------------------------------------

@dataclass
class ConfirmArgsOutcome:
    """confirm 参数生成结果。

    ``source``：generated＝模型产出直接可用；degraded＝校验未过但保留了模型
    的合规对话；fallback＝整份改用模板。调用方按它记录日志，不必自己推断。
    """

    args: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    source: str = "generated"
    usages: list[Any] = field(default_factory=list)


def sanitize_confirm_args(
    args: dict[str, Any] | None,
    *,
    npc_name: str,
    get_npc: Optional[str] = None,
    finish_npc: Optional[str] = None,
    allow_return: Optional[bool] = None,
) -> dict[str, Any] | None:
    """宽松归一化（不硬打回）：字符串化 + 空对话补占位，优先保住 LLM 的产出。

    占位说话人按场次取：接取对话用接取 NPC、完成对话用完成 NPC（都用当前 NPC 会在
    跨 NPC 委托里把完成对话说成接取人）。回访开关宽松布尔化；不启用配对时（接取与
    完成同一人，或开关为假）两组回访对话与回访描述一律清空，残留给的值不保留。
    """
    if not isinstance(args, dict):
        return None
    enter_npc = (get_npc or npc_name or "").strip() or npc_name
    leave_npc = (finish_npc or npc_name or "").strip() or npc_name
    if allow_return is None:
        allow_return = enter_npc != leave_npc
    out: dict[str, Any] = {}
    title = args.get("title")
    desc = args.get("description")
    out["title"] = title.strip() if isinstance(title, str) else ""
    out["description"] = desc.strip() if isinstance(desc, str) else ""

    def _speaker(value: Any, default_npc: str) -> str:
        name = str(value or "").strip()
        # 模型爱把玩家写成「玩家」，或写成落盘占位符的变体（$PC_TITLE、（$PC）、$pc）；
        # 一律收敛成 $PC——AS2 按它换玩家角色名与立绘，写别的会被当成陌生 NPC。
        if name == "玩家" or PLAYER_NAME in name.upper():
            return PLAYER_NAME
        return name or default_npc

    def _dlg(
        key: str,
        fallback_text: str,
        default_npc: str,
        *,
        allow_empty: bool = False,
    ) -> list[dict[str, Any]]:
        rows = args.get(key)
        cleaned: list[dict[str, Any]] = []
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                text = str(r.get("text") or "").strip()
                if not text:
                    continue
                cleaned.append({
                    "name": _speaker(r.get("name"), default_npc),
                    "emotion": str(r.get("emotion") or "").strip(),
                    "text": text,
                })
        if not cleaned and not allow_empty:
            # 对话缺失不回炉重生成：补一条占位，保证发布流程可用
            cleaned = [{"name": default_npc, "emotion": "", "text": fallback_text}]
        return cleaned

    out["get_dialogue"] = _dlg("get_dialogue", "这份委托就交给你了。", enter_npc)
    out["finish_dialogue"] = _dlg("finish_dialogue", "干得漂亮，报酬一分不少。", leave_npc)

    flag = allow_return and coerce_return_for_reward(args.get("return_for_reward"))
    out["return_for_reward"] = flag
    if flag:
        # 回访两组不补占位：缺内容交给校验打回，宁可不启用配对也不写半套对话
        return_desc = args.get("return_description")
        out["return_description"] = (
            return_desc.strip() if isinstance(return_desc, str) else ""
        )
        out["return_get_dialogue"] = _dlg(
            "return_get_dialogue", "", leave_npc, allow_empty=True,
        )
        out["return_finish_dialogue"] = _dlg(
            "return_finish_dialogue", "", enter_npc, allow_empty=True,
        )
    else:
        out["return_description"] = ""
        out["return_get_dialogue"] = []
        out["return_finish_dialogue"] = []

    rewrite = bool(args.get("dialogue_rewrite"))
    out["dialogue_rewrite"] = rewrite
    out["dialogue_rewrite_instruction"] = (
        str(args.get("dialogue_rewrite_instruction") or "").strip() if rewrite else ""
    )
    return out


def degrade_confirm_args(
    draft: dict[str, Any],
    args: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    npc_name: str,
    get_npc: str,
    finish_npc: str,
    warnings: list[str],
) -> tuple[dict[str, Any], bool]:
    """两段式校验最终未通过：退回单任务，绝不写半套配对。

    返回 (args, 是否整份改用模板)。
    """
    base_broken = any(
        issue.get("field") in {"get_dialogue", "finish_dialogue"} for issue in issues
    )
    if base_broken:
        logger.warning("confirm 两段式校验失败且原对话不合规，整份改用模板兜底: %s", issues)
        warnings.append("两段式回访对话校验未通过，已改用模板文本发布单个任务。")
        return fallback_confirm_args(draft, npc_name), True
    logger.warning("confirm 两段式校验失败，退回单任务并保留原对话: %s", issues)
    warnings.append("两段式回访对话校验未通过，本次按单任务发布。")
    args["return_for_reward"] = False
    args["return_description"] = ""
    args["return_get_dialogue"] = []
    args["return_finish_dialogue"] = []
    return args, False


async def generate_confirm_args(
    llm: LLMClient,
    *,
    draft: dict[str, Any],
    npc_name: str,
    base_messages: list[dict[str, Any]],
    draft_summary: str,
    spoken_text: str = "",
    emotion: str = "",
    finish_npc_reference: str = "",
) -> ConfirmArgsOutcome:
    """生成 confirm 发布文本（title/description/两组或四组对话）。

    与聊天 Agent 同源前缀（调用方传入 base_messages），追加本轮过渡话与草案详情；
    单次、无工具、宽松归一化；仅当 LLM 调用整体失败或解析不出 JSON 时回退模板。
    说话人校验不过时额外做 **一次** 修正重试，仍不通过按 `degrade_confirm_args`
    降级（保留合规对话或整份换模板）。
    """
    get_npc, finish_npc = resolve_task_npcs(draft, npc_name)
    allow_return = get_npc != finish_npc
    user_prompt = (
        confirm_args_system(allow_return=allow_return) + "\n\n"
        + build_confirm_args_user_prompt(
            draft_summary=draft_summary,
            npc_name=npc_name,
            spoken_text=spoken_text,
            emotion=emotion,
            allow_return=allow_return,
            finish_npc_reference=finish_npc_reference,
        )
    )
    messages = [*base_messages, {"role": "user", "content": user_prompt}]
    warnings: list[str] = []
    usages: list[Any] = []
    source = "generated"

    def _parse_and_clean(raw: str) -> dict[str, Any] | None:
        return sanitize_confirm_args(
            parse_confirm_args(raw), npc_name=npc_name,
            get_npc=get_npc, finish_npc=finish_npc, allow_return=allow_return,
        )

    args: dict[str, Any] | None = None
    raw_text = ""
    try:
        result = await llm.chat(ChatRequest(messages=messages, purpose="subagent"))
        usages.append(result.usage)
        raw_text = result.content or ""
        args = _parse_and_clean(raw_text)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("confirm 发布文本生成失败，使用模板兜底: %s", exc)

    if args:
        issues = validate_confirm_dialogues(args, get_npc=get_npc, finish_npc=finish_npc)
        if issues:
            # 只对可机器判定的校验问题重试一次；调用异常/解析失败不重试（已在上方兜底）
            retry_messages = [
                *messages,
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": build_confirm_retry_prompt(issues)},
            ]
            retry_args: dict[str, Any] | None = None
            try:
                retry = await llm.chat(ChatRequest(
                    messages=retry_messages, purpose="subagent",
                ))
                usages.append(retry.usage)
                retry_args = _parse_and_clean(retry.content or "")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("confirm 修正重试失败，走降级: %s", exc)
            if retry_args:
                args = retry_args
                issues = validate_confirm_dialogues(
                    args, get_npc=get_npc, finish_npc=finish_npc,
                )
            if issues:
                args, used_fallback = degrade_confirm_args(
                    draft, args, issues, npc_name=npc_name,
                    get_npc=get_npc, finish_npc=finish_npc, warnings=warnings,
                )
                source = "fallback" if used_fallback else "degraded"

    if not args:
        args = fallback_confirm_args(draft, npc_name)
        source = "fallback"
    return ConfirmArgsOutcome(args=args, warnings=warnings, source=source, usages=usages)


# ---------------------------------------------------------------------------
# 依赖容器
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorDeps:
    """可注入依赖（生产用全局单例；测试注入 Fake）。"""

    memory: MemoryStore
    npc_manager: NPCManager
    registry: ToolRegistry
    game_data: Any
    engine: Any = None                       # RetrievalEngine（None 时检索降级）
    llm_factory: Callable[[LLMConfig], LLMClient] = LLMClient.for_config
    summary_llm_factory: Callable[[LLMConfig], LLMClient] = LLMClient.for_config
    merge_grace_ms: int = 1500
    subagent_timeout_s: int = 120
    task_max_rounds: int = 4
    search_max_rounds: int = 3
    draft_keep_turns: int = 3
    history_limit: int = 20
    # 聊天主 Agent prompt 里逐字保留的近期对话条数（服务端默认；请求带合法
    # 精确短期记忆档位时由该档位覆盖）
    recent_dialogue_limit: int = 8
    enable_summary: bool = True


def default_deps() -> OrchestratorDeps:
    """生产依赖（全部来自 startup 初始化的全局单例）。"""
    from services.memory.store import get_memory_store
    from services.npc.manager import get_npc_manager

    from core.config import get_settings

    settings = get_settings()
    engine = None
    try:
        from services.retrieval import get_retrieval_engine

        engine = get_retrieval_engine()
    except Exception:  # pragma: no cover
        logger.warning("检索引擎不可用，Tier-1 检索降级")
    game_data = None
    try:
        from services.game_data.registry import get_game_data_registry

        game_data = get_game_data_registry()
    except Exception:  # pragma: no cover
        logger.warning("游戏数据不可用，任务功能将受限")
    return OrchestratorDeps(
        memory=get_memory_store(),
        npc_manager=None,  # type: ignore[arg-type]  # 运行时经 get_npc_manager()
        registry=get_tool_registry(),
        game_data=game_data,
        engine=engine,
        merge_grace_ms=settings.merge_grace_ms,
        subagent_timeout_s=settings.subagent_timeout_s,
        task_max_rounds=settings.subagent_task_max_rounds,
        search_max_rounds=settings.subagent_search_max_rounds,
        draft_keep_turns=settings.draft_keep_turns,
    )


# ---------------------------------------------------------------------------
# 会话级锁（同会话两请求不交叉，03 §9.6）
# ---------------------------------------------------------------------------

_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


async def get_session_lock(session_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _SESSION_LOCKS[session_id] = lock
        return lock


def reset_session_locks() -> None:
    """测试用。"""
    _SESSION_LOCKS.clear()


# ---------------------------------------------------------------------------
# TurnOrchestrator
# ---------------------------------------------------------------------------

class TurnOrchestrator:
    """一个实例跑一个回合；run() 产出 SSE 事件流。"""

    def __init__(
        self,
        *,
        session_id: str,
        npc_name: str,
        query: str,
        player_identity: str = "",
        progress_stage: Optional[int] = None,
        current_emotion: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        send_image: bool = True,
        agent_enabled: bool = True,
        summarize_interval: Optional[int] = None,
        long_term_memory_enabled: Optional[bool] = None,
        deps: Optional[OrchestratorDeps] = None,
    ) -> None:
        """send_image 仅是「允许带图」开关（默认允许）；是否真带图由 07 §4 规则
        在上下文装配时判定（vision Profile + 未被标记不支持 + 立绘资产可用）。

        agent_enabled=False 时本回合只做纯聊天：不下发任何路由工具、
        prompt 不含路由说明、模型仍硬发的路由调用一律忽略。
        任务发布临时关闭名单内的 NPC（services/npc/task_publish_policy.py）
        不论 agent_enabled 一律同走纯聊天闸门，prompt 换成不可派活约束。
        summarize_interval 是前端「精确短期记忆长度」档位（10/30/100/500，对应
        短/中/长/几乎无限）；非白名单值一律按未提供处理，回落到服务端默认。
        long_term_memory_enabled 是前端每次请求带的跨对话长期记忆开关原值
        （None＝未提供）；游戏内（embedded 口径）恒定启用、不受理关闭，
        关闭时本回合既不注入长期记忆也不下发长期记忆工具、不写库。
        """
        self.session_id = session_id
        self.npc_name = npc_name
        self.query = query
        self.player_identity = player_identity
        self.progress_stage = progress_stage
        self.current_emotion = current_emotion
        self.llm_config = llm_config or LLMConfig()
        self.send_image = send_image
        self.agent_enabled = agent_enabled
        self.long_term_memory_enabled = long_term_memory_enabled
        self.task_publish_disabled = is_task_publish_disabled(npc_name)
        self.precise_memory_limit = (
            summarize_interval if summarize_interval in PRECISE_MEMORY_OPTIONS else None
        )
        self.deps = deps

        self._ctx: Optional[TurnContext] = None
        self._usage: dict[str, Any] | None = None
        # 本轮已把「刚发布的委托」块拼进 result_block 的 task_id：
        # 同一轮若再重建消息，user core 里不再重复注入（一个 Turn 一个实例，天生本轮有效）。
        self._published_block_task_id: Optional[int] = None

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[SSEEvent]:
        deps = self._resolve_deps()
        if deps.npc_manager is None:
            from services.npc.manager import get_npc_manager

            deps.npc_manager = await get_npc_manager()  # type: ignore[assignment]
        if deps.game_data is None:
            try:
                from services.game_data.registry import get_game_data_registry

                deps.game_data = get_game_data_registry()
            except Exception:  # pragma: no cover
                logger.warning("游戏数据不可用，任务功能将受限")
        lock = await get_session_lock(self.session_id)
        async with lock:
            async for ev in self._run_locked(deps):
                yield ev

    def _resolve_deps(self) -> OrchestratorDeps:
        return self.deps if self.deps is not None else default_deps()

    @property
    def _routing_enabled(self) -> bool:
        """本回合是否允许路由（Agent 能力开启，且该 NPC 不在任务发布关闭名单）。"""
        return self.agent_enabled and not self.task_publish_disabled

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    async def _run_locked(self, deps: OrchestratorDeps) -> AsyncIterator[SSEEvent]:
        memory = deps.memory
        self._usage = None
        collected_text: list[str] = []
        handle = None
        merge: Optional[MergeCoordinator] = None
        route_kind: Optional[str] = None
        draft_touched = False  # 本回合任务被触碰（confirm/cancel 成功也算）

        try:
            # ---- steer：取消同会话未完成的旧子 Agent（01 §4）----
            sess = await get_session_subagents(self.session_id)
            sess.cancel_unfinished()

            # ---- BUILD_CONTEXT ----
            # 精确短期记忆档位同时决定「从库里读多少条」与「逐字进 prompt 多少条」；
            # 未提供（或非法）时两者各回落到服务端默认，行为与档位引入前一致。
            history_limit = self.precise_memory_limit or deps.history_limit
            dialogue_limit = self.precise_memory_limit or deps.recent_dialogue_limit
            ctx = await assemble_context(
                session_id=self.session_id,
                npc_name=self.npc_name,
                player_query=self.query,
                player_identity=self.player_identity,
                progress_stage=self.progress_stage,
                current_emotion=self.current_emotion,
                llm_config=self.llm_config,
                memory=memory,
                npc_manager=deps.npc_manager,
                game_data=deps.game_data,
                engine=deps.engine,
                history_limit=history_limit,
                send_image=self.send_image,
                long_term_memory_enabled=self.long_term_memory_enabled,
            )
            self._ctx = ctx

            llm = deps.llm_factory(self.llm_config)

            # 触发三：委派工具按注册闸下发（后台有子 Agent 在跑 / 未绑定存档＝完全不注册）；
            # 任务兜底表述按条件闸二选一——仅当当轮真的拼了已完成委托、且其中有未挂
            # 重写标记的委托时才给，否则任务相关表述一个字都不出现（§4.0.2-3）。
            delegate_tool = delegate_tool_or_none(
                include_task_fallback=ctx.finished_task_block.has_unmarked,
            )
            # 触发二：关系跨档提示只在工具真的下发那一轮注入（不注册就完全不提），
            # 注入即扣一轮；工具没下发时原样留着，等能下发的那一轮再给。
            relationship_hint = (
                await self._take_relationship_hint(memory) if delegate_tool is not None else ""
            )

            # ---- BURST_STREAMING（调用 #1）----
            # system 分层（从稳定到易变）：世界观 → 输出规则/路由说明（跨 NPC）
            # → 单 NPC 扮演块（含 meta 协议）；user1 = 历史 → 会话态 → RAG（同回合
            # 多次调用共享）；user2 = 玩家当轮话 + 立绘说明/情绪连贯提示（+图片 part 在尾）。
            base_messages = self._build_base_messages(
                ctx, dialogue_limit=dialogue_limit, relationship_hint=relationship_hint,
            )
            player_message = build_player_message(
                self.query,
                npc_name=self.npc_name,
                has_image=bool(ctx.image_data_url),
                current_emotion=self.current_emotion,
            )
            # 立绘图片只进聊天调用 #1 的末条 user 消息（文本 part 前、图片 part 后）；
            # base_messages 保持纯文本，merge/补救/confirm 参数等调用天然无图
            messages = [
                *base_messages,
                {"role": "user", "content": build_image_message_content(player_message, ctx.image_data_url)},
            ]

            # 聊天 Agent 的工具表 = 路由表（判据写在工具描述里，见 services/tools/route/）
            # ＋ 长期记忆工具（B3-4：不在全局注册表里，按作用域当前状态手拼，
            # 三形态同名不同形；关闭时既不下发也不注入）
            # ＋ 对话修改委派工具（同样手拼，注册闸不通过时既不下发也不提示）。
            # Agent 能力关闭或任务发布临时关闭时不下发路由工具：模型连路由入口都没有。
            chat_tools = (
                self._routing_tools(
                    deps.registry,
                    has_pending_draft=self._routing_enabled and ctx.pending_draft_row is not None,
                    player_can_challenge=ctx.player_can_challenge,
                )
                if self._routing_enabled else []
            )
            if ctx.long_term is not None:
                chat_tools = [*chat_tools, build_chat_tool(ctx.long_term)]
            if delegate_tool is not None:
                chat_tools = [*chat_tools, delegate_tool]

            stream = llm.chat_stream(ChatRequest(
                messages=messages,
                tools=chat_tools or None,
                purpose="chat",
                send_image=self.send_image,
            ))
            meta, events = await split_meta_events(
                stream, list(ctx.npc_state.emotions or ["普通"]),
            )

            # 情绪/好感先于正文：解析到 meta 即更新并发事件
            yield await self._emit_meta(meta, deps)

            # 正文直发。路由声明只在 finish 事件里聚合出来（client 不流式吐 tool_calls），
            # 所以「路由轮不说话」由提示词与工具描述约束，这里只在模型违规时留痕。
            tool_sink: list[dict[str, Any]] = []
            async for ev in self._forward_body(events, collected_text, tool_sink=tool_sink):
                yield ev
            # 路由闸门关闭时即便模型硬发路由调用也一律忽略：本回合只有纯聊天。
            # （工具表已经没下发路由工具，这里是防越权的第二道闸。）
            route = _parse_route(tool_sink) if self._routing_enabled else None
            route_kind = route.kind if route is not None else None
            spoken_text = "".join(collected_text)

            # 长期记忆工具不在全局注册表里（要按作用域状态换形），这里直接落库；
            # 与路由互不干扰——路由轮同样可以顺带更新记忆。
            recorded = await self._apply_long_term_calls(deps, ctx, tool_sink)
            # 对话修改委派同理（工具是手拼的、不进全局注册表），受理即入队后台子 Agent
            delegated = await self._apply_delegate_calls(
                deps, tool_sink, registered=delegate_tool is not None,
            )

            if route is not None and spoken_text.strip():
                logger.warning(
                    "会话 %s：路由轮（%s）仍输出了台词 %.60s——提示词要求该轮只路由不说话",
                    self.session_id, route.kind, spoken_text,
                )
            if route is None and self._routing_enabled and ctx.pending_draft_row is not None:
                # 观测点：有草案时玩家常在「接/不接」，模型漏发路由会导致草案静默过期，
                # 玩家侧只看到「过期的委托草案已取消」，必须留痕便于排查
                logger.warning(
                    "会话 %s：存在待确认草案但模型未路由（玩家消息: %.60s）——"
                    "若玩家在确认/拒绝，本次会按纯聊天处理并累计草案过期计数",
                    self.session_id, self.query,
                )

            if route is None:
                # ---- 纯聊天路径：正文已直发，无后台动作 ----
                # 两个手拼工具不算「越权路由调用」：记忆工具由 _apply_long_term_calls 落库，
                # 委派工具由 _apply_delegate_calls 受理（未下发时那边已单独留痕）。
                stray_calls = [
                    tc for tc in tool_sink
                    if tool_call_name(tc) not in (CHAT_TOOL_NAME, DELEGATE_TOOL_NAME)
                ]
                if stray_calls:
                    names = [tool_call_name(tc) for tc in stray_calls]
                    if self._routing_enabled:
                        logger.info("纯聊天轮出现非路由工具调用，忽略: %s", names)
                    elif self.agent_enabled:
                        logger.warning("该 NPC 任务发布已临时关闭，模型仍硬发路由调用，忽略: %s", names)
                    else:
                        logger.warning("Agent 能力已关闭，模型仍硬发路由调用，忽略: %s", names)
                if not spoken_text.strip():
                    # 一个字台词都没出（只落库、或直接空返），玩家侧只剩空气泡。
                    # 补一次不带任何工具的调用：手拼工具不再注册、prompt 里只给
                    # 落库与已派出的结果——从根上断掉「发工具调用就不说话」这条路径。
                    async for ev in self._empty_reply_rescue(
                        llm, base_messages, meta, collected_text, recorded, delegated,
                    ):
                        yield ev

            elif route.kind in ("task_confirm", "task_cancel"):
                # ---- SYNC_ACTION：路由轮按契约不说话，后端办完再让 NPC 开口 ----
                is_confirm = route.kind == "task_confirm"
                tool_name = "confirm_agent_task" if is_confirm else "cancel_agent_task"
                ui_hint = route.ui_hint or ("发布委托" if is_confirm else "取消委托")
                draft_row = ctx.pending_draft_row
                yield tool_status_event(tool_name, "running", ui_hint=ui_hint)

                # reply = (结果块, 指令, 静态兜底话术, 事后系统通知)
                if not is_confirm and draft_row is None:
                    # 无草案可撤：不视为失败，让 NPC 顺着话头自然回应
                    yield tool_status_event(tool_name, "success", ui_hint=ui_hint)
                    reply = (
                        "【后台结果】你手上并没有正在跟他谈的委托，没什么可撤销的。",
                        TASK_CANCEL_NOOP_INSTRUCTION,
                        "嗯？我这儿可没什么活儿要办。",
                        "",
                    )
                elif not is_confirm:
                    execute_cancel_agent_task(
                        {"draft_id": draft_row.draft_id},
                        pending_draft=draft_row.draft,
                    )
                    draft_touched = True
                    await memory.delete_draft(self.session_id, draft_id=draft_row.draft_id)
                    await memory.reset_rounds_without_task(self.session_id)
                    yield tool_status_event(tool_name, "success", ui_hint=ui_hint)
                    reply = (
                        "【后台结果】这份委托已经撤销，玩家手上不再有它。",
                        TASK_CANCEL_MERGE_INSTRUCTION,
                        "行，那这事就算了。",
                        "委托已取消",
                    )
                elif draft_row is None:
                    yield tool_status_event(tool_name, "failed", ui_hint=ui_hint)
                    fail_reason = "当前没有待确认的委托草案"
                    yield system_notice_event(f"委托确认失败：{fail_reason}")
                    reply = self._fail_reply_args(fail_reason)
                else:
                    confirm_args, confirm_warnings = await self._generate_confirm_args(
                        llm, ctx, draft_row, deps.game_data,
                        base_messages=base_messages, spoken_text=spoken_text,
                        emotion=meta.emotion,
                    )
                    outcome = execute_confirm_agent_task(
                        confirm_args,
                        pending_draft=draft_row.draft,
                        npc_name=self.npc_name,
                        player_progress=ctx.progress_stage or 1,
                        npc_affinity=ctx.favorability,
                        game_data=deps.game_data,
                        rag_context_text=ctx.rag_context_text,
                        npc_states=ctx.npc_states,
                        publish_session_id=self.session_id,
                        generation_warnings=confirm_warnings,
                    )
                    if outcome.payload.get("status") == "confirmed":
                        draft_touched = True
                        await memory.delete_draft(self.session_id, draft_id=draft_row.draft_id)
                        await memory.reset_rounds_without_task(self.session_id)
                        yield tool_status_event(tool_name, "success", ui_hint=ui_hint)
                        # 发布轮就要给达成要点：拼进本轮结果块，并登记后续几轮的供给期
                        published_block = await self._publish_task_highlight(
                            ctx, deps, task_id=outcome.payload.get("task_id"),
                        )
                        base_text = (
                            "【后台结果】这份委托已经正式派给玩家了"
                            "（内容就是上面会话状态里那份待确认草案）。"
                        )
                        if published_block:
                            base_text = f"{base_text}\n\n{published_block}"
                        reply = (
                            base_text,
                            TASK_CONFIRM_MERGE_INSTRUCTION,
                            "这事就这么定了，去办吧。",
                            "委托已发布",
                        )
                    else:
                        # confirm 失败：system_notice 给真实原因 + 1 次补救回复（03 §4.2）
                        yield tool_status_event(tool_name, "failed", ui_hint=ui_hint)
                        fail_reason = self._extract_fail_reason(outcome.payload)
                        yield system_notice_event(f"委托确认失败：{fail_reason}")
                        reply = self._fail_reply_args(fail_reason)

                # 成功路径的系统行排在 NPC 台词之后（先听人说、再看到系统行，
                # 与重构前观感一致）；失败路径相反，真实原因已经先发了
                result_block, instruction, fallback_text, notice = reply
                async for ev in self._sync_action_reply(
                    llm, base_messages, meta, spoken_text, result_block,
                    instruction, fallback_text, collected_text,
                ):
                    yield ev
                if notice:
                    yield system_notice_event(notice)

            elif route.kind == "search":
                # ---- 检索：编排层先做关键词预取（检索子 Agent 的第一步）----
                # 等待文案优先用模型自带的（「回忆中」「翻查记录中」），没给才回落「查资料中」
                yield agent_status_event("search", "searching", route.ui_hint or "查资料中")
                prefetch = await self._run_search_prefetch(route, ctx, deps)

                deep = route.deep_search and bool(route.query.strip())
                if route.deep_search and not deep:
                    logger.warning(
                        "会话 %s：route_search 开了 deep_search 但没填 query，按常规检索处理",
                        self.session_id,
                    )

                if deep:
                    # 深度检索：预取结果 + 聊天 Agent 已有的 Tier-1 材料交给子 Agent，
                    # 由它决定继续补查还是直接总结。done/failed 由子 Agent 终态事件发。
                    handle = self._launch_subagent(
                        route, ctx, deps, spoken_text=spoken_text, prefetch=prefetch,
                    )
                    sess.register(handle)
                    merge = MergeCoordinator(
                        handle=handle,
                        llm=llm,
                        base_messages=base_messages,
                        npc_name=self.npc_name,
                        player_query=self.query,
                        spoken_text=spoken_text,
                        emotion=meta.emotion,
                        grace_s=deps.merge_grace_ms / 1000.0,
                        timeout_s=float(deps.subagent_timeout_s),
                    )
                    async for ev in merge.stream():
                        yield ev
                    outcome = merge.outcome
                    self._usage = accumulate_usage(self._usage, outcome.usage)
                else:
                    # 常规检索：不委派子 Agent，结果直接拼进下一轮聊天调用。
                    # _stream_simple 不下发 tools，模型没有可调用的东西，只能开口。
                    from services.orchestrator.search_prefetch import build_prefetch_result_block

                    yield agent_status_event("search", "done", "查资料完成")
                    async for ev in self._sync_action_reply(
                        llm,
                        base_messages,
                        meta,
                        spoken_text,
                        build_prefetch_result_block(prefetch.text),
                        SEARCH_MERGE_INSTRUCTION,
                        "这我也不太清楚。",
                        collected_text,
                    ):
                        yield ev

            else:
                # ---- 后台子 Agent（fire-and-steer）----
                handle = self._launch_subagent(route, ctx, deps, spoken_text=spoken_text)
                sess.register(handle)

                # ---- MERGE_WAIT ----
                merge = MergeCoordinator(
                    handle=handle,
                    llm=llm,
                    base_messages=base_messages,
                    npc_name=self.npc_name,
                    player_query=self.query,
                    spoken_text=spoken_text,
                    emotion=meta.emotion,
                    grace_s=deps.merge_grace_ms / 1000.0,
                    timeout_s=float(deps.subagent_timeout_s),
                )
                async for ev in merge.stream():
                    yield ev
                outcome = merge.outcome
                self._usage = accumulate_usage(self._usage, outcome.usage)
                if outcome.status == "final" and outcome.agent == "task" and outcome.payload.get("draft"):
                    await memory.upsert_draft(
                        self.session_id,
                        outcome.payload["draft"],
                        bargain_count=int(outcome.payload.get("bargain_count") or 0),
                    )
                    notice = (
                        "任务草案已更新，等待确认"
                        if handle.kind == "task_update"
                        else "任务草案已拟定，等待确认"
                    )
                    yield system_notice_event(notice)

            # ---- POST_PROCESS ----
            # 记忆分段：被 agent 等待（拟定委托/检索）中断过的连续说话各存一条
            # assistant 记录 —— 首段正文（collected_text）一段，子 Agent 汇合后的
            # 每次说话爆发（过渡语 #2a / 汇合 #2b / FAIL_REPLY）各一段。前端历史
            # 渲染按消息分气泡，落库分段即历史气泡分段。
            reply_segments: list[str] = []
            first_segment = "".join(collected_text)
            if first_segment.strip():
                reply_segments.append(first_segment)
            if merge is not None:
                reply_segments.extend(
                    seg for seg in merge.outcome.reply_segments if seg.strip()
                )
            await memory.add_message(self.session_id, "user", self.query)
            for segment_text in reply_segments:
                await memory.add_message(self.session_id, "assistant", segment_text)

            if not draft_touched:
                if route_kind is None:
                    # 岔开话题：草案触碰计数 +1，达到上限自动过期取消（03 §4.2）
                    counter = await memory.increment_rounds_without_task(self.session_id)
                    if (
                        ctx.pending_draft_row is not None
                        and counter >= deps.draft_keep_turns
                    ):
                        await memory.delete_draft(
                            self.session_id, draft_id=ctx.pending_draft_row.draft_id,
                        )
                        await memory.reset_rounds_without_task(self.session_id)
                        yield system_notice_event("过期的委托草案已取消")
                elif route_kind in ("task_draft", "task_update"):
                    await memory.reset_rounds_without_task(self.session_id)

            await self._maybe_summarize(deps)

            yield done_event(self.session_id, self._usage)

        except asyncio.CancelledError:
            raise
        except LLMError as exc:
            logger.warning("回合 LLM 错误: %s", exc)
            yield error_event(
                code=type(exc).__name__,
                message=str(exc),
                retryable=exc.retryable,
            )
            yield done_event(self.session_id, self._usage)
        except Exception as exc:
            logger.exception("回合处理异常")
            yield error_event("internal_error", f"{type(exc).__name__}: {exc}", retryable=False)
            yield done_event(self.session_id, self._usage)

    # ------------------------------------------------------------------
    # 组装与转发
    # ------------------------------------------------------------------

    def _build_base_messages(
        self, ctx: TurnContext, *, dialogue_limit: int, relationship_hint: str = "",
    ) -> list[dict[str, Any]]:
        """[system 静态, user shared core]（前缀缓存对齐；tail 与玩家话在末条 user）。

        ``relationship_hint`` 是触发二的好感度跨档提示：只在委派工具真的下发那一轮
        注入，且只在模型尚未调用委派期间持续 RELATIONSHIP_HINT_ROUNDS 轮。
        """
        # 路由闸门关闭时待确认草案不进 prompt：模型既无法确认也无法取消，
        # 摆在它面前只会诱导它空口答应。草案仍照常装配，交给过期计数清理。
        has_pending_draft = self._routing_enabled and ctx.pending_draft_row is not None
        from services.integration.state import get_integration_state

        embedded = get_integration_state().mode == "embedded"
        system = build_static_system(
            npc_name=self.npc_name,
            state=ctx.npc_state,
            same_faction_npcs=ctx.same_faction_block,
            scene_block=ctx.scene_block,
            scene_supplement=ctx.scene_supplement,
            has_shop=ctx.has_shop,
            shop_reward_types=ctx.shop_reward_types,
            player_can_challenge=ctx.player_can_challenge,
            has_pending_draft=has_pending_draft,
            agent_enabled=self.agent_enabled,
            task_publish_disabled=self.task_publish_disabled,
            equipment_block=self._equipment_block(),
            embedded=embedded,
        )
        # 发布轮已经把这块拼进 result_block 了：同一轮若再重建消息，不能在这里再来一遍
        published_block = ctx.published_task_block
        if (
            published_block
            and ctx.published_task_id is not None
            and ctx.published_task_id == self._published_block_task_id
        ):
            published_block = ""
        shared = build_user_shared_core(
            retrieved_context=ctx.rag_context_text,
            mentioned_npcs_str=ctx.mentioned_npcs_str,
            summary_str=self._summary_text(ctx),
            history_str=self._history_block(ctx, limit=dialogue_limit),
            player_identity=self.player_identity,
            progress_desc=ctx.progress_desc,
            favorability=ctx.favorability,
            relationship_level=ctx.relationship_level,
            pending_draft_summary=ctx.pending_draft_summary if has_pending_draft else "",
            save_info=ctx.save_info,
            relationship_hint=relationship_hint,
            long_term_block=format_long_term_block(ctx.long_term) if ctx.long_term else "",
            finished_task_block=build_finished_task_block(
                ctx.finished_task_block.text,
                has_long_term_facts=bool(ctx.long_term and ctx.long_term.facts),
            ),
            active_task_block=ctx.active_task_block,
            published_task_block=published_block,
            item_ledger_block=ctx.item_ledger_block,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": shared},
        ]

    def _equipment_block(self) -> str:
        """NPC 固定层装备块（units.json ＋ items 解析）。任何异常一律降级为空串：
        units.json 缺失或名字对不上，都不该让这轮对话失败。"""
        try:
            from services.npc.npc_equipment import build_npc_equipment_block

            return build_npc_equipment_block(
                npc_name=self.npc_name, game_data=self._resolve_deps().game_data,
            )
        except Exception as exc:
            logger.warning("装备块装配失败，降级为空: %s", exc)
            return ""

    async def _publish_task_highlight(
        self, ctx: TurnContext, deps: OrchestratorDeps, *, task_id: Any,
    ) -> str:
        """发布轮：拼「刚发布的委托」块，并登记后续轮次的供给期。

        构建失败（读不到任务记录 / 缺标题）＝空串且**不写记录**，只记 warning：
        宁可不给，也不给半截内容让模型自己补全。轮次只写进会话库，不进提示词。
        """
        try:
            tid = int(task_id)
        except (TypeError, ValueError):
            logger.warning(
                "会话 %s：confirm 成功但 payload 缺有效 task_id，跳过达成要点", self.session_id,
            )
            return ""
        data_root = getattr(deps.game_data, "data_root", None)
        if data_root is None:
            logger.warning("会话 %s：游戏数据根不可用，跳过发布轮达成要点", self.session_id)
            return ""
        slot_key = ctx.save_info.get("slot_key") if isinstance(ctx.save_info, dict) else None
        try:
            from services.agent_tools.finished_tasks import build_published_task_block

            block = await asyncio.to_thread(
                lambda: build_published_task_block(
                    data_root=data_root,
                    slot_key=slot_key,
                    npc_name=self.npc_name,
                    task_id=tid,
                    game_data=deps.game_data,
                    rag_context_text=ctx.rag_context_text,
                )
            )
        except Exception as exc:
            logger.warning("会话 %s：发布轮达成要点拼接失败，降级为空: %s", self.session_id, exc)
            return ""
        if not block:
            return ""
        # 本轮已拼进 result_block：同一轮若重建消息，user core 里不重复注入
        self._published_block_task_id = tid
        from core.config import get_settings

        total = int(getattr(get_settings(), "published_task_highlight_turns", 5))
        try:
            await deps.memory.published_task_highlight_write(
                self.session_id, tid, max(0, total - 1),
            )
        except Exception as exc:
            logger.warning("会话 %s：达成要点供给期写入失败（本轮仍注入）: %s", self.session_id, exc)
        return block

    @staticmethod
    def _raw_summary(ctx: TurnContext) -> str:
        """滚动摘要原文（永久模糊记忆），无摘要时为空串。"""
        summary = getattr(ctx, "summary", None)
        return str(summary).strip() if summary else ""

    def _summary_text(self, ctx: TurnContext) -> str:
        """早期对话滚动摘要（与长期记忆同属跨轮稀疏层，两块相邻）。标题沿用 v2 口径。

        澄清句不可省：摘要现在紧贴长期记忆块，不点明它只总结本轮较早的对话，
        模型会把两块当成同一份记忆。
        """
        summary = self._raw_summary(ctx)
        if not summary:
            return ""
        return (
            "当前对话历史较长，早期对话已整理为以下摘要"
            "（这是本轮较早的对话内容的摘要，近期对话原文在其后的对话记录里给出）：\n"
            f"{summary}"
        )

    def _history_block(self, ctx: TurnContext, *, limit: int) -> str:
        """近期对话逐字块（聊天主 Agent 用）。

        两点沿用 v2 口径：① 带引导语，其中「在保持人物性格与情节连贯的前提下
        继续对话」是每轮一次、且位置紧贴生成点的入戏提醒，v3 一度整块裸奔；
        ② 无历史记录时整块不注入，不给模型一个空对话框。
        引导语按有无摘要二选一，与 v2 完全一致。
        """
        lines = [
            f"玩家：{m.content}" if m.role == "user" else f"{self.npc_name}：{m.content}"
            for m in ctx.history[-limit:]
        ]
        if not lines:
            return ""
        header = (
            "以下是最近的对话记录（按时间从早到晚排列），"
            "请结合上述摘要与近期记录，在保持人物性格与情节连贯的前提下继续对话："
            if self._raw_summary(ctx)
            else "下面是你与玩家之间的对话历史（按时间从早到晚排列），"
            "请在保持人物性格与情节连贯的前提下继续对话："
        )
        return header + "\n" + "\n".join(lines)

    async def _forward_body(
        self, events: AsyncIterator[Any], sink: list[str],
        *, tool_sink: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[SSEEvent]:
        """转发调用 #1 的正文流（content/usage；finish 时提取 tool_calls 后终止）。"""
        async for ev in events:
            if ev.kind == "content" and ev.text:
                sink.append(ev.text)
                yield content_event(ev.text)
            elif ev.kind == "usage" and ev.usage:
                self._usage = accumulate_usage(self._usage, ev.usage)
            elif ev.kind == "finish":
                if tool_sink is not None and ev.tool_calls:
                    tool_sink.extend(ev.tool_calls)
                return

    async def _emit_meta(self, meta: Meta, deps: OrchestratorDeps) -> SSEEvent:
        if meta.favorability_change:
            old_level = self._ctx.npc_state.relationship_level
            state = await deps.npc_manager.update_favorability(
                self.npc_name, meta.favorability_change,
            )
            self._ctx.npc_state = state
            if state.relationship_level and state.relationship_level != old_level:
                # 触发二：跨档提示只能作用于下一轮——本轮的工具表与请求都已经发出去了，
                # 落库后由 _take_relationship_hint 在下轮取用。
                await self._store_relationship_hint(
                    deps.memory, old_level, state.relationship_level,
                )
        else:
            state = self._ctx.npc_state
        return meta_event(
            emotion=meta.emotion,
            favorability_change=meta.favorability_change,
            favorability=state.favorability,
            relationship_level=state.relationship_level,
        )

    # ------------------------------------------------------------------
    # 路由表下发
    # ------------------------------------------------------------------

    @staticmethod
    def _routing_tools(
        registry: ToolRegistry, *,
        has_pending_draft: bool,
        player_can_challenge: bool = False,
    ) -> list[dict[str, Any]]:
        """聊天主 Agent 的工具表 = 路由表（判据写在工具 description 里）。

        按有无待确认草案二选一：无草案时 confirm/cancel/update 都无从下手，
        摆出来只会诱导模型空调用；有草案时反过来要给足五个入口——四个任务侧
        （confirm/cancel/update/draft）＋ 检索（草案在途时仍可查设定，检索既不取消
        也不提前过期草案）。
        顺序固定（ROUTE_TOOLS_* 的定义序），保证 system/tools 前缀稳定，
        有利于前缀缓存与逐轮 diff。

        player_can_challenge 只作用于 route_task_draft：可切磋时它的描述多一句
        「切磋/对战也能路由」，不可切磋时描述里不出现切磋字样（见该工具模块）。
        该值是稀疏量（玩家升级跨过切磋门槛才翻），不会逐轮抖动。
        """
        names = ROUTE_TOOLS_DRAFT if has_pending_draft else ROUTE_TOOLS_NEW
        out: list[dict[str, Any]] = []
        for name in names:
            tool = registry.get(name)
            if tool is None:
                continue
            if name == "route_task_draft":
                out.append(tool.to_openai_tool(
                    player_can_challenge=player_can_challenge,
                ))
            else:
                out.append(tool.to_openai_tool())
        return out

    async def _apply_long_term_calls(
        self, deps: OrchestratorDeps, ctx: TurnContext, tool_sink: list[dict[str, Any]],
    ) -> list[str]:
        """落库本轮聊天 Agent 的长期记忆调用（工具不在注册表里，这里直接执行）。

        写库失败不阻断回合：记忆是附加能力，玩家这轮的对话必须照常完成。
        返回「结果＋写进去的内容」逐条列表，供空正文兜底轮告诉 NPC 已经记了什么。
        """
        snapshot = ctx.long_term
        if snapshot is None or not tool_sink:
            return []
        recorded: list[str] = []
        for tool_call in tool_sink:
            if tool_call_name(tool_call) != CHAT_TOOL_NAME:
                continue
            try:
                result = await apply_chat_tool(
                    deps.memory, tool_call, snapshot=snapshot, session_id=self.session_id,
                )
            except Exception as exc:
                logger.warning("会话 %s 长期记忆写入失败（忽略）: %s", self.session_id, exc)
                continue
            if result:
                logger.info(
                    "会话 %s 长期记忆更新（作用域 %s / %s）：%s",
                    self.session_id, snapshot.slot_key, snapshot.npc, result,
                )
                recorded.append(_long_term_record_line(result, tool_call))
        return recorded

    async def _apply_delegate_calls(
        self,
        deps: OrchestratorDeps,
        tool_sink: list[dict[str, Any]],
        *,
        registered: bool,
    ) -> list[str]:
        """落地本轮的对话修改委派（触发二、三共用同一个工具，不在全局注册表里）。

        注册闸不通过时一律不受理：工具没下发却硬发调用属于越权，只留痕。
        受理过就清掉跨档提示——§4.0.2-2 要求「一旦调用立刻去掉提示词」。
        返回逐条结果文本，供空正文兜底轮告诉 NPC 已经派了什么；
        任何失败只记日志，绝不阻断本轮对话（§4.0.1 独立链路）。
        """
        calls = [tc for tc in tool_sink if tool_call_name(tc) == DELEGATE_TOOL_NAME]
        if not calls:
            return []
        if not registered:
            logger.warning(
                "会话 %s：委派工具本轮未下发（注册闸不通过），模型仍硬发调用，忽略",
                self.session_id,
            )
            return []
        try:
            results = await apply_delegate_calls(
                calls,
                npc_name=self.npc_name,
                session_id=self.session_id,
                llm_config=self.llm_config,
            )
        except Exception as exc:
            logger.warning("会话 %s 对话修改委派执行失败（忽略）: %s", self.session_id, exc)
            return []
        try:
            await deps.memory.dialogue_hint_clear(self.session_id)
        except Exception as exc:
            logger.warning("会话 %s 跨档提示清除失败（忽略）: %s", self.session_id, exc)
        return results

    async def _take_relationship_hint(self, memory: MemoryStore) -> str:
        """取本会话待提示的好感度跨档文案并扣掉一轮（触发二）。

        只在委派工具真的下发那一轮调用：注册闸不通过时既不注入也不扣轮，
        否则三轮会在模型手上根本没有工具的情况下白白烧掉。
        """
        try:
            hint = await memory.dialogue_hint_read(self.session_id)
            if hint is None:
                return ""
            await memory.dialogue_hint_write(
                self.session_id, hint.hint, hint.rounds_left - 1,
            )
            return hint.hint
        except Exception as exc:
            logger.warning("会话 %s 跨档提示读取失败（本轮不提示）: %s", self.session_id, exc)
            return ""

    async def _store_relationship_hint(
        self, memory: MemoryStore, old_level: str, new_level: str,
    ) -> None:
        """落一条跨档提示，持续 RELATIONSHIP_HINT_ROUNDS 轮，模型调用过即清除。"""
        try:
            await memory.dialogue_hint_write(
                self.session_id,
                build_relationship_hint(old_level, new_level),
                RELATIONSHIP_HINT_ROUNDS,
            )
        except Exception as exc:
            logger.warning("会话 %s 跨档提示写入失败（忽略）: %s", self.session_id, exc)

    def _recent_dialogue_text(
        self, ctx: TurnContext, spoken_text: str, *, limit: int = SUBAGENT_RECENT_MESSAGES,
    ) -> str:
        """子 Agent 用的近期对话逐字块（聊天主 Agent 走 _history_block）。

        固定预算 SUBAGENT_RECENT_MESSAGES（近 4 轮 + 本轮已流出正文，含过渡话），
        不随玩家选的精确短期记忆档位变化；无记录时给占位行，因为子 Agent 的
        prompt 里这块有标题、留空会让它以为装配出错。
        """
        lines: list[str] = []
        for m in ctx.history[-limit:]:
            lines.append(f"玩家：{m.content}" if m.role == "user" else f"{self.npc_name}：{m.content}")
        if spoken_text.strip():
            lines.append(f"{self.npc_name}：{spoken_text.strip()}")
        return "\n".join(lines) or "（暂无对话记录）"

    # ------------------------------------------------------------------
    # 同步动作
    # ------------------------------------------------------------------

    def _finish_npc_reference(
        self, ctx: TurnContext, game_data: Any, finish_npc: str,
    ) -> str:
        """完成 NPC 的写作参考（过往台词/情绪标签/与玩家的关系）。

        台词取该 NPC 的全部日常台词＋前 6 条任务台词（含本存档覆盖层），
        只读；读取失败降级为空串。
        只在完成 NPC ≠ 当前说话人时由调用方触发。
        """
        from services.dialogue.view import build_view
        from services.npc.scene_context import npc_state_of

        name = (finish_npc or "").strip()
        if not name:
            return ""
        state = npc_state_of(getattr(ctx, "npc_states", None), name)
        lines: list[str] = []
        save_info = getattr(ctx, "save_info", None)
        slot_key = (
            str(save_info.get("slot_key") or "")
            if isinstance(save_info, dict) else ""
        )
        try:
            lines = build_view(game_data=game_data, slot_key=slot_key).npc_line_samples(name)
        except Exception as exc:
            logger.warning("完成 NPC 台词样本读取失败，降级为空: %s", exc)
        return build_finish_npc_reference(
            npc_name=name,
            lines=lines,
            emotions=list(getattr(state, "emotions", None) or []) if state is not None else [],
            favorability=state.favorability if state is not None else None,
            relationship_level=state.relationship_level if state is not None else "",
        )

    async def _generate_confirm_args(
        self,
        llm: LLMClient,
        ctx: TurnContext,
        draft_row: TaskDraftRow,
        game_data: Any,
        *,
        base_messages: list[dict[str, Any]],
        spoken_text: str = "",
        emotion: str = "",
    ) -> tuple[dict[str, Any], list[str]]:
        """confirm 发布文本生成（终端与插话共用模块级 `generate_confirm_args`）。

        与聊天 Agent **同源前缀**（世界观/扮演/RAG/会话态/近期历史），追加本轮
        过渡话与草案详情——保证生成的对话贴合对话脉络。单次、无工具、宽松归一化；
        仅当 LLM 调用整体失败或解析不出 JSON 时回退模板（优先保证发布流程可用）。

        完成 NPC ≠ 当前说话人时，末尾再拼该 NPC 的过往台词/情绪/关系参考：
        base_messages 里的资料与台词样本全是当前 NPC 的，不补这段模型只能凭名字
        脑补对方的说话方式。

        说话人校验不过时额外做 **一次** 修正重试（带具体问题），仍不通过就降级：
        原两组对话本身合规 → 保留模型产出、关掉回访配对；原两组也不合规 → 整份换
        模板兜底。返回 (args, warnings)，warnings 回给调用方合并进确认结果。
        """
        draft = draft_row.draft
        finish_reference = ""
        _, finish_npc = resolve_task_npcs(draft, self.npc_name)
        if _norm(finish_npc) != _norm(self.npc_name):
            finish_reference = await asyncio.to_thread(
                self._finish_npc_reference, ctx, game_data, finish_npc,
            )
        summary = _detailed_draft_summary(
            draft, game_data, rag_context_text=ctx.rag_context_text,
            npc_states=ctx.npc_states,
        )
        outcome = await generate_confirm_args(
            llm,
            draft=draft,
            npc_name=self.npc_name,
            base_messages=base_messages,
            draft_summary=summary,
            spoken_text=spoken_text,
            emotion=emotion,
            finish_npc_reference=finish_reference,
        )
        for usage in outcome.usages:
            self._usage = accumulate_usage(self._usage, usage)
        args = outcome.args
        args["draft_id"] = draft_row.draft_id
        return args, outcome.warnings

    def _degrade_confirm_args(
        self,
        draft: dict[str, Any],
        args: dict[str, Any],
        issues: list[dict[str, str]],
        *,
        get_npc: str,
        finish_npc: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        """两段式校验最终未通过：退回单任务，绝不写半套配对。"""
        return degrade_confirm_args(
            draft, args, issues, npc_name=self.npc_name,
            get_npc=get_npc, finish_npc=finish_npc, warnings=warnings,
        )[0]

    def _sanitize_confirm_args(
        self,
        args: dict[str, Any] | None,
        *,
        get_npc: Optional[str] = None,
        finish_npc: Optional[str] = None,
        allow_return: Optional[bool] = None,
    ) -> dict[str, Any] | None:
        """宽松归一化（实现见模块级 `sanitize_confirm_args`）。"""
        return sanitize_confirm_args(
            args, npc_name=self.npc_name, get_npc=get_npc,
            finish_npc=finish_npc, allow_return=allow_return,
        )

    @staticmethod
    def _extract_fail_reason(payload: dict[str, Any]) -> str:
        """confirm 失败的结构化原因（system_notice 用真实原因）。"""
        issues = payload.get("issues") or []
        if issues:
            first = issues[0] if isinstance(issues[0], dict) else {}
            return str(first.get("message") or "草案校验未通过")
        message = payload.get("message")
        return str(message or "未知原因")

    @staticmethod
    def _fail_reply_args(fail_reason: str) -> tuple[str, str, str, str]:
        """confirm 失败的补救回复参数（真实原因已先发 system_notice，这里只让 NPC 收场）。"""
        return (
            f"【真实原因（对玩家只说人话，不要提系统/校验等词汇）】{fail_reason}",
            CONFIRM_FAIL_INSTRUCTION,
            "这份委托刚才出了点问题，改天再谈吧。",
            "",
        )

    async def _sync_action_reply(
        self,
        llm: LLMClient,
        base_messages: list[dict[str, Any]],
        meta: Meta,
        spoken_text: str,
        result_block: str,
        instruction: str,
        fallback_text: str,
        sink: list[str],
    ) -> AsyncIterator[SSEEvent]:
        """同步动作（confirm/cancel）办完后让 NPC 开口。

        路由轮按契约不说话，所以这里必须补一次生成——否则玩家只看到一条
        system_notice 就没了下文。与汇合 #2 同构（同前缀 + 情绪回显 + 结果块 +
        指令），差别是不用等子 Agent、也不需要过渡语。
        """
        prompt = build_merge_user_prompt(
            npc_name=self.npc_name,
            player_query=self.query,
            spoken_text=spoken_text,
            emotion=meta.emotion,
            result_block=result_block,
            instruction=instruction,
        )
        before = len(sink)
        async for ev in self._stream_simple(llm, base_messages, prompt, sink):
            yield ev
        if len(sink) == before:
            # 生成失败或空回复：静态兜底，玩家侧不能留空白
            sink.append(fallback_text)
            yield content_event(fallback_text)

    async def _stream_simple(
        self,
        llm: LLMClient,
        base_messages: list[dict[str, Any]],
        user_prompt: str,
        sink: list[str],
    ) -> AsyncIterator[SSEEvent]:
        messages = [*base_messages, {"role": "user", "content": user_prompt}]
        try:
            stream = llm.chat_stream(ChatRequest(
                messages=messages, purpose="chat", send_image=False,
            ))
            _meta, events = await split_meta_events(stream, [])
            async for ev in events:
                if ev.kind == "content" and ev.text:
                    sink.append(ev.text)
                    yield content_event(ev.text)
                elif ev.kind == "usage" and ev.usage:
                    self._usage = accumulate_usage(self._usage, ev.usage)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("流式回复调用失败: %s", exc)

    async def _empty_reply_rescue(
        self,
        llm: LLMClient,
        base_messages: list[dict[str, Any]],
        meta: Meta,
        sink: list[str],
        recorded: list[str],
        delegated: list[str],
    ) -> AsyncIterator[SSEEvent]:
        """空正文兜底：纯聊天轮没说出一个字时，再开一次**不带任何工具**的调用。

        模型会把「调用路由工具的那一轮不要写台词」套到长期记忆工具上，发完调用
        就静默。这里复用的 _stream_simple 不下发 tools，工具连定义都不在，
        模型没有可调的东西，只能开口；已经落库/已经派出的结果用文字告诉它，
        免得它重复记或重复派。
        生成失败或还是空回复时给静态收场话，玩家侧不能留空白。
        """
        prompt = build_empty_reply_user_prompt(
            player_query=self.query,
            emotion=meta.emotion,
            recorded="；".join(recorded),
            delegated="；".join(delegated),
        )
        before = len(sink)
        async for ev in self._stream_simple(llm, base_messages, prompt, sink):
            yield ev
        if len(sink) == before:
            sink.append(EMPTY_REPLY_FALLBACK_TEXT)
            yield content_event(EMPTY_REPLY_FALLBACK_TEXT)

    # ------------------------------------------------------------------
    # 子 Agent 启动
    # ------------------------------------------------------------------

    async def _run_search_prefetch(
        self, route: Route, ctx: TurnContext, deps: OrchestratorDeps,
    ) -> Any:
        """route_search 的编排层预取（同步检索放线程池，避免阻塞事件循环）。

        检索范围由 route.sources 决定；结果与聊天 Agent 已有的 Tier-1 材料按节点 id
        去重。任何失败都降级为空结果，后续仍会让 NPC 开口，不阻断回合。
        """
        from services.orchestrator.search_prefetch import SearchPrefetch, run_search_prefetch

        keywords = (route.keywords or self.query).strip()
        if not keywords:
            return SearchPrefetch(sources=route.sources)

        def _run() -> Any:
            return run_search_prefetch(
                engine=deps.engine,
                game_data=deps.game_data,
                keywords=keywords,
                sources=route.sources,
                npc_name=ctx.npc_name,
                npc_titles=list(ctx.npc_state.titles or []),
                npc_faction=ctx.npc_state.faction,
                forbidden=ctx.forbidden,
                existing_bundle=ctx.retrieval_bundle,
            )

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            logger.warning("会话 %s：检索预取失败，降级为空: %s", self.session_id, exc)
            return SearchPrefetch(sources=route.sources)

    def _launch_subagent(
        self,
        route: Route,
        ctx: TurnContext,
        deps: OrchestratorDeps,
        *,
        spoken_text: str = "",
        prefetch: Any = None,
    ) -> Any:
        """按路由声明启动后台子 Agent（fire-and-steer）。

        任务侧三个入口（draft/update，含 update 无草案时的降级）共享一套 NPC 身份
        与能力参数——报酬能力（商店类型）和切磋可行性是任务 Agent 自己选型时要用
        的硬边界，聊天主 Agent 只负责说「往哪个方向派」。
        """
        progress = ctx.progress_stage or 1
        affinity = ctx.favorability
        npc_titles = list(ctx.npc_state.titles or [])
        npc_states = ctx.npc_states
        llm = deps.llm_factory(self.llm_config)
        recent_dialogue = self._recent_dialogue_text(ctx, spoken_text)
        # 子 Agent 的对话预算固定 8 条，但永久模糊记忆（滚动摘要）另算：它体积恒定
        # 且不随档位膨胀，能让任务 Agent 知道「这玩家早先谈过什么」，故一并带上，
        # 在 prompt 里排在最近 8 条之前（详见 subagents/prompts.build_task_user）。
        memory_summary = self._raw_summary(ctx)

        if route.kind == "search":
            return SearchRunner.launch(
                llm=llm,
                registry=deps.registry,
                query=route.query or route.keywords or self.query,
                player_query=self.query,
                npc_name=self.npc_name,
                npc_faction=ctx.npc_state.faction or "",
                npc_titles=npc_titles,
                npc_challenge=ctx.npc_state.challenge,
                player_progress=progress,
                npc_affinity=affinity,
                npc_states=npc_states,
                game_data=deps.game_data,
                retrieve_fn=ctx.retrieve_fn,
                existing_material=ctx.rag_context_text,
                first_round_material=(prefetch.text if prefetch is not None else ""),
                max_rounds=deps.search_max_rounds,
                ui_hint=route.ui_hint,
            )

        task_kwargs: dict[str, Any] = dict(
            llm=llm,
            registry=deps.registry,
            recent_dialogue=recent_dialogue,
            memory_summary=memory_summary,
            npc_name=self.npc_name,
            npc_faction=ctx.npc_state.faction or "",
            npc_titles=npc_titles,
            npc_challenge=ctx.npc_state.challenge,
            player_progress=progress,
            player_level=ctx.player_level,
            progress_desc=ctx.progress_desc,
            npc_affinity=affinity,
            npc_states=npc_states,
            game_data=deps.game_data,
            rag_context_text=ctx.rag_context_text,
            retrieve_fn=ctx.retrieve_fn,
            max_rounds=deps.task_max_rounds,
        )
        new_draft_kwargs = {
            **task_kwargs,
            "has_shop": ctx.has_shop,
            "shop_reward_types": ctx.shop_reward_types,
            "player_can_challenge": ctx.player_can_challenge,
            "pending_draft": None,
            "bargain_count": 0,
            "draft_commit_valid": False,
        }

        draft_row = ctx.pending_draft_row
        if route.kind == "task_update" and (
            draft_row is None or not isinstance(draft_row.draft, dict) or not draft_row.draft
        ):
            # 无草案可更新：降级为按备注重拟（路由错发的容错）
            logger.info(
                "会话 %s：task_update 但无待修改草案，降级为新草案", self.session_id,
            )
            return TaskRunner.launch(
                kind="task_draft",
                direction=route.note or self.query,
                player_query=self.query,
                note="",
                **new_draft_kwargs,
            )

        if route.kind == "task_update":
            return TaskRunner.launch(
                kind="task_update",
                direction="",
                player_query=self.query,
                note=route.note,
                pending_draft=draft_row.draft,
                bargain_count=draft_row.bargain_count,
                draft_commit_valid=True,
                **task_kwargs,
            )

        # task_draft：note 是聊天主 Agent 提炼的诉求要点，缺省时回落到玩家原话；
        # task_type / reward_types 由任务 Agent 自己 prepare 选型，聊天侧不再决定
        return TaskRunner.launch(
            kind="task_draft",
            direction=route.note or self.query,
            player_query=self.query,
            note="",
            **new_draft_kwargs,
        )

    # ------------------------------------------------------------------
    # 摘要触发
    # ------------------------------------------------------------------

    async def _maybe_summarize(self, deps: OrchestratorDeps) -> None:
        if not deps.enable_summary:
            return
        try:
            count = await deps.memory.count_messages(self.session_id)
            if not should_summarize(count):
                return
            worker = get_summary_worker()
            if worker is None:
                return
            # 作用域在本轮就解析好交给 worker：摘要延后执行期间玩家换存档/解绑，
            # 基础长期记忆也不会写进另一个作用域。
            snapshot = self._ctx.long_term if self._ctx is not None else None
            worker.submit(SummaryRequest(
                session_id=self.session_id,
                npc_name=self.npc_name,
                llm_config=self.llm_config,
                long_term_scope=snapshot.slot_key if snapshot is not None else "",
            ))
        except Exception as exc:  # 摘要不阻断回合
            logger.warning("摘要提交失败（忽略）: %s", exc)
