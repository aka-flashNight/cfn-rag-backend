"""玩家设置项在编排层的落地：Agent 能力开关 + 精确短期记忆档位。

两项都是前端「终端设置」面板的请求级字段（NPCChatRequest.agent_enabled /
summarize_interval），语义边界：
- 开关关闭 = 本回合纯聊天（不下发任何路由工具、prompt 不含路由说明、
  模型硬发的 route_* 调用一律忽略）；
- 档位 = 逐字进 prompt 的近期对话条数，同时决定从库里读多少条；
- 两者都不传时行为与字段引入前完全一致（服务端默认 20 条读库 / 8 条进 prompt）。
"""

from __future__ import annotations

from services.orchestrator.turn import SUBAGENT_RECENT_MESSAGES, TurnOrchestrator
from services.tools.base import ROUTE_TOOLS_DRAFT, ROUTE_TOOLS_NEW
from tests.orchestrator.conftest import (
    MERGE_REPLY_TASK,
    NPC_NAME,
    VALID_DRAFT,
    collect,
    compressed_kinds,
    contents_of,
    notices_of,
    phases_of,
    tc,
)


def _orch(env, query: str, **kwargs) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id,
        npc_name=NPC_NAME,
        query=query,
        deps=env.deps,
        **kwargs,
    )


def _system_prompt(env) -> str:
    return env.fake.stream_requests[0].messages[0]["content"]


def _shared_core(env) -> str:
    return env.fake.stream_requests[0].messages[1]["content"]


def _chat_tool_names(env) -> list[str]:
    return [t["function"]["name"] for t in (env.fake.stream_requests[0].tools or [])]


async def _seed_history(env, count: int) -> None:
    """按 玩家/NPC 交替塞入带序号的历史消息（序号零填充，便于子串断言）。"""
    for i in range(1, count + 1):
        role = "user" if i % 2 == 1 else "assistant"
        await env.store.add_message(env.session_id, role, f"历史消息{i:02d}")


# ---------------------------------------------------------------------------
# Agent 能力开关
# ---------------------------------------------------------------------------

async def test_agent_disabled_offers_no_tool(env):
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯，聊什么都行。")
    await collect(_orch(env, "你好", agent_enabled=False))
    assert env.fake.stream_requests[0].tools is None


async def test_agent_enabled_by_default_offers_new_task_routes(env):
    """不传开关 = 既有行为：聊天主 Agent 拿到路由表（无草案分支）。"""
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯，聊什么都行。")
    await collect(_orch(env, "你好"))
    assert _chat_tool_names(env) == list(ROUTE_TOOLS_NEW)


async def test_pending_draft_widens_route_table(env):
    """有待确认草案时换成草案分支表：confirm/cancel/update 三个入口才有意义。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯，聊什么都行。")
    await collect(_orch(env, "在吗"))
    assert _chat_tool_names(env) == list(ROUTE_TOOLS_DRAFT)


async def test_agent_disabled_system_prompt_drops_delegation(env):
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="今天没活。")
    await collect(_orch(env, "给我个任务", agent_enabled=False))

    system = _system_prompt(env)
    assert "当前无法派发委托" in system
    for forbidden in ("prepare_task_context", "route_", "路由",
                      "任务发布硬约束", "reward_types"):
        assert forbidden not in system


async def test_agent_disabled_prompt_guides_player_to_open_permission(env):
    """玩家坚持要活时，NPC 要引导他去终端打开接取委托的权限（沿用 v2 口径）。"""
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="终端权限还关着呢。")
    await collect(_orch(env, "我就要接活", agent_enabled=False))

    system = _system_prompt(env)
    assert "接取委托的权限" in system
    assert "终端" in system
    # 引导也要在人设里说：给出角色口吻的范例，避免模型直接报界面术语
    assert "你那终端的委托权限还关着呢" in system


async def test_agent_disabled_ignores_model_declared_route(env):
    """模型越权硬发 route_task_draft：按纯聊天放行正文，不起子 Agent、不落草案。"""
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0}',
        text="这事我记下了。",
        tool_calls=[tc("route_task_draft", {"note": "玩家想要个活"})],
    )
    events = await collect(_orch(env, "给我安排个活", agent_enabled=False))

    assert compressed_kinds(events) == ["meta", "content", "done"]
    assert phases_of(events) == []
    assert "".join(contents_of(events)) == "这事我记下了。"
    assert await env.store.get_draft(env.session_id) is None
    # 只发生了聊天主调用，没有任何子 Agent / 补救调用
    assert len(env.fake.stream_requests) == 1
    assert env.fake.chat_requests == []


async def test_agent_disabled_hides_pending_draft_and_lets_it_expire(env):
    """残留草案不进 prompt（模型无法确认它），但仍照常累计过期计数被清理。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    for i in range(3):
        env.fake.add_stream(meta='{"emo":"普通","fav":0}', text=f"闲聊{i}。")
        await collect(_orch(env, "聊点别的", agent_enabled=False))

    assert "食材收集委托" not in _shared_core(env)
    assert "待确认的任务草案" not in _shared_core(env)
    # 关闭态连路由表都不下发，残留草案也不会让 confirm/cancel 入口冒出来
    assert env.fake.stream_requests[0].tools is None
    assert await env.store.get_draft(env.session_id) is None


async def test_agent_disabled_confirm_route_does_not_publish(env, monkeypatch):
    """关闭态下 route_task_confirm 也不能发布任务：不写任务文件、草案保留待过期。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    monkeypatch.setattr(env.game_data, "data_root", env.tmp_path / "data")

    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0}',
        text="行，那就这样。",
        tool_calls=[tc("route_task_confirm", {})],
    )
    events = await collect(_orch(env, "我接了", agent_enabled=False))

    assert compressed_kinds(events) == ["meta", "content", "done"]
    assert notices_of(events) == []
    assert not (env.tmp_path / "data" / "task" / "agent_tasks.json").exists()
    assert await env.store.get_draft(env.session_id) is not None


# ---------------------------------------------------------------------------
# 精确短期记忆档位
# ---------------------------------------------------------------------------

async def test_default_window_keeps_last_eight_messages(env):
    await _seed_history(env, 30)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "还记得前面说的吗")
    await collect(orch)

    core = _shared_core(env)
    assert "历史消息23" in core
    assert "历史消息22" not in core
    # 读库用服务端默认上限，prompt 窗口另由 recent_dialogue_limit 决定
    assert len(orch._ctx.history) == env.deps.history_limit


async def test_precise_memory_tier_widens_prompt_and_db_read(env):
    await _seed_history(env, 14)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "还记得前面说的吗", summarize_interval=10)
    await collect(orch)

    core = _shared_core(env)
    assert "历史消息05" in core
    assert "历史消息04" not in core
    # 档位同时收紧读库条数：只取最近 10 条
    assert len(orch._ctx.history) == 10
    assert orch.precise_memory_limit == 10


async def test_precise_memory_almost_unlimited_keeps_everything(env):
    await _seed_history(env, 40)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "全都记得吗", summarize_interval=500)
    await collect(orch)

    core = _shared_core(env)
    assert "历史消息01" in core
    assert "历史消息40" in core
    assert len(orch._ctx.history) == 40


async def test_invalid_tier_falls_back_to_server_default(env):
    await _seed_history(env, 14)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "在吗", summarize_interval=7)
    await collect(orch)

    assert orch.precise_memory_limit is None
    core = _shared_core(env)
    assert "历史消息07" in core
    assert "历史消息06" not in core


async def test_tier_does_not_change_subagent_dialogue_budget(env):
    """档位只作用于聊天主 Agent；子 Agent 的近期对话仍用固定预算。"""
    await _seed_history(env, 30)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "都记得吗", summarize_interval=500)
    await collect(orch)

    ctx = orch._ctx
    assert len(ctx.history) == 30
    # 聊天主 Agent 的 prompt 拿到全部 30 条
    assert "历史消息01" in _shared_core(env)

    # 子 Agent 用的默认预算仍是 SUBAGENT_RECENT_MESSAGES 条
    subagent_text = orch._recent_dialogue_text(ctx, "")
    oldest_kept = 30 - SUBAGENT_RECENT_MESSAGES + 1
    assert f"历史消息{oldest_kept:02d}" in subagent_text
    assert f"历史消息{oldest_kept - 1:02d}" not in subagent_text


# ---------------------------------------------------------------------------
# 子 Agent 的永久模糊记忆
# ---------------------------------------------------------------------------

SUMMARY_TEXT = "玩家早先替铁匠跑过一趟腿，还问过废城的路。"


async def _drive_task_draft(env, **orch_kwargs) -> TurnOrchestrator:
    """跑一轮完整的 task_draft（路由 → prepare → draft → 汇合），返回编排器实例。"""
    # 宽限期给足，避免过渡语多插一次流式调用打乱 stream_requests 的下标
    env.deps.merge_grace_ms = 500
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0}',
        text="",
        tool_calls=[tc("route_task_draft", {"note": "玩家想找点活干"})],
    )
    env.fake.add_chat(tool_calls=[tc("prepare_task_context", {
        "task_type": "资源收集",
        "reward_types": {"regular": ["金币"], "optional": []},
    }, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", {
        "task_type": "资源收集",
        "title": "食材收集委托",
        "rewards": [{"item_name": "金币", "count": 25000}],
    })])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    orch = _orch(env, "有什么活给我干吗", **orch_kwargs)
    await collect(orch)
    return orch


def _subagent_user_prompt(env) -> str:
    """任务 Agent 第一次调用的 user prompt（messages[0] 是 system）。"""
    return env.fake.chat_requests[0].messages[1]["content"]


async def test_subagent_gets_fuzzy_memory_before_recent_dialogue(env):
    """滚动摘要进任务 Agent，且排在最近对话块之前。"""
    await env.store.save_summary(env.session_id, SUMMARY_TEXT, 30)
    await _seed_history(env, 30)
    await _drive_task_draft(env)

    prompt = _subagent_user_prompt(env)
    assert SUMMARY_TEXT in prompt
    assert "更早的对话印象" in prompt
    assert prompt.index(SUMMARY_TEXT) < prompt.index("【最近对话")


async def test_subagent_fuzzy_memory_does_not_widen_recent_window(env):
    """档位调到「几乎无限」也不会撑大子 Agent 的近期对话预算。"""
    await env.store.save_summary(env.session_id, SUMMARY_TEXT, 30)
    await _seed_history(env, 30)
    await _drive_task_draft(env, summarize_interval=500)

    prompt = _subagent_user_prompt(env)
    assert SUMMARY_TEXT in prompt
    oldest_kept = 30 - SUBAGENT_RECENT_MESSAGES + 1
    assert f"历史消息{oldest_kept:02d}" in prompt
    assert f"历史消息{oldest_kept - 1:02d}" not in prompt
    # 聊天主 Agent 那边确实拿到了全部 30 条（档位生效，不是两处都没生效）
    assert "历史消息01" in _shared_core(env)


async def test_subagent_prompt_omits_memory_block_when_no_summary(env):
    """没有滚动摘要时不注入空块。"""
    await _seed_history(env, 6)
    await _drive_task_draft(env)

    prompt = _subagent_user_prompt(env)
    assert "更早的对话印象" not in prompt
    assert "【最近对话" in prompt
