"""草案在途时仍可检索（工作项 3）。

口径：有待确认草案时工具表给足**五个**入口（四个任务侧 + route_search）；
检索不改草案生命周期——既不取消、也不重置/推进 `rounds_without_task`，
草案过期仍只由「连续 N 轮纯聊天」触发。
"""

from __future__ import annotations

from services.orchestrator.search_prefetch import SearchPrefetch
from services.orchestrator.turn import TurnOrchestrator
from services.tools.base import ROUTE_TOOL_NAMES, ROUTE_TOOLS_DRAFT, ROUTE_TOOLS_NEW
from tests.orchestrator.conftest import NPC_NAME, VALID_DRAFT, collect, notices_of, tc

TASK_TOOLS = ("route_task_confirm", "route_task_cancel", "route_task_update", "route_task_draft")


def _orch(env, query: str) -> TurnOrchestrator:
    return TurnOrchestrator(session_id=env.session_id, npc_name=NPC_NAME, query=query, deps=env.deps)


def _tool_names(req) -> list[str]:
    return [t["function"]["name"] for t in (req.tools or [])]


def _patch_prefetch(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.orchestrator.search_prefetch.run_search_prefetch",
        lambda **kwargs: SearchPrefetch(text="【世界观】安迪曾是A兵团佣兵。", sources=("knowledge",)),
    )


async def _draft_turn(env) -> None:
    """跑一轮带草案的纯聊天（用来观察草案是否还在、计数是否推进）。"""
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯，那你再想想。")
    await collect(_orch(env, "嗯……"))


def test_route_tool_tables_are_fixed_and_complete():
    """工具表定义是固定顺序的常量：有草案时＝四个任务侧 + 检索；无草案时只有两个入口。"""
    assert ROUTE_TOOLS_DRAFT == (*TASK_TOOLS, "route_search")
    assert ROUTE_TOOLS_NEW == ("route_task_draft", "route_search")
    # 两张表都是全量表的子集（不存在「表里有、注册表却没有」的名字）
    assert set(ROUTE_TOOLS_DRAFT) <= set(ROUTE_TOOL_NAMES)
    assert set(ROUTE_TOOLS_NEW) <= set(ROUTE_TOOL_NAMES)


async def test_pending_draft_offers_search_tool(env):
    """有草案时本轮工具表同时包含 route_search 与四个任务工具（顺序固定）。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, "你好"))

    names = _tool_names(env.fake.stream_requests[0])
    assert names == list(ROUTE_TOOLS_DRAFT)
    assert "route_search" in names


async def test_search_does_not_touch_draft_or_counter(env, monkeypatch):
    """草案在途时检索：草案还在、rounds_without_task 不被推进也不被重置。"""
    _patch_prefetch(monkeypatch)
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="", tool_calls=[tc("route_search", {"keywords": "安迪"})],
    )
    env.fake.add_stream(text="安迪以前是A兵团的佣兵。")
    await collect(_orch(env, "安迪以前是谁？"))

    assert await env.store.get_draft(env.session_id) is not None
    assert env.store.get_rounds_without_task_sync(env.session_id) == 0


async def test_repeated_search_never_expires_draft(env, monkeypatch):
    """连续多次检索不会把草案清掉（过期只看纯聊天轮次）。"""
    _patch_prefetch(monkeypatch)
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    for i in range(5):
        env.fake.add_stream(
            meta='{"emo":"普通","fav":0}',
            text="",
            tool_calls=[tc("route_search", {"keywords": f"安迪{i}"})],
        )
        env.fake.add_stream(text="我再查查。")
        await collect(_orch(env, f"那{ i }呢？"))

    assert await env.store.get_draft(env.session_id) is not None
    assert env.store.get_rounds_without_task_sync(env.session_id) == 0


async def test_pure_chat_still_expires_after_search(env, monkeypatch):
    """检索之后连续 3 轮纯聊天，仍按原逻辑过期（draft_keep_turns 语义不变）。"""
    _patch_prefetch(monkeypatch)
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="", tool_calls=[tc("route_search", {"keywords": "安迪"})],
    )
    env.fake.add_stream(text="安迪以前是A兵团的佣兵。")
    await collect(_orch(env, "安迪以前是谁？"))
    assert env.store.get_rounds_without_task_sync(env.session_id) == 0

    await _draft_turn(env)
    assert env.store.get_rounds_without_task_sync(env.session_id) == 1
    await _draft_turn(env)
    assert await env.store.get_draft(env.session_id) is not None

    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="行吧。")
    events = await collect(_orch(env, "算了不说了"))
    assert "过期的委托草案已取消" in notices_of(events)
    assert await env.store.get_draft(env.session_id) is None


async def test_routing_guide_mentions_search_while_draft_pending():
    """提示词也要说清：草案在途时仍可检索，检索不会取消或提前过期草案。"""
    from services.orchestrator.prompts import build_routing_guide

    guide = build_routing_guide(has_pending_draft=True)
    assert "route_search" in guide
    assert "检索不会取消草案，也不会让它提前过期" in guide
    # 四个任务侧入口仍逐条在
    for tool in TASK_TOOLS:
        assert tool in guide
