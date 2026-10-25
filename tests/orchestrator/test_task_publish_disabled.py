"""无驻地 NPC 任务发布临时关闭（用户裁定 2026-10-11，临时措施）。

验收口径：与「Agent 能力关闭」同闸门（不下发路由工具、prompt 无路由说明、
模型硬发的路由调用一律忽略），但提示词换成 NO_TASK_PUBLISH_CONSTRAINT——
以人设口吻直接婉拒，不引导玩家去改终端权限（玩家侧没有可开的闸）。
"""

from __future__ import annotations

from services.npc.task_publish_policy import is_task_publish_disabled
from services.orchestrator.prompts import NO_TASK_PUBLISH_CONSTRAINT
from services.orchestrator.turn import TurnOrchestrator
from services.tools.base import ROUTE_TOOLS_NEW
from tests.orchestrator.conftest import (
    VALID_DRAFT,
    collect,
    compressed_kinds,
    contents_of,
    notices_of,
    tc,
)

DISABLED_NPC = "lochan"
NORMAL_NPC = "冷兵器商人"


def _orch(env, query: str, npc_name: str, **kwargs) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id,
        npc_name=npc_name,
        query=query,
        deps=env.deps,
        **kwargs,
    )


def _system_prompt(env) -> str:
    return env.fake.stream_requests[0].messages[0]["content"]


def test_policy_matches_normalized_name():
    assert is_task_publish_disabled(DISABLED_NPC)
    assert is_task_publish_disabled("LoChan")
    assert not is_task_publish_disabled(NORMAL_NPC)


async def test_disabled_npc_offers_no_tool_despite_agent_enabled(env):
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯，聊什么都行。")
    orch = _orch(env, "你好", DISABLED_NPC)
    await collect(orch)
    assert orch.task_publish_disabled is True
    assert env.fake.stream_requests[0].tools is None


async def test_disabled_npc_system_prompt_replaces_terminal_guidance(env):
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="我这儿没活。")
    await collect(_orch(env, "给我个任务", DISABLED_NPC))

    system = _system_prompt(env)
    assert NO_TASK_PUBLISH_CONSTRAINT.strip() in system
    for forbidden in ("prepare_task_context", "route_", "路由", "任务发布硬约束",
                      "reward_types", "接取委托的权限", "终端"):
        assert forbidden not in system


async def test_disabled_npc_ignores_model_declared_route(env):
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}',
        text="这事我记下了。",
        tool_calls=[tc("route_task_draft", {"note": "玩家想要个活"})],
    )
    events = await collect(_orch(env, "给我安排个活", DISABLED_NPC))

    assert compressed_kinds(events) == ["meta", "content", "done"]
    assert "".join(contents_of(events)) == "这事我记下了。"
    assert await env.store.get_draft(env.session_id) is None
    assert len(env.fake.stream_requests) == 1
    assert env.fake.chat_requests == []


async def test_disabled_npc_hides_pending_draft_and_lets_it_expire(env):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    for i in range(3):
        env.fake.add_stream(meta='{"emo":"普通","fav":0}', text=f"闲聊{i}。")
        await collect(_orch(env, "聊点别的", DISABLED_NPC))

    shared = env.fake.stream_requests[0].messages[1]["content"]
    assert "食材收集委托" not in shared
    assert env.fake.stream_requests[0].tools is None
    assert await env.store.get_draft(env.session_id) is None


async def test_disabled_npc_confirm_route_does_not_publish(env, monkeypatch):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    monkeypatch.setattr(env.game_data, "data_root", env.tmp_path / "data")

    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}',
        text="行，那就这样。",
        tool_calls=[tc("route_task_confirm", {})],
    )
    events = await collect(_orch(env, "我接了", DISABLED_NPC))

    assert compressed_kinds(events) == ["meta", "content", "done"]
    assert notices_of(events) == []
    assert not (env.tmp_path / "data" / "task" / "agent_tasks.json").exists()
    assert await env.store.get_draft(env.session_id) is not None


async def test_normal_npc_keeps_routes(env):
    """对照组：不在名单的 NPC 不受影响，仍拿到路由表。"""
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "你好", NORMAL_NPC)
    await collect(orch)
    assert orch.task_publish_disabled is False
    names = [t["function"]["name"] for t in (env.fake.stream_requests[0].tools or [])]
    assert names == list(ROUTE_TOOLS_NEW)
