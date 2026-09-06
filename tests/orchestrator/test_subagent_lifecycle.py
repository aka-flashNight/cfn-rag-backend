"""子 Agent 生命周期与会话并发测试（03 §9.5/§9.6）。"""

from __future__ import annotations

import asyncio

from services.orchestrator.turn import TurnOrchestrator
from services.subagents import count_pending_subagent_tasks, get_session_subagents
from tests.orchestrator.conftest import (
    NPC_NAME,
    collect,
    contents_of,
    phases_of,
    tc,
)


def _orch(env, query: str) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id,
        npc_name=NPC_NAME,
        query=query,
        deps=env.deps,
    )


async def test_stale_subagent_cancelled_by_new_turn(env):
    """回合中玩家新消息到来 → 旧子 Agent 被取消，task 数归零（03 §9.5）。"""
    env.deps.merge_grace_ms = 50
    env.deps.subagent_timeout_s = 0.4

    # 回合 1：task_draft 的第一轮 LLM 调用挂起 5s → 汇合等待超时收场
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":{"kind":"task_draft","direction":"收集"}}',
        text="我想想……",
    )
    env.fake.add_chat(delay=5.0, tool_calls=[tc("prepare_task_context", {
        "task_type": "资源收集",
        "reward_types": {"regular": ["金币"], "optional": []},
    })])
    env.fake.add_stream(text="今天手头的事都派完了，改天吧。")

    events1 = await collect(_orch(env, "给我个活"))
    assert ("task", "failed") in phases_of(events1)
    assert events1[-1].event == "done"
    # 回合 1 已结束，但挂起的 TaskRunner 任务仍在
    assert count_pending_subagent_tasks() == 1

    # 回合 2：新回合启动时取消旧子 Agent
    env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text="好。")
    events2 = await collect(_orch(env, "先不聊任务"))
    assert events2[0].event == "meta"
    await asyncio.sleep(0.1)  # 等取消生效
    assert count_pending_subagent_tasks() == 0
    bucket = await get_session_subagents(env.session_id)
    assert bucket.task is None and bucket.search is None


async def test_session_lock_serializes_concurrent_turns(env):
    """同会话两请求不交叉：第二个回合的全部事件在第一个回合 done 之后（03 §9.6）。"""
    env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text="回答甲。")
    env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text="回答乙。")

    order: list[str] = []

    async def run_one(query: str, tag: str) -> None:
        async for ev in _orch(env, query).run():
            if ev.event == "done":
                order.append(f"{tag}-done")

    await asyncio.gather(run_one("甲问题", "a"), run_one("乙问题", "b"))
    assert order == ["a-done", "b-done"]  # 先启动者先获得会话锁


async def test_npc_favorability_not_lost_under_concurrency(env):
    """并发好感度更新不丢失（修 E1 的回合级验证）。"""
    env.fake.add_stream(meta='{"emo":"微笑","fav":2,"act":null}', text="甲。")
    env.fake.add_stream(meta='{"emo":"微笑","fav":3,"act":null}', text="乙。")

    async def run_one(query: str) -> None:
        async for _ in _orch(env, query).run():
            pass

    await asyncio.gather(run_one("甲"), run_one("乙"))
    state = await env.npc_manager.get(NPC_NAME)
    assert state.favorability == 30 + 2 + 3
