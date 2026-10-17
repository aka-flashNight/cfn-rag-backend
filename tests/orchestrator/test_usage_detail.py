"""usage 逐次明细的口径：求和值不变，另出一份逐次 calls。

为什么要逐次：一轮里常有多次 LLM 调用（角色轮、任务汇合轮、确认轮），
只有 cached_tokens 的总量时看不出到底哪一次命中了前缀缓存——
比如三次调用 2560/0/0 与 0/0/7680 的总量都是 7680，命中率却完全不同。

后台子 Agent 的每次调用也在内：它每轮的 usage 随终态事件带上来，由汇合层按时序
并进本回合的 calls（旧口径下子 Agent 整段用量被丢弃，实测一轮检索少计 25.5%）。
"""

from services.orchestrator.events import accumulate_usage
from services.orchestrator.turn import TurnOrchestrator
from tests.orchestrator.conftest import NPC_NAME, collect, tc


def test_each_call_gets_its_own_row_and_sums_stay_untouched():
    total = accumulate_usage(None, {
        "prompt_tokens": 2000, "completion_tokens": 100,
        "total_tokens": 2100, "cached_tokens": 1984,
    })
    total = accumulate_usage(total, {"prompt_tokens": 2500, "completion_tokens": 80, "cached_tokens": 0})
    total = accumulate_usage(total, {"prompt_tokens": 2600, "completion_tokens": 90, "cached_tokens": 512})

    assert total["prompt_tokens"] == 7100
    assert total["completion_tokens"] == 270
    assert total["total_tokens"] == 7370
    assert total["cached_tokens"] == 2496
    # 三次调用三行，各自能单独读出缓存命中量
    assert [c["cached_tokens"] for c in total["calls"]] == [1984, 0, 512]
    assert [c["prompt_tokens"] for c in total["calls"]] == [2000, 2500, 2600]


def test_reported_keys_only():
    """端点没给 cached_tokens 时，该行不凭空补 0——补了就像「确认未命中」而非「未上报」。"""
    total = accumulate_usage(None, {"prompt_tokens": 10, "completion_tokens": 5})
    assert total["calls"] == [{"prompt_tokens": 10, "completion_tokens": 5}]
    assert "cached_tokens" not in total


def test_subtotal_calls_are_flattened_not_nested():
    """子流程合计进来时展开其明细，不把整块合计再记成一行。"""
    sub = accumulate_usage(None, {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 64})
    sub = accumulate_usage(sub, {"prompt_tokens": 200, "completion_tokens": 20, "cached_tokens": 0})

    total = accumulate_usage(
        {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 512,
         "calls": [{"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 512}]},
        sub,
    )
    assert total["prompt_tokens"] == 1300
    assert total["completion_tokens"] == 80
    assert total["cached_tokens"] == 576
    assert total["calls"] == [
        {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 512},
        {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 64},
        {"prompt_tokens": 200, "completion_tokens": 20, "cached_tokens": 0},
    ]


def test_empty_incoming_keeps_total_as_is():
    assert accumulate_usage({"prompt_tokens": 1, "calls": [{"prompt_tokens": 1}]}, None) == {
        "prompt_tokens": 1, "calls": [{"prompt_tokens": 1}]
    }


# ---------------------------------------------------------------------------
# 子 Agent 的逐次 usage 进回合（真机走查取证：一段检索的 prompt 曾被整段丢弃）
# ---------------------------------------------------------------------------

async def test_subagent_usage_rows_reach_turn_detail(env):
    """路由轮 → 子 Agent 两轮 → 汇合轮：done 的 calls 应有四行，按调用时序排列。"""
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="",
        tool_calls=[tc("route_search", {"query": "安迪的过去"})],
        usage={"prompt_tokens": 6561, "completion_tokens": 44, "cached_tokens": 3136},
    )
    env.fake.add_chat(
        tool_calls=[tc("search_knowledge", {"keyword": "安迪 过去"})],
        usage={"prompt_tokens": 748, "completion_tokens": 34},
    )
    env.fake.add_chat(
        content="安迪曾在废城担任佣兵，后来加入A兵团。",
        usage={"prompt_tokens": 3456, "completion_tokens": 355, "cached_tokens": 704},
    )
    env.fake.add_stream(
        text="安迪以前是A兵团的佣兵，这事我知道一些。",
        usage={"prompt_tokens": 5699, "completion_tokens": 262},
    )

    events = await collect(TurnOrchestrator(
        session_id=env.session_id, npc_name=NPC_NAME, query="你知道安迪以前的事吗？", deps=env.deps,
    ))
    usage = events[-1].data["usage"]

    assert [c["prompt_tokens"] for c in usage["calls"]] == [6561, 748, 3456, 5699]
    assert [c["completion_tokens"] for c in usage["calls"]] == [44, 34, 355, 262]
    assert usage["prompt_tokens"] == 16464
    assert usage["completion_tokens"] == 695
    assert usage["cached_tokens"] == 3840

    # 计量行不进提示词：汇合轮只该看到检索结论，看不到 usage_rows
    merge_prompt = "\n".join(
        str(m.get("content", "")) for m in env.fake.stream_requests[-1].messages
    )
    assert "usage_rows" not in merge_prompt


async def test_failed_subagent_usage_rows_still_counted(env):
    """子 Agent 中途失败：已经发生的调用照样进明细（失败前那一轮也烧了 token）。"""
    env.deps.merge_grace_ms = 50
    env.deps.subagent_timeout_s = 2
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="",
        tool_calls=[tc("route_search", {"query": "安迪的过去"})],
        usage={"prompt_tokens": 100, "completion_tokens": 10},
    )
    env.fake.add_chat(
        tool_calls=[tc("search_knowledge", {"keyword": "安迪 过去"})],
        usage={"prompt_tokens": 748, "completion_tokens": 34},
    )
    env.fake.add_stream(text="这事我还没打听过。", usage={"prompt_tokens": 200, "completion_tokens": 20})

    real_chat = env.fake.chat
    rounds = {"n": 0}

    async def first_ok_then_boom(req):
        rounds["n"] += 1
        if rounds["n"] == 1:
            return await real_chat(req)
        raise RuntimeError("model exploded")

    env.fake.chat = first_ok_then_boom

    events = await collect(TurnOrchestrator(
        session_id=env.session_id, npc_name=NPC_NAME, query="你知道安迪以前的事吗？", deps=env.deps,
    ))
    calls = events[-1].data["usage"]["calls"]

    assert rounds["n"] == 2
    assert [c["prompt_tokens"] for c in calls] == [100, 748, 200]
    assert any(e.event == "agent_status" and e.data.get("phase") == "failed" for e in events)
