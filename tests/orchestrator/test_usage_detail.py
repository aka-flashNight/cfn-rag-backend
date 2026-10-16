"""usage 逐次明细的口径：求和值不变，另出一份逐次 calls。

为什么要逐次：一轮里常有多次 LLM 调用（角色轮、任务汇合轮、确认轮），
只有 cached_tokens 的总量时看不出到底哪一次命中了前缀缓存——
比如三次调用 2560/0/0 与 0/0/7680 的总量都是 7680，命中率却完全不同。
"""

from services.orchestrator.events import accumulate_usage


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
