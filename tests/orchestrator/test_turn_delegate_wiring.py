"""阶段四 B4-4 编排层接线用例（触发二、三在 ``turn.py`` 里的那几个调用点）。

``services/dialogue/delegation.py`` 自身的受理规则另有
``tests/dialogue/test_dialogue_delegation.py``；这一层锁的是**聊天轮怎么用它**：
注册闸下发与否、跨档提示只在工具真下发那一轮注入并扣一轮、模型调用的落地与
提示清除、工具没下发却硬发调用时的忽略，以及空正文兜底轮怎么交代已派出的事。

这几条都是时序口径（改一处就可能静默失效），只靠函数级用例锁不住。
记忆库换成 tmp_path 下的临时库、后台 worker 换成假的：不起真子 Agent、不碰真实存档。
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from services.dialogue.delegation import (
    DELEGATE_TOOL_NAME,
    RELATIONSHIP_HINT_ROUNDS,
    TRIGGER_ATTITUDE,
)
from services.dialogue.worker import set_dialogue_worker
from services.integration.save_paths import shared_task_paths
from services.integration.state import get_integration_state
from services.memory.store import DIALOGUE_JOB_DELEGATED, set_memory_store
from services.orchestrator.turn import TurnOrchestrator
from tests.orchestrator.conftest import NPC_NAME, collect, tc

SLOT = "slot_turn_delegation"
HINT_TEXT = (
    "你的关系从 熟悉 变为了 朋友，"
    "可以考虑调用对话修改的子 Agent，调整你的对话措辞。"
)


class _FakeWorker:
    """假后台：只记录被受理的委派，绝不真跑子 Agent。"""

    def __init__(self) -> None:
        self.jobs: list = []
        self.running = False

    def submit(self, job) -> bool:
        self.jobs.append(job)
        return True


@pytest.fixture
def fake_worker():
    worker = _FakeWorker()
    set_dialogue_worker(worker)
    yield worker
    set_dialogue_worker(None)


@pytest.fixture
def env_store(env):
    """把编排层的临时库同时装进全局单例（delegation 侧按单例取库）。"""
    set_memory_store(env.store)
    yield env.store
    set_memory_store(None)


@pytest.fixture
def bound_slot():
    state = get_integration_state()

    def _bind(progress: dict | None = None) -> str:
        state.bind(SLOT, progress=progress or {"tasks_finished": []})
        return SLOT

    yield _bind
    state.unbind()


@pytest.fixture
def task_data(env, monkeypatch):
    """把任务文件指到 tmp 数据根，并写一条该 NPC 发布的委托（标记由参数决定）。"""
    data_root = env.tmp_path / "data"
    monkeypatch.setattr(env.game_data, "data_root", data_root)

    def _write(task_id: int = 200001, *, marked: bool = False) -> None:
        record = {"id": task_id, "title": f"$AGENT_TITLE_{task_id}", "get_npc": NPC_NAME}
        if marked:
            record["dialogue_rewrite"] = True
        tasks_path, text_path = shared_task_paths(data_root)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.write_text(
            json.dumps({"tasks": [record]}, ensure_ascii=False), encoding="utf-8",
        )
        text_path.write_text(
            json.dumps({f"$AGENT_TITLE_{task_id}": "取回祖传铁锤"}, ensure_ascii=False),
            encoding="utf-8",
        )

    return _write


def _orch(env, query: str, **kwargs) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id, npc_name=NPC_NAME, query=query, deps=env.deps, **kwargs,
    )


def _names(request) -> list[str]:
    return [t["function"]["name"] for t in (request.tools or [])]


def _delegate_tool(request) -> dict | None:
    return next((t for t in (request.tools or []) if t["function"]["name"] == DELEGATE_TOOL_NAME), None)


def _shared_core(env, index: int = 0) -> str:
    return env.fake.stream_requests[index].messages[1]["content"]


def _job_rows(env) -> list[dict]:
    with sqlite3.connect(str(env.tmp_path / "memory.db")) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM dialogue_jobs")]


def _delegate_call(reason: str = "他当着我的面把话挑明了") -> dict:
    return tc(DELEGATE_TOOL_NAME, {
        "change_type": TRIGGER_ATTITUDE, "reason": reason, "instruction": "往后说话冲一点",
    })


# ---------------------------------------------------------------------------
# 注册闸：下发与否，以及「不注册就一个字都不提」
# ---------------------------------------------------------------------------

async def test_tool_absent_when_unbound_and_hint_not_burned(env, env_store, fake_worker, bound_slot):
    """未绑定存档＝工具不下发，跨档提示也不注入、**不扣轮**（三轮不能白烧在模型没有工具的那几轮上）。"""
    await env.store.dialogue_hint_write(env.session_id, HINT_TEXT, RELATIONSHIP_HINT_ROUNDS)

    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, "你好"))
    assert DELEGATE_TOOL_NAME not in _names(env.fake.stream_requests[0])
    assert "对话修改的子 Agent" not in _shared_core(env)
    read = await env.store.dialogue_hint_read(env.session_id)
    assert read is not None and read.rounds_left == RELATIONSHIP_HINT_ROUNDS   # 没扣

    bound_slot()
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯，说吧。")
    await collect(_orch(env, "我又来了"))
    assert DELEGATE_TOOL_NAME in _names(env.fake.stream_requests[1])
    # 下发这一轮才注入并扣一轮
    assert HINT_TEXT in _shared_core(env, index=1)
    assert (await env.store.dialogue_hint_read(env.session_id)).rounds_left == RELATIONSHIP_HINT_ROUNDS - 1


async def test_tool_absent_while_subagent_is_running(env, env_store, fake_worker, bound_slot):
    """后台正在写覆盖层＝本轮完全不注册（避免两个写者同时改同一份覆盖层）。"""
    bound_slot()
    fake_worker.running = True
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, "你好"))

    assert _delegate_tool(env.fake.stream_requests[0]) is None
    assert "对话修改的子 Agent" not in _shared_core(env)


async def test_task_fallback_shape_follows_finished_block(
    env, env_store, fake_worker, bound_slot, task_data,
):
    """条件闸：当轮拼了**未挂标记**的已完成委托才有 task_id 与任务兜底表述；都带标记就一个字不给。"""
    task_data(marked=False)
    bound_slot(progress={"tasks_finished": [200001]})
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, "那委托我完成了"))

    tool = _delegate_tool(env.fake.stream_requests[0])
    assert tool is not None and "task_id" in tool["function"]["parameters"]["properties"]
    assert "重要任务" in json.dumps(tool, ensure_ascii=False)

    task_data(marked=True)
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, "再来一轮"))

    without = _delegate_tool(env.fake.stream_requests[1])
    assert without is not None
    assert "task_id" not in without["function"]["parameters"]["properties"]
    assert "重要任务" not in json.dumps(without, ensure_ascii=False)
    assert "已完成委托块" not in json.dumps(without, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 触发二：跨档 → 提示 → 三轮用尽
# ---------------------------------------------------------------------------

async def test_relationship_hint_lifecycle_across_rounds(env, env_store, fake_worker, bound_slot):
    """本轮跨档只落库（提示作用于下一轮），随后连续注入三轮，用尽后不再打扰。"""
    bound_slot()
    # meta 的 fav 单轮被 clamp 到 ±5（`services/llm/meta.py`），先把好感推到跨档门口：
    # 铁匠起始 30 → 45 仍落在「熟悉」（<50）区间内。
    pre = await env.npc_manager.update_favorability(NPC_NAME, 15)
    assert (pre.favorability, pre.relationship_level) == (45, "熟悉")

    env.fake.add_stream(meta='{"emo":"微笑","fav":5}', text="你这回倒是上心。")   # 45→50 → 朋友
    await collect(_orch(env, "我把锤子找回来了"))

    read = await env.store.dialogue_hint_read(env.session_id)
    assert read is not None and read.rounds_left == RELATIONSHIP_HINT_ROUNDS
    assert HINT_TEXT in read.hint
    # 跨档那一轮不注入（本轮工具表与请求都已发出）
    assert HINT_TEXT not in _shared_core(env)

    for index in range(1, RELATIONSHIP_HINT_ROUNDS + 1):
        env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
        await collect(_orch(env, f"聊第{index}句"))
        assert HINT_TEXT in _shared_core(env, index=index)
        left = await env.store.dialogue_hint_read(env.session_id)
        expected = RELATIONSHIP_HINT_ROUNDS - index
        if expected > 0:
            assert left.rounds_left == expected
        else:
            assert left is None

    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, "聊第五句"))
    # 上面共 1（跨档）+ RELATIONSHIP_HINT_ROUNDS（注入轮）次调用，本轮排在它们之后
    assert len(env.fake.stream_requests) == RELATIONSHIP_HINT_ROUNDS + 2
    assert HINT_TEXT not in _shared_core(env, index=-1)


async def test_no_hint_when_level_unchanged(env, env_store, fake_worker, bound_slot):
    """没跨档就落一条提示都没有（+5 仍在「熟悉」区间）。"""
    bound_slot()
    env.fake.add_stream(meta='{"emo":"微笑","fav":5}', text="行。")
    await collect(_orch(env, "顺手帮了点忙"))

    assert await env.store.dialogue_hint_read(env.session_id) is None


# ---------------------------------------------------------------------------
# 触发三：模型调用落地 / 越权忽略 / 兜底轮交代
# ---------------------------------------------------------------------------

async def test_delegate_call_lands_and_clears_hint(env, env_store, fake_worker, bound_slot):
    """聊天轮里的委派调用就地受理（入队后台），并把跨档提示撤掉——一旦调用立刻不再提示。"""
    bound_slot()
    await env.store.dialogue_hint_write(env.session_id, HINT_TEXT, RELATIONSHIP_HINT_ROUNDS)

    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="成，往后我照着这个劲儿说。",
        tool_calls=[_delegate_call()],
    )
    await collect(_orch(env, "你现在说话比以前随便多了"))

    assert len(fake_worker.jobs) == 1
    job = fake_worker.jobs[0]
    assert (job.npc_name, job.slot_key, job.source_trigger) == (NPC_NAME, SLOT, TRIGGER_ATTITUDE)
    assert job.reason == "他当着我的面把话挑明了" and job.instruction == "往后说话冲一点"

    rows = _job_rows(env)
    assert len(rows) == 1 and rows[0]["status"] == DIALOGUE_JOB_DELEGATED
    assert rows[0]["session_id"] == env.session_id
    assert await env.store.dialogue_hint_read(env.session_id) is None


async def test_spoken_round_with_call_gets_no_rescue(env, env_store, fake_worker, bound_slot):
    """又说话又调委派：不触发兜底轮，本轮只有一次调用。"""
    bound_slot()
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="知道了。", tool_calls=[_delegate_call()],
    )
    await collect(_orch(env, "把那事放心上"))

    assert len(env.fake.stream_requests) == 1
    assert len(fake_worker.jobs) == 1


async def test_stray_delegate_call_is_ignored_when_tool_not_registered(
    env, env_store, fake_worker,
):
    """工具没下发（未绑定存档）模型却硬发调用：一律不受理，也不留库里的痕迹。"""
    get_integration_state().unbind()
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="嗯。", tool_calls=[_delegate_call()],
    )
    await collect(_orch(env, "你去改改你以后说的话"))

    assert fake_worker.jobs == []
    assert _job_rows(env) == []


async def test_rescue_round_reports_delegation_without_tools(env, env_store, fake_worker, bound_slot):
    """只调委派、一个字没说：兜底轮**不带任何工具**，并用文字交代已派出什么。"""
    bound_slot()
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}', text="", tool_calls=[_delegate_call()],
    )
    env.fake.add_stream(text="行，我往后就这么说。")
    events = await collect(_orch(env, "你以后说话随意点"))

    rescue = env.fake.stream_requests[1]
    assert rescue.tools is None
    assert "【已派出】态度变化：已受理" in rescue.messages[-1]["content"]
    assert "".join(
        e.data.get("delta", "") for e in events if e.event == "content"
    ).endswith("行，我往后就这么说。")
    assert len(fake_worker.jobs) == 1
