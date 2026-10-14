"""DialogueWorker 后台串行 worker 测试（阶段四 B4-2，§4.0.1 独立链路）。

覆盖：串行执行、失败只记日志不影响后续委派与主链路、running 标志的置位与复位、
**总时限触发时子 Agent 任务被真取消**（不留孤儿）、以及 `_execute` 的真实装配路径
（NPC 资料＋长期记忆→子 Agent 材料块）。

worker 本体行为是被测对象：LLM、落盘、游戏数据一律打桩，不碰真实存档槽位。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import services.dialogue.worker as worker_module
from services.dialogue.worker import (
    DialogueJob,
    DialogueWorker,
    is_dialogue_agent_running,
    set_dialogue_worker,
)
from services.subagents.base import SubagentEvent, SubagentHandle


def _job(name: str) -> DialogueJob:
    return DialogueJob(npc_name=name, llm_config=None)  # type: ignore[arg-type]


@pytest.fixture
def worker():
    w = DialogueWorker()
    set_dialogue_worker(w)
    yield w
    set_dialogue_worker(None)


async def test_worker_runs_jobs_serially(worker):
    """串行执行：三个委派按提交顺序逐个执行，执行区间不重叠。"""
    order: list[str] = []
    active = 0
    overlap = False

    async def fake_execute(job: DialogueJob) -> None:
        nonlocal active, overlap
        if active:
            overlap = True
        active += 1
        order.append(job.npc_name)
        await asyncio.sleep(0.01)
        active -= 1

    worker._execute = fake_execute  # type: ignore[method-assign]
    worker.start()
    for name in ("甲", "乙", "丙"):
        assert worker.submit(_job(name)) is True
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()

    assert order == ["甲", "乙", "丙"]
    assert overlap is False


async def test_worker_failure_does_not_break_queue(worker):
    """失败只记日志：一个委派抛异常，后续委派照常执行，异常不外抛。"""
    calls: list[str] = []

    async def fake_execute(job: DialogueJob) -> None:
        calls.append(job.npc_name)
        if job.npc_name == "炸":
            raise RuntimeError("boom")

    worker._execute = fake_execute  # type: ignore[method-assign]
    worker.start()
    worker.submit(_job("炸"))
    worker.submit(_job("好"))
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()

    assert calls == ["炸", "好"]


async def test_running_flag_during_execution(worker):
    """running 标志：执行期间为 True（注册闸口径）， drained 后回到 False。"""
    assert is_dialogue_agent_running() is False
    gate = asyncio.Event()
    flags_inside: list[bool] = []

    async def fake_execute(job: DialogueJob) -> None:
        flags_inside.append(is_dialogue_agent_running())
        await gate.wait()

    worker._execute = fake_execute  # type: ignore[method-assign]
    worker.start()
    worker.submit(_job("测"))
    # 等 worker 真正拿起这个任务
    for _ in range(200):
        if worker.running:
            break
        await asyncio.sleep(0.005)
    assert worker.running is True
    assert flags_inside == [True]

    gate.set()
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()
    assert is_dialogue_agent_running() is False


async def test_submit_when_queue_full_drops_with_log(worker, caplog):
    """队列满：丢弃本次委派并记日志，不阻塞调用方。"""
    gate = asyncio.Event()

    async def fake_execute(job: DialogueJob) -> None:
        await gate.wait()

    worker._execute = fake_execute  # type: ignore[method-assign]
    worker.start()
    accepted = 0
    for i in range(20):
        if worker.submit(_job(f"丙{i}")):
            accepted += 1
    # 提交循环没有 await 点，worker 来不及消费：队列上限 16，前 16 个入队、其余丢弃
    assert accepted == 16
    assert caplog.text.count("对话修改队列已满") == 4
    gate.set()
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()


# ---------------------------------------------------------------------------
# _execute 真实路径（装配 + 终态 + 超时收柄）
# ---------------------------------------------------------------------------


def _fake_state() -> SimpleNamespace:
    return SimpleNamespace(
        sex="女", faction="铁匠铺", titles=["前治安官"],
        appearance="炉前系着烧出洞的围裙",
        favorability=42, relationship_level="信赖",
    )


def _patch_assembly(monkeypatch) -> None:
    """打桩 _execute 的外部依赖（NPC 状态、库、游戏数据），装配代码本身照跑。"""
    async def _get_npc_manager():
        return SimpleNamespace(get=lambda npc_name: _async_value(_fake_state()))

    async def _load_snapshot(store, *, slot_key, npc):
        return SimpleNamespace(
            is_empty=False,
            base="手艺人，认死理",
            facts=[SimpleNamespace(entry_no=1, content="他替我找回了丢失的龙鳞")],
        )

    monkeypatch.setattr("services.npc.manager.get_npc_manager", _get_npc_manager)
    monkeypatch.setattr("services.memory.store.get_memory_store", lambda: object())
    monkeypatch.setattr(worker_module, "load_snapshot", _load_snapshot)
    monkeypatch.setattr("services.game_data.registry.get_game_data_registry", lambda: "game-data")
    monkeypatch.setattr(worker_module, "LLMClient", SimpleNamespace(for_config=lambda cfg: "llm"))
    monkeypatch.setattr(worker_module, "_tool_registry", lambda: None)


def _async_value(value):
    async def _value():
        return value
    return _value()


async def test_execute_assembles_materials_and_completes(worker, monkeypatch):
    """真实装配路径：NPC 资料与长期记忆拼成材料块交给子 Agent，正常结束不留任务。"""
    launched: list[dict] = []

    class _OkRunner:
        @staticmethod
        def launch(**kwargs):
            launched.append(kwargs)
            events: asyncio.Queue[SubagentEvent] = asyncio.Queue()

            async def _finish():
                events.put_nowait(SubagentEvent(kind="final", payload={"changed": True}))

            return SubagentHandle(
                kind="dialogue", agent="dialogue", coro=_finish(), events=events,
            )

    monkeypatch.setattr(worker_module, "DialogueRunner", _OkRunner)
    _patch_assembly(monkeypatch)

    worker.start()
    assert worker.submit(DialogueJob(
        npc_name="铁匠", llm_config=None, source_trigger="重要任务",  # type: ignore[arg-type]
        source_task_id=12345, reason="他把丢的龙鳞找回来了", instruction="语气更敬重些",
    )) is True
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await worker.stop()

    assert len(launched) == 1
    call = launched[0]
    assert call["npc_name"] == "铁匠"
    assert call["game_data"] == "game-data"
    assert call["source_task_id"] == 12345
    assert (
        "你现在扮演游戏角色「铁匠」（性别：女）（阵营：铁匠铺）（身份或称呼：前治安官）。"
        in call["profile_block"]
    )
    assert "【你的形象】\n炉前系着烧出洞的围裙" in call["profile_block"]
    assert "当前好感度 42 / 100，关系「信赖」。" in call["relationship_block"]
    assert "你对他的基础印象：手艺人，认死理" in call["relationship_block"]
    assert "1. 他替我找回了丢失的龙鳞" in call["relationship_block"]


async def test_total_timeout_cancels_runner_task(worker, monkeypatch, caplog):
    """总时限触发＝连子 Agent 一起收掉：孤儿不得在「已放弃本次」之后继续跑完三轮。

    回归的是超时只取消等待方、不取消子任务的那条缺陷——两个写者并发时后写会抹掉先写，
    且 running 标志提前复位会把注册闸误开。
    """
    outcomes: list[str] = []
    handles: list[SubagentHandle] = []

    class _HangingRunner:
        @staticmethod
        def launch(**_kwargs):
            async def _hang():
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    outcomes.append("cancelled")
                    raise
                outcomes.append("completed")

            handles.append(SubagentHandle(kind="dialogue", agent="dialogue", coro=_hang()))
            return handles[-1]

    monkeypatch.setattr(worker_module, "DIALOGUE_JOB_TIMEOUT_S", 0.2)
    monkeypatch.setattr(worker_module, "DialogueRunner", _HangingRunner)
    _patch_assembly(monkeypatch)

    worker.start()
    assert worker.submit(_job("铁匠")) is True
    await asyncio.wait_for(worker._queue.join(), timeout=5)
    await asyncio.sleep(0.05)  # 让被取消的子 Agent 跑完收尾
    await worker.stop()

    assert outcomes == ["cancelled"]
    assert handles[0].task.cancelled() is True
    assert is_dialogue_agent_running() is False
    assert "超过总上限" in caplog.text
