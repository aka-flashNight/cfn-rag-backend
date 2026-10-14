"""DialogueWorker 后台串行 worker 测试（阶段四 B4-2，§4.0.1 独立链路）。

覆盖：串行执行、失败只记日志不影响后续委派与主链路、running 标志的置位与复位。
_execute 以实例属性打桩（不触碰真实 LLM/落盘），worker 本体行为为被测对象。
"""

from __future__ import annotations

import asyncio

import pytest

from services.dialogue.worker import (
    DialogueJob,
    DialogueWorker,
    is_dialogue_agent_running,
    set_dialogue_worker,
)


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
