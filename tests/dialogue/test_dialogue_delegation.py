"""阶段四 B4-4 触发批正式用例（幂等 / dismissed / 频率限制 / 注册闸 / 委派材料）。

委派受理（``services/dialogue/delegation.py``）与 ``dialogue_jobs`` 状态机此前零专门
用例。数据层改的是覆盖层，这一层决定的却是「一次触发到底会不会变成一次子 Agent 执行」
——成本、重复改写、串档都在这里挡住，所以跟数据层同批补齐。

后台 worker 一律换成假的：委派是 fire-and-forget，本文件锁的是受理侧，
真子 Agent 的流程与状态回写另有 ``test_dialogue_runner.py`` / ``test_dialogue_worker.py``。
用例只碰 tmp_path 下的库与假数据根，不触碰真实存档。
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest
from services.agent_tools.finished_tasks import MARKED_TASK_HINT
from services.dialogue import delegation
from services.dialogue.delegation import (
    DELEGATE_TOOL_NAME,
    DIALOGUE_DELEGATE_COOLDOWN_S,
    DIALOGUE_DELEGATE_PER_SESSION,
    RELATIONSHIP_HINT_ROUNDS,
    TRIGGER_ATTITUDE,
    TRIGGER_IMPORTANT_TASK,
    TRIGGER_KEY_INFO,
    TRIGGER_RELATIONSHIP,
    apply_delegate_calls,
    build_delegate_tool,
    build_relationship_hint,
    delegate_tool_or_none,
    handle_task_completed,
    submit_delegation,
)
from services.dialogue.worker import set_dialogue_worker
from services.integration.save_paths import agent_save_task_paths
from services.integration.state import get_integration_state
from services.memory.store import (
    DIALOGUE_JOB_DELEGATED,
    DIALOGUE_JOB_DISMISSED,
    DIALOGUE_JOB_FAILED,
    MemoryStore,
    set_memory_store,
)
from tests.dialogue.conftest import NPC, SLOT

TASK_ID = 200001
OTHER_SLOT = "slot_dialogue_other"


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class _FakeWorker:
    """记录被受理的委派，不执行任何东西（队列满的场景由测试翻 accept）。"""

    def __init__(self) -> None:
        self.jobs: list = []
        self.running = False
        self.accept = True

    def submit(self, job) -> bool:
        if not self.accept:
            return False
        self.jobs.append(job)
        return True


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "memory.db"


@pytest.fixture
def store(db_path):
    """把用例的临时库装进全局单例（delegation 与 worker 都按单例取库）。"""
    instance = MemoryStore(db_path=db_path)
    set_memory_store(instance)
    yield instance
    instance.close()
    set_memory_store(None)


@pytest.fixture
def fake_worker():
    worker = _FakeWorker()
    set_dialogue_worker(worker)
    yield worker
    set_dialogue_worker(None)


def _rows(db_path) -> list[dict]:
    """直读 dialogue_jobs（公开接口只给按任务/按作用域的计数，这里要看整表）。"""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM dialogue_jobs ORDER BY created_at")]


def _row_by_job(db_path, job_id: str) -> dict:
    matches = [row for row in _rows(db_path) if row["job_id"] == job_id]
    assert len(matches) == 1, f"job_id={job_id} 应有且只有一条记录，实得 {len(matches)}"
    return matches[0]


def _backdate_delegated(db_path, job_id: str, ts: float) -> None:
    """把 delegated_at 推到冷却窗之外（不靠放宽常量或睡眠来验时间逻辑）。"""
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE dialogue_jobs SET delegated_at = ? WHERE job_id = ?", (ts, job_id))
        conn.commit()


@pytest.fixture
def task_root(tmp_path, monkeypatch):
    """一条挂了对话重写标记的 Agent 委托（触发一的材料来源），外加未标记的那条。"""
    data_root = tmp_path / "data"
    tasks_path, text_path = agent_save_task_paths(data_root, SLOT)
    _write(tasks_path, {"tasks": [
        {
            "id": TASK_ID,
            "title": f"$AGENT_TITLE_{TASK_ID}",
            "chain": "委托",
            "get_npc": NPC,
            "finish_npc": NPC,
            "get_conversation": f"$AGENT_GET_{TASK_ID}",
            "finish_conversation": f"$AGENT_FINISH_{TASK_ID}",
            "rewards": ["精铁×3", "800 金币"],
            "finish_requirements": ["废矿坑#普通"],
            "dialogue_rewrite": True,
            "dialogue_rewrite_instruction": "语气更敬重些",
            "publish_session_id": "sess_publish",
        },
        {   # 发布时漏挂标记的委托（触发一该把它挡下来）
            "id": TASK_ID + 1,
            "title": f"$AGENT_TITLE_{TASK_ID + 1}",
            "chain": "委托",
            "get_npc": NPC,
            "finish_npc": NPC,
        },
    ]})
    _write(text_path, {
        f"$AGENT_TITLE_{TASK_ID}": "取回祖传铁锤",
        f"$AGENT_DESCRIPTION_{TASK_ID}": "锤子在废矿坑，路上当心。",
        f"$AGENT_GET_{TASK_ID}": [
            {"name": "$PC", "text": "锤子我给你找回来。"},
            {"name": NPC, "text": "好，我在炉子边等你。"},
        ],
        f"$AGENT_FINISH_{TASK_ID}": [
            {"name": NPC, "text": "锤子回来了，这份情我记着。"},
        ],
        f"$AGENT_TITLE_{TASK_ID + 1}": "没挂标记的委托",
    })
    monkeypatch.setattr(delegation, "_data_root", lambda: data_root)
    return data_root


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------

async def test_task_completion_delegates_only_once(task_root, db_path, store, fake_worker, bound_slot):
    """同存档同任务只委派一次：第二次完成事件静默跳过，不留下第二个子 Agent。"""
    bound_slot()
    first = await handle_task_completed(slot_key=SLOT, task_id=TASK_ID)
    assert first.accepted is True and first.job_id
    assert len(fake_worker.jobs) == 1

    row = await store.dialogue_job_find_by_task(SLOT, TASK_ID)
    assert (row.job_id, row.status, row.trigger) == (first.job_id, DIALOGUE_JOB_DELEGATED, TRIGGER_IMPORTANT_TASK)
    assert row.npc == NPC                                              # 权威 NPC 取自落盘记录的 get_npc
    assert row.task_id == TASK_ID and row.session_id is None           # HTTP 事件没有聊天会话

    second = await handle_task_completed(slot_key=SLOT, task_id=TASK_ID)
    assert second.accepted is False and "已委派过，静默跳过" in second.detail
    assert len(fake_worker.jobs) == 1                                   # 没有重复执行
    assert len(_rows(db_path)) == 1                                     # 也没有第二条记录


async def test_idempotency_index_is_scoped_per_slot_and_ignores_null_task(db_path, store):
    """硬幂等的键＝(slot_key, task_id)：换存档不算重复，无任务的触发不受约束。"""
    first = await store.dialogue_job_create(slot_key="slotA", npc=NPC, trigger=TRIGGER_KEY_INFO, task_id=TASK_ID)
    same_slot = await store.dialogue_job_create(slot_key="slotA", npc=NPC, trigger=TRIGGER_KEY_INFO, task_id=TASK_ID)
    other_slot = await store.dialogue_job_create(slot_key="slotB", npc=NPC, trigger=TRIGGER_KEY_INFO, task_id=TASK_ID)

    assert first is not None and same_slot is None and other_slot is not None

    a1 = await store.dialogue_job_create(slot_key="slotA", npc=NPC, trigger=TRIGGER_RELATIONSHIP)
    a2 = await store.dialogue_job_create(slot_key="slotA", npc=NPC, trigger=TRIGGER_RELATIONSHIP)
    assert a1 is not None and a2 is not None                            # task_id 为 NULL 不进该索引
    assert await store.dialogue_job_find_by_task("slotA", 999999) is None
    assert len(_rows(db_path)) == 4


async def test_dismissed_row_occupies_the_task_index(task_root, db_path, store, fake_worker, bound_slot):
    """当前口径：为同一任务留痕的 dismissed 记录同样占住唯一索引，此后该任务在本存档
    内不再有任何委派机会——**包括聊天侧「发布时漏标记」的兜底调用**。

    这条与「任务只兜发布时漏标记的遗漏」的口径存在张力（兜底恰好要用于漏标记的任务）。
    本用例只如实钉住现状，改不改属机制取舍，已登记待裁定。
    """
    bound_slot()
    unmarked = await handle_task_completed(slot_key=SLOT, task_id=TASK_ID + 1, npc=NPC)
    assert unmarked.accepted is False and "没挂对话重写标记" in unmarked.detail
    assert (await store.dialogue_job_find_by_task(SLOT, TASK_ID + 1)).status == DIALOGUE_JOB_DISMISSED

    fallback = await submit_delegation(
        npc_name=NPC, trigger=TRIGGER_IMPORTANT_TASK,
        reason="玩家说他把那委托做完了", task_id=TASK_ID + 1, session_id="sess_chat",
    )
    assert fallback.accepted is False and "已委派过，静默跳过" in fallback.detail
    assert fake_worker.jobs == []
    assert len(_rows(db_path)) == 1                                     # 兜底那次连留痕都写不进去


# ---------------------------------------------------------------------------
# HTTP 入口（游戏侧唯一能碰到的那一面）
# ---------------------------------------------------------------------------

def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.integration_api import router

    app = FastAPI()
    app.include_router(router, prefix="/integration")
    return TestClient(app)


def test_task_completed_route_passthrough_and_repeat_push(task_root, db_path, store, fake_worker, bound_slot):
    """``POST /integration/task-completed`` 只做转发：字段名口径、不受理也是 200、
    重复推送同样 200＋accepted=False（游戏侧只管推，不需要区分原因）。"""
    bound_slot()
    client = _client()

    first = client.post("/integration/task-completed", json={"slot_key": SLOT, "task_id": TASK_ID})
    assert first.status_code == 200
    body = first.json()
    assert (body["ok"], body["accepted"]) == (True, True)
    assert body["job_id"] and body["detail"] == "已受理"
    assert len(fake_worker.jobs) == 1

    again = client.post("/integration/task-completed", json={"slot_key": SLOT, "task_id": TASK_ID})
    assert again.status_code == 200
    assert again.json()["accepted"] is False and "已委派过，静默跳过" in again.json()["detail"]
    assert len(fake_worker.jobs) == 1


def test_task_completed_route_rejects_malformed_payload(task_root, db_path, store, fake_worker, bound_slot):
    """请求体校验在路由层就挡住：任务 id 必须为正、槽位键不能带路径分隔符。"""
    bound_slot()
    client = _client()

    assert client.post("/integration/task-completed", json={"slot_key": SLOT, "task_id": 0}).status_code == 422
    assert client.post(
        "/integration/task-completed", json={"slot_key": "../evil", "task_id": TASK_ID},
    ).status_code == 422
    assert client.post("/integration/task-completed", json={"task_id": TASK_ID}).status_code == 422
    assert fake_worker.jobs == [] and _rows(db_path) == []


# ---------------------------------------------------------------------------
# dismissed / 早退路径
# ---------------------------------------------------------------------------

async def test_unmarked_task_is_dismissed(task_root, db_path, store, fake_worker, bound_slot):
    """不够格（发布时没挂标记）→ 自动委派不受理，但落一条 dismissed 供事后核查。"""
    bound_slot()
    outcome = await handle_task_completed(slot_key=SLOT, task_id=TASK_ID + 1, npc=NPC)

    assert outcome.accepted is False and "没挂对话重写标记" in outcome.detail
    # 留痕行是补写的，不回传 job_id（调用方拿不到它，只能按任务查）
    assert outcome.job_id == ""
    assert fake_worker.jobs == []
    row = await store.dialogue_job_find_by_task(SLOT, TASK_ID + 1)
    assert row.status == DIALOGUE_JOB_DISMISSED and row.task_id == TASK_ID + 1
    assert row.npc == NPC and row.trigger == TRIGGER_IMPORTANT_TASK
    assert row.delegated_at is None                                     # 没真委派就不占用冷却
    assert row.detail == outcome.detail


async def test_event_from_another_slot_is_dropped_before_recording(task_root, db_path, store, fake_worker, bound_slot):
    """非绑定槽位的事件（换档后旧存档才推来）整条忽略：既不执行也不落库。"""
    bound_slot()
    outcome = await handle_task_completed(slot_key=OTHER_SLOT, task_id=TASK_ID)

    assert outcome.accepted is False and "槽位" in outcome.detail
    assert outcome.job_id == "" and fake_worker.jobs == []
    assert _rows(db_path) == []
    assert await store.dialogue_job_find_by_task(OTHER_SLOT, TASK_ID) is None


async def test_missing_task_record_is_dismissed(task_root, db_path, store, fake_worker, bound_slot):
    """任务记录读不到（已被删/不属本存档）→ dismissed 留痕，不凭空起子 Agent。"""
    bound_slot()
    outcome = await submit_delegation(
        npc_name=NPC, trigger=TRIGGER_IMPORTANT_TASK, reason="完成了", task_id=424242,
    )
    assert outcome.accepted is False and "读不到" in outcome.detail
    assert fake_worker.jobs == []
    assert (await store.dialogue_job_find_by_task(SLOT, 424242)).status == DIALOGUE_JOB_DISMISSED


async def test_initial_checks_without_any_record(db_path, store, fake_worker, bound_slot):
    """初判本身就不成立的两种早退：未知更改类型、重要任务没给 id——连记录都不建，不留无主痕迹。"""
    bound_slot()
    unknown = await submit_delegation(npc_name=NPC, trigger="随便写的类型", reason="发生了事")
    no_id = await submit_delegation(npc_name=NPC, trigger=TRIGGER_IMPORTANT_TASK, reason="完成了")

    assert "未知的更改类型" in unknown.detail
    assert "没给任务 id" in no_id.detail
    assert (unknown.job_id, no_id.job_id) == ("", "")
    assert fake_worker.jobs == []
    assert _rows(db_path) == []


async def test_missing_reason_is_dismissed(db_path, store, fake_worker, bound_slot):
    """触发二、三没写发生了什么 → 判断不了关键性，落 dismissed（task_id 为空，不占幂等索引）。"""
    bound_slot()
    no_reason = await submit_delegation(npc_name=NPC, trigger=TRIGGER_ATTITUDE)

    assert no_reason.accepted is False and "没写发生了什么" in no_reason.detail
    assert no_reason.job_id == ""
    rows = _rows(db_path)
    assert len(rows) == 1
    assert (rows[0]["status"], rows[0]["trigger"], rows[0]["task_id"]) == (
        DIALOGUE_JOB_DISMISSED, TRIGGER_ATTITUDE, None,
    )
    assert fake_worker.jobs == []


async def test_missing_npc_or_slot_leaves_no_record(db_path, store, fake_worker, bound_slot):
    """缺 NPC 名 / 未绑定存档：覆盖层无处落盘，写侧 fail-closed，受理侧也不留痕
    （留痕行按 (存档, NPC) 才查得到，这两样都缺时写了也没人看得见）。"""
    bound_slot()
    no_npc = await submit_delegation(npc_name="  ", trigger=TRIGGER_KEY_INFO, reason="有事发生")
    assert no_npc.accepted is False and "缺少 NPC 名" in no_npc.detail
    assert len(_rows(db_path)) == 0

    get_integration_state().unbind()
    unbound = await submit_delegation(npc_name=NPC, trigger=TRIGGER_KEY_INFO, reason="换了档之后发生的事")
    assert unbound.accepted is False and "无处落盘" in unbound.detail
    assert unbound.job_id == ""
    assert len(_rows(db_path)) == 0
    assert fake_worker.jobs == []


async def test_queue_full_marks_record_failed(task_root, db_path, store, fake_worker, bound_slot):
    """后台队列已满：记录推到 failed 并写明原因，不把「已受理」骗给调用方。"""
    bound_slot()
    fake_worker.accept = False
    outcome = await submit_delegation(npc_name=NPC, trigger=TRIGGER_KEY_INFO, reason="有要紧事")

    assert outcome.accepted is False and "后台队列已满" in outcome.detail
    row = _row_by_job(db_path, outcome.job_id)
    assert row["status"] == DIALOGUE_JOB_FAILED and "队列已满" in row["detail"]
    assert fake_worker.jobs == []


# ---------------------------------------------------------------------------
# 频率限制
# ---------------------------------------------------------------------------

async def test_cooldown_blocks_second_delegation_for_same_npc(db_path, store, fake_worker, bound_slot):
    """冷却窗内同存档同 NPC 只放行一次；被挡的那次落 dismissed 且不占用冷却计时。"""
    bound_slot()
    first = await submit_delegation(npc_name=NPC, trigger=TRIGGER_KEY_INFO, reason="他知道了我的旧事")
    assert first.accepted is True

    second = await submit_delegation(npc_name=NPC, trigger=TRIGGER_ATTITUDE, reason="态度也跟着变了")
    assert second.accepted is False and "冷却窗内" in second.detail
    assert len(fake_worker.jobs) == 1
    blocked = _row_by_job(db_path, second.job_id)
    assert blocked["status"] == DIALOGUE_JOB_DISMISSED and blocked["delegated_at"] is None
    assert await store.dialogue_job_delegated_since(SLOT, NPC, time.time() - 600) == 1


async def test_cooldown_is_per_npc_and_expires(db_path, store, fake_worker, bound_slot):
    """冷却按 (存档, NPC) 计算，窗过后同一 NPC 可以再次委派。"""
    bound_slot()
    first = await submit_delegation(npc_name=NPC, trigger=TRIGGER_KEY_INFO, reason="他知道了我的旧事")
    other_npc = await submit_delegation(npc_name="另一个NPC", trigger=TRIGGER_KEY_INFO, reason="同样的事")
    assert first.accepted is True and other_npc.accepted is True        # 不同 NPC 互不挡

    _backdate_delegated(db_path, first.job_id, time.time() - DIALOGUE_DELEGATE_COOLDOWN_S - 1)
    again = await submit_delegation(npc_name=NPC, trigger=TRIGGER_ATTITUDE, reason="又发生了一件事")
    assert again.accepted is True
    assert len(fake_worker.jobs) == 3


async def test_per_session_cap(db_path, store, fake_worker, bound_slot):
    """同一聊天会话累计委碰到顶就挡；换会话与 HTTP 事件（无会话）不受该上限约束。"""
    bound_slot()
    for index in range(DIALOGUE_DELEGATE_PER_SESSION):
        outcome = await submit_delegation(
            npc_name=f"路人{index}", trigger=TRIGGER_KEY_INFO,
            reason=f"第{index}起事", session_id="sess_cap",
        )
        assert outcome.accepted is True
    assert await store.dialogue_job_session_delegated("sess_cap") == DIALOGUE_DELEGATE_PER_SESSION

    over = await submit_delegation(npc_name="路人X", trigger=TRIGGER_KEY_INFO, reason="再一起", session_id="sess_cap")
    assert over.accepted is False and "本会话委派已达上限" in over.detail
    assert len(fake_worker.jobs) == DIALOGUE_DELEGATE_PER_SESSION

    other_session = await submit_delegation(
        npc_name="路人Y", trigger=TRIGGER_KEY_INFO, reason="别的会话", session_id="sess_other",
    )
    assert other_session.accepted is True
    assert len(fake_worker.jobs) == DIALOGUE_DELEGATE_PER_SESSION + 1


# ---------------------------------------------------------------------------
# 注册闸与工具形状（§4.0.4：注册了才提，不注册一个字都不提）
# ---------------------------------------------------------------------------

def test_register_gate_hides_tool(store, fake_worker, bound_slot):
    """未绑定存档或后台正在改对话时，工具整件不下发。"""
    get_integration_state().unbind()
    assert delegate_tool_or_none(include_task_fallback=True) is None      # 未绑定存档

    bound_slot()
    tool = delegate_tool_or_none(include_task_fallback=True)
    assert tool is not None and tool["function"]["name"] == DELEGATE_TOOL_NAME
    assert tool["function"]["parameters"]["required"] == ["change_type", "reason"]

    fake_worker.running = True
    assert delegate_tool_or_none(include_task_fallback=True) is None      # 已有子 Agent 在写覆盖层
    fake_worker.running = False
    assert delegate_tool_or_none(include_task_fallback=True) is not None


def test_register_gate_counts_running_not_queued(bound_slot, fake_worker):
    """注册闸口径＝正在执行（``worker.running``）：排队中的委派不挡注册，串行由队列自己保证。"""
    bound_slot()
    fake_worker.jobs.append(object())                        # 已入队、尚未执行
    assert fake_worker.running is False
    assert delegate_tool_or_none(include_task_fallback=True) is not None


def test_task_fallback_wording_appears_only_with_task_context():
    """「任务兜底」那几句和 task_id 参数只在当轮有已完成带标记委托时才出现。"""
    with_task = json.dumps(build_delegate_tool(include_task_fallback=True), ensure_ascii=False)
    without = json.dumps(build_delegate_tool(include_task_fallback=False), ensure_ascii=False)

    assert "task_id" in with_task and TRIGGER_IMPORTANT_TASK in with_task
    assert "当你首次获知某任务已经完成" in with_task
    assert "task_id" not in without
    assert TRIGGER_IMPORTANT_TASK not in without
    assert "当你首次获知" not in without
    # 三种通用类型两种形状都给全（否则模型选不到该走的那条）
    for kind in (TRIGGER_RELATIONSHIP, TRIGGER_ATTITUDE, TRIGGER_KEY_INFO):
        assert kind in without
    assert json.loads(without)["function"]["parameters"]["properties"]["change_type"]["enum"] == \
        [TRIGGER_RELATIONSHIP, TRIGGER_ATTITUDE, TRIGGER_KEY_INFO]


# ---------------------------------------------------------------------------
# 委派材料（触发装配给子 Agent 的那几块）
# ---------------------------------------------------------------------------

async def test_task_material_reaches_worker_job(task_root, db_path, store, fake_worker, bound_slot):
    """触发一交给后台的材料：任务详情＋发布那场对话（摘要＋摘要后原文）＋发布时挂的指示。"""
    bound_slot()
    await store.add_message("sess_publish", "user", "我要订一把刀。")
    await store.add_message("sess_publish", "assistant", "刀要等龙鳞。")
    await store.save_summary("sess_publish", "玩家来订刀，我让他先去寻龙鳞。", 2)
    await store.add_message("sess_publish", "user", "龙鳞我带来了。")
    await store.add_message("sess_publish", "assistant", "好，剩下的交给我。")

    outcome = await handle_task_completed(slot_key=SLOT, task_id=TASK_ID)
    assert outcome.accepted is True

    job = fake_worker.jobs[0]
    assert (job.npc_name, job.slot_key, job.source_trigger, job.source_task_id) == \
        (NPC, SLOT, TRIGGER_IMPORTANT_TASK, TASK_ID)
    assert job.job_id == outcome.job_id

    # 权威 NPC 取自落盘记录的 get_npc，而不是事件里带的名字
    assert "取回祖传铁锤" in job.task_block
    assert "任务说明：锤子在废矿坑，路上当心。" in job.task_block
    assert "报酬：精铁×3、800 金币" in job.task_block
    assert "任务前：" in job.task_block and "任务后：" in job.task_block
    assert "任务达成内容（玩家达成）：通关废矿坑（普通）" in job.task_block
    # 聊天侧那句「勿重复调用」提醒不进子 Agent 材料（上一批修的泄漏，钉住不回潮）
    assert MARKED_TASK_HINT not in job.task_block

    assert job.instruction == "语气更敬重些"                       # 发布时挂的指示兜底进委派
    assert "重要委托" in job.reason and f"任务 id {TASK_ID}" in job.reason

    assert "【当时的对话摘要】" in job.publish_block
    assert "玩家来订刀，我让他先去寻龙鳞。" in job.publish_block
    assert "【摘要之后的对话原文】" in job.publish_block
    assert "龙鳞我带来了。" in job.publish_block
    assert "我要订一把刀。" not in job.publish_block                # 摘要之前的原文不再重复拼


async def test_apply_delegate_calls_lands_tool_invocation(store, fake_worker, bound_slot):
    """触发三：模型在聊天轮里的委派调用逐条落地，结果文本回给兜底轮；无关调用忽略。"""
    bound_slot()
    calls = [{
        "id": "c1", "type": "function",
        "function": {
            "name": DELEGATE_TOOL_NAME,
            "arguments": json.dumps({
                "change_type": TRIGGER_ATTITUDE,
                "reason": "他当着我的面把虎妙佣兵的套装拆了",
                "instruction": "少用敬语，说话冲一点",
            }, ensure_ascii=False),
        },
    }]
    results = await apply_delegate_calls(calls, npc_name=NPC, session_id="sess_chat")

    assert results == [f"{TRIGGER_ATTITUDE}：已受理"]
    job = fake_worker.jobs[0]
    assert (job.source_trigger, job.instruction, job.slot_key) == (TRIGGER_ATTITUDE, "少用敬语，说话冲一点", SLOT)

    ignored = await apply_delegate_calls(
        [{"id": "c2", "type": "function", "function": {"name": "publish_task", "arguments": "{}"}}],
        npc_name=NPC, session_id="sess_chat",
    )
    assert ignored == []
    assert len(fake_worker.jobs) == 1                               # 非委派工具不触发改写

    bad_kind = await apply_delegate_calls(
        [{"id": "c3", "type": "function", "function": {
            "name": DELEGATE_TOOL_NAME, "arguments": json.dumps({"change_type": "不重要", "reason": "有事"})}}],
        npc_name=NPC, session_id="sess_chat",
    )
    assert len(bad_kind) == 1 and "未知的更改类型" in bad_kind[0]


# ---------------------------------------------------------------------------
# 触发二：好感度跨档提示的存续（只拼提示，调不调由模型决定）
# ---------------------------------------------------------------------------

async def test_relationship_hint_wording_and_rounds(store):
    """跨档提示文案逐字口径；持续三轮，用完或被模型调用过就消失。"""
    assert build_relationship_hint("陌生", "友好") == (
        "你的关系从 陌生 变为了 友好，"
        "可以考虑调用对话修改的子 Agent，调整你的对话措辞。"
    )

    hint = build_relationship_hint("友好", "挚友")
    await store.dialogue_hint_write("sess_h", hint, RELATIONSHIP_HINT_ROUNDS)
    left = RELATIONSHIP_HINT_ROUNDS
    while left > 0:
        read = await store.dialogue_hint_read("sess_h")
        assert read is not None and read.hint == hint and read.rounds_left == left
        left -= 1
        await store.dialogue_hint_write("sess_h", hint, left)       # 注入一轮就扣一轮
    assert await store.dialogue_hint_read("sess_h") is None         # 轮数用尽，不再打扰

    await store.dialogue_hint_write("sess_h", hint, RELATIONSHIP_HINT_ROUNDS)
    await store.dialogue_hint_clear("sess_h")                       # 模型已经调用过委派
    assert await store.dialogue_hint_read("sess_h") is None


async def test_relationship_hint_edges(store, db_path):
    """没有该会话、空文案、会话 id 缺失都读不到提示；同会话重写＝覆盖不叠加。"""
    assert await store.dialogue_hint_read("sess_none") is None
    assert await store.dialogue_hint_read("") is None

    await store.dialogue_hint_write("", "没有会话 id 的提示", 3)
    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_dialogue_hints").fetchone()[0] == 0

    await store.dialogue_hint_write("sess_e", "   ", 3)
    assert await store.dialogue_hint_read("sess_e") is None

    await store.dialogue_hint_write("sess_e", "第一次的提示", 3)
    await store.dialogue_hint_write("sess_e", "换档后的新提示", 2)
    read = await store.dialogue_hint_read("sess_e")
    assert (read.hint, read.rounds_left) == ("换档后的新提示", 2)
    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_dialogue_hints").fetchone()[0] == 1
