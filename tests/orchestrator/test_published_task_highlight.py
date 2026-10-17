"""发布后持续供给「任务达成要点」（工作项 1）。

口径（改这块先读）：
- 发布轮：块拼进本轮 **result_block**，并登记供给期（rounds_left＝总轮数-1）；
- 后续轮：块进 **user 侧共享 core**（已完成 → 进行中 → 刚发布）；轮次只存在于会话库，
  提示词里**不出现**「第几轮/共几轮」；
- 任务已完成＝立刻清记录、不再注入；记录只在会话库，不落 status/progress/npc_state_db；
- 同一轮 user core 不重复注入（发布轮已进 result_block）。
"""

from __future__ import annotations

import json

import pytest

from core.config import get_settings
from services.integration.save_paths import agent_save_task_paths, shared_task_paths
from services.integration.state import get_integration_state
from services.orchestrator.turn import TurnOrchestrator
from tests.orchestrator.conftest import (
    NPC_NAME,
    VALID_DRAFT,
    collect,
    tc,
)

SLOT = "slot_hl"
BLOCK_HEAD = "【你刚发布的委托（玩家尚未完成）】"

CONFIRM_ARGS = {
    "title": "食材收集委托",
    "description": "收集食材并提交给铁匠，报酬金币。",
    "get_dialogue": [{"name": NPC_NAME, "title": "铁匠", "emotion": "", "text": "这份委托就交给你了。"}],
    "finish_dialogue": [{"name": NPC_NAME, "title": "铁匠", "emotion": "", "text": "干得漂亮，报酬一分不少。"}],
}


@pytest.fixture
def integration_state():
    state = get_integration_state()
    state.unbind()
    yield state
    state.unbind()


@pytest.fixture
def data_root(env, monkeypatch):
    """任务写盘落点重定向到临时目录（不污染真实资源）。"""
    root = env.tmp_path / "data"
    monkeypatch.setattr(env.game_data, "data_root", root)
    return root


def _orch(env, query: str) -> TurnOrchestrator:
    return TurnOrchestrator(session_id=env.session_id, npc_name=NPC_NAME, query=query, deps=env.deps)


def _bind(integration_state, root, **progress) -> None:
    integration_state.bind(
        slot_key=SLOT, character={"name": "林三"}, progress=progress,
    )


async def _publish(env) -> int:
    """走完一次 confirm 发布，返回新任务 id（绑定存档落槽目录，未绑定落共享文件）。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    env.fake.add_stream(meta='{"emo":"微笑","fav":0}', text="", tool_calls=[tc("route_task_confirm", {})])
    env.fake.add_chat(content=json.dumps(CONFIRM_ARGS, ensure_ascii=False))
    env.fake.add_stream(text="这事就这么定了，去办吧。")
    await collect(_orch(env, "接！"))

    root = env.tmp_path / "data"
    slot = get_integration_state().slot_key
    tasks_path, _ = (
        agent_save_task_paths(root, slot) if slot else shared_task_paths(root)
    )
    doc = json.loads(tasks_path.read_text(encoding="utf-8"))
    return int(doc["tasks"][0]["id"])


async def _chat(env, query: str = "刚才那份委托怎么样了？") -> str:
    """跑一轮纯聊天，返回本轮 user 侧共享 core（messages[1]）。"""
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, query))
    return env.fake.stream_requests[-1].messages[1]["content"]


async def test_publish_turn_injects_into_result_block(env, integration_state, data_root):
    """发布轮就要给：块进收场调用的结果块，同轮 user core 不重复。"""
    _bind(integration_state, data_root)
    task_id = await _publish(env)

    reply_messages = env.fake.stream_requests[1].messages
    reply_prompt = reply_messages[-1]["content"]
    assert BLOCK_HEAD in reply_prompt
    assert "任务达成内容（玩家达成）" in reply_prompt
    assert "完成 NPC：" in reply_prompt
    assert "不要重复发布同类委托" in reply_prompt
    # 同一轮的 user core 里没有第二份
    assert BLOCK_HEAD not in reply_messages[1]["content"]

    record = env.store.published_task_highlight_read_sync(env.session_id)
    assert record is not None
    assert record.task_id == task_id
    assert record.rounds_left == get_settings().published_task_highlight_turns - 1


async def test_following_turns_inject_then_expire(env, integration_state, data_root, monkeypatch):
    """后续轮进 user core，每轮递减；用尽后整块消失，且文本不带轮次计数。"""
    monkeypatch.setattr(get_settings(), "published_task_highlight_turns", 3)
    _bind(integration_state, data_root)
    await _publish(env)

    core = await _chat(env)
    assert BLOCK_HEAD in core
    assert "此前你发布的委托「食材收集委托」" in core
    # 文本稳定：不暴露「第几轮/共几轮」
    assert "轮" not in BLOCK_HEAD
    assert env.store.published_task_highlight_read_sync(env.session_id).rounds_left == 1

    core = await _chat(env)
    assert BLOCK_HEAD in core
    assert env.store.published_task_highlight_read_sync(env.session_id) is None

    core = await _chat(env)
    assert BLOCK_HEAD not in core


async def test_finished_task_clears_highlight_immediately(env, integration_state, data_root):
    """玩家已完成＝立刻撤掉记录，不再按「尚未完成」说话。"""
    _bind(integration_state, data_root)
    task_id = await _publish(env)
    assert env.store.published_task_highlight_read_sync(env.session_id) is not None

    _bind(integration_state, data_root, tasks_finished=[task_id])
    core = await _chat(env)
    assert BLOCK_HEAD not in core
    assert env.store.published_task_highlight_read_sync(env.session_id) is None


async def test_active_block_does_not_repeat_highlight_task(env, integration_state, data_root):
    """highlight 期内同一 task_id 不再被「尚在途中」块列一遍；别的在途委托照常列出。"""
    _bind(integration_state, data_root)
    task_id = await _publish(env)

    # 再造一条更早发布的在途委托（直接写进本存档文件，模拟历史数据）
    tasks_path, text_path = agent_save_task_paths(data_root, SLOT)
    doc = json.loads(tasks_path.read_text(encoding="utf-8"))
    doc["tasks"].append({
        "id": task_id + 1,
        "title": f"$AGENT_TITLE_{task_id + 1}",
        "get_npc": NPC_NAME,
        "finish_npc": NPC_NAME,
        "chain": "委托",
    })
    tasks_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    texts = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else {}
    texts[f"$AGENT_TITLE_{task_id + 1}"] = "护送商队"
    text_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")

    _bind(integration_state, data_root, tasks_active=[task_id, task_id + 1])
    core = await _chat(env)

    assert BLOCK_HEAD in core
    assert "【你发布的委托尚在途中（玩家尚未完成）】" in core
    assert "护送商队" in core
    # highlight 的那条只在「刚发布」块出现一次
    assert core.count("食材收集委托") == 1
    assert f"任务 id {task_id}）仍在进行中" not in core
    assert core.count("仍在进行中") == 1  # 只有那条历史委托进「尚在途中」


async def test_bound_but_no_data_root_keeps_record(env, integration_state, monkeypatch, tmp_path):
    """数据根不可用＝不注入也**不消耗轮次**，等下一轮再说。"""
    monkeypatch.setattr(env.game_data, "data_root", tmp_path / "data")
    _bind(integration_state, tmp_path / "data")
    await _publish(env)
    before = env.store.published_task_highlight_read_sync(env.session_id).rounds_left

    monkeypatch.setattr(env.game_data, "data_root", None)
    core = await _chat(env)
    assert BLOCK_HEAD not in core
    after = env.store.published_task_highlight_read_sync(env.session_id)
    assert after is not None and after.rounds_left == before


async def test_standalone_publish_writes_shared_and_still_highlights(env, data_root):
    """未绑定存档也能用：发布落共享文件，块照样进 result_block，但记录照常写（无槽位依赖）。"""
    await _publish(env)
    reply_prompt = env.fake.stream_requests[1].messages[-1]["content"]
    assert BLOCK_HEAD in reply_prompt
    assert (env.tmp_path / "data" / "task" / "agent_tasks.json").exists()
