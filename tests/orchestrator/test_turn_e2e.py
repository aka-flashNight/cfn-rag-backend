"""TurnOrchestrator 七条主路径 e2e（对应 docs/v3-developer/03 §9、09-前端适配说明 §2）。

Fake LLM 脚本化驱动：纯聊天 / 发任务（含中间结果）/ 确认 / 取消 / 讨价还价 /
岔开话题保留与过期 / 搜索；另覆盖宽限路径、confirm 失败补救。
"""

from __future__ import annotations

import json

import pytest

from services.orchestrator.turn import TurnOrchestrator
from tests.orchestrator.conftest import (
    MERGE_REPLY_TASK,
    NPC_NAME,
    VALID_DRAFT,
    collect,
    compressed_kinds,
    contents_of,
    notices_of,
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


# ---------------------------------------------------------------------------
# 1. 纯聊天：meta → content×N → done
# ---------------------------------------------------------------------------

async def test_pure_chat(env):
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":null}',
        text="诶，是你啊。今天有什么想聊的？",
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    events = await collect(_orch(env, "你好呀"))

    assert events[0].event == "meta"
    meta = events[0].data
    assert meta["emotion"] == "微笑"
    assert meta["favorability_change"] == 0
    assert meta["favorability"] == 30  # fav=0 无变化，meta 仍先于正文
    assert meta["relationship_level"] == "熟悉"

    assert events[-1].event == "done"
    assert events[-1].data["session_id"] == env.session_id
    assert events[-1].data["usage"]["prompt_tokens"] == 100

    body = "".join(contents_of(events))
    assert body == "诶，是你啊。今天有什么想聊的？"
    # 记忆落库：user + assistant（纯台词）
    history = await env.deps.memory.get_history(env.session_id)
    assert history[-2].role == "user" and history[-2].content == "你好呀"
    assert history[-1].role == "assistant" and history[-1].content == body


async def test_meta_without_meta_line_falls_back(env):
    """无 meta 行：默认情绪放行正文，不丢内容。"""
    env.fake.add_stream(text="嗯？有事吗？")
    events = await collect(_orch(env, "在吗"))
    assert events[0].event == "meta"
    assert events[0].data["emotion"] == "普通"
    assert "".join(contents_of(events)) == "嗯？有事吗？"


# ---------------------------------------------------------------------------
# 2. 发任务（含校验失败 → 中间结果过渡 → 修复成功 → 汇合）
# ---------------------------------------------------------------------------

async def test_task_draft_with_interim_rule(env):
    env.deps.merge_grace_ms = 200
    env.deps.subagent_timeout_s = 5
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":{"kind":"task_draft"}}',
        text="哦？想找活干？我看看手头有什么适合你的……",
        tool_calls=[tc("prepare_task_context", {
            "task_type": "资源收集",
            "reward_types": {"regular": ["金币"], "optional": []},
        }, call_id="p1")],
    )
    # TaskRunner（情形 1，候选池已预取）：非法草案（V1）→ 合法草案
    env.fake.add_chat(delay=0.05, tool_calls=[tc("draft_agent_task", {
        "task_type": "资源收集",
        "title": "食材收集委托",
        "rewards": [{"item_name": "不存在的神器", "count": 1}],
    })])
    env.fake.add_chat(delay=0.5, tool_calls=[tc("draft_agent_task", {
        "task_type": "资源收集",
        "title": "食材收集委托",
        "rewards": [{"item_name": "金币", "count": 25000}],
    })])
    # 过渡语 #2a（中间结果，只说一次、不含数字）+ 汇合 #2b
    env.fake.add_stream(text="委托内容还在调整，稍等一下。")
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "最近有什么活给我干吗？"))

    # 时序：meta → content(过渡) → agent_status(drafting) → agent_status(repairing)
    #       → content(过渡语) → agent_status(done) → content(详述) → system_notice → done
    assert events[0].event == "meta"
    phases = phases_of(events)
    assert phases[0] == ("task", "drafting")
    assert ("task", "repairing") in phases
    assert phases[-1] == ("task", "done")

    # 中间结果规则：过渡语只说一次、不含任何数字
    all_text = "".join(contents_of(events))
    assert all_text.count("稍等") == 1
    interim_chunks = [c for c in contents_of(events) if "稍等" in c]
    assert "".join(interim_chunks) == "委托内容还在调整，稍等一下。"
    assert not any(ch.isdigit() for ch in "".join(interim_chunks))

    # 汇合详述 + 系统通知
    assert MERGE_REPLY_TASK in all_text
    assert "任务草案已拟定，等待确认" in notices_of(events)
    assert events[-1].event == "done"

    # 草案入库（bargain_count 独立列，草案 JSON 不含内部字段）
    row = await env.deps.memory.get_draft(env.session_id)
    assert row is not None
    assert row.draft["rewards"] == [{"item_name": "金币", "count": 25000}]
    assert row.bargain_count == 0
    assert "bargain_count" not in row.draft
    assert "_draft_commit_valid" not in row.draft


async def test_task_draft_within_grace_no_interim(env):
    """宽限路径：TaskRunner 在宽限内完成 → 无过渡语直接汇合（03 §9.3）。"""
    env.deps.merge_grace_ms = 500
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":{"kind":"task_draft"}}',
        text="我想想给你安排点什么……",
        tool_calls=[tc("prepare_task_context", {
            "task_type": "资源收集",
            "reward_types": {"regular": ["金币"], "optional": []},
        }, call_id="p1")],
    )
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", {
        "task_type": "资源收集",
        "title": "食材收集委托",
        "rewards": [{"item_name": "金币", "count": 25000}],
    })])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "有什么活吗？"))
    assert phases_of(events) == [("task", "drafting"), ("task", "done")]
    assert all("稍等" not in c for c in contents_of(events))
    assert MERGE_REPLY_TASK in "".join(contents_of(events))
    assert "任务草案已拟定，等待确认" in notices_of(events)


# ---------------------------------------------------------------------------
# 3. 确认（HITL，同步动作）
# ---------------------------------------------------------------------------

async def test_task_confirm_success(env, monkeypatch):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    # confirm 落盘写入重定向到临时目录（不污染真实资源）
    monkeypatch.setattr(env.game_data, "data_root", env.tmp_path / "data")

    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":{"kind":"task_confirm"}}',
        text="好，就接了！",
    )
    env.fake.add_chat(content=json.dumps({
        "title": "食材收集委托",
        "description": "收集食材并提交给铁匠，报酬金币。",
        "get_dialogue": [{"name": NPC_NAME, "title": "铁匠", "emotion": "", "text": "这份委托就交给你了。"}],
        "finish_dialogue": [{"name": NPC_NAME, "title": "铁匠", "emotion": "", "text": "干得漂亮，报酬一分不少。"}],
    }, ensure_ascii=False))

    events = await collect(_orch(env, "接！"))

    assert compressed_kinds(events) == [
        "meta", "tool_status", "tool_status", "content", "system_notice", "done",
    ]
    running = events[1].data
    assert running["tool"] == "confirm_agent_task" and running["status"] == "running"
    success = events[2].data
    assert success["status"] == "success"
    assert "".join(contents_of(events)) == "好，就接了！"
    assert notices_of(events) == ["委托已发布"]

    # 草案清除 + 任务文件原子写入临时目录
    assert await env.store.get_draft(env.session_id) is None
    tasks_file = env.tmp_path / "data" / "task" / "agent_tasks.json"
    assert tasks_file.exists()
    doc = json.loads(tasks_file.read_text(encoding="utf-8"))
    assert any(200001 <= t["id"] <= 300000 for t in doc["tasks"])
    # 计数归零（任务轮触碰）
    assert env.store.get_rounds_without_task_sync(env.session_id) == 0


async def test_confirm_failure_discards_body_and_remediates(env, monkeypatch):
    """confirm 失败：丢弃已生成正文 → 补救调用 → system_notice 真实原因（03 §9.4）。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    import services.orchestrator.turn as turn_module
    from services.agent_tools.handlers import DraftOpOutcome

    def failing_confirm(*args, **kwargs):
        return DraftOpOutcome(
            result_json=json.dumps({"status": "error", "message": "草案校验未通过"}, ensure_ascii=False),
            payload={"status": "error", "message": "草案校验未通过"},
        )

    monkeypatch.setattr(turn_module, "execute_confirm_agent_task", failing_confirm)

    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":{"kind":"task_confirm"}}',
        text="好，就接了！",
    )
    env.fake.add_stream(text="这份委托刚才出了点问题，改天再谈吧。")

    events = await collect(_orch(env, "接！"))

    kinds = [e.event for e in events]
    # 正文被丢弃：tool_status(failed) 之前不允许出现 content；补救话术在其后
    assert kinds.index("content") > kinds.index("tool_status")
    assert events[2].data["status"] == "failed"
    assert "好，就接了！" not in "".join(contents_of(events))
    assert "草案校验未通过" in notices_of(events)[0]
    assert "这份委托刚才出了点问题" in "".join(contents_of(events))
    assert events[-1].event == "done"
    # 草案保留（失败不清草案）
    assert await env.store.get_draft(env.session_id) is not None


# ---------------------------------------------------------------------------
# 4. 取消
# ---------------------------------------------------------------------------

async def test_task_cancel(env):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0,"act":{"kind":"task_cancel"}}',
        text="那算了，不勉强。",
    )
    events = await collect(_orch(env, "算了不接了"))

    assert compressed_kinds(events) == [
        "meta", "tool_status", "tool_status", "content", "system_notice", "done",
    ]
    assert events[2].data["tool"] == "cancel_agent_task"
    assert notices_of(events) == ["委托已取消"]
    assert await env.store.get_draft(env.session_id) is None


# ---------------------------------------------------------------------------
# 5. 讨价还价（update 模式，bargain_count 独立列）
# ---------------------------------------------------------------------------

async def test_task_bargain_update(env):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0,"act":{"kind":"task_update","note":"玩家嫌奖励少"}}',
        text="嗯？你说说看。",
    )
    env.fake.add_chat(tool_calls=[tc("update_task_draft", {
        "draft_id": "testd001",
        "modify_fields": {"rewards": [{"item_name": "金币", "count": 25000}]},
    })])
    env.fake.add_stream(text="新方案：金币25000，这下行了吧？接还是不接？")

    events = await collect(_orch(env, "奖励太少了，加点"))

    assert phases_of(events) == [("task", "drafting"), ("task", "done")]
    assert "任务草案已更新，等待确认" in notices_of(events)
    row = await env.store.get_draft(env.session_id)
    assert row.draft["rewards"] == [{"item_name": "金币", "count": 25000}]
    assert row.bargain_count == 1  # 讨价还价计数（独立列）


# ---------------------------------------------------------------------------
# 6. 岔开话题：草案保留与过期取消
# ---------------------------------------------------------------------------

async def test_draft_expiry_after_keep_turns(env):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    for i in range(3):
        env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text=f"闲聊第{i}句。")

    # 第 1、2 次岔开：草案保留
    events1 = await collect(_orch(env, "今天天气如何"))
    assert notices_of(events1) == []
    assert await env.store.get_draft(env.session_id) is not None
    assert env.store.get_rounds_without_task_sync(env.session_id) == 1

    events2 = await collect(_orch(env, "你吃过饭了吗"))
    assert notices_of(events2) == []
    assert await env.store.get_draft(env.session_id) is not None
    assert env.store.get_rounds_without_task_sync(env.session_id) == 2

    # 第 3 次岔开：达到 draft_keep_turns → 自动取消 + 通知
    events3 = await collect(_orch(env, "算了聊点别的"))
    assert "过期的委托草案已取消" in notices_of(events3)
    assert await env.store.get_draft(env.session_id) is None
    assert events3[-1].event == "done"


# ---------------------------------------------------------------------------
# 7. 搜索并行
# ---------------------------------------------------------------------------

async def test_search_parallel(env):
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0,"act":{"kind":"search","query":"安迪的过去"}}',
        text="安迪啊……让我想想。",
    )
    env.fake.add_chat(tool_calls=[tc("search_knowledge", {"keyword": "安迪 过去"})])
    env.fake.add_chat(content="安迪曾在废城担任佣兵，后来加入了A兵团。")
    env.fake.add_stream(text="安迪以前是A兵团的佣兵，这事我知道一些。")

    events = await collect(_orch(env, "你知道安迪以前的事吗？"))

    phases = phases_of(events)
    assert phases == [("search", "searching"), ("search", "done")]
    assert "安迪以前是A兵团的佣兵" in "".join(contents_of(events))
    assert notices_of(events) == []  # 搜索无草案，无系统通知
    assert events[-1].event == "done"
    assert await env.store.get_draft(env.session_id) is None


# ---------------------------------------------------------------------------
# 汇合失败路径（极端：子 Agent 失败 → 人设话术，不发 error）
# ---------------------------------------------------------------------------

async def test_task_draft_failed_reply(env):
    """TaskRunner 失败（LLM 工具调用异常）：agent_status(failed) + 人设话术 + done。"""
    env.deps.merge_grace_ms = 50
    env.deps.subagent_timeout_s = 2
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":{"kind":"task_draft"}}',
        text="我想想……",
    )

    async def boom(req):
        raise RuntimeError("model exploded")

    env.fake.chat = boom  # TaskRunner 第一轮即异常
    env.fake.add_stream(text="今天手头的事都派完了，改天吧。")

    events = await collect(_orch(env, "给我个活"))
    phases = phases_of(events)
    assert ("task", "failed") in phases
    assert "今天手头的事都派完了" in "".join(contents_of(events))
    assert not any(e.event == "error" for e in events)  # 不甩错误码
    assert await env.store.get_draft(env.session_id) is None


@pytest.fixture
def _unused():
    return None

# ---------------------------------------------------------------------------
# 情形 2：交流阶段未调用 prepare → 任务 Agent 先重 prepare 再 draft（工具动态收窄）
# ---------------------------------------------------------------------------

async def test_task_draft_reprepare_when_prepare_missing(env):
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0,"act":{"kind":"task_draft"}}',
        text="我想想给你安排点什么……",  # 模型忘了调 prepare
    )
    env.fake.add_chat(tool_calls=[tc("prepare_task_context", {
        "task_type": "资源收集",
        "reward_types": {"regular": ["金币"], "optional": []},
    }, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", {
        "task_type": "资源收集",
        "title": "食材收集委托",
        "rewards": [{"item_name": "金币", "count": 25000}],
    })])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "有什么活吗？"))
    # 工具动态收窄：轮 1 只有 prepare，轮 2 只有 draft
    from services.llm import ChatRequest as _CR

    tool_names = [
        [t["function"]["name"] for t in (req.tools or [])]
        for req in env.fake.chat_requests
    ]
    assert tool_names[0] == ["prepare_task_context"]
    assert tool_names[1] == ["draft_agent_task"]
    assert phases_of(events) == [("task", "drafting"), ("task", "done")]
    assert MERGE_REPLY_TASK in "".join(contents_of(events))
    row = await env.deps.memory.get_draft(env.session_id)
    assert row is not None and row.draft["rewards"] == [{"item_name": "金币", "count": 25000}]


async def test_confirm_args_sanitizer_fills_placeholder(env):
    """confirm 发布文本归一化：LLM 缺对话时补占位，不回炉重生成（流程可用优先）。"""
    orch = _orch(env, "接！")
    args = orch._sanitize_confirm_args({
        "title": "食材收集委托",
        "description": "收集食材",
        "get_dialogue": [{"name": "铁匠", "text": "交给你了。"}],
        "finish_dialogue": "不是数组",
    })
    assert args is not None
    assert args["get_dialogue"][0]["text"] == "交给你了。"
    assert args["finish_dialogue"][0]["text"] == "干得漂亮，报酬一分不少。"  # 占位补齐
