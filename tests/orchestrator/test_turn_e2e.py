"""TurnOrchestrator 主路径 e2e（对应 docs/v3-developer/03 §9、09-前端适配说明 §2）。

Fake LLM 脚本化驱动：纯聊天 / 发任务（含过渡语）/ 确认 / 取消 / 讨价还价 /
岔开话题保留与过期 / 搜索；另覆盖宽限路径、confirm 失败补救、prepare 晚到。

v3 再次重构（路由工具化）后的口径：
- 调用 #1 的 meta 行只有 emo/fav，委派意图走 route_* 工具调用；
- 路由轮**不输出台词**，所以任务/确认/取消路径的第一段正文都来自后台办完后的
  第二次（或第三次）调用；
- prepare 由 TaskRunner 自己调，聊天主 Agent 的工具表里只有五个路由声明。
"""

from __future__ import annotations

import json
import logging

from services.orchestrator.turn import TurnOrchestrator
from services.subagents.prompts import build_direction_vague_note
from services.orchestrator.prompts import task_merge_hint
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

PREPARE_COLLECT = {
    "task_type": "资源收集",
    "reward_types": {"regular": ["金币"], "optional": []},
}
DRAFT_VALID = {
    "task_type": "资源收集",
    "title": "食材收集委托",
    "rewards": [{"item_name": "金币", "count": 25000}],
}


def _orch(env, query: str) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id,
        npc_name=NPC_NAME,
        query=query,
        deps=env.deps,
    )


def _route(env, name: str, args: dict | None = None, meta: str = '{"emo":"微笑","fav":0}'):
    """登记一次「只路由不说话」的调用 #1。"""
    env.fake.add_stream(meta=meta, text="", tool_calls=[tc(name, args or {})])


def _tool_names_per_round(env) -> list[list[str]]:
    """TaskRunner / SearchRunner 每轮拿到的工具表（动态收窄的证据）。"""
    return [
        [t["function"]["name"] for t in (req.tools or [])]
        for req in env.fake.chat_requests
    ]


# ---------------------------------------------------------------------------
# 1. 纯聊天：meta → content×N → done
# ---------------------------------------------------------------------------

async def test_pure_chat(env):
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0}',
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


async def test_routing_turn_that_still_speaks_is_logged(env, caplog):
    """模型在路由轮违规说话：正文照发（不能吞玩家已看到的内容），但必须留痕。"""
    env.deps.merge_grace_ms = 500
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0}',
        text="我想想给你安排点什么……",
        tool_calls=[tc("route_task_draft", {"note": "玩家想找点活干"})],
    )
    env.fake.add_chat(tool_calls=[tc("prepare_task_context", PREPARE_COLLECT, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", DRAFT_VALID)])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    with caplog.at_level(logging.WARNING, logger="services.orchestrator.turn"):
        events = await collect(_orch(env, "有什么活吗？"))

    assert "仍输出了台词" in caplog.text
    assert "我想想给你安排点什么" in "".join(contents_of(events))


# ---------------------------------------------------------------------------
# 2. 发任务（prepare → 校验失败 → 过渡语报方向 → 修复成功 → 汇合）
# ---------------------------------------------------------------------------

async def test_task_draft_with_interim_rule(env):
    env.deps.merge_grace_ms = 200
    env.deps.subagent_timeout_s = 5
    _route(env, "route_task_draft", {"note": "玩家想找点活干"})
    # TaskRunner 轮 1：prepare（选型由任务 Agent 自己决定）
    env.fake.add_chat(delay=0.05, tool_calls=[tc("prepare_task_context", PREPARE_COLLECT, call_id="p1")])
    # 轮 2：非法草案（奖励物品不在候选内）→ 校验失败
    env.fake.add_chat(delay=0.05, tool_calls=[tc("draft_agent_task", {
        "task_type": "资源收集",
        "title": "食材收集委托",
        "rewards": [{"item_name": "不存在的神器", "count": 1}],
    })])
    # 轮 3：合法草案
    env.fake.add_chat(delay=0.5, tool_calls=[tc("draft_agent_task", DRAFT_VALID)])
    # 过渡语 #2a（报方向、不含数字）+ 汇合 #2b
    env.fake.add_stream(text="缺些东西，得让人替我弄回来，稍等我一下。")
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "最近有什么活给我干吗？"))

    assert events[0].event == "meta"
    # prepare 完成推的是「拟定委托中」→ drafting；校验失败才是 repairing
    assert phases_of(events) == [
        ("task", "drafting"), ("task", "drafting"), ("task", "repairing"), ("task", "done"),
    ]

    # 中间结果规则：过渡语只说一次、不含任何数字
    all_text = "".join(contents_of(events))
    assert all_text.count("稍等") == 1
    interim_chunks = [c for c in contents_of(events) if "稍等" in c]
    assert "".join(interim_chunks) == "缺些东西，得让人替我弄回来，稍等我一下。"
    assert not any(ch.isdigit() for ch in "".join(interim_chunks))

    # 过渡语的种子是 prepare 选出的真实 task_type 对应的方向，不是「还在调整」
    interim_prompt = env.fake.stream_requests[1].messages[-1]["content"]
    assert build_direction_vague_note("资源收集") in interim_prompt
    assert "绝对不要" in interim_prompt

    # 汇合详述 + 按任务类型条件拼接的说明要点 + 系统通知
    assert MERGE_REPLY_TASK in all_text
    merge_prompt = env.fake.stream_requests[2].messages[-1]["content"]
    assert task_merge_hint("资源收集") in merge_prompt
    assert "任务草案已拟定，等待确认" in notices_of(events)
    assert events[-1].event == "done"

    # 草案入库（bargain_count 独立列，草案 JSON 不含内部字段）
    row = await env.deps.memory.get_draft(env.session_id)
    assert row is not None
    assert row.draft["rewards"] == [{"item_name": "金币", "count": 25000}]
    assert row.bargain_count == 0
    assert "bargain_count" not in row.draft
    assert "_draft_commit_valid" not in row.draft

    # 记忆分段：路由轮无正文 → 过渡语与汇合各存一条 assistant（前端两个气泡）
    history = await env.deps.memory.get_history(env.session_id)
    assert history[-3].role == "user" and history[-3].content == "最近有什么活给我干吗？"
    assert history[-2].content == "缺些东西，得让人替我弄回来，稍等我一下。"
    assert history[-1].content == MERGE_REPLY_TASK


async def test_task_draft_within_grace_no_interim(env):
    """宽限路径：TaskRunner 在宽限内完成 → 无过渡语直接汇合（03 §9.3）。"""
    env.deps.merge_grace_ms = 500
    _route(env, "route_task_draft", {"note": "玩家想找点活干"})
    env.fake.add_chat(tool_calls=[tc("prepare_task_context", PREPARE_COLLECT, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", DRAFT_VALID)])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "有什么活吗？"))
    assert phases_of(events) == [
        ("task", "drafting"), ("task", "drafting"), ("task", "done"),
    ]
    # 只有两次流式调用：路由 #1 + 汇合 #2（没有过渡语 #2a）
    assert len(env.fake.stream_requests) == 2
    assert MERGE_REPLY_TASK in "".join(contents_of(events))
    assert "任务草案已拟定，等待确认" in notices_of(events)


async def test_transition_line_when_prepare_finishes_after_grace(env):
    """prepare 晚于宽限期完成 → 过渡语照说（bridge 排水），玩家不会对着空白等草案。"""
    env.deps.merge_grace_ms = 50
    env.deps.subagent_timeout_s = 5
    _route(env, "route_task_draft", {"note": "玩家想找点活干"})
    env.fake.add_chat(delay=0.3, tool_calls=[tc("prepare_task_context", PREPARE_COLLECT, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", DRAFT_VALID)])
    env.fake.add_stream(text="缺些东西，得让人替我弄回来，稍等。")
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "有什么活吗？"))

    assert len(env.fake.stream_requests) == 3  # 路由 + 过渡 + 汇合
    assert build_direction_vague_note("资源收集") in env.fake.stream_requests[1].messages[-1]["content"]
    assert "缺些东西" in "".join(contents_of(events))
    assert MERGE_REPLY_TASK in "".join(contents_of(events))
    assert phases_of(events)[-1] == ("task", "done")


async def test_task_draft_narrows_tools_and_injects_pricing_card(env):
    """prepare 在任务 Agent 内：工具表轮内收窄，定价卡按真实 task_type 事后补投。"""
    env.deps.merge_grace_ms = 500
    _route(env, "route_task_draft", {"note": "玩家想找点活干"})
    env.fake.add_chat(tool_calls=[tc("prepare_task_context", PREPARE_COLLECT, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", DRAFT_VALID)])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    await collect(_orch(env, "有什么活吗？"))

    assert _tool_names_per_round(env) == [["prepare_task_context"], ["draft_agent_task"]]
    # launch 时 task_type 未知 → 定价卡本体不在 system 里（只有一句「候选回来后给你」）
    card_header = "【奖励定价规则（与后端校验完全一致，按此计算即可通过）】"
    first_round = env.fake.chat_requests[0].messages
    assert card_header not in first_round[0]["content"]
    # prepare 回来后追加一条 user 消息补投（不改写 system，前缀缓存不受影响）
    second_round = env.fake.chat_requests[1].messages
    assert second_round[0]["content"] == first_round[0]["content"]
    injected = [m for m in second_round if m["role"] == "user" and card_header in m["content"]]
    assert len(injected) == 1
    assert "资源收集" in injected[0]["content"]


# ---------------------------------------------------------------------------
# 3. 确认（同步动作：路由轮静默 → 办完 → NPC 收场）
# ---------------------------------------------------------------------------

async def test_task_confirm_success(env, monkeypatch):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    # confirm 落盘写入重定向到临时目录（不污染真实资源）
    monkeypatch.setattr(env.game_data, "data_root", env.tmp_path / "data")

    _route(env, "route_task_confirm")
    env.fake.add_chat(content=json.dumps({
        "title": "食材收集委托",
        "description": "收集食材并提交给铁匠，报酬金币。",
        "get_dialogue": [{"name": NPC_NAME, "title": "铁匠", "emotion": "", "text": "这份委托就交给你了。"}],
        "finish_dialogue": [{"name": NPC_NAME, "title": "铁匠", "emotion": "", "text": "干得漂亮，报酬一分不少。"}],
    }, ensure_ascii=False))
    env.fake.add_stream(text="这事就这么定了，去办吧。")

    events = await collect(_orch(env, "接！"))

    # 系统行排在 NPC 台词之后（先听人说、再看到系统行）
    assert compressed_kinds(events) == [
        "meta", "tool_status", "tool_status", "content", "system_notice", "done",
    ]
    running = events[1].data
    assert running["tool"] == "confirm_agent_task" and running["status"] == "running"
    success = events[2].data
    assert success["status"] == "success"
    assert "".join(contents_of(events)) == "这事就这么定了，去办吧。"
    assert notices_of(events) == ["委托已发布"]

    # 动作后回复的指令：不许再问一次接不接
    reply_prompt = env.fake.stream_requests[1].messages[-1]["content"]
    assert "不要再问一次他接不接" in reply_prompt

    # 草案清除 + 任务文件原子写入临时目录
    assert await env.store.get_draft(env.session_id) is None
    tasks_file = env.tmp_path / "data" / "task" / "agent_tasks.json"
    assert tasks_file.exists()
    doc = json.loads(tasks_file.read_text(encoding="utf-8"))
    assert any(200001 <= t["id"] <= 300000 for t in doc["tasks"])
    # 计数归零（任务轮触碰）
    assert env.store.get_rounds_without_task_sync(env.session_id) == 0


async def test_confirm_failure_notices_then_remediates(env, monkeypatch):
    """confirm 失败：system_notice 给真实原因 → 补救回复 → 草案保留（03 §9.4）。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    import services.orchestrator.turn as turn_module
    from services.agent_tools.handlers import DraftOpOutcome

    def failing_confirm(*args, **kwargs):
        return DraftOpOutcome(
            result_json=json.dumps({"status": "error", "message": "草案校验未通过"}, ensure_ascii=False),
            payload={"status": "error", "message": "草案校验未通过"},
        )

    monkeypatch.setattr(turn_module, "execute_confirm_agent_task", failing_confirm)

    _route(env, "route_task_confirm")
    env.fake.add_chat(content=json.dumps({"title": "食材收集委托"}, ensure_ascii=False))
    env.fake.add_stream(text="这份委托刚才出了点问题，改天再谈吧。")

    events = await collect(_orch(env, "接！"))

    kinds = [e.event for e in events]
    assert kinds.index("content") > kinds.index("system_notice")
    assert events[2].data["status"] == "failed"
    assert "草案校验未通过" in notices_of(events)[0]
    assert "这份委托刚才出了点问题" in "".join(contents_of(events))
    assert events[-1].event == "done"
    # 草案保留（失败不清草案）
    assert await env.store.get_draft(env.session_id) is not None


async def test_confirm_reply_falls_back_when_llm_silent(env, monkeypatch):
    """动作后回复生成失败/空回复 → 静态兜底话术，玩家侧不留空白。"""
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    monkeypatch.setattr(env.game_data, "data_root", env.tmp_path / "data")

    _route(env, "route_task_confirm")
    env.fake.add_chat(content=json.dumps({"title": "食材收集委托"}, ensure_ascii=False))
    env.fake.add_stream(text="")  # 模型没吐出任何正文

    events = await collect(_orch(env, "接！"))
    assert "".join(contents_of(events)) == "这事就这么定了，去办吧。"
    assert notices_of(events) == ["委托已发布"]


# ---------------------------------------------------------------------------
# 4. 取消
# ---------------------------------------------------------------------------

async def test_task_cancel(env):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    _route(env, "route_task_cancel", meta='{"emo":"普通","fav":0}')
    env.fake.add_stream(text="行，那这事就算了。")

    events = await collect(_orch(env, "算了不接了"))

    assert compressed_kinds(events) == [
        "meta", "tool_status", "tool_status", "content", "system_notice", "done",
    ]
    assert events[2].data["tool"] == "cancel_agent_task"
    assert notices_of(events) == ["委托已取消"]
    assert await env.store.get_draft(env.session_id) is None


async def test_task_cancel_without_draft_is_not_a_failure(env):
    """无草案可撤：不算失败，NPC 顺着话头自然回应，不发系统行。"""
    _route(env, "route_task_cancel", meta='{"emo":"普通","fav":0}')
    env.fake.add_stream(text="嗯？我这儿可没什么活儿要办。")

    events = await collect(_orch(env, "那个活我不干了"))

    assert events[2].data["status"] == "success"
    assert notices_of(events) == []
    assert "".join(contents_of(events)) == "嗯？我这儿可没什么活儿要办。"


# ---------------------------------------------------------------------------
# 5. 讨价还价（update 模式，bargain_count 独立列）
# ---------------------------------------------------------------------------

async def test_task_bargain_update(env):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)
    _route(env, "route_task_update", {"note": "玩家嫌奖励少"}, meta='{"emo":"普通","fav":0}')
    env.fake.add_chat(tool_calls=[tc("update_task_draft", {
        "draft_id": "testd001",
        "modify_fields": {"rewards": [{"item_name": "金币", "count": 25000}]},
    })])
    env.fake.add_stream(text="新方案：金币25000，这下行了吧？接还是不接？")

    events = await collect(_orch(env, "奖励太少了，加点"))

    assert phases_of(events) == [("task", "drafting"), ("task", "done")]
    assert _tool_names_per_round(env) == [["update_task_draft"]]
    assert "任务草案已更新，等待确认" in notices_of(events)
    row = await env.store.get_draft(env.session_id)
    assert row.draft["rewards"] == [{"item_name": "金币", "count": 25000}]
    assert row.bargain_count == 1  # 讨价还价计数（独立列）


async def test_task_update_without_draft_degrades_to_new_draft(env):
    """路由错发 task_update 而实际无草案 → 降级为新草案，不让回合空转。"""
    env.deps.merge_grace_ms = 500
    _route(env, "route_task_update", {"note": "玩家想要点活"}, meta='{"emo":"普通","fav":0}')
    env.fake.add_chat(tool_calls=[tc("prepare_task_context", PREPARE_COLLECT, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", DRAFT_VALID)])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "给我个活"))

    assert _tool_names_per_round(env) == [["prepare_task_context"], ["draft_agent_task"]]
    assert "任务草案已拟定，等待确认" in notices_of(events)
    assert await env.store.get_draft(env.session_id) is not None


# ---------------------------------------------------------------------------
# 6. 岔开话题：草案保留与过期取消
# ---------------------------------------------------------------------------

async def test_draft_expiry_after_keep_turns(env):
    await env.store.upsert_draft(env.session_id, dict(VALID_DRAFT), bargain_count=0)

    for i in range(3):
        env.fake.add_stream(meta='{"emo":"普通","fav":0}', text=f"闲聊第{i}句。")

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
# 7. 搜索
# ---------------------------------------------------------------------------

async def test_search_parallel(env):
    _route(env, "route_search", {"query": "安迪的过去"}, meta='{"emo":"普通","fav":0}')
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
    _route(env, "route_task_draft", {"note": "玩家想要个活"})

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


# ---------------------------------------------------------------------------
# 工具/参数解析的容错
# ---------------------------------------------------------------------------

async def test_route_with_broken_arguments_still_dispatches(env):
    """路由参数是坏 JSON：路由本身仍生效，note 按空处理（回落到玩家原话）。"""
    env.deps.merge_grace_ms = 500
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":0}',
        text="",
        tool_calls=[{
            "id": "c0",
            "type": "function",
            "function": {"name": "route_task_draft", "arguments": "{不是 JSON"},
        }],
    )
    env.fake.add_chat(tool_calls=[tc("prepare_task_context", PREPARE_COLLECT, call_id="p1")])
    env.fake.add_chat(tool_calls=[tc("draft_agent_task", DRAFT_VALID)])
    env.fake.add_stream(text=MERGE_REPLY_TASK)

    events = await collect(_orch(env, "有什么活吗？"))

    assert "任务草案已拟定，等待确认" in notices_of(events)
    # 玩家原话进了任务 Agent 的方向块
    assert "有什么活吗？" in env.fake.chat_requests[0].messages[1]["content"]


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
